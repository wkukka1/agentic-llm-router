"""Build LLM profiles: `model_id -> profile_text` (+ structured facts).

A profile is a short natural-language description of a model, in the style of the
NIRT / IRT-Router "LLM descriptions". It has up to two parts:

1. **curated** -- a hand-written ``feature`` sentence plus structured facts
   (context window, params, release, price) from ``configs/model_profiles.yaml``,
   or a template built from the model registry when no entry exists.
2. **empirical** (optional, ``profiles.include_empirical``) -- one paragraph
   summarising the model's *observed* behaviour in this dataset: overall
   correctness, correctness by benchmark family, multiple-choice skill above
   chance, mean cost, and pairwise win-rate. Set ``include_empirical: false``
   for description-only profiles exactly as in the paper.

The profile text is later embedded by the ``irt`` encoder pathway (BERT); the
resulting vector is what Phase 1 uses to initialise ``theta_m`` for cold-start
models. Nothing here trains or fits anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config, section
from ..data import schemas
from ..data.model_registry import model_info
from ..data.response_matrix import read_responses

PROFILE_COLUMNS = [
    "model_id", "model_name", "provider", "family", "version",
    "has_curated", "feature", "profile_text", "profile_version",
    "n_observations", "n_datasets", "structured", "empirical",
]


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #
def _profiles_cfg(cfg: Config) -> dict:
    return section(cfg, "profiles")


def load_profile_entries(cfg: Config) -> dict:
    pc = _profiles_cfg(cfg)
    path = cfg.resolve(pc.get("config_path", "configs/model_profiles.yaml"))
    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("profiles", {}) or {}


def _task_family_map(cfg: Config) -> dict[str, str]:
    pc = _profiles_cfg(cfg)
    fams = pc.get("task_families", {}) or {}
    out: dict[str, str] = {}
    for family, names in fams.items():
        for n in names:
            out[str(n).lower()] = family
    return out


def _family_of(dataset: str, fam_map: dict[str, str]) -> Optional[str]:
    d = str(dataset).lower()
    if d in fam_map:
        return fam_map[d]
    for key, fam in fam_map.items():
        if key in d:
            return fam
    return None


# --------------------------------------------------------------------------- #
# empirical stats                                                             #
# --------------------------------------------------------------------------- #
def empirical_stats(responses: pd.DataFrame, model_id: str, cfg: Config) -> dict:
    fam_map = _task_family_map(cfg)
    g = responses[responses["model_id"] == model_id]
    if g.empty:
        return {}

    corr = g[g["metric_type"].isin(schemas.MetricType.CORRECTNESS)].copy()
    stats: dict = {
        "n_observations": int(len(g)),
        "n_datasets": int(g["dataset"].nunique()),
    }

    if not corr.empty:
        eff = corr["corrected_score"].where(
            corr["corrected_score"].notna(),
            pd.to_numeric(corr["score"], errors="coerce"),
        )
        corr = corr.assign(_eff=pd.to_numeric(eff, errors="coerce"))
        stats["overall_correctness"] = round(float(corr["_eff"].mean()), 4)
        corr["_family"] = corr["dataset"].map(lambda d: _family_of(d, fam_map))
        by_fam = (
            corr.dropna(subset=["_family"])
            .groupby("_family")["_eff"].mean().round(4).to_dict()
        )
        if by_fam:
            stats["correctness_by_family"] = {k: float(v) for k, v in by_fam.items()}

        mc = corr[corr["is_multiple_choice"].fillna(False) & corr["n_choices"].notna()]
        if not mc.empty:
            chance = 1.0 / mc["n_choices"].astype(float)
            raw = pd.to_numeric(mc["score"], errors="coerce")
            stats["mc_skill_above_chance"] = round(float((raw - chance).mean()), 4)
            stats["mc_n"] = int(len(mc))

    cost = pd.to_numeric(g["cost"], errors="coerce").dropna()
    if not cost.empty:
        stats["mean_cost"] = round(float(cost.mean()), 6)
        stats["median_cost"] = round(float(cost.median()), 6)

    pw = g[g["metric_type"].isin(schemas.MetricType.PAIRWISE)]
    if not pw.empty:
        wr: dict[str, dict] = {}
        for src, sg in pw.groupby("source"):
            s = pd.to_numeric(sg["score"], errors="coerce")
            wr[str(src)] = {
                "win_rate": round(float(s.mean()), 4),   # wins + 0.5*ties
                "n": int(len(sg)),
            }
        stats["pairwise"] = wr

    return stats


# --------------------------------------------------------------------------- #
# text rendering                                                              #
# --------------------------------------------------------------------------- #
_FAM_LABEL = {
    "math": "math", "code": "coding", "knowledge": "knowledge",
    "reasoning": "reasoning", "language_zh": "Chinese-language", "chat": "chat",
}


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def render_profile_text(
    model_id: str,
    entry: Optional[dict],
    stats: dict,
    include_empirical: bool = True,
) -> tuple[str, str, dict]:
    """Return (feature_text, full_profile_text, structured_facts)."""
    info = model_info(model_id)
    entry = entry or {}

    feature = _clean(entry.get("feature", "")) if entry.get("feature") else ""
    if not feature:
        bits = [f"{info.model_name}"]
        if info.provider and info.provider != "unknown":
            bits.append(f"is a language model from {info.provider}")
        else:
            bits.append("is a language model")
        if info.family:
            bits.append(f"in the {info.family} family")
        feature = _clean(" ".join(bits)) + "."

    structured = {
        "params_b": entry.get("params_b"),
        "context_window": entry.get("context_window"),
        "release": entry.get("release"),
        "modality": entry.get("modality"),
        "input_price_per_1k": entry.get("input_price_per_1k"),
        "output_price_per_1k": entry.get("output_price_per_1k"),
        "provider": info.provider,
        "family": info.family,
        "version": info.version,
    }

    facts: list[str] = []
    if structured["params_b"]:
        facts.append(f"approximately {structured['params_b']}B parameters")
    if structured["context_window"]:
        facts.append(f"{int(structured['context_window'])}-token context window")
    if structured["release"]:
        facts.append(f"released {structured['release']}")
    if structured["input_price_per_1k"] is not None:
        facts.append(
            f"priced at ${structured['input_price_per_1k']}/${structured['output_price_per_1k']} "
            f"per 1K input/output tokens"
        )
    facts_sent = (" It has " + ", ".join(facts) + ".") if facts else ""

    emp_sent = ""
    if include_empirical and stats:
        parts: list[str] = []
        if "overall_correctness" in stats:
            parts.append(f"overall correctness {stats['overall_correctness']:.2f}")
        by_fam = stats.get("correctness_by_family") or {}
        if by_fam:
            fam_txt = ", ".join(
                f"{_FAM_LABEL.get(k, k)} {v:.2f}"
                for k, v in sorted(by_fam.items(), key=lambda kv: -kv[1])
            )
            parts.append(f"by area {fam_txt}")
        if "mc_skill_above_chance" in stats:
            parts.append(
                f"multiple-choice skill above chance {stats['mc_skill_above_chance']:+.2f}"
            )
        if "mean_cost" in stats:
            parts.append(f"mean cost per query ${stats['mean_cost']:.4f}")
        pw = stats.get("pairwise") or {}
        for src, d in pw.items():
            label = "human-preference" if "arena" in src else "LLM-judge"
            parts.append(f"{label} win-rate {d['win_rate']:.2f} (n={d['n']})")
        if parts:
            emp_sent = (
                f" Observed behaviour in this dataset across {stats.get('n_datasets', 0)} "
                f"tasks: " + "; ".join(parts) + "."
            )

    full = _clean(feature + facts_sent + emp_sent)
    return feature, full, structured


# --------------------------------------------------------------------------- #
# build                                                                       #
# --------------------------------------------------------------------------- #
def build_model_profiles(
    cfg: Config, responses: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if responses is None:
        responses = read_responses(cfg)
    pc = _profiles_cfg(cfg)
    include_emp = bool(pc.get("include_empirical", True))
    version = int(pc.get("version", 1))
    entries = load_profile_entries(cfg)

    model_ids = sorted(responses["model_id"].dropna().unique().tolist())
    rows = []
    for mid in model_ids:
        stats = empirical_stats(responses, mid, cfg)
        entry = entries.get(mid)
        feature, full, structured = render_profile_text(
            mid, entry, stats, include_empirical=include_emp
        )
        info = model_info(mid)
        rows.append(
            {
                "model_id": mid,
                "model_name": info.model_name,
                "provider": info.provider,
                "family": info.family,
                "version": info.version,
                "has_curated": bool(entry and entry.get("feature")),
                "feature": feature,
                "profile_text": full,
                "profile_version": version,
                "n_observations": stats.get("n_observations", 0),
                "n_datasets": stats.get("n_datasets", 0),
                "structured": structured,
                "empirical": stats,
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def write_model_profiles(df: pd.DataFrame, cfg: Config) -> Path:
    out_dir = cfg.path("processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_profiles.parquet"
    enc = df.copy()
    for col in ("structured", "empirical"):
        enc[col] = enc[col].map(lambda v: json.dumps(v, default=str))
    enc.to_parquet(path, index=False)
    return path


def read_model_profiles(cfg: Config) -> pd.DataFrame:
    path = cfg.path("processed") / "model_profiles.parquet"
    df = pd.read_parquet(path)
    for col in ("structured", "empirical"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) and v else {})
    return df

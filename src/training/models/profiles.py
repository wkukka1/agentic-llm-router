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

from router.config import Config, section
from ..data import schemas
from ..data.families import family_of as _family_of
from ..data.families import task_family_map as _task_family_map
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


# --------------------------------------------------------------------------- #
# empirical stats                                                             #
# --------------------------------------------------------------------------- #
def empirical_stats(responses: pd.DataFrame, model_id: str, cfg: Config, *,
                    group: Optional[pd.DataFrame] = None) -> dict:
    """``group`` (this model's rows, pre-split by the caller) skips the full-frame
    filter -- :func:`build_model_profiles` passes one ``groupby`` group per model."""
    fam_map = _task_family_map(cfg)
    g = group if group is not None else responses[responses["model_id"] == model_id]
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
    in_price, out_price = structured["input_price_per_1k"], structured["output_price_per_1k"]
    if in_price is not None and out_price is not None:
        facts.append(f"priced at ${in_price}/${out_price} per 1K input/output tokens")
    elif in_price is not None:
        facts.append(f"priced at ${in_price} per 1K input tokens")
    elif out_price is not None:
        facts.append(f"priced at ${out_price} per 1K output tokens")
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
    # default off: the empirical addendum summarises ALL responses (validation /
    # test outcomes and cold-start models' own rows) and would leak them into
    # the embedded profile a missing key used to silently enable it.
    include_emp = bool(pc.get("include_empirical", False))
    version = int(pc.get("version", 1))
    entries = load_profile_entries(cfg)

    model_ids = sorted(responses["model_id"].dropna().unique().tolist())
    groups = dict(tuple(responses.groupby("model_id", sort=False)))
    rows = []
    for mid in model_ids:
        stats = empirical_stats(responses, mid, cfg, group=groups.get(mid))
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


# --------------------------------------------------------------------------- #
# embeddings                                                                  #
# --------------------------------------------------------------------------- #
def build_profile_embeddings(
    cfg: Config,
    pathway: Optional[str] = None,
    *,
    restart: bool = False,
    text_column: str = "profile_text",
):
    """Embed each model's profile text (:func:`build_model_profiles`, built if
    missing) with the ``pathway`` encoder -> a ``model_id -> vector``
    :class:`~router.embeddings.EmbeddingStore`.

    ``text_column`` ``"feature"`` embeds the curated description only;
    ``"profile_text"`` (default) includes the empirical addendum when
    ``profiles.include_empirical`` is set.
    """
    from router.embeddings import available_pathways, build_store, default_store_dir, load_encoder

    profiles = _current_profiles(cfg)
    profiles = profiles.sort_values("model_id").reset_index(drop=True)

    pw = pathway or cfg.get("embedding.default_pathway") or available_pathways(cfg)[0]
    encoder = load_encoder(cfg, pathway=pw)
    out_dir = default_store_dir(cfg, "model_profile", pw)
    # require_fingerprints: a store whose texts (or text_column) differ is
    # re-encoded -- profile stores are tiny, so a legacy store is simply rebuilt
    return build_store(
        out_dir, profiles["model_id"].tolist(), profiles[text_column].tolist(), encoder,
        id_field="model_id", resume=not restart, force=restart,
        manifest_extra={"text_column": text_column}, require_fingerprints=True,
    )


def _current_profiles(cfg: Config) -> pd.DataFrame:
    """Profiles rendered from the CURRENT config + responses.

    Re-renders every call (cheap) and rewrites ``model_profiles.parquet`` only
    when the rendered text differs, so editing ``configs/model_profiles.yaml``,
    toggling ``profiles.include_empirical`` or rebuilding ``responses.parquet``
    is picked up instead of reusing the stale parquet.
    """
    try:
        existing = read_model_profiles(cfg)
    except FileNotFoundError:
        existing = None
    try:
        fresh = build_model_profiles(cfg)
    except FileNotFoundError:
        if existing is None:
            raise
        return existing          # no responses table to re-render from
    key = ["model_id", "feature", "profile_text"]
    if existing is None or not (
        existing[key].sort_values("model_id").reset_index(drop=True)
        .equals(fresh[key].sort_values("model_id").reset_index(drop=True))
    ):
        write_model_profiles(fresh, cfg)
    return fresh

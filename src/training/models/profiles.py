"""Build LLM profiles: `model_id -> profile_text` (+ structured facts).

A profile is a short natural-language description of a model, in the style of the
NIRT / IRT-Router "LLM descriptions". It has up to three parts:

1. **curated** -- a hand-written ``feature`` sentence plus structured facts
   (context window, params, release, price) from ``configs/model_profiles.yaml``,
   or a template built from the model registry when no entry exists.
2. **public benchmarks + editorial** (``benchmarks``, ``strengths``,
   ``weaknesses``) -- :func:`resolve_benchmarks` merges whichever of the HF
   Open LLM Leaderboard (primary), Chatbot Arena Elo, and BenchLM
   (supplementary) an entry opts into (via ``hf_repo_id``/``arena_name``/
   ``benchlm_slug``), plus a manual YAML fallback; ``strengths``/
   ``weaknesses`` are hand-written editorial bullets. None of this is derived
   from this project's own response data -- see
   ``training.data.public_benchmarks`` -- so it can describe a model being
   tested zero-shot without leaking its answer key.
3. **empirical** (optional, ``profiles.include_empirical``) -- one paragraph
   summarising the model's *observed* behaviour in THIS dataset: overall
   correctness, correctness by benchmark family, multiple-choice skill above
   chance, mean cost, and pairwise win-rate. Set ``include_empirical: false``
   for description-only profiles exactly as in the paper. Never conflated
   with part 2 -- this is the one part that IS project-data-derived and
   leakage-sensitive.

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
from ..data.public_benchmarks import (
    BENCHLM_ATTRIBUTION,
    arena_elo_rating,
    benchlm_scores,
    hf_open_llm_leaderboard,
    merge_benchmarks,
)
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


# public (non-empirical) benchmark key -> human-readable label for rendering.
_BENCH_LABEL = {
    "mmlu": "MMLU", "gsm8k": "GSM8K", "arc": "ARC", "hellaswag": "HellaSwag",
    "truthfulqa": "TruthfulQA", "winogrande": "WinoGrande", "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA", "musr": "MUSR", "ifeval": "IFEval", "bbh": "BBH", "math_lvl5": "MATH Lvl 5",
    "arena_elo": "Chatbot Arena Elo", "overall": "BenchLM overall", "agentic": "BenchLM agentic",
    "coding": "BenchLM coding", "reasoning": "BenchLM reasoning", "knowledge": "BenchLM knowledge",
    "multimodal_grounded": "BenchLM multimodal",
}


def _benchmark_label(key: str) -> str:
    return _BENCH_LABEL.get(key, key.replace("_", " ").title())


def _benchmarks_sentence(benchmarks: dict) -> str:
    """``benchmarks`` (:func:`resolve_benchmarks`'s merged, structured output)
    -> one rendered sentence, with the BenchLM attribution its CC BY-NC 4.0
    license requires appended once if any key came from that source."""
    if not benchmarks:
        return ""
    parts: list[str] = []
    uses_benchlm = False
    for key in sorted(benchmarks):
        rec = benchmarks[key]
        txt = f"{_benchmark_label(key)} {rec['value']}"
        if rec.get("unit") == "elo" and rec.get("snapshot_date"):
            txt += f" ({rec['snapshot_date']} snapshot)"
        parts.append(txt)
        uses_benchlm = uses_benchlm or rec.get("source") == "benchlm"
    sent = " Public benchmark scores: " + ", ".join(parts) + "."
    if uses_benchlm:
        sent += f" {BENCHLM_ATTRIBUTION}."
    return sent


def _editorial_sentence(entry: dict) -> str:
    """Hand-written ``strengths``/``weaknesses`` -> one rendered sentence pair.
    Kept visually/structurally distinct from benchmark facts -- these are a
    subjective editorial prior, not an objective measurement (see
    ``structured["strengths_source"]``/``weaknesses_source"``)."""
    parts: list[str] = []
    strengths = entry.get("strengths") or []
    weaknesses = entry.get("weaknesses") or []
    if strengths:
        parts.append("Strengths: " + ", ".join(strengths) + ".")
    if weaknesses:
        parts.append("Weaknesses: " + ", ".join(weaknesses) + ".")
    return (" " + " ".join(parts)) if parts else ""


# Profile ablation variants (1d): which already-resolved pieces get folded
# into the RENDERED TEXT -- `structured` always stays the complete record
# regardless of variant; only `feature`/`full` differ, so each variant can be
# embedded (build_profile_embeddings(text_column=...)) and compared to
# isolate which ingredient actually moves downstream eval numbers.
#   benchmarks: True = all, False = none, "no_arena_elo" / "arena_elo_only" = filtered.
VARIANTS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")
_VARIANT_SPECS = {
    "P0": {"benchmarks": False, "editorial": False, "redact_name": False},
    "P1": {"benchmarks": True, "editorial": False, "redact_name": False},
    "P2": {"benchmarks": False, "editorial": True, "redact_name": False},
    "P3": {"benchmarks": True, "editorial": True, "redact_name": False},   # full/default
    "P4": {"benchmarks": "no_arena_elo", "editorial": True, "redact_name": False},  # Arena-Elo-blind
    "P5": {"benchmarks": "arena_elo_only", "editorial": False, "redact_name": False},
    "P6": {"benchmarks": True, "editorial": True, "redact_name": True},
}
_REDACTED_FEATURE = "This model is an AI language model."


def _filter_benchmarks_for_variant(benchmarks: dict, mode) -> dict:
    if mode is False:
        return {}
    if mode == "no_arena_elo":
        return {k: v for k, v in benchmarks.items() if k != "arena_elo"}
    if mode == "arena_elo_only":
        return {k: v for k, v in benchmarks.items() if k == "arena_elo"}
    return dict(benchmarks)


def variant_benchmarks(benchmarks: dict, variant: str) -> dict:
    """Public wrapper over :func:`_filter_benchmarks_for_variant` -- which
    benchmark keys a given ablation variant actually renders into text.
    Used by ``evaluation.nirt.pairwise_coldstart``'s E4 guard to assert
    ``"arena_elo" not in variant_benchmarks(structured["benchmarks"], "P4")``
    before scoring, rather than trusting the filter silently."""
    return _filter_benchmarks_for_variant(benchmarks, _VARIANT_SPECS[variant]["benchmarks"])


def render_profile_text(
    model_id: str,
    entry: Optional[dict],
    stats: dict,
    include_empirical: bool = True,
    benchmarks: Optional[dict] = None,
    variant: str = "P3",
) -> tuple[str, str, dict]:
    """Return (feature_text, full_profile_text, structured_facts).

    ``benchmarks`` is :func:`resolve_benchmarks`'s already-resolved, merged
    output (public, non-project benchmark scores) -- resolving it requires
    I/O (HTTP/HF Hub fetches), so it is computed once by the caller
    (:func:`build_model_profiles`), exactly like ``stats`` (the data-derived
    empirical block) is computed by the caller rather than here.

    ``variant`` (one of :data:`VARIANTS`, default ``"P3"`` = today's full
    profile) selects which pieces are folded into the RENDERED TEXT for the
    ablation study (1d) -- ``structured`` is always the complete record
    regardless of variant.
    """
    info = model_info(model_id)
    entry = entry or {}
    benchmarks = benchmarks or {}
    spec = _VARIANT_SPECS[variant]

    if spec["redact_name"]:
        feature = _REDACTED_FEATURE
    elif entry.get("feature"):
        feature = _clean(entry["feature"])
    else:
        feature = ""
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
        "benchmarks": benchmarks,
        "benchmark_scores": {k: v["value"] for k, v in benchmarks.items()},
        "benchmark_scores_source": {k: v["source"] for k, v in benchmarks.items()},
        "strengths": list(entry.get("strengths") or []),
        "weaknesses": list(entry.get("weaknesses") or []),
        "strengths_source": "editorial",
        "weaknesses_source": "editorial",
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

    bench_sent = _benchmarks_sentence(_filter_benchmarks_for_variant(benchmarks, spec["benchmarks"]))
    edit_sent = _editorial_sentence(entry) if spec["editorial"] else ""
    full = _clean(feature + facts_sent + bench_sent + edit_sent + emp_sent)
    return feature, full, structured


# --------------------------------------------------------------------------- #
# public benchmark resolution                                                 #
# --------------------------------------------------------------------------- #
def _manual_benchmark_record(raw: Optional[dict], default_source: str = "manual") -> dict:
    """A hand-transcribed YAML block (``openlm_scores:`` / ``model_card_scores:``
    / the generic last-resort ``benchmark_scores:``) -> the same
    structured-record shape the automated sources return. Each value may be a
    bare number (wrapped with a ``percent``/``default_source`` default) or
    already a ``{value, unit, source, version, snapshot_date}`` dict for a
    model whose author wants to record more provenance by hand. There is no
    automated fetch for OpenLM/model-card tiers -- OpenLM.ai has no API or
    data export (confirmed directly), and a model card is definitionally
    hand-read -- so both are entered the same way as the final-fallback tier,
    just under their own YAML key so :func:`resolve_benchmarks` can slot them
    into their own precedence position."""
    if not raw:
        return {}
    out: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            out[key] = {
                "value": float(value["value"]), "unit": value.get("unit", "percent"),
                "source": value.get("source", default_source), "version": value.get("version"),
                "snapshot_date": value.get("snapshot_date"),
            }
        else:
            out[key] = {"value": float(value), "unit": "percent", "source": default_source,
                        "version": None, "snapshot_date": None}
    return out


def resolve_benchmarks(entry: Optional[dict], cfg: Optional[Config] = None) -> dict:
    """Merge every public benchmark source configured for this profile entry.

    None of these are derived from this project's own response data, so none
    of them can leak into a zero-shot cold-start/generalization test (see
    ``training.data.public_benchmarks``). Precedence (only matters on an
    actual key collision -- in practice the sources mostly use disjoint key
    namespaces):

    1. ``openlm_scores:`` -- hand-transcribed from OpenLM.ai's leaderboard
       (no API/export exists there, confirmed directly -- this is why it's a
       manual YAML block like model cards, not an automated lookup).
    2. ``model_card_scores:`` -- hand-transcribed from the model's own
       official announcement/technical report.
    3. ``benchlm_slug:`` -> BenchLM (automated, supplementary -- covers
       proprietary/API-only models no open-weight leaderboard can).
    4. ``hf_repo_id:`` -> HF Open LLM Leaderboard (automated, open-weight
       models only).
    5. ``benchmark_scores:`` -- generic last-resort manual fallback.

    ``arena_name:`` -> Chatbot Arena Elo is a separate, orthogonal signal
    (not a general capability benchmark, and the deliberate target of the
    Arena-Elo-blind ablation variant) -- always resolved if configured,
    outside this five-tier chain entirely.
    """
    entry = entry or {}
    openlm_rec = _manual_benchmark_record(entry.get("openlm_scores"), default_source="openlm")
    model_card_rec = _manual_benchmark_record(entry.get("model_card_scores"),
                                              default_source="model_card")
    benchlm_rec = benchlm_scores(entry["benchlm_slug"], cfg=cfg) if entry.get("benchlm_slug") else None
    hf_rec = hf_open_llm_leaderboard(entry["hf_repo_id"], cfg=cfg) if entry.get("hf_repo_id") else None
    manual_rec = _manual_benchmark_record(entry.get("benchmark_scores"))
    arena_rec = arena_elo_rating(entry["arena_name"], cfg=cfg) if entry.get("arena_name") else None
    return merge_benchmarks(openlm_rec, model_card_rec, benchlm_rec, hf_rec, manual_rec, arena_rec)


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
        benchmarks = resolve_benchmarks(entry, cfg)
        feature, full, structured = render_profile_text(
            mid, entry, stats, include_empirical=include_emp, benchmarks=benchmarks,
        )
        info = model_info(mid)
        row = {
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
        # one rendered-text column per ablation variant (1d) -- same inputs,
        # re-rendered with a different subset folded in; cheap (no I/O: stats/
        # benchmarks are already resolved above) and gives each variant its
        # own build_profile_embeddings(text_column=...) store to compare.
        for variant in VARIANTS:
            _, variant_text, _ = render_profile_text(
                mid, entry, stats, include_empirical=include_emp, benchmarks=benchmarks,
                variant=variant,
            )
            row[f"profile_text_{variant}"] = variant_text
        rows.append(row)
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS + [f"profile_text_{v}" for v in VARIANTS])


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


def build_variant_profile_embeddings(
    cfg: Config,
    pathway: Optional[str] = None,
    variant: str = "P3",
    *,
    restart: bool = False,
):
    """:func:`build_profile_embeddings` for one ablation variant (1d) ->
    its OWN persistent store, keyed as if ``kind="model_profile_{variant}"``.

    ``default_store_dir`` keys storage only on ``(kind, pathway)``, not
    ``text_column`` -- calling :func:`build_profile_embeddings` directly
    with a different ``text_column`` under the SAME pathway would silently
    overwrite the previous variant's store in place, which would make E3's
    "compare all seven variants" sweep impossible (only the last-built one
    would survive on disk). Suffixing ``kind`` by variant instead of
    ``pathway`` keeps the real encoder pathway (e.g. ``"irt"``) intact --
    only the storage location changes, not which encoder runs.
    """
    from router.embeddings import available_pathways, build_store, default_store_dir, load_encoder

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")

    profiles = _current_profiles(cfg)
    profiles = profiles.sort_values("model_id").reset_index(drop=True)

    pw = pathway or cfg.get("embedding.default_pathway") or available_pathways(cfg)[0]
    encoder = load_encoder(cfg, pathway=pw)
    out_dir = default_store_dir(cfg, f"model_profile_{variant}", pw)
    text_column = f"profile_text_{variant}"
    return build_store(
        out_dir, profiles["model_id"].tolist(), profiles[text_column].tolist(), encoder,
        id_field="model_id", resume=not restart, force=restart,
        manifest_extra={"text_column": text_column, "variant": variant}, require_fingerprints=True,
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

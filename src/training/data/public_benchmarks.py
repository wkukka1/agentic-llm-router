"""Public (non-project) benchmark score lookups for LLM profiles.

Every lookup here is sourced from somewhere outside this project's own
response data, so none of it can leak into a zero-shot cold-start /
generalization test the way the data-derived ``empirical`` block in
``training.models.profiles`` could (see that module's docstring).

Three *automated* sources live here -- ``benchlm_scores`` (supplementary:
proprietary/API-only models no open-weight leaderboard can index),
``hf_open_llm_leaderboard`` (open-weight models only), and
``arena_elo_rating`` (its own, orthogonal signal, outside the general
benchmark precedence chain). They are the 3rd/4th/separate tiers of the full
five-tier precedence ``training.models.profiles.resolve_benchmarks``
assembles -- OpenLM.ai (tier 1) and official model cards (tier 2) rank higher
but have no API/export to automate (OpenLM.ai's leaderboard has neither,
confirmed directly), so they're hand-transcribed into YAML instead; see that
function's docstring for the full ordering. Every value returned here is a
structured record ``{"value": float, "unit": str, "source": str, "version":
Optional[str], "snapshot_date": Optional[str]}``, never a bare float, so
provenance (which benchmark version, which dated snapshot) is never silently
lost; :func:`merge_benchmarks` unions these -- not overrides -- since in
practice they report disjoint key namespaces.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from router.config import Config, load_config
from training.data.download import fetch as _download

Record = dict[str, dict]

# BenchLM data is CC BY-NC 4.0 -- every profile that uses it must carry this
# credit (configs/model_profiles.yaml rendering + docs/data_sources.md).
BENCHLM_ATTRIBUTION = "Data from BenchLM.ai (https://benchlm.ai)"

# ``open-llm-leaderboard/contents`` column name -> our benchmark key. Only
# columns present and non-null for a given row are ever returned -- a missing
# benchmark is omitted, never fabricated as 0. Verified against a live
# download (2026-10-07, 4576 rows): the dataset's CURRENT benchmark suite is
# v3 (IFEval/BBH/MATH Lvl 5/GPQA/MUSR/MMLU-PRO) -- it has no MMLU/GSM8K/ARC/
# HellaSwag/TruthfulQA/Winogrande columns at all (that was a wrong v1-era
# assumption from before this was checked against the live schema).
_HF_LEADERBOARD_COLUMNS = {
    "IFEval": "ifeval", "BBH": "bbh", "MATH Lvl 5": "math_lvl5",
    "GPQA": "gpqa", "MUSR": "musr", "MMLU-PRO": "mmlu_pro",
}


def hf_leaderboard_record(row: dict) -> Record:
    """One ``open-llm-leaderboard/contents`` row -> ``{key: record}`` for
    whichever benchmark columns are present and not null."""
    out: Record = {}
    for col, key in _HF_LEADERBOARD_COLUMNS.items():
        value = row.get(col)
        if value is None:
            continue
        out[key] = {
            "value": float(value), "unit": "percent", "source": "hf_open_llm_leaderboard",
            "version": "v3", "snapshot_date": None,
        }
    return out


# BenchLM's own camelCase category key -> our snake_case convention. Verified
# against a live download of https://benchlm.ai/data/models.json (2026-10-07):
# real top-level shape is {"items": [...], "generatedAt": ..., ...}, NOT the
# {"models": [...]} this module originally (wrongly) assumed.
_BENCHLM_CATEGORY_KEYS = {
    "agentic": "agentic", "coding": "coding", "reasoning": "reasoning",
    "multimodalGrounded": "multimodal_grounded", "knowledge": "knowledge",
    "multilingual": "multilingual", "instructionFollowing": "instruction_following",
    "math": "math",
}


def benchlm_record(entry: dict, *, snapshot_date: Optional[str] = None) -> Record:
    """One BenchLM ``items[]`` entry -> ``{key: record}`` for the overall
    score plus whichever of its category scores are present.

    The item's own top-level ``displayScore`` is the authoritative overall
    score (the file's own ``scoreFields.displayScore.role`` is
    ``"published-overall-score"``) -- NOT ``scores.displayScore`` /
    ``scores.overallScore``, which the file itself labels
    ``"legacy-compatibility-score"`` and can disagree with substantially
    (seen live: 38.49 vs. 54 for the same model). Category scores live
    under ``scores.displayCategoryScores`` (camelCase keys).
    """
    version = entry.get("publicRankingMode")
    out: Record = {}
    overall = entry.get("displayScore")
    if overall is not None:
        out["overall"] = {"value": float(overall), "unit": "benchlm_score", "source": "benchlm",
                           "version": version, "snapshot_date": snapshot_date}
    categories = ((entry.get("scores") or {}).get("displayCategoryScores")) or {}
    for raw_key, value in categories.items():
        if value is None:
            continue
        key = _BENCHLM_CATEGORY_KEYS.get(raw_key, raw_key)
        out[key] = {"value": float(value), "unit": "benchlm_score", "source": "benchlm",
                    "version": version, "snapshot_date": snapshot_date}
    return out


def arena_elo_record(elo: float, *, snapshot_date: str) -> Record:
    """A raw Chatbot Arena Elo rating -> its structured record under key
    ``arena_elo``. ``snapshot_date`` must be pinned by the caller -- Elo is a
    live, continuously recalculated rating (see ``docs/data_sources.md`` §2's
    opponent-confounding caveat for the same reason raw battle rows must never
    mix snapshots)."""
    return {"arena_elo": {"value": float(elo), "unit": "elo", "source": "lmarena_hf",
                           "version": None, "snapshot_date": snapshot_date}}


def merge_benchmarks(*sources: Optional[Record]) -> Record:
    """Union every source's keys; on an actual key collision the EARLIEST
    (highest-precedence) source wins. In practice the three sources use
    disjoint key namespaces, so this is almost always a plain union -- the
    precedence rule only matters if two sources ever report the same key."""
    merged: Record = {}
    for source in sources:
        if not source:
            continue
        for key, record in source.items():
            merged.setdefault(key, record)
    return merged


# --------------------------------------------------------------------------- #
# fixture/cache-backed lookups                                               #
# --------------------------------------------------------------------------- #
def lookup_hf_leaderboard(hf_repo_id: str, *, table_path: Path) -> Optional[Record]:
    """``table_path``: a JSON file holding a list of ``open-llm-leaderboard/contents``
    rows (each with a ``fullname`` column), as cached by the fetch layer."""
    rows = json.loads(Path(table_path).read_text(encoding="utf-8"))
    for row in rows:
        if row.get("fullname") == hf_repo_id:
            rec = hf_leaderboard_record(row)
            return rec or None
    return None


def lookup_benchlm(benchlm_slug: str, *, models_path: Path) -> Optional[Record]:
    """``models_path``: the cached ``https://benchlm.ai/data/models.json`` file
    (real shape: ``{"items": [...], "generatedAt": "<ISO timestamp>", ...}``)."""
    data = json.loads(Path(models_path).read_text(encoding="utf-8"))
    generated_at = data.get("generatedAt")
    snapshot_date = generated_at[:10] if generated_at else None
    for entry in data.get("items", []):
        if entry.get("slug") == benchlm_slug:
            rec = benchlm_record(entry, snapshot_date=snapshot_date)
            return rec or None
    return None


def lookup_arena_elo(arena_name: str, *, table_path: Path, snapshot_date: str) -> Optional[Record]:
    """``table_path``: a cached, dated Chatbot Arena leaderboard CSV from the
    ``lmarena-ai/chatbot-arena-leaderboard`` HF Space -- one specific snapshot,
    never mixed with another date's file."""
    with Path(table_path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("Model") == arena_name:
                elo = row.get("Arena Elo rating")
                if elo in (None, "", "-"):
                    return None
                return arena_elo_record(float(elo), snapshot_date=snapshot_date)
    return None


# --------------------------------------------------------------------------- #
# fetch + cache (thin I/O glue -- the logic above is what's unit-tested)      #
# --------------------------------------------------------------------------- #
def _public_benchmarks_cfg(cfg: Config) -> Config:
    return cfg.get("public_benchmarks") or Config({})


def fetch_benchlm_models_json(cfg: Optional[Config] = None, *, force: bool = False) -> Path:
    cfg = cfg or load_config()
    pc = _public_benchmarks_cfg(cfg)
    dest = cfg.resolve(pc.get("benchlm_local_path", "data/raw/public_benchmarks/benchlm_models.json"))
    if force and dest.exists():
        dest.unlink()
    return _download("https://benchlm.ai/data/models.json", dest)


def fetch_arena_leaderboard_csv(cfg: Optional[Config] = None, *, snapshot_date: str = "2023-12-15",
                                 force: bool = False) -> Path:
    """Downloads one PINNED commit of the ``lmarena-ai/chatbot-arena-leaderboard``
    HF Space's leaderboard CSV -- ``revision``/``filename`` must be an explicit,
    human-verified (commit_sha, filename) pair in ``configs/phase0.yaml``'s
    ``public_benchmarks.arena_snapshots.<snapshot_date>``, never guessed from
    the date string, since the Space's current filename does not reliably
    follow a `leaderboard_table_<date>.csv` pattern across every snapshot."""
    cfg = cfg or load_config()
    pc = _public_benchmarks_cfg(cfg)
    snaps = pc.get("arena_snapshots") or Config({})
    snap = snaps.get(snapshot_date)
    if snap is None:
        raise KeyError(
            f"no public_benchmarks.arena_snapshots entry for {snapshot_date!r} in config -- "
            "add {revision, filename} verified against the lmarena-ai/chatbot-arena-leaderboard "
            "HF Space before using this snapshot"
        )
    dest = cfg.resolve(f"data/raw/public_benchmarks/arena_leaderboard_{snapshot_date}.csv")
    if force and dest.exists():
        dest.unlink()
    if not dest.exists():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id="lmarena-ai/chatbot-arena-leaderboard", repo_type="space",
                                filename=snap["filename"], revision=snap["revision"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
    return dest


def _restricted_unpickle(path: Path):
    """Load an ``elo_results_<date>.pkl`` from the Arena Space without running
    arbitrary code: only pandas/numpy reconstructors (and ``builtins.slice``)
    resolve; plotly figures -- the only other globals those files reference --
    become inert stubs. Anything else raises."""
    import importlib
    import pickle

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("plotly"):
                return _Stub
            if module == "builtins" and name == "slice":
                return slice
            if module in ("pandas", "numpy") or module.startswith(("pandas.", "numpy.")):
                return getattr(importlib.import_module(module), name)
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    with Path(path).open("rb") as fh:
        return _Unpickler(fh).load()


def arena_category_elo_rows(elo_results: dict) -> list[dict]:
    """Flatten an unpickled ``elo_results`` dict into long rows
    ``{category, arena_key, rating, rating_q025, rating_q975, num_battles}`` --
    one per (text category, model). Vision tables are skipped."""
    rows: list[dict] = []
    for category, block in (elo_results.get("text") or {}).items():
        table = block.get("leaderboard_table_df")
        if table is None:
            continue
        for key, r in table.iterrows():
            rows.append({"category": str(category), "arena_key": str(key),
                         "rating": float(r["rating"]),
                         "rating_q025": float(r["rating_q025"]),
                         "rating_q975": float(r["rating_q975"]),
                         "num_battles": int(r["num_battles"])})
    return rows


def fetch_arena_category_elo(cfg: Optional[Config] = None, *, snapshot_date: str,
                             filename: str, revision: str, force: bool = False) -> Path:
    """Per-category Arena Elo (full, coding, math, chinese, ...) for one PINNED
    snapshot, cached as a long CSV at
    ``data/raw/public_benchmarks/arena_category_elo_<snapshot_date>.csv``.

    From 2024 on the Space's ``leaderboard_table_<date>.csv`` no longer carries
    an Elo column -- ratings live only in ``elo_results_<date>.pkl``, loaded
    here through :func:`_restricted_unpickle`. ``filename``/``revision`` are an
    explicit pair from config, never guessed from the date."""
    cfg = cfg or load_config()
    dest = cfg.resolve(f"data/raw/public_benchmarks/arena_category_elo_{snapshot_date}.csv")
    if force and dest.exists():
        dest.unlink()
    if not dest.exists():
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id="lmarena-ai/chatbot-arena-leaderboard", repo_type="space",
                               filename=filename, revision=revision)
        rows = arena_category_elo_rows(_restricted_unpickle(Path(path)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dest, index=False)
    return dest


def fetch_hf_leaderboard_table(cfg: Optional[Config] = None, *, force: bool = False) -> Path:
    """Caches the ``open-llm-leaderboard/contents`` dataset as a flat JSON list
    of row dicts (each with a ``fullname`` column) for :func:`lookup_hf_leaderboard`."""
    cfg = cfg or load_config()
    pc = _public_benchmarks_cfg(cfg)
    dest = cfg.resolve(pc.get("hf_leaderboard_local_path",
                               "data/raw/public_benchmarks/hf_open_llm_leaderboard_contents.json"))
    if force and dest.exists():
        dest.unlink()
    if not dest.exists():
        from datasets import load_dataset

        ds = load_dataset("open-llm-leaderboard/contents", split="train")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(ds.to_list()), encoding="utf-8")
    return dest


def hf_open_llm_leaderboard(hf_repo_id: str, *, cfg: Optional[Config] = None) -> Optional[Record]:
    """Tier-4 automated benchmark source -- see module docstring. ``None``
    for any model never submitted to the leaderboard (every proprietary API
    model)."""
    return lookup_hf_leaderboard(hf_repo_id, table_path=fetch_hf_leaderboard_table(cfg))


def benchlm_scores(benchlm_slug: str, *, cfg: Optional[Config] = None) -> Optional[Record]:
    """Tier-3 automated benchmark source, for proprietary/API-only models the
    leaderboard structurally can't cover. ``None`` if BenchLM doesn't track this model
    (expected for every 2023-era model -- see the plan's Context)."""
    return lookup_benchlm(benchlm_slug, models_path=fetch_benchlm_models_json(cfg))


def arena_elo_rating(arena_name: str, *, snapshot_date: str = "2023-12-15",
                      cfg: Optional[Config] = None) -> Optional[Record]:
    """The public Chatbot Arena Elo rating, pinned to one dated snapshot so
    every model scored from it is comparable (never mix snapshot dates)."""
    path = fetch_arena_leaderboard_csv(cfg, snapshot_date=snapshot_date)
    return lookup_arena_elo(arena_name, table_path=path, snapshot_date=snapshot_date)

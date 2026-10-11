"""Real battles for LLMs this project has NEVER trained on -- not the
split-forced-cold models of the richer-profiles workstream (which still have
their own correctness/profile rows, just held out of training), but models
with zero rows anywhere in ``responses.parquet``.

Source: ``lmarena-ai/arena-human-preference-140k`` (Aug 2025, CC BY 4.0), a
newer sibling of the ``arena-human-preference-55k`` dump this project's own
``chatbot_arena`` source already uses -- confirmed live to cover several
current frontier models (2026-10-08 session). Restricted to battles where
BOTH sides are in :data:`MODERN_POOL_ALIASES`, so no opponent profile beyond
the ones this module itself builds is ever needed -- a clean "brand-new vs.
brand-new" generalization test, no warm reference pool involved.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from router.config import Config, load_config

BATTLE_REPO = "lmarena-ai/arena-human-preference-140k"
BATTLE_FILES = [f"data/train-0000{i}-of-00007.parquet" for i in range(7)]

# dataset slug -> our configs/model_profiles.yaml model_id. Only models with
# BOTH a real curated YAML profile entry AND real battle rows in BATTLE_REPO
# are included (verified live, 2026-10-08). Three are checkpoint-dated/
# variant-named aliases, not exact string matches -- verified by hand, same
# discipline as this project's existing GPT-4-Turbo / qwen3 alias precedents:
#   * claude-sonnet-4-20250514 -- the dataset names it by release checkpoint
#     date; same model as our `claude-sonnet-4` entry.
#   * qwen3-235b-a22b-instruct-2507 -- dataset slug includes "instruct";
#     same checkpoint as our `qwen3-235b-a22b-2507` entry.
# `qwen3-235b-a22b-thinking-2507`, `gemini-2.5-pro`, `gemini-2.5-flash`,
# `deepseek-v3-0324`, `deepseek-r1-0528` are exact matches already.
MODERN_POOL_ALIASES: dict[str, str] = {
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "deepseek-v3-0324": "deepseek-v3-0324",
    "deepseek-r1-0528": "deepseek-r1-0528",
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507": "qwen3-235b-a22b-thinking-2507",
}

# score_a for model_a; "both_bad" is dropped (not a preference between the
# two, unlike a genuine tie) -- same 3-outcome scheme as the project's
# existing chatbot_arena source (winner in {a, b, tie}).
WINNER_SCORE: dict[str, float] = {"model_a": 1.0, "model_b": 0.0, "tie": 0.5}


def query_text_from_conversation(conversation_a) -> Optional[str]:
    """The first user turn's plain text, or ``None`` for a row with no text
    turn (e.g. an image-only prompt)."""
    for turn in conversation_a:
        if turn.get("role") != "user":
            continue
        for part in turn.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                return part["text"]
    return None


def filter_modern_pool_battles(raw: pd.DataFrame) -> pd.DataFrame:
    """``raw``: ``id, model_a, model_b, winner, conversation_a`` rows from
    BATTLE_REPO -> ``query_id, model_a, model_b, score_a, query_text`` for
    battles where both sides alias to a :data:`MODERN_POOL_ALIASES` target.
    Pure (no I/O) so it's unit-testable against a small fixture frame."""
    a = raw["model_a"].map(MODERN_POOL_ALIASES)
    b = raw["model_b"].map(MODERN_POOL_ALIASES)
    keep = a.notna() & b.notna() & raw["winner"].isin(WINNER_SCORE)
    out = raw.loc[keep, ["id", "winner", "conversation_a"]].copy()
    out["model_a"] = a[keep]
    out["model_b"] = b[keep]
    out["score_a"] = out["winner"].map(WINNER_SCORE)
    out["query_text"] = out["conversation_a"].map(query_text_from_conversation)
    out = out.dropna(subset=["query_text"])
    return out.rename(columns={"id": "query_id"})[
        ["query_id", "model_a", "model_b", "score_a", "query_text"]
    ].reset_index(drop=True)


def download_modern_pool_battles(cfg: Optional[Config] = None, *, force: bool = False) -> pd.DataFrame:
    """Cached locally as one small parquet -- BATTLE_REPO is ~1.6GB and this
    filters it down to a few thousand rows, so the cache avoids re-fetching
    it (and re-paying the conversation-column parse cost) every call."""
    cfg = cfg or load_config()
    dest = cfg.resolve("data/raw/public_benchmarks/modern_pool_battles.parquet")
    if not force and dest.exists():
        return pd.read_parquet(dest)

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    frames = []
    for f in BATTLE_FILES:
        path = hf_hub_download(repo_id=BATTLE_REPO, repo_type="dataset", filename=f)
        table = pq.read_table(path, columns=["id", "model_a", "model_b", "winner", "conversation_a"])
        frames.append(table.to_pandas())
    raw = pd.concat(frames, ignore_index=True)
    out = filter_modern_pool_battles(raw)

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    return out


def modern_pool_profile_texts(cfg: Optional[Config] = None) -> dict[str, str]:
    """``{model_id: profile_text}`` for every :data:`MODERN_POOL_ALIASES`
    target, rendered straight from ``configs/model_profiles.yaml``.

    NOT via :func:`training.models.profiles.build_model_profiles` -- that
    function only profiles model_ids present in ``responses.parquet``, and
    these models have zero rows there by construction (never trained on,
    never even loaded). Renders via
    :func:`training.models.profiles.render_zero_shot_profile_text` (shared
    with ``router.routing.model_profiles.embed_ad_hoc_profile``'s live ad hoc
    path) -- there IS no empirical data for a model this project has never
    scored, so zero-shot rendering is correct here, not a workaround.
    """
    from training.models.profiles import load_profile_entries, render_zero_shot_profile_text

    cfg = cfg or load_config()
    entries = load_profile_entries(cfg)
    return {
        model_id: render_zero_shot_profile_text(cfg, model_id, entries.get(model_id))
        for model_id in sorted(set(MODERN_POOL_ALIASES.values()))
    }

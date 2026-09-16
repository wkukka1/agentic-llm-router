"""RouterBench model-response TEXT, for the anchor-judge pipeline (docs/anchor_judge.md).

RouterBench's raw pickles carry each model's actual generated answer in a
``<native_model>|model_response`` column, but ``loaders._melt_routerbench``
filters those out (only the score and ``|total_cost`` columns are kept) when
building ``responses.parquet`` -- no other code path needs response text
today. Rather than widen the core response schema for one eval-only
pipeline, this module reads the raw pickle directly and reconstructs
``query_id`` with :func:`training.data.loaders.routerbench_query_ids` --
the EXACT SAME function ``_melt_routerbench`` uses -- so ids are
byte-identical to ``responses.parquet`` / ``nirt_observations.parquet`` and
join 1:1.

RouterBench repeats a few prompts verbatim. The response matrix averages their
scores (``collapse_duplicate_observations``), so no single response text
matches the gold label -- those (query_id, model_id) pairs are dropped here.
"""

from __future__ import annotations

import ast
import warnings

import pandas as pd

from training.data.loaders import routerbench_model_columns, routerbench_query_ids
from training.data.model_registry import canonical_model_id
from training.data.normalize import sanitize_text

_KEY = ["query_id", "model_id"]


def _render_response(raw: object) -> str:
    """Unwrap RouterBench's ``str(list[str])`` response storage.

    Unlike ``render_prompt``, only a list whose elements are all ``str`` is
    unwrapped, so an answer that is itself a list literal (``"[1, 2, 3]"``) is
    kept verbatim; any parse failure (including deep nesting) falls back to the
    raw text.
    """
    s = str(raw)
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)) and all(isinstance(t, str) for t in parsed):
            return sanitize_text("\n\n".join(parsed))
    return sanitize_text(s)


def _load_all_rows(cfg, shot: str) -> pd.DataFrame:
    """Every raw (query_id, model_id, response_text) row, duplicates included."""
    if shot != "0shot":
        raise NotImplementedError(
            f"load_routerbench_response_text only supports shot='0shot' today, got {shot!r}"
        )

    src_cfg = cfg.sources.routerbench
    local_dir = cfg.resolve(src_cfg.local_dir)
    pkl = local_dir / f"routerbench_{shot}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"RouterBench file not found: {pkl}\nRun: python scripts/data/download_routerbench.py"
        )
    wide = pd.read_pickle(pkl)

    model_cols = routerbench_model_columns(wide)
    query_id = routerbench_query_ids(wide)   # shot="0shot" -- the guard above enforces it

    frames: list[pd.DataFrame] = []
    for m in model_cols:
        resp_col = f"{m}|model_response"
        if resp_col not in wide.columns:
            continue
        frames.append(pd.DataFrame({
            "query_id": query_id,
            "model_id": canonical_model_id(m),
            "response_text": wide[resp_col].map(_render_response),
        }))
    if not frames:
        return pd.DataFrame(columns=["query_id", "model_id", "response_text"])
    return pd.concat(frames, ignore_index=True)


def load_routerbench_response_text(cfg, *, shot: str = "0shot") -> pd.DataFrame:
    """One row per (query_id, model_id): the model's raw generated answer text.

    Pairs that occur more than once (duplicate prompts) are dropped entirely;
    see :func:`duplicated_routerbench_query_ids`.

    Only the 0-shot pathway is supported (the anchor-judge pipeline samples
    from 0-shot-only queries, per ``router.cli.zeroshot_only``) -- ``shot``
    is accepted for forward compatibility but non-"0shot" values raise until
    there's an actual caller.
    """
    df = _load_all_rows(cfg, shot)
    dup = df.duplicated(_KEY, keep=False)
    if dup.any():
        n_pairs = df.loc[dup, _KEY].drop_duplicates().shape[0]
        warnings.warn(
            f"dropped {n_pairs} (query_id, model_id) pairs with duplicate RouterBench prompts "
            f"({df.loc[dup, 'query_id'].nunique()} queries): their gold score is an average "
            f"that matches no single response text",
            stacklevel=2,
        )
        df = df[~dup].reset_index(drop=True)
    return df


def duplicated_routerbench_query_ids(cfg, *, shot: str = "0shot") -> set[str]:
    """Query ids whose response text :func:`load_routerbench_response_text` drops."""
    df = _load_all_rows(cfg, shot)
    return set(df.loc[df.duplicated(_KEY, keep=False), "query_id"])


def response_text_lookup(
    cfg, query_ids=None, *, shot: str = "0shot"
) -> dict[tuple[str, str], str]:
    """``{(query_id, model_id): response_text}``, optionally restricted to ``query_ids``."""
    df = load_routerbench_response_text(cfg, shot=shot)
    if query_ids is not None:
        keep = set(query_ids)
        df = df[df["query_id"].isin(keep)]
    return {(qid, mid): text for qid, mid, text in
            zip(df["query_id"], df["model_id"], df["response_text"])}

"""RouterBench model-response TEXT, for the anchor-judge pipeline (docs/anchor_judge.md).

RouterBench's raw pickles carry each model's actual generated answer in a
``<native_model>|model_response`` column, but ``loaders._melt_routerbench``
filters those out (only the score and ``|total_cost`` columns are kept) when
building ``responses.parquet`` -- no other code path needs response text
today. Rather than widen the core response schema for one eval-only
pipeline, this module reads the raw pickle directly and reconstructs
``query_id`` with the EXACT SAME two calls ``_melt_routerbench`` uses
(``make_query_id(Source.ROUTERBENCH, eval_name, query)`` +
``canonical_model_id(native_model)``), so ids are byte-identical to
``responses.parquet`` / ``nirt_observations.parquet`` and join 1:1.
"""

from __future__ import annotations

import pandas as pd

from training.data import schemas
from training.data.model_registry import canonical_model_id
from training.data.normalize import make_query_id, render_prompt

_META_COLS = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}


def load_routerbench_response_text(cfg, *, shot: str = "0shot") -> pd.DataFrame:
    """One row per (query_id, model_id): the model's raw generated answer text.

    Only the 0-shot pathway is supported (the anchor-judge pipeline samples
    from 0-shot-only queries, per ``router.cli.zeroshot_only``) -- ``shot``
    is accepted for forward compatibility but non-"0shot" values raise until
    there's an actual caller.
    """
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

    model_cols = [
        c for c in wide.columns
        if c not in _META_COLS
        and "|" not in c
        and f"{c}|total_cost" in wide.columns
    ]

    query = wide["prompt"].map(render_prompt)
    eval_name = wide["eval_name"].astype(str)
    query_id = pd.Series(
        [make_query_id(schemas.Source.ROUTERBENCH, e, q) for e, q in zip(eval_name, query)],
        index=wide.index,
    )

    frames: list[pd.DataFrame] = []
    for m in model_cols:
        resp_col = f"{m}|model_response"
        if resp_col not in wide.columns:
            continue
        text = wide[resp_col].map(render_prompt)
        frames.append(pd.DataFrame({
            "query_id": query_id,
            "model_id": canonical_model_id(m),
            "response_text": text,
        }))
    if not frames:
        return pd.DataFrame(columns=["query_id", "model_id", "response_text"])
    return pd.concat(frames, ignore_index=True)


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

"""Per-cell token counts and fitted per-model token prices.

Cost decomposes exactly as ``cost = p_in(m) * L_in + p_out(m) * L_out``. The
output-length cost head predicts ``L_out``; this module supplies its labels
(``output_tokens``), the known-ex-ante ``L_in``, and the per-model prices.

* IRT-Router (and lm-eval) rows already carry ``input_tokens`` / ``output_tokens``
  in the response table.
* RouterBench reports only ``total_cost``. :func:`routerbench_tokens` derives the
  counts by tokenising the prompt the model was sent and its recorded response
  text (a GPT-2 byte-level BPE, ``roberta-base``'s). The exact tokenizer differs
  per model; :func:`fit_token_prices` absorbs that scale into effective
  per-model prices fitted against the real cost (R^2 ~0.97-0.99 per model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from router.config import Config

TOKENIZER = "roberta-base"
ROUTERBENCH_TOKENS = "routerbench_tokens.parquet"
_KEY = ["query_id", "model_id"]


def count_tokens(texts, tokenizer, batch: int = 2048) -> np.ndarray:
    """Token count per string (no special tokens, no truncation)."""
    texts = [str(t) for t in texts]
    out = np.empty(len(texts), dtype=np.int64)
    for i in range(0, len(texts), batch):
        ids = tokenizer(texts[i:i + batch], add_special_tokens=False)["input_ids"]
        out[i:i + batch] = [len(x) for x in ids]
    return out


def routerbench_tokens(cfg: Config, *, tokenizer=None) -> pd.DataFrame:
    """``(query_id, model_id, input_tokens, output_tokens)`` for every configured
    RouterBench shot. ``input_tokens`` counts the NATIVE prompt (few-shot
    exemplars included -- that is what the model was sent and billed for), while
    ``query_id`` uses exactly the loader's construction so rows join 1:1.
    Duplicate (query, model) rows are averaged, as the response matrix does."""
    from .loaders import routerbench_model_columns, routerbench_query_ids
    from .model_registry import canonical_model_id
    from .normalize import render_prompt, render_response

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        tokenizer.model_max_length = 10 ** 9      # counting only; silence the length warning
    src = cfg.sources.routerbench
    local_dir = cfg.resolve(src.local_dir)
    shots = list(src.shots)
    zero = pd.read_pickle(local_dir / "routerbench_0shot.pkl")
    zero_text = {str(s): render_prompt(p) for s, p in zip(zero["sample_id"], zero["prompt"])}

    frames = []
    for shot in shots:
        wide = zero if shot == "0shot" else pd.read_pickle(local_dir / f"routerbench_{shot}.pkl")
        qids = routerbench_query_ids(wide, shot, zero_text).to_numpy()
        l_in = count_tokens(wide["prompt"].map(render_prompt), tokenizer)
        for m in routerbench_model_columns(wide):
            col = f"{m}|model_response"
            if col not in wide.columns:
                continue
            frames.append(pd.DataFrame({
                "query_id": qids, "model_id": canonical_model_id(m),
                "input_tokens": l_in,
                "output_tokens": count_tokens(wide[col].map(render_response), tokenizer),
            }))
    out = pd.concat(frames, ignore_index=True)
    return out.groupby(_KEY, as_index=False)[["input_tokens", "output_tokens"]].mean()


def build_routerbench_tokens(cfg: Config) -> Path:
    path = cfg.path("processed") / ROUTERBENCH_TOKENS
    routerbench_tokens(cfg).to_parquet(path, index=False)
    return path


def response_tokens(responses: pd.DataFrame) -> pd.DataFrame:
    """Per-cell mean recorded ``input_tokens`` / ``output_tokens`` from the
    response table (correctness rows only, grouped like the NIRT observation
    table's ``cost``). Cells with no recorded counts are NaN."""
    from . import schemas

    rows = responses[responses["metric_type"].isin(schemas.MetricType.CORRECTNESS)]
    cols = [c for c in ("input_tokens", "output_tokens") if c in rows.columns]
    num = rows[_KEY + cols].copy()
    for c in cols:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    out = num.groupby(_KEY, as_index=False)[cols].mean()
    for c in ("input_tokens", "output_tokens"):
        if c not in out.columns:
            out[c] = np.nan
    return out


def cell_token_table(data, cfg: Optional[Config] = None) -> pd.DataFrame:
    """Recorded counts, filled from the derived RouterBench table where the
    response table has none. Run ``scripts/data/build_routerbench_tokens.py``
    first for RouterBench coverage."""
    rec = response_tokens(data.responses)
    cfg = cfg or data.cfg
    path = cfg.path("processed") / ROUTERBENCH_TOKENS
    if not path.exists():
        return rec
    rb = pd.read_parquet(path)
    m = rec.merge(rb, on=_KEY, how="outer", suffixes=("", "_rb"))
    for c in ("input_tokens", "output_tokens"):
        m[c] = m[c].fillna(m[f"{c}_rb"])
    return m[_KEY + ["input_tokens", "output_tokens"]]


def fit_token_prices(cells: pd.DataFrame, *, min_rows: int = 20) -> dict:
    """Per-model ``{model_id: {"p_in", "p_out", "r2", "n"}}`` (USD per token):
    non-negative least squares ``cost ~ p_in * input_tokens + p_out *
    output_tokens`` (no intercept) over rows with all three finite. Pass TRAIN
    cells only."""
    from scipy.optimize import nnls

    df = cells.copy()
    for c in ("cost", "input_tokens", "output_tokens"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[["cost", "input_tokens", "output_tokens"]]).all(axis=1)]
    out = {}
    for m, g in df.groupby(df["model_id"].astype(str)):
        if len(g) < min_rows:
            continue
        X = g[["input_tokens", "output_tokens"]].to_numpy(np.float64)
        y = g["cost"].to_numpy(np.float64)
        (a, b), _ = nnls(X, y)
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ((y - X @ [a, b]) ** 2).sum() / ss if ss > 0 else float("nan")
        out[m] = {"p_in": float(a), "p_out": float(b), "r2": float(r2), "n": int(len(g))}
    return out

"""Output-length cost head: buckets, labels, per-model priors and per-row inputs.

``cost_mode: length`` predicts a distribution over output-token buckets per
(query, model) and prices it: ``E[C] = p_in(m) * L_in + p_out(m) * E[L_out]``.
Everything here that is estimated (bucket values, per-model priors, token
prices) is fitted on the TRAIN split and frozen into the checkpoint's model
config, so evaluation never re-derives it from the split being scored.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

#: lower edges of the output-token buckets: [0,1), [1,2), [2,3), [3,5), [5,9), ...,
#: [2049, inf). Log-spaced above 2 -- RouterBench's median answer is 1-2 tokens.
BUCKET_EDGES = (0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049)
_KEY = ["query_id", "model_id"]


def length_labels(out_tokens, edges=BUCKET_EDGES) -> np.ndarray:
    """Bucket index per row; ``-1`` where the count is missing."""
    t = np.asarray(out_tokens, np.float64)
    lab = np.searchsorted(np.asarray(edges, np.float64), t, side="right") - 1
    return np.where(np.isfinite(t) & (t >= 0), lab, -1).astype(np.int64)


def bucket_values(out_tokens, edges=BUCKET_EDGES) -> list[float]:
    """Train mean output length within each bucket (the bucket's lower edge when
    a bucket is empty) -- the values ``E[L_out]`` is taken over."""
    t = np.asarray(out_tokens, np.float64)
    lab = length_labels(t, edges)
    vals = []
    for k, lo in enumerate(edges):
        sel = lab == k
        vals.append(float(t[sel].mean()) if sel.any() else float(lo))
    return vals


def length_prior_table(model_ids, labels, n_buckets: int = len(BUCKET_EDGES),
                       alpha: float = 1.0) -> dict:
    """``{"per_model": {m: [log p_k]}, "default": [log p_k]}`` -- Laplace-smoothed
    train bucket frequencies per model, and pooled (for models with no labels)."""
    labels = np.asarray(labels)
    mids = np.asarray(model_ids).astype(str)
    ok = labels >= 0

    def logp(lab):
        c = np.bincount(lab, minlength=n_buckets).astype(np.float64) + alpha
        return np.log(c / c.sum()).tolist()

    # plain str keys: the table is written into the checkpoint's config.yaml
    per_model = {str(m): logp(labels[ok & (mids == m)]) for m in np.unique(mids[ok])}
    return {"per_model": per_model, "default": logp(labels[ok])}


def attach_tokens(arr, token_table: pd.DataFrame):
    """Fill ``arr.in_tokens`` / ``arr.out_tokens`` by (query_id, model_id)."""
    cells = pd.DataFrame({"query_id": np.asarray(arr.query_ids).astype(str),
                          "model_id": np.asarray(arr.model_ids).astype(str)})
    tt = token_table.copy()
    tt["query_id"] = tt["query_id"].astype(str)
    tt["model_id"] = tt["model_id"].astype(str)
    m = cells.merge(tt.drop_duplicates(_KEY), on=_KEY, how="left")
    return replace(arr, in_tokens=pd.to_numeric(m["input_tokens"], errors="coerce").to_numpy(np.float32),
                   out_tokens=pd.to_numeric(m["output_tokens"], errors="coerce").to_numpy(np.float32))


def attach_length_inputs(arr, model_cfg: dict):
    """Per-row ``length_prior`` (log) and ``token_price`` ``[p_in, p_out]`` from a
    length-mode checkpoint's frozen tables. Unseen models get the pooled prior;
    unpriced models get NaN prices (their expected cost is undefined)."""
    if (model_cfg or {}).get("cost_mode") != "length":
        return arr
    pri, prices = model_cfg["length_priors"], model_cfg.get("token_prices", {})
    mids = np.asarray(arr.model_ids).astype(str)
    prior = np.array([pri["per_model"].get(m, pri["default"]) for m in mids], np.float32)
    price = np.array([[prices[m]["p_in"], prices[m]["p_out"]] if m in prices else [np.nan, np.nan]
                      for m in mids], np.float64).reshape(-1, 2)
    return replace(arr, length_prior=prior, token_price=price)


def fit_length_tables(tr_arr, edges=BUCKET_EDGES) -> dict:
    """Everything a length-mode model needs, from the TRAIN arrays only:
    ``length_buckets`` (edges + values), ``length_priors``, ``token_prices``."""
    from training.data.tokens import fit_token_prices

    if tr_arr.out_tokens is None:
        raise ValueError("cost.mode=length needs token counts on the arrays "
                         "(training.nirt.baseline.length.attach_tokens)")
    labels = length_labels(tr_arr.out_tokens, edges)
    if not (labels >= 0).any():
        raise ValueError("cost.mode=length: no train row has an output-token count")
    cells = pd.DataFrame({"model_id": np.asarray(tr_arr.model_ids).astype(str),
                          "cost": tr_arr.cost, "input_tokens": tr_arr.in_tokens,
                          "output_tokens": tr_arr.out_tokens})
    return {
        "length_buckets": {"edges": list(edges), "values": bucket_values(tr_arr.out_tokens, edges)},
        "length_priors": length_prior_table(tr_arr.model_ids, labels, len(edges)),
        "token_prices": fit_token_prices(cells),
    }

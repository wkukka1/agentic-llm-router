"""Materialised arrays for the Phase 1 baseline + shared batched inference.

``build_arrays`` joins one split into contiguous numpy (``e_q``, ``e_m``/``midx``,
``r_q``, warm-up ``nbr``, binary ``y`` + graded ``y_soft``). ``to_tensors`` /
``batched_forward`` are the single inference path used by the trainer, evaluator,
cold-start and inspection code.

Binary label: ``y = 1[score_effective >= binary_threshold]`` (default 0.5).
``mc_accuracy`` is already {0,1}; RouterBench graded ``accuracy`` is thresholded,
not cast. Arena/Judge are excluded (``nirt_observations`` is correctness-only).
The graded score is kept as ``y_soft``; Phase 0 data is never overwritten.
Both come from :func:`graded_labels` -- the one label definition shared with the
cold-start evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from router.config import Config, require_choice

SCORE_KINDS = ("effective", "raw", "corrected")


@dataclass
class BaselineArrays:
    e_q: np.ndarray
    midx: np.ndarray
    y: np.ndarray
    y_soft: np.ndarray
    query_ids: np.ndarray
    model_ids: np.ndarray
    e_m: Optional[np.ndarray] = None
    r_q: Optional[np.ndarray] = None
    nbr: Optional[np.ndarray] = None
    cost: Optional[np.ndarray] = None
    price_feats: Optional[np.ndarray] = None    # (n, 2) log input/output $/1M, cost-probe ablation
    cost_offset: Optional[np.ndarray] = None    # (n,) train per-model mean log cost (residual probe)
    in_tokens: Optional[np.ndarray] = None      # (n,) input tokens (known before routing)
    out_tokens: Optional[np.ndarray] = None     # (n,) realized output tokens (length-head label)
    length_prior: Optional[np.ndarray] = None   # (n, n_buckets) per-model log bucket prior
    token_price: Optional[np.ndarray] = None    # (n, 2) USD per token [p_in, p_out]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def query_dim(self) -> int:
        return int(self.e_q.shape[1])

    @property
    def profile_dim(self) -> int:
        return int(self.e_m.shape[1]) if self.e_m is not None else 0

    @property
    def relevance_dim(self) -> int:
        return int(self.r_q.shape[1]) if self.r_q is not None else 0

    def imbalance_report(self) -> dict:
        y = self.y
        per_model = {str(m): int((self.model_ids == m).sum()) for m in np.unique(self.model_ids)}
        pos_per_model = {str(m): round(float(y[self.model_ids == m].mean()), 4)
                         for m in np.unique(self.model_ids)}
        _, qc = np.unique(self.query_ids, return_counts=True)
        return {
            "n": int(len(y)),
            "positive_rate": round(float(y.mean()), 4),
            "negative_rate": round(1 - float(y.mean()), 4),
            "n_models": len(per_model),
            "n_queries": int(len(qc)),
            "obs_per_query": {"min": int(qc.min()), "mean": round(float(qc.mean()), 2), "max": int(qc.max())},
            "obs_per_model": per_model,
            "pos_rate_per_model": pos_per_model,
        }


def graded_labels(score, binary_threshold: float = 0.5):
    """``(y_soft, y, finite)`` from a graded score.

    ``y_soft`` is clipped to ``[0, 1]`` (the support every response head
    assumes; negative targets appear with ``chance_correction.clip: false``),
    ``y = 1[y_soft >= binary_threshold]``, and ``finite`` flags rows whose score
    was a real number -- callers drop the others (a NaN target poisons every
    parameter through Adam)."""
    s = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(s)
    y_soft = np.clip(np.where(finite, s, 0.0), 0.0, 1.0)
    y = (y_soft >= float(binary_threshold)).astype(np.float32)
    return y_soft.astype(np.float32), y, finite


def _join(store, query_ids, fill):
    """(n, store.dim) matrix of ``store`` rows for ``query_ids``; missing -> ``fill``.

    Query ids repeat once per scored model, so each distinct id is looked up once."""
    query_ids = np.asarray(query_ids)
    out = np.full((len(query_ids), store.dim), fill, dtype=np.float32)
    if len(query_ids) == 0:
        return out
    uniq, inv = np.unique(query_ids.astype(str), return_inverse=True)
    urows = np.fromiter((store.row_of(q) if q in store else -1 for q in uniq),
                        dtype=np.int64, count=len(uniq))
    rows = urows[inv]
    hit = rows >= 0
    out[hit] = np.asarray(store.matrix)[rows[hit]]
    return out


def build_arrays(
    cfg: Config,
    *,
    split: str,
    data=None,
    pathway: Optional[str] = None,
    binary_threshold: float = 0.5,
    score_kind: str = "effective",
    use_relevance: bool = True,
    use_warmup: bool = True,
    model_index: Optional[dict] = None,
    relevance_dim: Optional[int] = None,
) -> BaselineArrays:
    """``score_kind="effective"`` uses the observation table's ``target`` (built
    with the effective score); ``raw`` / ``corrected`` re-derive the score from
    the correctness view. Rows with a non-finite score are dropped and counted
    in ``meta``."""
    from training.data.facade import load_training_data

    require_choice(score_kind, SCORE_KINDS, field="response.score_kind")
    d = data or load_training_data(cfg)
    pathway = pathway or cfg.get("data.pathway", "irt")

    ds = d.nirt_dataset(split=split, pathway=pathway)
    g = ds.gather()
    e_q = np.ascontiguousarray(g["query_embedding"], dtype=np.float32)
    e_m = np.ascontiguousarray(g["model_embedding"], dtype=np.float32)
    query_ids = np.asarray(ds.query_ids)
    model_ids = np.asarray(ds.model_ids).astype(str)

    score = np.asarray(g["target"], dtype=np.float64)
    if score_kind != "effective":
        key = d.correctness(split=split, models="warm", score_kind=score_kind).set_index(
            ["query_id", "model_id"])["score"].to_dict()
        score = np.array([key.get((q, m), np.nan) for q, m in zip(query_ids, model_ids)], np.float64)
    n_out_of_range = int((np.isfinite(score) & ((score < 0) | (score > 1))).sum())
    y_soft, y, keep = graded_labels(score, binary_threshold)
    n_nonfinite = int((~keep).sum())
    cost = np.ascontiguousarray(g["cost"], dtype=np.float32)

    if model_index is None:
        model_index = {m: i for i, m in enumerate(sorted(np.unique(model_ids[keep])))}
    else:
        # A provided model_index pins the model->row mapping of a trained
        # checkpoint. Drop observations for models the checkpoint never saw
        # (e.g. re-evaluating a 9-model checkpoint after the dataset grew to 11
        # for a later pool-expansion phase) rather than crash on an OOB index.
        keep = keep & np.array([m in model_index for m in model_ids], dtype=bool)
    if not keep.all():
        e_q, e_m, y_soft, y, cost = e_q[keep], e_m[keep], y_soft[keep], y[keep], cost[keep]
        query_ids, model_ids = query_ids[keep], model_ids[keep]
    midx = np.array([model_index.get(m, -1) for m in model_ids], dtype=np.int64)

    r_q = nbr = None
    if use_relevance:
        from training.taxonomy import load_relevance

        rel = load_relevance(cfg)
        if rel is not None:
            if relevance_dim is not None and rel.dim != relevance_dim:
                # The checkpoint was trained against an earlier taxonomy (a different
                # cluster count); its w_r layer is a fixed relevance_dim -> theta_dim
                # matrix and cannot consume today's re-clustered r_q. Fail here with a
                # clear message instead of a bare matmul shape error deep in the model.
                raise ValueError(
                    f"relevance store at dim {rel.dim} does not match this checkpoint's "
                    f"trained relevance_dim ({relevance_dim}) -- the taxonomy was "
                    "re-clustered after this checkpoint was trained; retrain the "
                    "checkpoint against the current taxonomy/relevance store, or point "
                    "cfg at the taxonomy artifacts it was originally trained with"
                )
            r_q = _join(rel, query_ids, fill=1.0 / rel.dim)   # uniform for unclustered queries
    if use_warmup:
        from training.retrieval import load_warmup

        wu = load_warmup(cfg, pathway)
        if wu is not None:
            nbr = _join(wu, query_ids, fill=0.0)              # zero rows fall back to e_q in the model

    return BaselineArrays(
        e_q=e_q, midx=midx, y=y, y_soft=y_soft, query_ids=query_ids, model_ids=model_ids,
        e_m=e_m, r_q=r_q, nbr=nbr,
        cost=cost,
        meta={
            "split": split, "pathway": pathway, "binary_threshold": float(binary_threshold),
            "score_kind": score_kind, "n_dropped_no_embedding": int(ds.dropped),
            "n_dropped_nonfinite_score": n_nonfinite,
            "n_clipped_out_of_range": n_out_of_range,
            "model_index": model_index,
            "relevance_available": r_q is not None, "warmup_available": nbr is not None,
        },
    )


# --------------------------------------------------------------------------- #
# shared inference                                                            #
# --------------------------------------------------------------------------- #
# USD per query standing in for a genuine $0 (a free API, e.g. glm_4_flash) so its
# log is finite. ~18x below the cheapest paid call in the data ($1.84e-6).
# Usage-accounting gaps are imputed or NaN upstream (pricing.impute_missing_usage),
# never 0, so they are never mistaken for free calls.
FREE_COST_FLOOR = 1e-7


def log_cost(cost: Optional[np.ndarray], n: int) -> np.ndarray:
    """``log(max(cost, FREE_COST_FLOOR))`` where finite and >= 0, else NaN
    (missing / unpriced cells)."""
    if cost is None:
        return np.full(n, np.nan, np.float32)
    c = np.asarray(cost, np.float64)
    ok = np.isfinite(c) & (c >= 0)
    return np.where(ok, np.log(np.where(ok, np.maximum(c, FREE_COST_FLOOR), 1.0)),
                    np.nan).astype(np.float32)


def cost_offset_table(model_ids, cost) -> dict:
    """``{"per_model": {model_id: mean log cost}, "default": global mean log cost}``
    over finite non-negative costs (0 floored). Callers pass the TRAIN split only -- this table
    is the residual probe's fixed baseline and is stored in the checkpoint."""
    lc = log_cost(cost, len(model_ids))
    ok = np.isfinite(lc)
    if not ok.any():
        raise ValueError("cost.residual: the train split has no finite positive cost")
    s = pd.Series(lc[ok].astype(np.float64)).groupby(np.asarray(model_ids).astype(str)[ok]).mean()
    return {"per_model": {str(k): float(v) for k, v in s.items()},
            "default": float(lc[ok].astype(np.float64).mean())}


def attach_cost_offset(arr: BaselineArrays, model_cfg: dict) -> BaselineArrays:
    """Fill ``arr.cost_offset`` from a residual checkpoint's ``cost_offsets`` table
    (no-op for a non-residual model). Models absent from the table -- never seen
    with a cost in training -- get the global train mean."""
    from dataclasses import replace

    table = (model_cfg or {}).get("cost_offsets")
    if not (model_cfg or {}).get("cost_residual") or table is None:
        return arr
    pm, default = table["per_model"], float(table["default"])
    off = np.array([pm.get(str(m), default) for m in arr.model_ids], dtype=np.float32)
    return replace(arr, cost_offset=off)


def to_tensors(arr: BaselineArrays, *, model_params: str, use_relevance: bool, use_warmup: bool):
    """``BaselineArrays`` -> dict of torch tensors (``e_q, ref, y, y_soft, r_q, nbr,
    log_cost, price_feats``)."""
    import torch

    ref = arr.e_m if model_params == "projected" else arr.midx
    return {
        "e_q": torch.from_numpy(arr.e_q),
        "ref": torch.from_numpy(ref),
        "y": torch.from_numpy(arr.y),
        "y_soft": torch.from_numpy(arr.y_soft),
        "r_q": torch.from_numpy(arr.r_q) if use_relevance and arr.r_q is not None else None,
        "nbr": torch.from_numpy(arr.nbr) if use_warmup and arr.nbr is not None else None,
        "log_cost": torch.from_numpy(log_cost(arr.cost, len(arr.y))),
        "price_feats": (torch.from_numpy(np.asarray(arr.price_feats, np.float32))
                        if arr.price_feats is not None else None),
        "cost_offset": (torch.from_numpy(np.asarray(arr.cost_offset, np.float32))
                        if arr.cost_offset is not None else None),
        **_length_tensors(arr),
    }


def _length_tensors(arr: BaselineArrays) -> dict:
    """Length-mode cost head inputs (all ``None`` when the arrays carry none)."""
    import torch

    from .length import length_labels

    def opt(x, dtype):
        return torch.from_numpy(np.ascontiguousarray(x, dtype)) if x is not None else None

    return {
        "length_label": (torch.from_numpy(length_labels(arr.out_tokens))
                         if arr.out_tokens is not None else None),
        "length_prior": opt(arr.length_prior, np.float32),
        "in_tokens": opt(arr.in_tokens, np.float64),
        "token_price": opt(arr.token_price, np.float64),
    }


def batched_forward(model, source, *, fields=("proba",), batch: int = 16384, device: str = "cpu",
                    y=None, level: float = 0.9, extra_levels=(), seed: int = 0):
    """Run ``model`` over ``source`` (a ``BaselineArrays`` or a ``to_tensors`` dict) in
    chunks; return ``{field: np.ndarray}``.

    Fields: ``logit``, ``base_irt``, ``a_q``, ``b_q`` (from ``Output``);
    ``proba`` (``head.as_probability``), ``mean`` / ``std`` / ``lower`` / ``upper``
    (predictive; ``lower``/``upper`` at central ``level``), ``nll`` (needs ``y``),
    and any response-head param name (``sigma``, ``kappa``, ``pi0``, ...).
    Cost probe (``use_cost_head`` models): ``cost_mu`` / ``cost_log_sigma`` (log-USD)
    and ``cost`` (expected USD, the lognormal mean).

    ``extra_levels`` adds ``lower@<lv>`` / ``upper@<lv>`` for more central levels.
    Every interval of a batch (both ends, every level) comes from ONE Monte
    Carlo draw for the sampling heads, under a fixed ``seed``, so ``lower`` and
    ``upper`` are consistent and reproducible."""
    import torch

    from router.nirt.components import CostHead

    from .response_head import ResponseHead

    t = source if isinstance(source, dict) else to_tensors(
        source, model_params=model.model_params,
        use_relevance=model.use_relevance, use_warmup=model.use_warmup)
    head = model.response_head
    n = len(t["e_q"])
    yt = None if y is None else torch.as_tensor(np.asarray(y, np.float32))
    levels = ([float(level)] if any(f in ("lower", "upper") for f in fields) else []) \
        + [float(lv) for lv in extra_levels]
    extra_keys = {float(lv): (f"lower@{float(lv):g}", f"upper@{float(lv):g}") for lv in extra_levels}
    acc: dict[str, list] = {f: [] for f in fields}
    for lo_key, hi_key in extra_keys.values():
        acc[lo_key], acc[hi_key] = [], []
    analytic = type(head).interval is not ResponseHead.interval
    devs = [torch.device(device)] if str(device).startswith("cuda") else []
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            pf, co, lp = t.get("price_feats"), t.get("cost_offset"), t.get("length_prior")
            o = model(t["e_q"][sl].to(device), t["ref"][sl].to(device),
                      t["r_q"][sl].to(device) if t["r_q"] is not None else None,
                      t["nbr"][sl].to(device) if t["nbr"] is not None else None,
                      pf[sl].to(device) if pf is not None else None,
                      co[sl].to(device) if co is not None else None,
                      lp[sl].to(device) if lp is not None else None)
            r = o.response
            iv: dict[float, tuple] = {}
            if levels:
                uniq = list(dict.fromkeys(levels))
                if analytic:
                    iv = {lv: head.interval(r, lv) for lv in uniq}
                else:
                    qs = [p for lv in uniq for p in ((1 - lv) / 2, (1 + lv) / 2)]
                    Q = head.quantiles(r, qs)
                    iv = {lv: (Q[2 * j], Q[2 * j + 1]) for j, lv in enumerate(uniq)}
            for f in fields:
                if f == "proba":
                    v = head.as_probability(r)
                elif f == "mean":
                    v = head.mean(r)
                elif f == "std":
                    v = head.stddev(r)
                elif f in ("lower", "upper"):
                    v = iv[float(level)][0 if f == "lower" else 1]
                elif f == "nll":
                    v = head.nll(yt[sl].to(device), r)
                elif f in ("logit", "base_irt", "a_q", "b_q", "cost_mu", "cost_log_sigma"):
                    v = getattr(o, f)
                    if v is None:
                        raise ValueError(f"field {f!r} unavailable: model has no cost head")
                elif f == "cost" and o.length_logits is not None:
                    # exact expected cost under the predicted length distribution
                    if t.get("in_tokens") is None or t.get("token_price") is None:
                        raise ValueError("length-mode 'cost' needs in_tokens and token_price "
                                         "(attach_tokens + attach_length_inputs)")
                    e_len = model.cost_head.expected_length(o.length_logits).double()
                    tp = t["token_price"][sl].to(device)
                    v = t["in_tokens"][sl].to(device) * tp[:, 0] + e_len * tp[:, 1]
                elif f == "expected_length":
                    if o.length_logits is None:
                        raise ValueError("field 'expected_length' needs a length-mode cost head")
                    v = model.cost_head.expected_length(o.length_logits)
                elif f == "length_logits":
                    v = o.length_logits
                elif f == "cost":
                    if o.cost_mu is None:
                        raise ValueError("field 'cost' unavailable: model has no cost head")
                    v = CostHead.expected_cost(o.cost_mu, o.cost_log_sigma)
                else:
                    v = r.params[f]
                acc[f].append(v.cpu().numpy())
            for lv, (lo_key, hi_key) in extra_keys.items():
                acc[lo_key].append(iv[lv][0].cpu().numpy())
                acc[hi_key].append(iv[lv][1].cpu().numpy())
    return {f: (np.concatenate(a) if a else np.zeros(0, np.float32)) for f, a in acc.items()}


def checkpoint_matrix(ckpt, cfg: Config, split: str, field: str = "proba", *,
                      level: float = 0.9):
    """Predicted ``[query_id x model_id]`` DataFrame for a saved Phase-1/2 checkpoint.

    ``field`` -> ``proba`` (Bernoulli P(correct)), ``mean`` (E[Y]), ``lower`` /
    ``upper`` (predictive interval at ``level``), or any response-head param.
    The one place the ``load_baseline_run`` -> ``build_arrays`` -> ``batched_forward`` ->
    pivot chain lives -- was copy-pasted into ``route_compare``, ``route_eval`` and
    ``pool_expansion.battery``.
    """
    import pandas as pd

    from router.nirt.frames import pivot_qm
    from .checkpoint import load_baseline_run

    model, blob, s = load_baseline_run(ckpt)
    ev = build_arrays(
        cfg, split=split, pathway=s["pathway"], binary_threshold=s["binary_threshold"],
        score_kind=s["score_kind"], use_relevance=model.use_relevance,
        use_warmup=model.use_warmup, model_index=blob["model_index"],
        relevance_dim=model.relevance_dim if model.use_relevance else None,
    )
    ev = attach_cost_offset(ev, blob["model_cfg"])
    if blob["model_cfg"].get("cost_mode") == "length":
        from training.data.facade import load_training_data
        from training.data.tokens import cell_token_table

        from .length import attach_length_inputs, attach_tokens

        ev = attach_length_inputs(
            attach_tokens(ev, cell_token_table(load_training_data(cfg), cfg)), blob["model_cfg"])
    v = batched_forward(model, ev, fields=(field,), level=level)[field]
    return pivot_qm(
        pd.DataFrame({"query_id": ev.query_ids, "model_id": ev.model_ids, field: v}), field
    )

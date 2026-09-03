"""Materialised arrays for the Phase 1 baseline + shared batched inference.

``build_arrays`` joins one split into contiguous numpy (``e_q``, ``e_m``/``midx``,
``r_q``, warm-up ``nbr``, binary ``y`` + graded ``y_soft``). ``to_tensors`` /
``batched_forward`` are the single inference path used by the trainer, evaluator,
cold-start and inspection code.

Binary label: ``y = 1[score_effective >= binary_threshold]`` (default 0.5).
``mc_accuracy`` is already {0,1}; RouterBench graded ``accuracy`` is thresholded,
not cast. Arena/Judge are excluded (``nirt_observations`` is correctness-only).
The graded score is kept as ``y_soft``; Phase 0 data is never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import Config


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


def _join(store, query_ids, fill):
    """(n, store.dim) matrix of ``store`` rows for ``query_ids``; missing -> ``fill``."""
    rows = np.array([store.row_of(q) if q in store else -1 for q in query_ids])
    out = np.full((len(query_ids), store.dim), fill, dtype=np.float32)
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
) -> BaselineArrays:
    from ..data.phase1 import load_phase1

    d = data or load_phase1(cfg)
    pathway = pathway or cfg.get("data.pathway", "irt")

    ds = d.nirt_dataset(split=split, pathway=pathway)
    g = ds.gather()
    e_q = np.ascontiguousarray(g["query_embedding"], dtype=np.float32)
    e_m = np.ascontiguousarray(g["model_embedding"], dtype=np.float32)
    y_soft = np.ascontiguousarray(g["target"], dtype=np.float32)
    query_ids = np.asarray(ds.query_ids)
    model_ids = np.asarray(ds.model_ids).astype(str)

    if score_kind == "raw":
        key = d.correctness(split=split, models="warm", score_kind="raw").set_index(
            ["query_id", "model_id"])["score"].to_dict()
        y_soft = np.array([key.get((q, m), np.nan) for q, m in zip(query_ids, model_ids)], np.float32)

    y = (y_soft.astype(np.float64) >= float(binary_threshold)).astype(np.float32)
    cost = np.ascontiguousarray(g["cost"], dtype=np.float32)

    if model_index is None:
        model_index = {m: i for i, m in enumerate(sorted(np.unique(model_ids)))}
    else:
        # A provided model_index pins the model->row mapping of a trained
        # checkpoint. Drop observations for models the checkpoint never saw
        # (e.g. re-evaluating a 9-model checkpoint after the dataset grew to 11
        # for a later pool-expansion phase) rather than crash on an OOB index.
        keep = np.array([m in model_index for m in model_ids])
        if not keep.all():
            e_q, e_m, y_soft, y, cost = e_q[keep], e_m[keep], y_soft[keep], y[keep], cost[keep]
            query_ids, model_ids = query_ids[keep], model_ids[keep]
    midx = np.array([model_index.get(m, -1) for m in model_ids], dtype=np.int64)

    r_q = nbr = None
    if use_relevance:
        from ..taxonomy import load_relevance

        rel = load_relevance(cfg)
        if rel is not None:
            r_q = _join(rel, query_ids, fill=1.0 / rel.dim)   # uniform for unclustered queries
    if use_warmup:
        from ..retrieval import load_warmup

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
            "model_index": model_index,
            "relevance_available": r_q is not None, "warmup_available": nbr is not None,
        },
    )


# --------------------------------------------------------------------------- #
# shared inference                                                            #
# --------------------------------------------------------------------------- #
def to_tensors(arr: BaselineArrays, *, model_params: str, use_relevance: bool, use_warmup: bool):
    """``BaselineArrays`` -> dict of torch tensors (``e_q, ref, y, y_soft, r_q, nbr``)."""
    import torch

    ref = arr.e_m if model_params == "projected" else arr.midx
    return {
        "e_q": torch.from_numpy(arr.e_q),
        "ref": torch.from_numpy(ref),
        "y": torch.from_numpy(arr.y),
        "y_soft": torch.from_numpy(arr.y_soft),
        "r_q": torch.from_numpy(arr.r_q) if use_relevance and arr.r_q is not None else None,
        "nbr": torch.from_numpy(arr.nbr) if use_warmup and arr.nbr is not None else None,
    }


def batched_forward(model, source, *, fields=("proba",), batch: int = 16384, device: str = "cpu",
                    y=None, level: float = 0.9):
    """Run ``model`` over ``source`` (a ``BaselineArrays`` or a ``to_tensors`` dict) in
    chunks; return ``{field: np.ndarray}``.

    Fields: ``logit``, ``base_irt``, ``a_q``, ``b_q`` (from ``Output``);
    ``proba`` (``head.as_probability``), ``mean`` / ``std`` / ``lower`` / ``upper``
    (predictive; ``lower``/``upper`` at central ``level``), ``nll`` (needs ``y``),
    and any response-head param name (``sigma``, ``kappa``, ``pi0``, ...)."""
    import torch

    t = source if isinstance(source, dict) else to_tensors(
        source, model_params=model.model_params,
        use_relevance=model.use_relevance, use_warmup=model.use_warmup)
    head = model.response_head
    n = len(t["e_q"])
    yt = None if y is None else torch.as_tensor(np.asarray(y, np.float32))
    acc: dict[str, list] = {f: [] for f in fields}
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            o = model(t["e_q"][sl].to(device), t["ref"][sl].to(device),
                      t["r_q"][sl].to(device) if t["r_q"] is not None else None,
                      t["nbr"][sl].to(device) if t["nbr"] is not None else None)
            r = o.response
            for f in fields:
                if f == "proba":
                    v = head.as_probability(r)
                elif f == "mean":
                    v = head.mean(r)
                elif f == "std":
                    v = head.stddev(r)
                elif f in ("lower", "upper"):
                    lo, hi = head.interval(r, level)
                    v = lo if f == "lower" else hi
                elif f == "nll":
                    v = head.nll(yt[sl].to(device), r)
                elif f in ("logit", "base_irt", "a_q", "b_q"):
                    v = getattr(o, f)
                else:
                    v = r.params[f]
                acc[f].append(v.cpu().numpy())
    return {f: np.concatenate(a) for f, a in acc.items()}

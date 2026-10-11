"""Cost-aware routing benchmark: routers vs an always-reference-model policy vs the oracle.

Each source (RouterBench, IRT-Router) is scored on its own dense test pool -- a
query only has realized scores for its own source's models -- and the per-query
picks are then concatenated into one combined benchmark.

Every policy picks one model per query; it is graded on the REALIZED quality
``Y[q, m]`` and REALIZED cost ``C[q, m]`` of that pick:

* router ``name``@lam   ``argmax_m  pred_quality[q, m] - lam * pred_cost[q, m]``
* reference             a fixed column (GPT-4 by default)
* oracle@lam            ``argmax_m  Y[q, m] - lam * C[q, m]``, ties to the cheapest model
* oracle_best_quality   best realized score, ties to the cheapest model
* oracle_match_reference  cheapest model scoring >= the reference on that query

The oracle rows read ground truth and exist only to grade the routers.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from router.nirt.routing_decision import routing_decision

REFERENCE_MODELS = {"routerbench": "gpt-4-1106-preview", "irt_router": "gpt_4o"}
DEFAULT_LAMS = (0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 1000.0)
_TIE = 1e-12


def baseline_cell_predictions(ckpt, cfg, data, split: str) -> tuple[pd.DataFrame, dict]:
    """Long ``(query_id, model_id, quality_pred, cost_pred, cost[, expected_length,
    output_tokens])`` frame for every ``split`` cell, from a ``BaselineNIRT``
    checkpoint with a cost head, plus the checkpoint's model config."""
    from training.data.tokens import cell_token_table
    from training.nirt.baseline.checkpoint import load_baseline_run
    from training.nirt.baseline.data import attach_cost_offset, batched_forward, build_arrays
    from training.nirt.baseline.length import attach_length_inputs, attach_tokens

    model, blob, s = load_baseline_run(ckpt)
    if model.cost_head is None:
        raise SystemExit(f"{ckpt}: no cost head (train with --cost)")
    ev = build_arrays(cfg, split=split, data=data, pathway=s["pathway"],
                      binary_threshold=s["binary_threshold"], score_kind=s["score_kind"],
                      use_relevance=model.use_relevance, use_warmup=model.use_warmup,
                      model_index=blob["model_index"],
                      relevance_dim=model.relevance_dim if model.use_relevance else None)
    ev = attach_cost_offset(ev, blob["model_cfg"])   # residual probes: train-split log C(m)
    length = blob["model_cfg"].get("cost_mode") == "length"
    if length:   # tokens + frozen train-split priors / prices
        ev = attach_length_inputs(attach_tokens(ev, cell_token_table(data, cfg)), blob["model_cfg"])
    f = batched_forward(model, ev, fields=("mean", "cost") + (("expected_length",) if length else ()))
    out = pd.DataFrame({"query_id": ev.query_ids, "model_id": ev.model_ids,
                        "quality_pred": f["mean"], "cost_pred": f["cost"], "cost": ev.cost})
    if length:
        out["expected_length"] = f["expected_length"]
        out["output_tokens"] = ev.out_tokens
    return out, blob["model_cfg"]


def per_model_mean_cost(obs: pd.DataFrame) -> pd.Series:
    """Train-split mean realized cost per model -- the static ``C(m)`` of the
    IRT-Router paper's routing policy."""
    tr = obs[obs["split"] == "train"]
    cost = pd.to_numeric(tr["cost"], errors="coerce")
    ok = np.isfinite(cost) & (cost >= 0)
    return cost[ok].groupby(tr.loc[ok, "model_id"].astype(str)).mean()


def source_picks(Y: np.ndarray, C: np.ndarray, models: Sequence[str], reference: str,
                 routers: Mapping[str, tuple[np.ndarray, np.ndarray]],
                 lams: Sequence[float] = DEFAULT_LAMS) -> dict:
    """``{policy: (realized_quality, realized_cost, picked_reference)}``, one entry
    per query, for every policy in the module docstring on one dense ``[Q, M]`` pool.
    ``routers`` maps a name to ``(pred_quality, pred_cost)``; ``pred_cost`` may be
    ``[Q, M]`` or per-model ``[M]``.

    Every input must be finite and the pool non-empty (``ValueError`` otherwise):
    ``routing_decision`` silently masks a non-finite cost only when ``lam > 0``, so
    a missing ``C(m)`` would drop that model from every ``lam > 0`` row alone."""
    Y, C = np.asarray(Y, np.float64), np.asarray(C, np.float64)
    if len(Y) == 0:
        raise ValueError("source_picks: no queries in the pool")
    arrays = {"Y": Y, "C": C} | {f"{n}.{part}": np.asarray(a, np.float64)
                                 for n, (pq, pc) in routers.items()
                                 for part, a in (("pred_quality", pq), ("pred_cost", pc))}
    bad = [n for n, a in arrays.items() if not np.isfinite(a).all()]
    if bad:
        raise ValueError(f"source_picks: non-finite values in {bad}")
    qi, ri = np.arange(len(Y)), list(models).index(reference)

    def take(sel):
        return Y[qi, sel], C[qi, sel], sel == ri

    def cheapest_at_max(U):   # argmax U, ties to the cheapest model (not the first column)
        return np.where(U >= U.max(1, keepdims=True) - _TIE, C, np.inf).argmin(1)

    picks = {"reference": take(np.full(len(Y), ri)),
             "oracle_best_quality": take(cheapest_at_max(Y)),
             "oracle_match_reference": take(np.where(Y >= Y[:, [ri]] - _TIE, C, np.inf).argmin(1))}
    for lam in lams:
        picks[f"oracle@{lam:g}"] = take(cheapest_at_max(Y - lam * C))
        for name, (pq, pc) in routers.items():
            picks[f"{name}@{lam:g}"] = take(routing_decision(pq, lam=lam, model_costs=pc))
    return picks


def combine(by_source: Mapping[str, dict]) -> dict:
    """Concatenate per-source picks (same policy keys) into one benchmark."""
    keys = next(iter(by_source.values())).keys()
    return {k: tuple(np.concatenate([p[k][i] for p in by_source.values()]) for i in range(3))
            for k in keys}


def summarize(picks: Mapping[str, tuple]) -> dict:
    """Per policy: mean quality, cost per 1k queries, both relative to the
    reference policy, and the share of queries sent to the reference model."""
    rq, rc = float(picks["reference"][0].mean()), float(picks["reference"][1].mean())
    out = {}
    for k, (y, c, is_ref) in picks.items():
        q, cm = float(np.mean(y)), float(np.mean(c))
        out[k] = {"quality": q, "cost_per_1k": 1e3 * cm, "q_vs_reference": q - rq,
                  "cost_vs_reference": cm / rc - 1 if rc else float("nan"),
                  "reference_share": float(np.mean(is_ref))}
    return out

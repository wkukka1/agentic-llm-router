"""Synthetic multidimensional-IRT recovery gate for the Phase 1 baseline.

``make_synthetic`` draws a well-specified world (``theta_true ~ N(0,I)``,
``a_true = softplus(.)``, ``b_true ~ N(0,1)``, ``y ~ Bernoulli(sigma(a.theta-b))``);
``recovery_report`` trains ``BaselineNIRT`` on it and checks recovery up to the
unidentifiable rotation/scale: predictive BCE near Bayes-optimal, and Spearman of
recovered vs true difficulty / ability ordering. A model that fails this should
not be trusted on the real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticIRT:
    e_q: np.ndarray
    theta_true: np.ndarray   # (M, K)
    a_true: np.ndarray       # (Q, K)
    b_true: np.ndarray       # (Q,)
    p: np.ndarray            # (Q, M)
    y: np.ndarray            # (Q, M) in {0, 1}
    query_ids: list
    model_ids: list
    K: int

    def bayes_bce(self) -> float:
        pp = np.clip(self.p, 1e-7, 1 - 1e-7)
        return float(-(self.p * np.log(pp) + (1 - self.p) * np.log(1 - pp)).mean())


def make_synthetic(n_queries=1500, n_models=12, K=4, q_dim=32, seed=0) -> SyntheticIRT:
    rng = np.random.default_rng(seed)
    e_q = rng.standard_normal((n_queries, q_dim)).astype(np.float32)
    a_true = np.log1p(np.exp(e_q @ (rng.standard_normal((q_dim, K)) / np.sqrt(q_dim))
                             + 0.3 * rng.standard_normal((n_queries, K)))) + 0.05
    b_true = (e_q @ (rng.standard_normal((q_dim, 1)) / np.sqrt(q_dim))).ravel() \
        + 0.5 * rng.standard_normal(n_queries)
    theta_true = rng.standard_normal((n_models, K))
    p = 1.0 / (1.0 + np.exp(-(np.einsum("qk,mk->qm", a_true, theta_true) - b_true[:, None])))
    return SyntheticIRT(
        e_q=e_q, theta_true=theta_true, a_true=a_true, b_true=b_true, p=p,
        y=(rng.random(p.shape) < p).astype(np.float32),
        query_ids=[f"sq{i}" for i in range(n_queries)],
        model_ids=[f"sm{j}" for j in range(n_models)], K=K)


def to_arrays(syn: SyntheticIRT, split_frac=0.8, seed=0):
    """``(train_arrays, val_arrays)`` as :class:`BaselineArrays` (no relevance / warm-up)."""
    from .baseline_data import BaselineArrays

    Q, M = syn.y.shape
    order = np.random.default_rng(seed).permutation(Q)
    cut = int(Q * split_frac)
    model_index = {m: i for i, m in enumerate(syn.model_ids)}

    def _pack(qidx):
        qi, mi = np.repeat(qidx, M), np.tile(np.arange(M), len(qidx))
        return BaselineArrays(
            e_q=syn.e_q[qi], midx=mi.astype(np.int64),
            y=syn.y[qi, mi].astype(np.float32), y_soft=syn.p[qi, mi].astype(np.float32),
            query_ids=np.array([syn.query_ids[i] for i in qi]),
            model_ids=np.array([syn.model_ids[j] for j in mi]),
            e_m=np.zeros((len(qi), 1), dtype=np.float32),
            meta={"model_index": model_index, "binary_threshold": 0.5, "score_kind": "synthetic",
                  "relevance_available": False, "warmup_available": False,
                  "split": "synthetic", "pathway": "synthetic"})

    return _pack(order[:cut]), _pack(order[cut:])


def recovery_report(model, syn: SyntheticIRT, val_arrays) -> dict:
    from scipy.stats import spearmanr

    from .baseline_data import batched_forward
    from .metrics import prediction_metrics

    fwd = batched_forward(model, val_arrays, fields=("proba", "b_q"))
    pm = prediction_metrics(val_arrays.y, fwd["proba"])
    theta_hat = model.theta.weight.detach().numpy()

    # difficulty ordering: recovered b vs true b, one row per val query
    pos = {q: k for k, q in enumerate(syn.query_ids)}
    rows = list({q: i for i, q in enumerate(val_arrays.query_ids)}.values())
    b_true = np.array([syn.b_true[pos[val_arrays.query_ids[i]]] for i in rows])
    b_rho = float(spearmanr(fwd["b_q"][rows], b_true).statistic)

    # ability ordering: align theta_hat -> theta_true by least squares, then Spearman
    theta_aligned = theta_hat @ np.linalg.lstsq(theta_hat, syn.theta_true, rcond=None)[0]
    abil_rho = float(spearmanr(theta_aligned.mean(1), syn.theta_true.mean(1)).statistic)
    per_dim = [float(spearmanr(theta_aligned[:, k], syn.theta_true[:, k]).statistic)
               for k in range(syn.K)]

    return {
        "prediction": {k: pm[k] for k in ("bce", "brier", "accuracy", "auc", "spearman_r")},
        "bayes_bce": syn.bayes_bce(),
        "bce_gap_vs_bayes": pm["bce"] - syn.bayes_bce(),
        "difficulty_spearman": b_rho,
        "ability_spearman_1d": abil_rho,
        "ability_spearman_per_dim": per_dim,
        "passed": bool(pm["bce"] - syn.bayes_bce() < 0.05 and b_rho > 0.6 and abil_rho > 0.6),
    }

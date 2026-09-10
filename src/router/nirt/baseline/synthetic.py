"""Synthetic IRT recovery gates for the NIRT model.

Two well-specified worlds share the dataclass, ``to_arrays`` and the
``recovery_report`` dispatch:

* :func:`make_synthetic` -- Bernoulli: ``y ~ Bernoulli(sigma(a . theta - b))``.
  Recovery = predictive BCE near Bayes-optimal + Spearman of recovered vs true
  difficulty / ability ordering (up to the unidentifiable rotation/scale).
* :func:`make_synthetic_continuous` -- a continuous response head (``normal`` /
  ``beta`` / ``zoib``) on top of the same IRT core plus the head's own noise
  parameters. Recovery additionally checks the dispersion (sigma / kappa) ordering
  and finite NLL.

A model that fails its gate should not be trusted on the real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import BaselineArrays


@dataclass
class Synthetic:
    e_q: np.ndarray
    theta_true: np.ndarray            # (M, K)
    b_true: np.ndarray               # (Q,)
    y: np.ndarray                    # (Q, M)  -- {0,1} for bernoulli, [0,1] otherwise
    query_ids: list
    model_ids: list
    K: int
    head: str = "bernoulli"
    a_true: Optional[np.ndarray] = None      # (Q, K)  -- bernoulli only
    p: Optional[np.ndarray] = None           # (Q, M)  -- bernoulli true P(correct)
    z: Optional[np.ndarray] = None           # (Q, M)  -- a.theta - b (continuous)
    disp_true: Optional[np.ndarray] = None   # (Q,)    -- sigma / kappa (continuous)
    pi_true: Optional[np.ndarray] = None     # (Q, 3)  -- zoib mixture weights

    def bayes_bce(self) -> float:
        pp = np.clip(self.p, 1e-7, 1 - 1e-7)
        return float(-(self.p * np.log(pp) + (1 - self.p) * np.log(1 - pp)).mean())


# --------------------------------------------------------------------------- #
# world generators (RNG draw order is load-bearing -- do not "tidy")          #
# --------------------------------------------------------------------------- #
def make_synthetic(n_queries=1500, n_models=12, K=4, q_dim=32, seed=0) -> Synthetic:
    rng = np.random.default_rng(seed)
    e_q = rng.standard_normal((n_queries, q_dim)).astype(np.float32)
    a_true = np.log1p(np.exp(e_q @ (rng.standard_normal((q_dim, K)) / np.sqrt(q_dim))
                             + 0.3 * rng.standard_normal((n_queries, K)))) + 0.05
    b_true = (e_q @ (rng.standard_normal((q_dim, 1)) / np.sqrt(q_dim))).ravel() \
        + 0.5 * rng.standard_normal(n_queries)
    theta_true = rng.standard_normal((n_models, K))
    p = 1.0 / (1.0 + np.exp(-(np.einsum("qk,mk->qm", a_true, theta_true) - b_true[:, None])))
    return Synthetic(
        e_q=e_q, theta_true=theta_true, a_true=a_true, b_true=b_true, p=p,
        y=(rng.random(p.shape) < p).astype(np.float32),
        query_ids=[f"sq{i}" for i in range(n_queries)],
        model_ids=[f"sm{j}" for j in range(n_models)], K=K, head="bernoulli")


def _core(n_queries, n_models, K, q_dim, rng):
    e_q = rng.standard_normal((n_queries, q_dim)).astype(np.float32)
    a = np.log1p(np.exp(e_q @ (rng.standard_normal((q_dim, K)) / np.sqrt(q_dim)))) + 0.1
    b = (e_q @ (rng.standard_normal((q_dim, 1)) / np.sqrt(q_dim))).ravel()
    theta = rng.standard_normal((n_models, K))
    z = np.einsum("qk,mk->qm", a, theta) - b[:, None]
    return e_q, a, b, theta, z


def make_synthetic_continuous(head: str, *, n_queries=1500, n_models=12, K=4, q_dim=32,
                              seed=0) -> Synthetic:
    rng = np.random.default_rng(seed)
    e_q, a, b, theta, z = _core(n_queries, n_models, K, q_dim, rng)
    w = rng.standard_normal(q_dim) / np.sqrt(q_dim)
    drive = e_q @ w                                    # per-query dispersion driver
    pi_true = None

    if head == "normal":
        sigma = np.exp(-1.0 + 0.9 * drive).clip(0.05, 3.0)
        y = z + sigma[:, None] * rng.standard_normal(z.shape)
        disp = sigma
    elif head == "beta":
        kappa = np.exp(1.5 + 1.2 * np.tanh(drive))
        mu = 1 / (1 + np.exp(-z))
        y = rng.beta(mu * kappa[:, None], (1 - mu) * kappa[:, None])
        disp = kappa
    elif head == "zoib":
        kappa = np.exp(1.5 + 1.0 * np.tanh(drive))
        mu = 1 / (1 + np.exp(-z))
        logits = np.stack([-1.2 + drive, -1.5 - drive, np.full_like(drive, 1.0)], axis=1)
        pi = np.exp(logits - logits.max(1, keepdims=True))
        pi_true = pi / pi.sum(1, keepdims=True)         # (Q, 3): pi0, pi1, pic
        comp = np.array([rng.choice(3, p=pi_true[q]) for q in range(n_queries)])
        beta_s = rng.beta(mu * kappa[:, None], (1 - mu) * kappa[:, None])
        y = np.where(comp[:, None] == 0, 0.0, np.where(comp[:, None] == 1, 1.0, beta_s))
        disp = kappa
    else:
        raise ValueError(head)

    return Synthetic(
        e_q=e_q, theta_true=theta, z=z, b_true=b, disp_true=disp,
        y=np.clip(y, 0.0, 1.0).astype(np.float32), pi_true=pi_true,
        query_ids=[f"sq{i}" for i in range(n_queries)],
        model_ids=[f"sm{j}" for j in range(n_models)], K=K, head=head)


# --------------------------------------------------------------------------- #
# arrays                                                                      #
# --------------------------------------------------------------------------- #
def to_arrays(syn: Synthetic, split_frac=0.8, seed=0):
    """``(train_arrays, val_arrays)`` as :class:`BaselineArrays` (no relevance / warm-up)."""
    Q, M = syn.y.shape
    order = np.random.default_rng(seed).permutation(Q)
    cut = int(Q * split_frac)
    model_index = {m: i for i, m in enumerate(syn.model_ids)}
    bernoulli = syn.head == "bernoulli"

    def _pack(qidx):
        qi, mi = np.repeat(qidx, M), np.tile(np.arange(M), len(qidx))
        ys = syn.y[qi, mi].astype(np.float32)
        y = ys if bernoulli else (ys >= 0.5).astype(np.float32)
        y_soft = syn.p[qi, mi].astype(np.float32) if bernoulli else ys
        return BaselineArrays(
            e_q=syn.e_q[qi], midx=mi.astype(np.int64), y=y, y_soft=y_soft,
            query_ids=np.array([syn.query_ids[i] for i in qi]),
            model_ids=np.array([syn.model_ids[j] for j in mi]),
            e_m=np.zeros((len(qi), 1), dtype=np.float32),
            meta={"model_index": model_index, "binary_threshold": 0.5, "score_kind": "synthetic",
                  "relevance_available": False, "warmup_available": False,
                  "split": "synthetic", "pathway": "synthetic"})

    return _pack(order[:cut]), _pack(order[cut:])


# --------------------------------------------------------------------------- #
# recovery report                                                             #
# --------------------------------------------------------------------------- #
def recovery_report(model, syn: Synthetic, val_arrays) -> dict:
    return (_recovery_bernoulli if syn.head == "bernoulli" else _recovery_continuous)(
        model, syn, val_arrays
    )


def _recovery_bernoulli(model, syn: Synthetic, val_arrays) -> dict:
    from scipy.stats import spearmanr

    from router.nirt.metrics import prediction_metrics

    from .data import batched_forward

    fwd = batched_forward(model, val_arrays, fields=("proba", "b_q"))
    pm = prediction_metrics(val_arrays.y, fwd["proba"])
    theta_hat = model.theta.weight.detach().numpy()

    pos = {q: k for k, q in enumerate(syn.query_ids)}
    rows = list({q: i for i, q in enumerate(val_arrays.query_ids)}.values())
    b_true = np.array([syn.b_true[pos[val_arrays.query_ids[i]]] for i in rows])
    b_rho = float(spearmanr(fwd["b_q"][rows], b_true).statistic)

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


def _recovery_continuous(model, syn: Synthetic, val_arrays) -> dict:
    from scipy.stats import spearmanr

    from .data import batched_forward

    disp_field = "sigma" if syn.head == "normal" else "kappa"
    fw = batched_forward(model, val_arrays, fields=("mean", "b_q", disp_field, "nll", "std"),
                         y=val_arrays.y_soft)
    pos = {q: k for k, q in enumerate(syn.query_ids)}
    qi = np.array([pos[q] for q in val_arrays.query_ids])
    mi = np.array([syn.model_ids.index(m) for m in val_arrays.model_ids])

    if syn.head == "normal":
        mu_true = syn.z[qi, mi]
    elif syn.head == "beta":
        mu_true = 1 / (1 + np.exp(-syn.z[qi, mi]))
    else:  # zoib: E[Y] = pi1 + pic * sigmoid(z)
        mb = 1 / (1 + np.exp(-syn.z[qi, mi]))
        mu_true = syn.pi_true[qi, 1] + syn.pi_true[qi, 2] * mb
    mean_corr = float(np.corrcoef(fw["mean"], mu_true)[0, 1])
    rmse = float(np.sqrt(np.mean((fw["mean"] - mu_true) ** 2)))

    rows = list({q: i for i, q in enumerate(val_arrays.query_ids)}.values())
    b_true = np.array([syn.b_true[pos[val_arrays.query_ids[i]]] for i in rows])
    disp_true = np.array([syn.disp_true[pos[val_arrays.query_ids[i]]] for i in rows])
    b_rho = float(spearmanr(fw["b_q"][rows], b_true).statistic)
    disp_rho = float(spearmanr(fw[disp_field][rows], disp_true).statistic)

    theta_hat = model.theta.weight.detach().numpy()
    theta_al = theta_hat @ np.linalg.lstsq(theta_hat, syn.theta_true, rcond=None)[0]
    abil_rho = float(spearmanr(theta_al.mean(1), syn.theta_true.mean(1)).statistic)

    nll = float(np.mean(fw["nll"]))
    rep = {
        "head": syn.head, "mean_corr": mean_corr, "mean_rmse": rmse, "nll": nll,
        "difficulty_spearman": b_rho, "ability_spearman_1d": abil_rho,
        "dispersion_spearman": disp_rho, "nll_finite": bool(np.isfinite(nll)),
        "mean_std": float(np.mean(fw["std"])),
    }
    if syn.head == "zoib":
        b0 = batched_forward(model, val_arrays, fields=("pi0",))["pi0"]
        b1 = batched_forward(model, val_arrays, fields=("pi1",))["pi1"]
        rep["pred_pi0_mean"] = float(b0.mean())
        rep["pred_pi1_mean"] = float(b1.mean())
        rep["true_frac0"] = float((val_arrays.y_soft <= 0).mean())
        rep["true_frac1"] = float((val_arrays.y_soft >= 1).mean())
    rep["passed"] = bool(
        rep["nll_finite"] and mean_corr > 0.85 and b_rho > 0.5 and abil_rho > 0.5
        and abs(disp_rho) > 0.25)
    return rep


# back-compat aliases (some callers import the old dataclass names)
SyntheticIRT = Synthetic
SyntheticContinuous = Synthetic

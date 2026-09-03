"""Synthetic recovery gates for the Phase 2 continuous response heads.

For each head, draw a well-specified world with a known IRT core
(``theta_true``, ``a_true``, ``b_true`` -> ``z``) plus the head's own noise
parameters, train ``BaselineNIRT`` with that head, and check recovery (up to the
unidentifiable rotation/scale):

* predictive mean close to the true mean,
* difficulty / ability ordering (Spearman),
* the head's dispersion parameter ordering (sigma / kappa),
* finite NLL, sane uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline_data import BaselineArrays


@dataclass
class SyntheticContinuous:
    e_q: np.ndarray
    theta_true: np.ndarray       # (M, K)
    z: np.ndarray                # (Q, M)  a.theta - b
    b_true: np.ndarray           # (Q,)
    disp_true: np.ndarray        # (Q,)  sigma (normal) or kappa (beta/zoib)
    y: np.ndarray                # (Q, M) in [0, 1]
    pi_true: np.ndarray | None   # (Q, 3) for zoib else None
    query_ids: list
    model_ids: list
    K: int
    head: str


def _core(n_queries, n_models, K, q_dim, rng):
    e_q = rng.standard_normal((n_queries, q_dim)).astype(np.float32)
    a = np.log1p(np.exp(e_q @ (rng.standard_normal((q_dim, K)) / np.sqrt(q_dim)))) + 0.1
    b = (e_q @ (rng.standard_normal((q_dim, 1)) / np.sqrt(q_dim))).ravel()
    theta = rng.standard_normal((n_models, K))
    z = np.einsum("qk,mk->qm", a, theta) - b[:, None]
    return e_q, a, b, theta, z


def make_synthetic_continuous(head: str, *, n_queries=1500, n_models=12, K=4, q_dim=32,
                              seed=0) -> SyntheticContinuous:
    rng = np.random.default_rng(seed)
    e_q, a, b, theta, z = _core(n_queries, n_models, K, q_dim, rng)
    w = rng.standard_normal(q_dim) / np.sqrt(q_dim)
    drive = e_q @ w                                    # per-query dispersion driver
    pi_true = None

    if head == "normal":
        sigma = np.exp(-1.0 + 0.9 * drive).clip(0.05, 3.0)   # (Q,) log-linear in the driver
        y = z + sigma[:, None] * rng.standard_normal(z.shape)
        disp = sigma
    elif head == "beta":
        kappa = np.exp(1.5 + 1.2 * np.tanh(drive))      # (Q,)  ~ [1, 30]
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

    return SyntheticContinuous(
        e_q=e_q, theta_true=theta, z=z, b_true=b, disp_true=disp,
        y=np.clip(y, 0.0, 1.0).astype(np.float32), pi_true=pi_true,
        query_ids=[f"sq{i}" for i in range(n_queries)],
        model_ids=[f"sm{j}" for j in range(n_models)], K=K, head=head)


def to_arrays(syn: SyntheticContinuous, split_frac=0.8, seed=0):
    Q, M = syn.y.shape
    order = np.random.default_rng(seed).permutation(Q)
    cut = int(Q * split_frac)
    model_index = {m: i for i, m in enumerate(syn.model_ids)}

    def _pack(qidx):
        qi, mi = np.repeat(qidx, M), np.tile(np.arange(M), len(qidx))
        ys = syn.y[qi, mi].astype(np.float32)
        return BaselineArrays(
            e_q=syn.e_q[qi], midx=mi.astype(np.int64),
            y=(ys >= 0.5).astype(np.float32), y_soft=ys,
            query_ids=np.array([syn.query_ids[i] for i in qi]),
            model_ids=np.array([syn.model_ids[j] for j in mi]),
            e_m=np.zeros((len(qi), 1), dtype=np.float32),
            meta={"model_index": model_index, "binary_threshold": 0.5, "score_kind": "synthetic",
                  "relevance_available": False, "warmup_available": False,
                  "split": "synthetic", "pathway": "synthetic"})

    return _pack(order[:cut]), _pack(order[cut:])


def recovery_report(model, syn: SyntheticContinuous, val_arrays) -> dict:
    from scipy.stats import spearmanr

    from .baseline_data import batched_forward

    head = model.response_head
    disp_field = "sigma" if syn.head == "normal" else "kappa"
    fw = batched_forward(model, val_arrays, fields=("mean", "b_q", disp_field, "nll", "std"),
                         y=val_arrays.y_soft)
    pos = {q: k for k, q in enumerate(syn.query_ids)}
    qi = np.array([pos[q] for q in val_arrays.query_ids])
    mi = np.array([syn.model_ids.index(m) for m in val_arrays.model_ids])

    # true predictive mean E[Y] (identifiable only up to the IRT scale -> correlation)
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
        b0, b1 = batched_forward(model, val_arrays, fields=("pi0", "pi1"))["pi0"], \
            batched_forward(model, val_arrays, fields=("pi1",))["pi1"]
        rep["pred_pi0_mean"] = float(b0.mean())
        rep["pred_pi1_mean"] = float(b1.mean())
        rep["true_frac0"] = float((val_arrays.y_soft <= 0).mean())
        rep["true_frac1"] = float((val_arrays.y_soft >= 1).mean())
    rep["passed"] = bool(
        rep["nll_finite"] and mean_corr > 0.85 and b_rho > 0.5 and abil_rho > 0.5
        and abs(disp_rho) > 0.25)
    return rep

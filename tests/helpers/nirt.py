"""Synthetic worlds + fakes for the ``router.nirt`` (``NIRTModel``) family.

These back the model / training / routing / cold-start tests. The RNG draw order
in :func:`make_synthetic_irt` is load-bearing for recovery thresholds -- do not
"tidy" it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from helpers.stores import make_store


def make_synthetic_irt(n_queries=1200, q_dim=16, n_models=6, K=2, seed=0,
                       orientation="query_latent", informative_profiles=False):
    """A well-specified IRT world: ``y ~ Bernoulli(sigma(a . theta - b))`` with the
    latent vector on the query (``query_latent``) or the model (``model_latent``).

    Returns ``(train_obs, val_obs, q_store, m_store)``; an 80/20 query split.
    """
    rng = np.random.default_rng(seed)
    Eq = rng.standard_normal((n_queries, q_dim)).astype(np.float32)
    W = rng.standard_normal((q_dim, K)) / np.sqrt(q_dim)

    if orientation == "query_latent":
        theta_q = Eq @ W                                  # (n_queries, K)
        a_m = np.abs(rng.standard_normal((n_models, K))) + 0.5
        b_m = rng.standard_normal(n_models) * 0.5
        def logit(i, j): return float(a_m[j] @ theta_q[i] - b_m[j])
        prof_signal = np.hstack([a_m, b_m[:, None]])
    else:  # model_latent
        A_q = np.abs(Eq @ W) + 0.3                        # query discrimination (n_queries, K)
        b_q = (Eq @ rng.standard_normal((q_dim, 1))).ravel() * 0.5
        theta_m = rng.random((n_models, K))               # model ability in [0, 1]
        def logit(i, j): return float(A_q[i] @ theta_m[j] - b_q[i])
        prof_signal = theta_m

    Em = rng.standard_normal((n_models, 5)).astype(np.float32)
    if informative_profiles:
        Em = np.hstack([prof_signal, 0.05 * rng.standard_normal((n_models, 3))]).astype(np.float32)

    qids = [f"q{i}" for i in range(n_queries)]
    mids = [f"m{j}" for j in range(n_models)]
    rows = []
    for i in range(n_queries):
        for j in range(n_models):
            p = 1.0 / (1.0 + np.exp(-logit(i, j)))
            rows.append((qids[i], mids[j], float(rng.random() < p), p))
    obs = pd.DataFrame(rows, columns=["query_id", "model_id", "target", "p_true"])
    obs["metric_type"] = "accuracy"
    obs["source"] = "synth"
    obs["cost"] = np.nan

    cut = int(n_queries * 0.8)
    train_obs = obs[obs["query_id"].isin(qids[:cut])].reset_index(drop=True)
    val_obs = obs[obs["query_id"].isin(qids[cut:])].reset_index(drop=True)
    return train_obs, val_obs, make_store(qids, Eq, "query_id"), make_store(mids, Em, "model_id")


def nirt_datasets(train_obs, val_obs, q_store, m_store):
    """``(train_ds, val_ds)`` -- accepts the 4-tuple from :func:`make_synthetic_irt`."""
    from router.data.nirt import NIRTDataset

    return (
        NIRTDataset(train_obs, q_store, m_store),
        NIRTDataset(val_obs, q_store, m_store),
    )


def make_split_obs(n_queries=900, n_models=9, K=2, seed=0):
    """A synthetic observation table with train/validation/test splits + per-model cost."""
    rng = np.random.default_rng(seed)
    Eq = rng.standard_normal((n_queries, 12))
    theta = Eq @ rng.standard_normal((12, K))          # strong latent signal
    a = np.abs(rng.standard_normal((n_models, K))) + 1.0
    b = rng.standard_normal(n_models)
    model_cost = np.linspace(0.0001, 0.004, n_models)
    qids = [f"q{i}" for i in range(n_queries)]
    mids = [f"m{j}" for j in range(n_models)]
    parts = np.array(["train"] * int(n_queries * 0.6)
                     + ["validation"] * int(n_queries * 0.2)
                     + ["test"] * (n_queries - int(n_queries * 0.6) - int(n_queries * 0.2)))
    rng.shuffle(parts)
    rows = []
    for i in range(n_queries):
        for j in range(n_models):
            p = 1.0 / (1.0 + np.exp(-(a[j] @ theta[i] - b[j])))
            rows.append((qids[i], mids[j], float(rng.random() < p), parts[i],
                         "accuracy", "synth", float(model_cost[j])))
    return pd.DataFrame(rows, columns=["query_id", "model_id", "target", "split",
                                       "metric_type", "source", "cost"])


class FakeTrainingData:
    """A partial ``TrainingData`` stand-in.

    Each accessor returns whatever store/frame it was given, or ``None`` -- which
    is exactly what the real facade does when an artifact is missing, so the code
    under test raises its own ``FileNotFoundError``. Pass ``query_dim`` to
    synthesise a query store sized to ``observations``.
    """

    def __init__(self, observations=None, *, query_store=None, profile_store=None,
                 feature_store=None, battles=None, query_dim=None):
        if query_dim and query_store is None and observations is not None:
            qids = sorted(observations["query_id"].unique())
            mat = np.random.default_rng(0).standard_normal((len(qids), query_dim))
            query_store = make_store(qids, mat, "query_id")
        self.observations = observations
        self._battles = battles
        self._query_store = query_store
        self._profile_store = profile_store
        self._feature_store = feature_store

    def nirt_observations(self):
        return self.observations

    def query_embeddings(self, pathway=None):
        return self._query_store

    def profile_embeddings(self, pathway=None):
        return self._profile_store

    def query_features(self, name=None):
        return self._feature_store

    def pairwise(self, source=None, split=None, models="all"):
        df = self._battles
        return df[df["source"] == source].copy() if source else df.copy()


def nirt_cfg(**model_over):
    """The compact ``fit`` config used by the ``router.nirt`` training tests."""
    model = {"dim": 1, "model_params": "projected", "query_hidden": 64,
             "constrain_discrimination": "auto"}
    model.update(model_over)
    return {
        "seed": 0,
        "data": {"pathway": "irt"},
        "model": model,
        "train": {"loss": "soft_bce", "lr": 1e-2, "weight_decay": 0.0,
                  "batch_size": 512, "epochs": 3, "patience": 10, "val_metric": "bce"},
    }


def toy_pairwise_arrays(n=40, q_dim=10, m_dim=7, seed=0):
    """A ``PairwiseArrays`` of random Arena/Judge-style battles."""
    from training.nirt.pairwise import PairwiseArrays

    rng = np.random.default_rng(seed)
    return PairwiseArrays(
        e_q=rng.standard_normal((n, q_dim)).astype(np.float32),
        e_a=rng.standard_normal((n, m_dim)).astype(np.float32),
        e_b=rng.standard_normal((n, m_dim)).astype(np.float32),
        y=rng.integers(0, 2, n).astype(np.float32),
        n_dropped=0,
    )

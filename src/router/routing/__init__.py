"""Modular routing layer: one :class:`Router` interface, many strategies.

A **router** turns per-``(query, model)`` quality predictions into a model choice
per query. This package fixes the contract (:class:`Router`) and the shared
selection machinery so a new strategy -- NIRT, k-NN, an MLP head, a bandit, an
LLM-judge cascade -- only has to answer "how good is model ``m`` for query
``q``". Comparing routers against oracle ground truth needs labels, so it
lives in :func:`evaluation.routing.oracle.compare_routers`, not here.

    from router.routing import NIRTRouter, KNNRouter
    from training.data.facade import load_training_data
    from router.config import load_config
    from evaluation.nirt.routing import eval_matrices
    from evaluation.routing.oracle import compare_routers

    d = load_training_data(load_config())
    true_df, cost_df = eval_matrices(d, split="test")
    qids = list(true_df.index)

    routers = [
        NIRTRouter.from_run("nirt-2d-projected", data=d),
        KNNRouter(d, k=5),
    ]

    routers[0].route(qids, lam=0.3).to_frame()          # the decision
    summary, _ = compare_routers(routers, true_df, cost_df, lam=0.3)

Register a new strategy so it is reachable by name::

    from router.routing import Router, register

    @register
    class BanditRouter(Router):
        kind = "bandit"
        def predict_scores(self, query_ids):
            ...

    build_router("bandit", model_ids=[...])
"""

from __future__ import annotations

from .base import Router, RoutingResult, UnsupportedCandidatePolicy
from .registry import REGISTRY, build_router, register
from .routers import (
    KNNRouter,
    MatrixRouter,
    MLPRouter,
    NIRTRouter,
    RandomRouter,
)

__all__ = [
    "Router",
    "RoutingResult",
    "UnsupportedCandidatePolicy",
    "REGISTRY",
    "register",
    "build_router",
    "MatrixRouter",
    "NIRTRouter",
    "KNNRouter",
    "MLPRouter",
    "RandomRouter",
]

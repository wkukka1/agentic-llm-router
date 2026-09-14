"""The :class:`Router` interface -- one contract every routing strategy implements.

A router is a **decision layer** over a fixed candidate pool: given some queries it
predicts each model's quality ``E[Y | q, m]`` and then picks one model per query.
Only :meth:`Router.predict_scores` is abstract -- the pool bookkeeping and the
``argmax`` / cost-aware selection
(:func:`router.nirt.routing_decision.routing_decision`) are shared here so a
new strategy is just "how do I score ``(query, model)``". Oracle-relative
scoring needs ground truth, so it is not a method on ``Router`` -- see
:func:`evaluation.routing.oracle.evaluate_router`.

    from router.routing import NIRTRouter

    r = NIRTRouter.from_run("nirt-2d-projected")        # a concrete strategy
    res = r.route(query_ids, lam=0.3)                   # -> RoutingResult
    res.to_frame()                                      # query_id, selected_model_id, ...

Writing a new router (e.g. a bandit, an LLM-judge cascade)::

    from router.routing import Router, register

    @register
    class MyRouter(Router):
        kind = "my_router"
        def predict_scores(self, query_ids):
            ...  # -> DataFrame [query_id x model_id] of predicted quality

``query_ids`` are ids into the project's prebuilt query set / embedding stores --
the same currency the rest of the stack uses. Strategies that can also score
**raw text** (an encoder in front of the scorer) override
:meth:`predict_scores_text` and set :attr:`can_route_text` -- that is the entry
point the agentic orchestrator (:mod:`router.agentic`) uses for live prompts.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..nirt.routing_decision import routing_decision

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..constraints import RoutingConstraints
    from ..models.artifacts import RouterModelArtifact

__all__ = ["Router", "RoutingResult", "UnsupportedCandidatePolicy"]


class UnsupportedCandidatePolicy(enum.Enum):
    """What :meth:`Router.filter_candidates` does with a candidate this router
    has no score for (not in :attr:`Router.model_ids`)."""

    ERROR = "error"
    DROP = "drop"
    SCORE_WITH_PRIOR = "score_with_prior"


# --------------------------------------------------------------------------- #
# result                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class RoutingResult:
    """What a router decided for a batch of queries.

    ``scores`` is the dense ``[Q x M]`` predicted-quality matrix the decision was
    made from (rows = ``query_ids``, cols = ``model_ids``); ``selected`` is the
    per-query column index that was picked.
    """

    query_ids: list[str]
    model_ids: list[str]
    scores: np.ndarray
    selected: np.ndarray
    lam: float = 0.0
    model_costs: Optional[np.ndarray] = None

    @property
    def selected_model_ids(self) -> list[str]:
        return [self.model_ids[i] for i in self.selected]

    @property
    def selected_scores(self) -> np.ndarray:
        """Predicted quality of the chosen model, per query."""
        return self.scores[np.arange(len(self.selected)), self.selected]

    def model_mix(self) -> dict[str, int]:
        """How often each model was chosen (only models chosen at least once)."""
        idx, counts = np.unique(self.selected, return_counts=True)
        return {self.model_ids[i]: int(c) for i, c in zip(idx, counts)}

    def to_frame(self) -> pd.DataFrame:
        """One row per query: the chosen model and its predicted score."""
        return pd.DataFrame(
            {
                "query_id": self.query_ids,
                "selected_model_id": self.selected_model_ids,
                "selected_score": self.selected_scores,
            }
        )

    def score_frame(self) -> pd.DataFrame:
        """The full ``[query_id x model_id]`` predicted-quality matrix."""
        return pd.DataFrame(self.scores, index=self.query_ids, columns=self.model_ids)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _resolve_costs(
    costs: Optional[Sequence[float] | Mapping[str, float] | np.ndarray],
    model_ids: Sequence[str],
) -> Optional[np.ndarray]:
    """Coerce a per-model cost spec to a length-``M`` vector in ``model_ids`` order."""
    if costs is None:
        return None
    if isinstance(costs, Mapping):
        missing = [m for m in model_ids if m not in costs]
        if missing:
            raise KeyError(f"model_costs is missing entries for {missing}")
        return np.array([float(costs[m]) for m in model_ids], dtype=np.float64)
    arr = np.asarray(costs, dtype=np.float64).ravel()
    if arr.shape != (len(model_ids),):
        raise ValueError(
            f"model_costs has length {arr.shape[0]}, expected {len(model_ids)} "
            f"(one per model, in model_ids order)"
        )
    return arr


# --------------------------------------------------------------------------- #
# the interface                                                               #
# --------------------------------------------------------------------------- #
class Router(abc.ABC):
    """Abstract base class for a routing strategy over a fixed candidate pool.

    Subclasses implement :meth:`predict_scores`. Everything else -- selection,
    cost-awareness -- is provided. Oracle-relative evaluation is a free
    function in ``evaluation``, not a method here (it needs ground truth).
    """

    #: short, stable identifier for the strategy (used by the registry and as the
    #: default ``name``). Override in every concrete subclass.
    kind: ClassVar[str] = "router"

    #: ``True`` when the subclass implements :meth:`predict_scores_text` (scoring
    #: from raw prompt text via an encoder, no prebuilt id required).
    can_route_text: ClassVar[bool] = False

    def __init__(self, model_ids: Sequence[str], *, name: Optional[str] = None,
                 unsupported_policy: UnsupportedCandidatePolicy = UnsupportedCandidatePolicy.DROP):
        if not len(list(model_ids)):
            raise ValueError("a router needs a non-empty candidate pool")
        self._model_ids = [str(m) for m in model_ids]
        self.name = name or self.kind
        self.unsupported_policy = unsupported_policy

    # -- pool ----------------------------------------------------------------
    @property
    def model_ids(self) -> list[str]:
        """The candidate pool, in the column order :meth:`predict_scores` returns."""
        return list(self._model_ids)

    def supported_models(self) -> list[str]:
        return self.model_ids

    def filter_candidates(self, candidates: Sequence, constraints: "RoutingConstraints") -> list:
        """Drop candidates this router can't score, and any missing a
        required capability. Concrete and inherited -- every strategy gets the
        same filtering, only :meth:`predict_scores` varies.

        ``candidates`` may be bare model-id strings or objects exposing
        ``model_id`` / ``capabilities`` (e.g. an ``LLMProfile``); duck-typed so
        this works before :mod:`router.llm` is wired to any router.
        """
        supported = set(self.supported_models())
        required = set(getattr(constraints, "required_capabilities", None) or [])
        out = []
        for c in candidates:
            model_id = str(getattr(c, "model_id", c))
            if model_id not in supported:
                if self.unsupported_policy is UnsupportedCandidatePolicy.ERROR:
                    raise ValueError(f"{model_id!r} is not a supported candidate for {self.name!r}")
                if self.unsupported_policy is UnsupportedCandidatePolicy.DROP:
                    continue
                # SCORE_WITH_PRIOR: kept, scored later with whatever prior predict_scores gives an
                # unseen column (typically NaN -> excluded by routing_decision's -inf fallback).
            if required and not required.issubset(set(getattr(c, "capabilities", []) or [])):
                continue
            out.append(c)
        return out

    # -- artifact lifecycle --------------------------------------------------
    def save(self, path: str) -> "RouterModelArtifact":
        """Persist this router's fitted state and return its artifact record.

        No concrete router implements this yet -- today's checkpoints are
        written directly by the training loop (``training.nirt.train.fit``)
        rather than through the served router object. Override in a subclass
        that owns state worth round-tripping through a
        :class:`~router.models.artifacts.RouterModelArtifact`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, artifact: "RouterModelArtifact") -> None:
        """Restore fitted state from an artifact record, in place.

        The working equivalent today is a classmethod constructor per router
        (e.g. :meth:`~router.routing.routers.NIRTRouter.from_run`) rather than
        mutating an existing instance from a typed artifact.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support load()")

    @property
    def default_model_costs(self) -> Optional[np.ndarray]:
        """Per-model cost vector used when :meth:`route` is called without an
        explicit ``model_costs`` (``None`` -> quality-only unless a cost is
        passed). Data-backed routers override this with a real cost signal."""
        return None

    # -- the one thing a subclass must provide ------------------------------
    @abc.abstractmethod
    def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        """Predicted quality ``E[Y | q, m]`` as a dense ``[query_id x model_id]``
        frame. Rows should cover the requested ``query_ids``; columns should be a
        superset of :attr:`model_ids`. Alignment / missing cells are handled by
        the base class."""

    def predict_scores_text(self, texts: Sequence[str], *, encoder=None) -> pd.DataFrame:
        """Predicted quality for raw prompt strings -- a ``[0..N-1 x model_id]``
        frame (rows positional, in ``texts`` order).

        Only strategies with an encoder-backed scorer implement this
        (:attr:`can_route_text`). The default raises; pre-resolve the text to a
        ``query_id`` and use :meth:`predict_scores` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot score raw text (can_route_text=False). "
            "Use NIRTRouter / KNNRouter, or resolve the prompt to a query_id first."
        )

    # -- shared machinery --------------------------------------------------
    def aligned_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        ids = [str(q) for q in query_ids]
        raw = self.predict_scores(ids)
        return raw.reindex(index=ids, columns=self._model_ids).astype(np.float64)

    def _route_from_scores(
        self,
        scores: pd.DataFrame,
        *,
        lam: float,
        model_costs,
        eligible,
    ) -> RoutingResult:
        scores = scores.reindex(columns=self._model_ids).astype(np.float64)
        costs = (
            _resolve_costs(model_costs, self._model_ids)
            if model_costs is not None
            else self.default_model_costs
        )
        elig = eligible
        if isinstance(eligible, pd.DataFrame):
            elig = eligible.reindex(index=scores.index, columns=self._model_ids).to_numpy(bool)

        mat = scores.to_numpy(np.float64)
        selected = routing_decision(mat, lam=lam, model_costs=costs, eligible=elig)
        return RoutingResult(
            query_ids=[str(q) for q in scores.index],
            model_ids=self.model_ids,
            scores=mat,
            selected=selected,
            lam=float(lam),
            model_costs=costs,
        )

    def route(
        self,
        query_ids: Sequence[str],
        *,
        lam: float = 0.0,
        model_costs: Optional[Sequence[float] | Mapping[str, float] | np.ndarray] = None,
        eligible: Optional[pd.DataFrame | np.ndarray] = None,
    ) -> RoutingResult:
        """Pick one model per query.

        ``lam`` weights cost in the utility ``pred - lam * C(m)`` (``lam = 0`` ->
        quality-only). ``model_costs`` overrides :attr:`default_model_costs`;
        ``eligible`` is an optional ``[Q x M]`` bool mask of selectable cells
        (a DataFrame is aligned by label, an array is taken positionally).
        """
        return self._route_from_scores(
            self.aligned_scores(query_ids),
            lam=lam, model_costs=model_costs, eligible=eligible,
        )

    __call__ = route

    def route_text(
        self,
        texts: Sequence[str],
        *,
        encoder=None,
        lam: float = 0.0,
        model_costs: Optional[Sequence[float] | Mapping[str, float] | np.ndarray] = None,
        eligible: Optional[pd.DataFrame | np.ndarray] = None,
    ) -> RoutingResult:
        """:meth:`route`, but from raw prompt text (via :meth:`predict_scores_text`).

        ``RoutingResult.query_ids`` are the positional indices ``"0"``, ``"1"``,
        ... of the input list.
        """
        scores = self.predict_scores_text(list(texts), encoder=encoder)
        scores = scores.copy()
        scores.index = [str(i) for i in range(len(scores))]
        return self._route_from_scores(
            scores, lam=lam, model_costs=model_costs, eligible=eligible,
        )

    # Oracle-relative evaluation needs ground truth and lives in
    # evaluation.routing.oracle.evaluate_router(router, true_df, cost_df, lam)
    # instead of a method here -- see that function's docstring.

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, n_models={len(self._model_ids)})"

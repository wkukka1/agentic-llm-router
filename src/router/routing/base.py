"""The :class:`RouterModel` interface -- one contract every routing strategy implements.

A router model is a **decision layer** over a fixed candidate pool: given some queries it
predicts each model's quality ``E[Y | q, m]`` and then picks one model per query.
Only :meth:`RouterModel.predict_scores` is abstract -- the pool bookkeeping and the
``argmax`` / cost-aware selection
(:func:`router.nirt.routing_decision.routing_decision`) are shared here so a
new strategy is just "how do I score ``(query, model)``". Oracle-relative
scoring needs ground truth, so it is not a method on ``RouterModel`` -- see
:func:`evaluation.routing.oracle.evaluate_router`.

    from router.routing import NIRTRouter

    r = NIRTRouter.from_run("nirt-2d-projected")        # a concrete strategy
    res = r.route(query_ids, lam=0.3)                   # -> RoutingResult
    res.to_frame()                                      # query_id, selected_model_id, ...

Writing a new router (e.g. a bandit, an LLM-judge cascade)::

    from router.routing import RouterModel, register

    @register
    class MyRouter(RouterModel):
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
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..nirt.routing_decision import (
    alpha_to_lam,
    cost_aware_utility,
    linear_cost,
    no_selectable_rows,
    routing_decision,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..constraints import RoutingConstraints
    from ..llm.cost import CostModel
    from ..models.artifacts import RouterModelArtifact

__all__ = ["RouterModel", "RoutingResult", "UnsupportedCandidatePolicy"]


class UnsupportedCandidatePolicy(enum.Enum):
    """What :meth:`RouterModel.filter_candidates` does with a candidate this router
    has no score for (not in :attr:`RouterModel.model_ids`)."""

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

    ``fallback`` flags rows where nothing was selectable (no finite score, or
    an all-False ``eligible`` row). Their ``selected`` is column 0 by
    convention; it is *not* a real choice and may even be an ineligible model,
    so callers must check this before dispatching. ``lam`` is the cost weight
    actually applied (``0`` when no cost vector resolved), and ``margin`` the
    quality the selection was allowed to give up to reach a cheaper model
    (``0`` = plain argmax, and likewise ``0`` when no cost vector resolved).
    Both record what was *applied*, not what was asked for, so a result can be
    read back without knowing how it was called. ``alpha`` is the
    IRT-Router-style reward blend requested of :meth:`RouterModel.route`
    (``None`` for an ordinary ``lam``-driven or quality-only route -- ``lam``
    still records the equivalent weight that was actually applied).
    """

    query_ids: list[str]
    model_ids: list[str]
    scores: np.ndarray
    selected: np.ndarray
    lam: float = 0.0
    model_costs: Optional[np.ndarray] = None
    fallback: Optional[np.ndarray] = None
    alpha: Optional[float] = None
    margin: float = 0.0

    @property
    def any_fallback(self) -> bool:
        return self.fallback is not None and bool(np.asarray(self.fallback).any())

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

    def row_scores(self, i: int = 0) -> dict[str, float]:
        """``{model_id: predicted quality}`` for query row ``i`` (NaN kept)."""
        return dict(zip(self.model_ids, map(float, np.asarray(self.scores, dtype=np.float64)[i])))


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _resolve_costs(
    costs: Optional[Sequence[float] | Mapping[str, float] | np.ndarray],
    model_ids: Sequence[str],
) -> Optional[np.ndarray]:
    """Coerce a per-model cost spec to a length-``M`` vector in ``model_ids`` order.

    Non-finite entries are rejected: a NaN cost makes that model's utility NaN,
    which ``argmax`` would pick whenever ``lam > 0``."""
    if costs is None:
        return None
    if isinstance(costs, pd.Series):    # labelled: align by index like a Mapping, not by position
        costs = costs.to_dict()
    if isinstance(costs, Mapping):
        missing = [m for m in model_ids if m not in costs]
        if missing:
            raise KeyError(f"model_costs is missing entries for {missing}")
        arr = np.array([float(costs[m]) for m in model_ids], dtype=np.float64)
    else:
        arr = np.asarray(costs, dtype=np.float64).ravel()
        if arr.shape != (len(model_ids),):
            raise ValueError(
                f"model_costs has length {arr.shape[0]}, expected {len(model_ids)} "
                f"(one per model, in model_ids order)"
            )
    bad = [m for m, c in zip(model_ids, arr) if not np.isfinite(c)]
    if bad:
        raise ValueError(f"model_costs must be finite; non-finite for {bad}")
    return arr


def _align_cost_frame(frame: pd.DataFrame, query_ids: Sequence[str],
                      model_ids: Sequence[str]) -> np.ndarray:
    """A ``[query_id x model_id]`` cost frame as a dense ``[Q, M]`` array in
    exactly this row / column order. Missing rows, columns or non-finite cells
    raise: a silently misaligned ``[Q, M]`` cost would route every query wrong."""
    frame = frame.copy()
    frame.index = [str(q) for q in frame.index]
    frame.columns = [str(m) for m in frame.columns]
    ids, cols = [str(q) for q in query_ids], [str(m) for m in model_ids]
    miss_q = [q for q in ids if q not in frame.index]
    miss_m = [m for m in cols if m not in frame.columns]
    if miss_q or miss_m:
        raise ValueError(f"cost matrix is missing queries {miss_q[:5]} / models {miss_m[:5]}")
    arr = frame.loc[ids, cols].to_numpy(np.float64)
    if not np.isfinite(arr).all():
        r, c = np.nonzero(~np.isfinite(arr))
        raise ValueError(f"cost matrix has {len(r)} non-finite cells, e.g. "
                         f"{[(ids[i], cols[j]) for i, j in zip(r[:5], c[:5])]}")
    return arr


# --------------------------------------------------------------------------- #
# the interface                                                               #
# --------------------------------------------------------------------------- #
class RouterModel(abc.ABC):
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
                 unsupported_policy: UnsupportedCandidatePolicy = UnsupportedCandidatePolicy.DROP,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None):
        ids = [str(m) for m in model_ids]   # materialise once: may be an iterator
        if not ids:
            raise ValueError("a router needs a non-empty candidate pool")
        self._model_ids = ids
        self.name = name or self.kind
        self.unsupported_policy = unsupported_policy
        #: set by RouterModelFactory.create when this router was built from a
        #: RouterModelArtifact; None for the many routers still built directly
        #: (from_run, raw kwargs) with no artifact behind them.
        self.cost_model = cost_model
        self.artifact = artifact

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
                # SCORE_WITH_PRIOR: kept. RouterModel.route can't score a column outside the pool;
                # Router.route gives it the pool-mean prediction with confidence 0.
            capabilities = getattr(c, "capabilities", None)
            # None means "unknown" (e.g. a no-registry _StubProfile, ROUTER-003) --
            # can't be checked, so it passes rather than being treated as "known,
            # has none" (an explicit [] still filters, same as before).
            if required and capabilities is not None and not required.issubset(set(capabilities)):
                continue
            out.append(c)
        return out

    def warmup(self) -> dict[str, float]:
        """Pay every lazy, first-request cost now (call once at startup), so no
        routed request does disk I/O or model loading. Returns seconds per stage.

        The default resolves :attr:`default_model_costs` and, for a
        text-capable router, runs one throwaway :meth:`route_text` -- which
        loads the encoder and any stores the scorer reads. Subclasses add
        finer-grained stages before delegating here."""
        import time

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        _ = self.default_model_costs
        timings["costs"] = time.perf_counter() - t0
        if self.can_route_text:
            t0 = time.perf_counter()
            self.route_text(["warmup"])
            timings["route_text"] = time.perf_counter() - t0
        return timings

    def cost_by_model(self) -> dict[str, float]:
        """``{model_id: per-query cost}`` from :attr:`default_model_costs`, for
        every pool model with a finite cost (empty when the router has none) --
        the one lookup callers use to price a pool model on the router's scale."""
        costs = self.default_model_costs
        if costs is None:
            return {}
        return {m: float(c) for m, c in zip(self._model_ids, costs) if math.isfinite(float(c))}

    @property
    def default_model_costs(self) -> Optional[np.ndarray]:
        """Per-model cost vector used when :meth:`route` is called without an
        explicit ``model_costs`` (``None`` -> quality-only unless a cost is
        passed). Data-backed routers override this with a real cost signal."""
        return None

    def predict_costs(self, query_ids: Sequence[str]) -> Optional[pd.DataFrame]:
        """Predicted expected cost ``E[C | q, m]`` (USD) as a ``[query_id x
        model_id]`` frame, or ``None`` when this router has no per-cell cost
        model (routing then uses :attr:`default_model_costs`)."""
        return None

    def decision_costs(self, query_ids: Sequence[str], model_costs=None) -> Optional[np.ndarray]:
        """The cost a :meth:`route` call on ``query_ids`` selects with: explicit
        ``model_costs`` (a per-model spec, or a ``[query_id x model_id]`` frame),
        else :meth:`predict_costs`, else :attr:`default_model_costs`. Returns
        ``[M]``, ``[Q, M]`` or ``None``."""
        ids = [str(q) for q in query_ids]
        if isinstance(model_costs, pd.DataFrame):
            return _align_cost_frame(model_costs, ids, self._model_ids)
        if model_costs is not None:
            return _resolve_costs(model_costs, self._model_ids)
        predicted = self.predict_costs(ids)
        if predicted is not None:
            return _align_cost_frame(predicted, ids, self._model_ids)
        return self.default_model_costs

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
        costs: Optional[np.ndarray],
        eligible,
        alpha: Optional[float] = None,
        margin: float = 0.0,
    ) -> RoutingResult:
        scores = scores.reindex(columns=self._model_ids).astype(np.float64)
        if lam and costs is None:
            warnings.warn(
                f"{self.name}: lam={lam} requested but no model cost vector resolved; "
                "routing on quality only (RoutingResult.lam records 0.0)",
                stacklevel=3,
            )
            lam = 0.0
        elig = eligible
        if isinstance(eligible, pd.DataFrame):
            # missing cells fail closed (reindex would insert NaN -> bool True)
            arr = eligible.reindex(index=scores.index, columns=self._model_ids,
                                   fill_value=False).to_numpy()
            elig = np.where(pd.isna(arr), False, arr).astype(bool)

        mat = scores.to_numpy(np.float64)
        if margin and costs is None:
            warnings.warn(
                f"{self.name}: margin={margin} requested but no model cost vector "
                "resolved; selecting on quality only (RoutingResult.margin records 0.0)",
                stacklevel=3,
            )
            margin = 0.0
        selected = routing_decision(mat, lam=lam, model_costs=costs, eligible=elig,
                                    margin=margin)
        # the same cost-aware utility routing_decision actually selected on --
        # not a cost-blind np.isfinite(mat) copy (RR-01), which misses a
        # non-finite lam forcing every cell to -inf after `pred - lam*cost`
        util = cost_aware_utility(mat, lam=lam, model_costs=costs, eligible=elig)
        return RoutingResult(
            query_ids=[str(q) for q in scores.index],
            model_ids=self.model_ids,
            scores=mat,
            selected=selected,
            lam=float(lam),
            model_costs=costs,
            fallback=no_selectable_rows(util),
            alpha=alpha,
            margin=float(margin),
        )

    def _resolve_alpha(
        self, alpha: Optional[float], lam: float, costs: Optional[np.ndarray],
    ) -> tuple[float, Optional[np.ndarray]]:
        """``alpha`` -> ``(lam, costs)`` for :meth:`_route_from_scores`.

        The IRT-Router reward ``argmax[alpha*pred - (1-alpha)*linear_cost(C)]``
        is the same ordering as ``lam``-routing on ``linear_cost``-normalized
        costs with ``lam = alpha_to_lam(alpha)`` (both constant-positive-rescale
        the per-query utility) -- so alpha-routing reuses the exact same
        decision rule, just with a different ``(lam, costs)`` pair. The
        normalization range is always the pool's :attr:`default_model_costs`
        (IRT-Router's ``pool_mean_costs``), even when ``costs`` itself is a
        per-cell predicted cost -- a stable scale across calls, not one that
        shifts with whichever queries are in this batch.
        """
        if alpha is None:
            return lam, costs
        if lam:
            raise ValueError("RouterModel.route(): pass either lam or alpha, not both")
        pool = self.default_model_costs
        if pool is None:
            raise ValueError(
                f"{self.name}: alpha={alpha} requested but this router has no "
                "default_model_costs to normalize against"
            )
        if costs is None:
            raise ValueError(
                f"{self.name}: alpha={alpha} requested but no model cost vector "
                "resolved for this call"
            )
        return alpha_to_lam(alpha), linear_cost(costs, pool)

    def route(
        self,
        query_ids: Sequence[str],
        *,
        lam: float = 0.0,
        alpha: Optional[float] = None,
        model_costs: Optional[Sequence[float] | Mapping[str, float] | np.ndarray] = None,
        eligible: Optional[pd.DataFrame | np.ndarray] = None,
        margin: float = 0.0,
    ) -> RoutingResult:
        """Pick one model per query.

        ``lam`` weights cost in the utility ``pred - lam * C`` (``lam = 0`` ->
        quality-only). ``C`` is :meth:`decision_costs`: ``model_costs`` if given
        (per-model, or a ``[query_id x model_id]`` frame), else
        :meth:`predict_costs`, else :attr:`default_model_costs`. ``eligible`` is
        an optional ``[Q x M]`` bool mask of selectable cells (a DataFrame is
        aligned by label, an array is taken positionally).

        ``alpha`` (mutually exclusive with ``lam``) picks instead by the
        IRT-Router-style reward blend ``alpha * quality - (1 - alpha) *
        linear_cost(C)``, ``alpha`` in ``(0, 1]`` -- ``1.0`` is quality-only,
        values near ``0`` are increasingly cost-averse. See
        :func:`router.nirt.routing_decision.alpha_to_lam`.
        """
        return self._route_scored(self.aligned_scores(query_ids), query_ids, lam=lam, alpha=alpha,
                                  model_costs=model_costs, eligible=eligible, margin=margin)

    def _route_scored(self, scores: pd.DataFrame, query_ids: Sequence[str], *, lam: float,
                      alpha: Optional[float], model_costs, eligible, margin: float) -> RoutingResult:
        """The cost choice and the pick, shared by :meth:`route` and
        :meth:`route_text`. ``scores`` are already aligned to ``query_ids``."""
        # quality-only routing never needs (possibly expensive) predicted costs
        costs = (self.decision_costs(query_ids, model_costs)
                 if lam or margin or alpha is not None or model_costs is not None
                 else self.default_model_costs)
        lam, costs = self._resolve_alpha(alpha, lam, costs)
        return self._route_from_scores(scores, lam=lam, costs=costs, eligible=eligible,
                                       alpha=alpha, margin=margin)

    def __call__(self, *args, **kwargs) -> RoutingResult:
        # not ``__call__ = route``: that binds the base function, so a subclass
        # overriding route() would be bypassed by ``router(...)``
        return self.route(*args, **kwargs)

    def route_text(
        self,
        texts: Sequence[str],
        *,
        encoder=None,
        lam: float = 0.0,
        alpha: Optional[float] = None,
        model_costs: Optional[Sequence[float] | Mapping[str, float] | np.ndarray] = None,
        eligible: Optional[pd.DataFrame | np.ndarray] = None,
        margin: float = 0.0,
    ) -> RoutingResult:
        """:meth:`route`, but from raw prompt text (via :meth:`predict_scores_text`).

        ``RoutingResult.query_ids`` are the positional indices ``"0"``, ``"1"``,
        ... of the input list. ``alpha`` is as in :meth:`route`. Costs resolve
        the same way :meth:`route` does (:meth:`decision_costs`, keyed by those
        positional ids) -- a router with no text-aware :meth:`predict_costs`
        override (every router today) falls straight through to
        :attr:`default_model_costs`, same as before; this just stops
        live-text routing from being permanently unable to use a per-cell
        cost model should one ever be added.
        """
        scores = self.predict_scores_text(list(texts), encoder=encoder)
        scores = scores.copy()
        scores.index = [str(i) for i in range(len(scores))]
        return self._route_scored(scores, list(scores.index), lam=lam, alpha=alpha,
                                  model_costs=model_costs, eligible=eligible, margin=margin)

    # Oracle-relative evaluation needs ground truth and lives in
    # evaluation.routing.oracle.evaluate_router(router, true_df, cost_df, lam)
    # instead of a method here -- see that function's docstring.

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, n_models={len(self._model_ids)})"

"""``AgenticRouter`` -- a :class:`~router.routing.base.RouterModel` plus an orchestrator.

Flow for a live prompt:

1. **route** the prompt against the candidate pool (encoder-backed, so no
   prebuilt ``query_id`` is needed) -> best model + its predicted quality.
2. **triage** on that signal (:mod:`router.agentic.triage`): can the best model
   answer directly, or should the request be decomposed?
3. **dispatch**
   * *single*  -> send the prompt to the routed model's client, return the answer.
   * *compound* -> hand it to an orchestrator that decomposes the request and
     routes each sub-task (recursively), then synthesizes one answer. The router
     is available to the orchestrator as a tool.

    from router.routing import NIRTRouter
    from router.agentic import AgenticRouter, ClientRegistry

    ar = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"))   # echo clients
    res = ar.run("Explain quicksort, then write it in Rust and analyse its complexity.")
    res.mode            # "orchestrated"
    res.answer
    res.summary()

For real model calls pass ``clients=ClientRegistry.langchain(pool)`` and an
LLM-driven orchestrator / triage (see ``docs/agentic_router.md``).

Multi-turn: ``run`` also takes a chat message list or a ``Conversation``. The
router scores all user turns joined (its training rendering), triage and
decomposition look at the latest turn only, and every routed call is sent the
earlier messages as history (:mod:`router.agentic.conversation`).

Failure handling: a model call that raises inside an orchestration is recorded
as a failed :class:`SubCall` (``error`` set, see ``AgenticResult.errors()``) so
the sub-answers already paid for survive; a failing *root* call still raises.
``max_calls`` caps the model calls one :meth:`run` may make across the whole
recursion tree (``max_depth`` alone bounds depth, not fan-out) -- routed calls
*and* the helper LLMs (``LLMTriage``, ``LLMDecomposer``, ``LLMSynthesizer``,
``QualityChecker``) all count against it.

``run()`` is safe to call concurrently on one shared ``AgenticRouter``
instance -- each call's routing cache and call count live in a context-local
state (thread- or asyncio-task-isolated), not on the instance.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import hashlib
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np

from decompose.classifiers.prompt_heads import StructureClassifier

from ..routing.base import RouterModel, RoutingResult
from .conversation import (
    AgenticInput,
    as_messages,
    history_before_latest,
    history_chars,
    latest_user_turn,
    routing_text,
    with_history,
)
from .llm_clients import ClientRegistry, LLMClient, LLMResponse
from .decompose import (
    LLMDecomposer,
    LLMSynthesizer,
    NaiveDecomposer,
    SubTask,
    concat_synthesizer,
    looks_atomic,
)
from .result import AgenticResult, CallBudgetExceeded, SubCall, total_cost
from .triage import HeuristicTriage, LLMTriage, TriageDecision, best_from_result

__all__ = ["AgenticRouter", "flush_bookkeeping"]

Decomposer = Callable[[str], list]
Synthesizer = Callable[..., str]
Triage = Callable[..., TriageDecision]


@dataclass
class _RunState:
    """One :meth:`AgenticRouter.run` call's state: routes already computed
    (so a sub-task triage isn't repeated) and the shared call budget.

    Lives in a :class:`contextvars.ContextVar`, not on the ``AgenticRouter``
    instance (AGENTICROUTER-004) -- each ``run()`` call (its own thread,
    asyncio task, or a re-entrant nested call) gets an isolated state, so
    concurrent runs can no longer wipe each other's cached triage decisions
    or share a call counter."""

    decisions: dict[str, TriageDecision] = field(default_factory=dict)
    calls: int = 0
    max_calls: Optional[int] = None
    #: messages before the latest user turn; every routed call is sent this
    #: history ahead of its own prompt (see ``conversation.with_history``)
    history: tuple = ()
    #: this call's IRT-Router-style reward blend override (``run(..., alpha=)``),
    #: in (0, 1] -- ``None`` keeps the agent's static ``lam``. Set once at the
    #: root and read by every sub-task's re-triage too (AgenticRouter.route_decision),
    #: so one request's cost preference holds through the whole recursion.
    alpha: Optional[float] = None
    #: seconds per stage of the root request (``route``/``triage``/``dispatch``/``total``)
    timings: dict[str, float] = field(default_factory=dict)
    #: one summary dict per plan an orchestrator ran (``router.execution.routed``)
    plans: list = field(default_factory=list)
    #: one dict per routed helper call (decompose / synthesize), see AgenticRouter.helper_call
    helper_calls: list = field(default_factory=list)
    #: the plan orchestrator's final quality check (see AgenticResult.quality_check)
    quality_check: Optional[dict] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def use_call(self, n: int = 1, *, enforce: bool = True) -> None:
        """Count ``n`` model call(s) against the budget. Every LLM
        invocation a run makes -- routed, triage, decompose, synthesize --
        goes through here, not just the routed calls (AGENTICROUTER-003).
        ``enforce=False`` counts without raising, for call sites that can't
        safely raise mid-call. Locked: an orchestrator may run sub-calls on
        worker threads sharing this state."""
        with self._lock:
            if enforce and self.max_calls is not None and self.calls + n > self.max_calls:
                raise CallBudgetExceeded(f"max_calls={self.max_calls} model calls exhausted for this run")
            self.calls += n


def _call_meta(resp, expected: Optional[dict] = None) -> dict:
    """A routed call's ``LLMResponse`` (+ its pre-call cost estimate) -> ``SubCall.meta``."""
    return {**(resp.meta or {}), **(expected or {}), "latency": resp.latency,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}


def _run_sink(sink: Callable[[AgenticResult], None], result: AgenticResult) -> None:
    """Call a result sink on the bookkeeping thread; a failing sink is logged,
    never raised into (or silently lost from) the request path."""
    try:
        sink(result)
    except Exception:  # pragma: no cover - exercised via the logging test
        logging.getLogger(__name__).exception("AgenticRouter result sink failed")


def _write_debug_trace(result: AgenticResult, debug: bool | str) -> None:
    """Write ``result.debug_trace()`` to disk for ``run(..., debug=...)``.
    ``debug=True`` auto-names the file under ``debug_traces/``; a string is
    used as the path verbatim. A write failure is logged, never raised --
    a debugging aid must not be able to break the request it's observing."""
    try:
        path = _debug_trace_path(result) if debug is True else Path(debug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.debug_trace(), encoding="utf-8")
    except Exception:
        logging.getLogger(__name__).exception("AgenticRouter debug trace write failed")


def _debug_trace_path(result: AgenticResult) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(result.prompt.encode("utf-8")).hexdigest()[:8]
    return Path("debug_traces") / f"{stamp}-{digest}.txt"


@dataclass
class _Request:
    """One request, normalized: its messages, the latest user turn, and the
    per-run state it will execute under."""

    messages: list
    latest: str
    conversation_id: Optional[str]
    state: _RunState


_BOOKKEEPING: Optional[ThreadPoolExecutor] = None
_BOOKKEEPING_LOCK = threading.Lock()


def flush_bookkeeping(timeout: Optional[float] = None) -> None:
    """Block until all post-response work submitted so far (result sinks,
    ``stream(on_result=)`` callbacks) has run -- e.g. before reading the run
    database in a test or at the end of a batch script."""
    _bookkeeping_executor().submit(lambda: None).result(timeout=timeout)


def _bookkeeping_executor() -> ThreadPoolExecutor:
    """One background thread for post-response work (``stream(on_result=)``),
    created on first use -- off the token path, in submission order."""
    global _BOOKKEEPING
    with _BOOKKEEPING_LOCK:
        if _BOOKKEEPING is None:
            _BOOKKEEPING = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentic-bookkeeping")
        return _BOOKKEEPING


_current_run: "contextvars.ContextVar[Optional[_RunState]]" = contextvars.ContextVar(
    "agentic_router_current_run", default=None
)


def _standalone_scope(method):
    """A public helper called outside :meth:`AgenticRouter.run` gets a run state
    of its own for the call, removed afterwards -- never one that outlives it
    and counts against (or pollutes) the next caller."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if _current_run.get() is not None:
            return method(self, *args, **kwargs)
        token = _current_run.set(_RunState(max_calls=self.max_calls))
        try:
            return method(self, *args, **kwargs)
        finally:
            _current_run.reset(token)
    return wrapper


class AgenticRouter:
    def __init__(
        self,
        router: RouterModel,
        clients: Optional[ClientRegistry | dict[str, LLMClient]] = None,
        *,
        triage: Triage | None = None,
        orchestrator: object = None,
        decomposer: Decomposer | None = None,
        synthesizer: Synthesizer | None = None,
        encoder: object = None,
        lam: float = 0.0,
        margin: float = 0.0,
        max_depth: int = 2,
        max_calls: int | None = None,
        query_resolver: Callable[[str], str] | None = None,
        result_sink: Callable[[AgenticResult], None] | None = None,
    ):
        self.router = router
        if not isinstance(clients, ClientRegistry):
            clients = ClientRegistry(clients or None)
        self.clients = clients
        # Default triage reads the structure signal rather than a phrase list:
        # the list missed ", then" and answered a three-part request in one shot.
        self.triage = triage or HeuristicTriage(structure=StructureClassifier())
        if orchestrator is None:
            from ..execution.routed import plan_orchestrator
            orchestrator = plan_orchestrator()
        self.orchestrator = orchestrator
        self.decomposer = decomposer or NaiveDecomposer()
        self.synthesizer = synthesizer or concat_synthesizer
        self.encoder = encoder
        self.lam = float(lam)
        # `margin` is the measured lever, not `lam` -- see
        # docs/prompt_decomposition_experiments.md. Default 0.0 keeps the
        # served behaviour identical until a caller opts in.
        self.margin = float(margin)
        self.max_depth = int(max_depth)
        self.max_calls = None if max_calls is None else int(max_calls)
        self._resolver = query_resolver
        #: called with every finished ``AgenticResult`` (``run`` and ``stream``)
        #: on the background bookkeeping thread -- e.g.
        #: ``router.tracing.run_store.sqlite_sink("data/runs/agentic_runs.sqlite")``
        self.result_sink = result_sink
        # AGENTICROUTER-003: route each helper's own LLM call through the same
        # per-run budget a routed call goes through (see _RunState.use_call).
        # Harmless no-op for helpers that don't expose ``on_call`` (the
        # heuristic/naive defaults, or a user-supplied plain callable).
        for helper in (self.triage, self.decomposer, self.synthesizer):
            if hasattr(helper, "on_call"):
                helper.on_call = self._use_call

    @classmethod
    def from_settings(cls, settings=None, *, router: Optional[RouterModel] = None) -> "AgenticRouter":
        """Build a router from :class:`~router.agentic.schemas.AgenticRouterSettings`
        -- an instance, a path to a settings YAML, or ``None`` for
        ``configs/agentic.yaml``. Loads the router run, the clients, every
        helper (with its prompt from the settings' ``prompts``) and the
        orchestrator, and warms up when ``settings.warmup`` is set.

        ``router`` reuses an already-loaded (and warmed) ``RouterModel``
        instead of loading ``settings.run`` again -- e.g. several agent
        configurations sharing one encoder in a service or a test."""
        import yaml

        from ..config import resolve_path
        from ..routing import NIRTRouter
        from .schemas import AgenticRouterSettings, load_settings

        s = settings if isinstance(settings, AgenticRouterSettings) else load_settings(settings)
        if router is None:
            router = NIRTRouter.from_run(s.run, runs_dir=s.runs_dir)

        clients = None
        if s.clients == "anthropic":
            with open(resolve_path(s.serve_map), encoding="utf-8") as fh:
                clients = ClientRegistry.anthropic(yaml.safe_load(fh), pool=router.model_ids)
        elif s.clients == "langchain":
            clients = ClientRegistry.langchain(router.model_ids)

        def chat(model_id, *, provider=None):
            from ..llm.adapters.langchain_adapter import chat_model
            return chat_model(model_id, provider=provider)

        prompts = s.prompts
        t = s.triage
        heuristic = HeuristicTriage(quality_threshold=t.quality_threshold,
                                    max_chars_single=t.max_chars_single,
                                    structure=StructureClassifier())
        triage = heuristic if t.kind == "heuristic" else LLMTriage(
            chat(t.model), fallback=heuristic, prompt=prompts[t.prompt],
            margin=math.inf if t.margin is None else t.margin)

        from ..execution.budget import ExecutionLimits
        from ..execution.routed import plan_orchestrator
        from .quality import QualityChecker

        # one shared model drives decompose + judge + synthesize
        shared_chat = chat(s.plan.model, provider=s.plan.provider)

        d = s.decomposer
        decomposer = LLMDecomposer(
            chat_model=shared_chat, max_parts=d.max_parts, max_prompt_chars=d.max_prompt_chars,
            fallback=NaiveDecomposer(max_parts=d.max_parts, min_part_chars=d.min_part_chars),
            plan_prompt=prompts[s.plan.planner_prompt])

        y = s.synthesizer
        synthesizer = LLMSynthesizer(chat_model=shared_chat, max_body_chars=y.max_body_chars,
                                     prompt=prompts[y.prompt])

        q = s.plan.quality_check
        checker = QualityChecker(chat_model=shared_chat, prompt=prompts[q.prompt],
                                 pass_score=q.pass_score)

        budget = math.inf if s.plan.budget is None else s.plan.budget
        orchestrator = plan_orchestrator(
            budget=budget, max_workers=s.plan.max_workers, sequential=s.plan.sequential,
            route_late=s.plan.route_late, checker=checker,
            length_budgets=s.plan.length_budgets,
            limits=ExecutionLimits(max_cost=budget, max_depth=s.max_depth,
                                   max_replans=s.plan.max_replans))

        sink = None
        if s.result_sink:
            from ..tracing.run_store import sqlite_sink
            sink = sqlite_sink(resolve_path(s.result_sink), router=s.run)

        agent = cls(router, clients, triage=triage, orchestrator=orchestrator,
                    decomposer=decomposer, synthesizer=synthesizer, lam=s.lam,
                    max_depth=s.max_depth, max_calls=s.max_calls, result_sink=sink)
        if s.warmup:
            agent.warmup()
        return agent

    def warmup(self) -> dict[str, float]:
        """Load everything a first request would otherwise pay for -- the
        router's encoder, stores and cost vector (:meth:`RouterModel.warmup`)
        -- so request latency is steady from the first prompt. Call once at
        startup. Returns seconds per stage plus ``total``."""
        import time

        t0 = time.perf_counter()
        timings = dict(self.router.warmup())
        timings["total"] = time.perf_counter() - t0
        return timings

    # ------------------------------------------------------------------ #
    # routing a single prompt                                            #
    # ------------------------------------------------------------------ #
    def _routing_knobs(self) -> tuple[Optional[float], float, np.ndarray]:
        """``(alpha, lam, eligible)`` for a routing call: this run's ``alpha``
        override (which replaces ``lam``), else the static ``lam``; only models
        :attr:`clients` can reach are eligible."""
        alpha = self._run_state().alpha
        return alpha, (0.0 if alpha is not None else self.lam), self.clients.eligible_mask(self.router.model_ids)

    def route_texts(self, prompts: list[str]) -> RoutingResult:
        """One router call over raw ``prompts`` with this agent's cost
        preference (``lam``/``alpha``/``margin``) and eligibility. The single
        place root prompts and plan steps are routed, so they cannot drift.
        Does not check for fallback rows; callers decide what an unscorable
        prompt means."""
        if not getattr(self.router, "can_route_text", False):
            raise TypeError(
                f"{type(self.router).__name__} cannot score raw text; pass "
                "query_resolver= or use NIRTRouter / KNNRouter"
            )
        alpha, lam, eligible = self._routing_knobs()
        return self.router.route_text(prompts, encoder=self.encoder, lam=lam, alpha=alpha,
                                      eligible=eligible, margin=self.margin)

    def route_decision(self, prompt: str) -> RoutingResult:
        """One-row :class:`RoutingResult` for ``prompt`` (encoder-backed, or via a
        ``query_resolver`` that maps text -> an existing ``query_id``).

        Uses this run's ``alpha`` override (``run(prompt, alpha=...)``) when
        one was given, else the agent's static ``lam``. Reads it from the
        current run's context-local state, so every sub-task's re-triage in
        one orchestrated request shares the same cost preference without it
        being threaded through every recursive call explicitly.

        Raises ``LookupError`` when the router has nothing selectable for the
        prompt (e.g. the resolved ``query_id`` has no scores) -- otherwise the
        pool's first model would silently answer.

        Deliberately scoped to the router's own narrow, dispatchable pool
        (:attr:`RouterModel.model_ids`) -- a routed call must be something
        :attr:`clients` can actually reach. See :meth:`score_all_models` for
        the broader, every-registered-model view; don't widen this method to
        use it."""
        if self._resolver is not None:
            alpha, lam, eligible = self._routing_knobs()
            rr = self.router.route([self._resolver(prompt)], lam=lam, alpha=alpha,
                                   eligible=eligible, margin=self.margin)
        else:
            rr = self.route_texts([prompt])
        if rr.any_fallback:
            raise LookupError(
                f"{self.router.name}: no usable score for this prompt "
                f"(query_id={rr.query_ids[0]!r}); refusing to dispatch to a default model"
            )
        return rr

    def score_all_models(
        self, prompt: str, *, impute_unknown: bool = False, knn_k: int = 10,
        knn_weighting: str = "similarity",
    ) -> RoutingResult:
        """Score ``prompt`` against every model currently *registered in the
        pool* -- every ``configs/model_profiles.yaml`` entry, unioned with
        this router's own trained pool -- not just the narrow, dispatchable
        set :meth:`route_decision` scores. A model with no profile embedding
        built yet is rendered + embedded on the fly
        (:mod:`router.routing.model_profiles`), with :attr:`encoder` reused as
        the embedder (profile and query text share one pathway here, so
        there's no second model to load); ``impute_unknown=True`` additionally
        kNN-blends any model with no training observations toward the trained
        pool's nearest profiles (see that module's caveat docstring before
        turning this on by default).

        Needs a :class:`~router.routing.routers.NIRTRouter` in
        ``model_params="projected"`` mode (cold-start capable by
        construction) -- raises ``TypeError`` otherwise.

        Scoring a model here never implies it can be *called*: dispatching
        still goes through :attr:`clients` (:class:`ClientRegistry`), which
        raises its own ``LookupError`` for an id with no registered adapter,
        independently of whether this method can score it.
        """
        import numpy as np

        from ..nirt.routing_decision import cost_aware_utility, no_selectable_rows, routing_decision
        from ..routing import NIRTRouter, model_profiles

        router = self.router
        if not isinstance(router, NIRTRouter):
            raise TypeError(
                f"score_all_models needs a NIRTRouter (cold-start pool scoring), "
                f"got {type(router).__name__}"
            )
        cfg = getattr(router._data, "cfg", None)
        pool_ids = sorted(set(router.model_ids) | set(model_profiles.list_profiled_model_ids(cfg)))
        scores = router.predict_pool_scores_text(
            [prompt], pool_ids, impute_unknown=impute_unknown, knn_k=knn_k,
            knn_weighting=knn_weighting, encoder=self.encoder, profile_encoder=self.encoder,
        )
        mat = scores.to_numpy(np.float64)
        util = cost_aware_utility(mat, lam=0.0, model_costs=None, eligible=None)
        return RoutingResult(
            query_ids=["0"], model_ids=list(scores.columns), scores=mat,
            selected=routing_decision(mat, lam=0.0, model_costs=None, eligible=None),
            lam=0.0, fallback=no_selectable_rows(util), alpha=None,
        )

    def _skip_decomposition(self, prompt: str, depth: int) -> bool:
        """The gate in :meth:`decompose_plan`: nothing for an LLM decomposer to split."""
        atomic = not self._is_multipart(prompt)
        if depth == 0:
            atomic = atomic and looks_atomic(prompt)
        return atomic

    @_standalone_scope
    def decompose_plan(self, prompt: str, depth: int, *, sequential: bool = False) -> list[SubTask]:
        """Split ``prompt`` into dependency-aware :class:`SubTask`\\ s --
        the one path the orchestrator decomposes through, so the policy for
        *whether* to spend an LLM call on it lives here, not in the decomposer.

        An :class:`LLMDecomposer` plans them (:meth:`LLMDecomposer.plan`), with
        the call skipped (one atomic ``SubTask``) when there is nothing to
        split:

        * ``depth == 0``: the request :func:`looks_atomic` (one sentence, no
          comma list) *and* has no lexical multi-part marker ("... and then ...").
        * ``depth > 0``: the sub-task is not lexically multi-part. It came from
          a decomposer told to make sub-tasks atomic, so a low router score
          alone is no reason to split it again.

        Any other decomposer (``NaiveDecomposer``, a plain callable) gives
        independent sub-tasks, each depending on the one before when
        ``sequential``."""
        if isinstance(self.decomposer, LLMDecomposer):
            if self._skip_decomposition(prompt, depth):
                return [SubTask(prompt)]
            return self.decomposer.plan(prompt)
        subs = [s for s in (self.decomposer(prompt) or []) if str(s).strip()] or [prompt]
        return [SubTask(s, (i,) if sequential and i else ()) for i, s in enumerate(subs)]

    def _is_multipart(self, prompt: str) -> bool:
        """The triage heuristic's lexical multi-part check (an ``LLMTriage``'s
        fallback heuristic, or default ``HeuristicTriage`` settings)."""
        heuristic = getattr(self.triage, "_fallback", self.triage)
        check = (getattr(heuristic, "is_multipart", None)
                 or HeuristicTriage(structure=StructureClassifier()).is_multipart)
        return check(prompt)

    @_standalone_scope
    def pick_helper_model(self, request: str, min_quality: float) -> tuple[str, float]:
        """The pool model an LLM helper (decomposer / synthesizer) runs on for
        ``request``: the cheapest model whose predicted quality is at least
        ``min_quality`` (ties -> higher score), else the top-scored model.

        Scores are the router's quality prediction for *answering* ``request``
        -- a proxy for model strength, not a measured decomposition skill. They
        come from this run's triage of ``request`` when there is one, so a
        helper costs no extra routing call.

        Deliberately scoped to :meth:`route_decision`'s narrow pool, like it
        -- a helper LLM must be something :attr:`clients` can actually call,
        unlike :meth:`score_all_models`'s broader, every-registered-model
        view."""
        known = self._run_state().decisions.get(request)
        scores = (known.scores if known is not None and known.scores
                  else self.route_decision(request).row_scores(0))
        scores = {m: float(q) for m, q in scores.items()
                  if math.isfinite(float(q)) and self.clients.can_serve(m)}
        if not scores:
            raise LookupError(f"{self.router.name}: no usable score to pick a helper model")
        above = [m for m, q in scores.items() if q >= min_quality]
        if not above:
            best = max(scores, key=scores.get)
            return best, scores[best]
        costs = self.router.cost_by_model()
        pick = min(above, key=lambda m: (costs.get(m, math.inf), -scores[m]))
        return pick, scores[pick]

    def helper_call(self, role: str, request: str, min_quality: float) -> Callable[[str], LLMResponse]:
        """A ``text -> response`` call on the pool model :meth:`pick_helper_model`
        chose for ``request`` -- the ``chat_for`` an LLM helper uses when it is
        ``kind: routed``. Each call is recorded in the run's ``helper_calls``
        (the helper itself counts it against ``max_calls`` via ``on_call``)."""
        model_id, q = self.pick_helper_model(request, min_quality)

        def call(text: str, **generation_kwargs) -> LLMResponse:
            resp = self.clients.complete(model_id, text, **generation_kwargs)
            self._run_state().helper_calls.append({
                "role": role, "request": request, "model_id": model_id,
                "predicted_quality": q, "min_quality": min_quality,
                "cost": resp.cost, "latency": resp.latency, "output": resp.content,
            })
            return resp

        return call

    @_standalone_scope
    def triage_prompt(self, prompt: str, *, routing_text: Optional[str] = None) -> TriageDecision:
        """Route ``routing_text`` (default: ``prompt``) and triage ``prompt``.

        The two differ for a conversation: the router scores every user turn
        (its training rendering), triage judges only the latest one. The first
        call in a run records the ``route``/``triage`` timings."""
        rs = self._run_state()
        t0 = time.perf_counter()
        rr = self.route_decision(routing_text if routing_text is not None else prompt)
        t1 = time.perf_counter()
        model_id, q, scores = best_from_result(rr)
        decision = self.triage(prompt, model_id=model_id, predicted_quality=q, scores=scores)
        rs.timings.setdefault("route", t1 - t0)
        rs.timings.setdefault("triage", time.perf_counter() - t1)
        rs.decisions[prompt] = decision
        return decision

    # ------------------------------------------------------------------ #
    # execution                                                          #
    # ------------------------------------------------------------------ #
    def run(self, prompt: AgenticInput, *, conversation_id: Optional[str] = None,
            alpha: Optional[float] = None, debug: bool | str = False) -> AgenticResult:
        """Triage the request and either answer it directly or orchestrate it.

        ``prompt`` is a bare string, a chat message list or a
        :class:`~decompose.conversation.Conversation` whose last message is the
        user turn to answer. The router scores every user turn; triage and
        decomposition see only the latest turn; each routed model call is sent
        the earlier messages as history (:mod:`router.agentic.conversation`).
        A bare string behaves exactly as a one-turn conversation with no history.

        ``alpha`` overrides this agent's static ``lam`` for just this call with
        an IRT-Router-style reward blend in ``(0, 1]`` (``1.0`` quality-only,
        near ``0`` cost-averse) -- e.g. different end users' cost preferences
        routed through one shared ``AgenticRouter``. Held for the whole call,
        including every sub-task's re-triage inside an orchestrated run. See
        :meth:`route_decision`.

        ``debug`` writes :meth:`AgenticResult.debug_trace` for this call to a
        text file: ``True`` auto-names it under ``debug_traces/``, a string is
        used as the path verbatim. A write failure is logged, never raised."""
        t_start = time.perf_counter()
        req = self._begin(prompt, conversation_id, alpha)
        token = _current_run.set(req.state)
        try:
            decision = self._triage_request(req)
            t0 = time.perf_counter()
            root = self._solve(req.latest, 0, decision=decision)
            req.state.timings["dispatch"] = time.perf_counter() - t0
        finally:
            _current_run.reset(token)
        result = self._finish(req, decision, root, t_start)
        if debug:
            _write_debug_trace(result, debug)
        self._emit(result)
        return result

    def stream(self, prompt: AgenticInput, *, conversation_id: Optional[str] = None,
               on_result: Optional[Callable[[AgenticResult], None]] = None,
               alpha: Optional[float] = None,
               ) -> Generator[str, None, AgenticResult]:
        """:meth:`run`, streaming the answer: yields text chunks as the routed
        model produces them; the generator's return value
        (``StopIteration.value``) is the :class:`AgenticResult`.

        Routing and triage happen before the first chunk (they pick the model);
        everything else waits for the stream to end -- pricing from the final
        usage (:meth:`LLMClient.stream`), building the ``SubCall`` /
        ``AgenticResult``, and ``on_result``, which runs on a background
        bookkeeping thread so a slow sink (tracing, logging, a ledger) never
        delays the caller. ``timings["ttft"]`` is request -> first chunk.

        An orchestrated request is solved first (sub-calls, synthesis) and its
        synthesized answer is yielded as one chunk. The per-run state lives in
        a context private to this generator, so interleaved streams on one
        agent stay independent. ``alpha`` is as in :meth:`run`."""
        t_start = time.perf_counter()
        req = self._begin(prompt, conversation_id, alpha)
        ctx = contextvars.copy_context()

        def setup() -> TriageDecision:
            _current_run.set(req.state)
            return self._triage_request(req)

        decision = ctx.run(setup)
        timings = req.state.timings
        t0 = time.perf_counter()
        if decision.mode == "single" or self.max_depth <= 0:
            model_id = decision.selected_model_id
            ctx.run(self._use_call)
            expected = self._expected_cost(model_id, req.latest, req.state.history)
            sent = with_history(req.state.history, req.latest)
            inner = self.clients.stream(model_id, sent)
            while True:
                try:
                    chunk = ctx.run(next, inner)
                except StopIteration as stop:
                    resp = stop.value
                    break
                timings.setdefault("ttft", time.perf_counter() - t_start)
                yield chunk
            timings.setdefault("ttft", time.perf_counter() - t_start)
            root = self._leaf(req.latest, model_id, 0, self._quality_of(decision), resp, expected,
                             sent_prompt=sent)
            timings["dispatch"] = time.perf_counter() - t0
        else:
            root = ctx.run(self._solve, req.latest, 0, decision=decision)
            timings["dispatch"] = time.perf_counter() - t0
            timings["ttft"] = time.perf_counter() - t_start
            yield root.answer
        result = self._finish(req, decision, root, t_start)
        if on_result is not None:
            _bookkeeping_executor().submit(on_result, result)
        self._emit(result)
        return result

    def _emit(self, result: AgenticResult) -> None:
        if self.result_sink is not None:
            _bookkeeping_executor().submit(_run_sink, self.result_sink, result)

    # -- request setup / result assembly shared by run() and stream() --
    def _begin(self, prompt: AgenticInput, conversation_id: Optional[str],
               alpha: Optional[float] = None) -> "_Request":
        messages = as_messages(prompt)
        if conversation_id is None:
            conversation_id = getattr(prompt, "conversation_id", None)
        return _Request(
            messages=messages, latest=latest_user_turn(messages),
            conversation_id=conversation_id,
            state=_RunState(max_calls=self.max_calls, history=history_before_latest(messages),
                            alpha=alpha),
        )

    def _result(self, req: "_Request", decision: TriageDecision, root: SubCall) -> AgenticResult:
        context = dict(
            conversation_id=req.conversation_id,
            turn_index=sum(1 for m in req.messages if m["role"] == "user"),
            timings=dict(req.state.timings),
            history=[dict(m) for m in req.state.history],
            plans=list(req.state.plans),
            helper_calls=list(req.state.helper_calls),
            quality_check=req.state.quality_check,
            triage_scores=dict(decision.scores),
        )
        if root.mode == "single":
            reason = decision.reason
            if decision.orchestrate:
                reason += " -- nothing to decompose, answered directly"
            return AgenticResult(
                prompt=req.latest, mode="single", answer=root.answer,
                triage_reason=reason,
                selected_model_id=root.model_id,
                # the model that actually answered's own score, not the pool
                # best decision.predicted_quality carries (AGENTICROUTER-001)
                predicted_quality=root.predicted_quality,
                steps=[root], cost=root.cost, **context,
            )
        return AgenticResult(
            prompt=req.latest, mode="orchestrated", answer=root.answer,
            triage_reason=decision.reason,
            selected_model_id=decision.selected_model_id,
            predicted_quality=decision.predicted_quality,
            steps=root.children, orchestrator=getattr(self.orchestrator, "name", "?"),
            cost=root.cost, **context,
        )

    def _triage_request(self, req: "_Request") -> TriageDecision:
        """The triage verdict for a request's latest turn, with its conversation as context."""
        return self.triage_prompt(req.latest, routing_text=routing_text(req.messages))

    def _finish(self, req: "_Request", decision: TriageDecision, root: SubCall,
                t_start: float) -> AgenticResult:
        """Stamp the request's total time and assemble its result (run and stream both end here)."""
        req.state.timings["total"] = time.perf_counter() - t_start
        return self._result(req, decision, root)

    # -- recursion unit ------------------------------------------------
    def _solve(self, prompt: str, depth: int, *, decision: TriageDecision | None = None) -> SubCall:
        """Solve one (sub-)prompt: a leaf ``SubCall`` (single) or a branch whose
        ``children`` are the orchestrator's sub-calls."""
        if decision is None:
            if depth >= self.max_depth:
                # the verdict would be ignored at the cap -> route only, skip triage
                return self._answer_directly(prompt, depth)
            decision = self.triage_prompt(prompt)

        if decision.mode == "single" or depth >= self.max_depth:
            return self._answer_directly(prompt, depth, model_id=decision.selected_model_id,
                                         predicted_quality=self._quality_of(decision))

        answer, children = self.orchestrator.run(prompt, self, depth)
        if (len(children) == 1 and children[0].prompt == prompt and not children[0].children
                and children[0].error is None
                and answer == children[0].answer):
            # nothing to decompose (and no synthesized answer of the orchestrator's
            # own) -> it collapses back to a single routed call at this depth
            children[0].depth = depth
            return children[0]
        return SubCall(
            prompt=prompt, model_id=decision.selected_model_id,
            predicted_quality=decision.predicted_quality, answer=answer,
            mode="orchestrated", depth=depth, children=children,
            # leaves only: a branch child's cost already includes its descendants
            cost=total_cost(n for c in children for n in c.leaves()),
        )

    @staticmethod
    def _quality_of(decision: TriageDecision) -> float:
        """Predicted quality of the model that will answer (not the pool best)."""
        return float(decision.scores.get(decision.selected_model_id, decision.predicted_quality))

    def _answer_directly(
        self, prompt: str, depth: int, *,
        model_id: str | None = None, predicted_quality: float = float("nan"),
    ) -> SubCall:
        """Route ``prompt`` (if the model was not already chosen) and invoke it."""
        if model_id is None:
            known = self._run_state().decisions.get(prompt)
            if known is not None:          # triage already routed this prompt
                model_id, predicted_quality = known.selected_model_id, self._quality_of(known)
            else:
                rr = self.route_decision(prompt)
                model_id = rr.selected_model_ids[0]
                predicted_quality = float(rr.selected_scores[0])
        return self._call_model(prompt, model_id, depth, predicted_quality=predicted_quality)

    def _call_model(self, prompt: str, model_id: str, depth: int, *,
                    predicted_quality: float = float("nan"), **generation_kwargs) -> SubCall:
        """Invoke ``model_id`` once, counting it against ``max_calls``. Inside an
        orchestration (``depth > 0``) a failure becomes a failed ``SubCall``; at
        the root it raises. ``generation_kwargs`` (e.g. ``max_tokens``) reach
        the model's adapter. A blank response (no text, no error from the
        adapter) is treated the same as a raised exception -- the model didn't
        answer, so this must not read as a successful empty answer."""
        history = self._run_state().history
        sent = with_history(history, prompt)
        expected: dict = {}
        try:
            self._use_call()
            expected = self._expected_cost(model_id, prompt, history)
            resp = self.clients.complete(model_id, sent, **generation_kwargs)
            if not resp.content or not resp.content.strip():
                raise RuntimeError(f"{model_id} returned no answer (empty response)")
        except Exception as exc:
            if depth == 0:
                raise
            return SubCall(
                prompt=prompt, model_id=model_id, predicted_quality=float(predicted_quality),
                answer=f"[error: {type(exc).__name__}: {exc}]", mode="single", depth=depth,
                cost=None, error=repr(exc),
                meta={**expected, "cost_source": "failed", "sent_prompt": sent,
                      **({"budget_exhausted": True} if isinstance(exc, CallBudgetExceeded) else {})},
            )
        return self._leaf(prompt, model_id, depth, predicted_quality, resp, expected, sent_prompt=sent)

    @staticmethod
    def _leaf(prompt: str, model_id: str, depth: int, predicted_quality: float,
              resp, expected: dict, *, sent_prompt: Optional[str] = None) -> SubCall:
        """The ``SubCall`` for one completed model call -- the single builder
        ``_call_model`` and ``stream`` share.

        ``cost`` is the call's actual cost when the client could price it
        (``meta["cost_source"] == "actual"``). A real call whose cost the
        client couldn't price (no profile prices, or no usage from the
        provider) is charged its pre-call estimate instead (``"estimate"``) --
        never silently ``0.0`` (AGENTICROUTER-002) -- and stays ``None`` only
        when there is no estimate either (``"unknown"``). ``sent_prompt`` (when
        it differs from ``prompt``, e.g. history or a length-budget line was
        prepended/appended) is the text actually sent to the model."""
        cost, source = resp.cost, "actual"
        if cost is None:
            cost = expected.get("expected_cost")
            source = "estimate" if cost is not None else "unknown"
        meta = {**_call_meta(resp, expected), "cost_source": source}
        if sent_prompt is not None:
            meta["sent_prompt"] = sent_prompt
        return SubCall(
            prompt=prompt, model_id=model_id, predicted_quality=float(predicted_quality),
            answer=resp.content, mode="single", depth=depth, cost=cost, meta=meta,
        )

    def _expected_cost(self, model_id: str, prompt: str, history) -> dict:
        """The pre-call estimate recorded next to a call's actual cost:
        ``expected_cost`` (USD or ``None``), ``expected_cost_source``
        (``profile`` / ``router_prior`` / ``unknown``, see
        :func:`router.llm.cost.estimate_call_cost`) and ``provider_model`` --
        the model that actually serves ``model_id`` (e.g. a Claude tier)."""
        from ..llm.cost import estimate_call_cost

        client = self.clients.get(model_id)
        chars = len(prompt) + history_chars(history)
        cost, source = estimate_call_cost(getattr(client, "profile", None), chars,
                                          fallback=self._router_cost(model_id))
        adapter = getattr(client, "_adapter", None)
        return {"expected_cost": cost, "expected_cost_source": source,
                "provider_model": getattr(adapter, "model", None)}

    def _router_cost(self, model_id: str) -> Optional[float]:
        """The router's own per-query cost for ``model_id`` (training-data
        mean, USD), or ``None``."""
        return self.router.cost_by_model().get(str(model_id))

    # -- per-run state (AGENTICROUTER-004: context-local, not instance state) --
    def _run_state(self) -> _RunState:
        """The active run's state. Falls back to a fresh, context-local one
        for calls made outside :meth:`run` (e.g. driving ``_solve``/
        ``_call_model`` directly, as the LangChain tool tests do): that state
        is not installed, so it cannot outlive the call."""
        rs = _current_run.get()
        return rs if rs is not None else _RunState(max_calls=self.max_calls)

    def _use_call(self, n: int = 1, *, enforce: bool = True) -> None:
        """Count ``n`` model call(s) against this run's shared budget. Every
        LLM invocation -- routed (``_call_model``) and helper (triage,
        decompose, synthesize, judge) -- goes through here (AGENTICROUTER-003)."""
        self._run_state().use_call(n, enforce=enforce)

    # ------------------------------------------------------------------ #
    # convenience                                                        #
    # ------------------------------------------------------------------ #
    def with_margin(self, margin: float) -> AgenticRouter:
        """A copy that spends ``margin`` quality to reach a cheaper model.

        Cost sensitivity belongs to the request, not the deployment -- the same
        service answers "summarise this changelog" and "write the migration",
        and they do not deserve the same budget. A copy rather than a mutable
        attribute so two concurrent callers cannot change each other's routing:

            router.with_margin(0.02).run(prompt)   # barely cheaper, barely worse
            router.with_margin(0.15).run(prompt)   # much cheaper, noticeably worse

        The margin is a quality budget, not a savings target. What it promises
        is the floor -- "no worse than ``margin`` below the best this router
        can predict" -- and that holds on any pool. What it buys varies by
        pool, so translate a savings target with
        ``evaluation.prompt_decomposition.policy_eval.margin_for_target_saving``
        rather than guessing.
        """
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        clone = copy.copy(self)
        clone.margin = float(margin)
        return clone

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"AgenticRouter(router={self.router.name!r}, "
                f"orchestrator={getattr(self.orchestrator, 'name', '?')!r}, "
                f"max_depth={self.max_depth})")

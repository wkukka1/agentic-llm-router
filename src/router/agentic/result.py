"""Structured trace returned by :class:`~router.agentic.router.AgenticRouter`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = ["SubCall", "AgenticResult", "total_cost"]


@dataclass
class SubCall:
    """One model invocation the orchestrator made (a leaf), or an orchestrated
    branch whose ``children`` are the sub-calls. ``error`` is set when the call
    failed; ``answer`` is then a placeholder."""

    prompt: str
    model_id: str
    predicted_quality: float
    answer: str
    mode: str = "single"                # "single" | "orchestrated"
    depth: int = 0
    cost: Optional[float] = None
    children: list["SubCall"] = field(default_factory=list)
    error: Optional[str] = None
    #: the call's ``LLMResponse.meta`` plus ``latency`` / ``input_tokens`` /
    #: ``output_tokens`` (leaves only) -- stop reason, cache reads, ttft ...
    meta: dict = field(default_factory=dict)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def leaves(self):
        """Nodes that made an actual model call. A branch's ``cost`` already
        sums its descendants, so totals must add leaves only."""
        return (n for n in self.walk() if not n.children)


def _render_subcall(node: SubCall, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines = [f"{pad}- [{node.mode}] model_id={node.model_id} "
             f"predicted_quality={node.predicted_quality} cost={node.cost}"]
    if node.error:
        lines.append(f"{pad}  error: {node.error}")
    step_scores = node.meta.get("step_scores")
    if step_scores:
        ranked = sorted(step_scores.items(), key=lambda kv: kv[1], reverse=True)
        lines.append(f"{pad}  considered: " + ", ".join(f"{m}={q:.3f}" for m, q in ranked))
    lines.append(f"{pad}  prompt: {node.prompt}")
    sent_prompt = node.meta.get("sent_prompt")
    if sent_prompt is not None and sent_prompt != node.prompt:
        lines.append(f"{pad}  sent: {sent_prompt}")
    lines.append(f"{pad}  answer: {node.answer}")
    escalation_reason = node.meta.get("escalation_reason")
    if escalation_reason:
        lines.append(f"{pad}  escalation_reason: {escalation_reason}")
    if node.meta:
        lines.append(f"{pad}  meta: {node.meta}")
    for child in node.children:
        lines.extend(_render_subcall(child, indent + 1))
    return lines


def total_cost(nodes: Iterable[SubCall]) -> Optional[float]:
    """Sum ``nodes``' costs (pass ``.leaves()`` -- a branch's own ``cost``
    already includes its descendants, so summing branches double-counts).

    A failed node (``error`` set) contributes ``0`` -- it didn't complete, so
    nothing was billed. A real call the client couldn't price already carries
    its pre-call *estimate* as ``cost`` (``meta["cost_source"] ==
    "estimate"``, see ``AgenticRouter._leaf``); a *successful* node with
    ``cost is None`` -- no actual cost and no estimate either -- makes the
    total ``None``: unresolved, not free (AGENTICROUTER-002: ``None`` must
    not collapse to ``0.0``, or an unpriced run silently reports as free)."""
    total = 0.0
    for n in nodes:
        if n.cost is not None:
            total += n.cost
        elif n.error is None:
            return None
    return total


@dataclass
class AgenticResult:
    prompt: str
    mode: str                            # "single" | "orchestrated"
    answer: str
    triage_reason: str
    selected_model_id: Optional[str] = None
    predicted_quality: Optional[float] = None
    steps: list[SubCall] = field(default_factory=list)
    orchestrator: Optional[str] = None
    cost: Optional[float] = None
    #: set when the request was a conversation (``run(messages, conversation_id=...)``)
    conversation_id: Optional[str] = None
    #: number of user turns in the request (1 for a bare prompt)
    turn_index: Optional[int] = None
    #: seconds per stage: ``route``, ``triage``, ``dispatch``, ``total``
    timings: dict = field(default_factory=dict)
    #: the conversation messages before ``prompt`` (empty for a bare prompt)
    history: list = field(default_factory=list)
    #: one dict per plan the plan orchestrator ran: estimates vs. what was spent
    plans: list = field(default_factory=list)
    #: one dict per ``kind: routed`` helper call (decompose / synthesize): role,
    #: chosen pool model, its predicted quality vs. the threshold, cost, output.
    #: Not included in ``cost``, which sums the answering leaves only.
    helper_calls: list = field(default_factory=list)
    #: the plan orchestrator's final quality check, when one ran: the best
    #: round's verdict (passed, score, weak_steps, issues, follow_ups, status),
    #: best_round, and every round with its escalations / follow-ups
    quality_check: Optional[dict] = None
    #: every pool model's predicted quality for ``prompt`` at triage time
    #: (``model_id -> score``) -- the comparison ``triage_reason`` was judged
    #: against, not just the winner's own score
    triage_scores: dict = field(default_factory=dict)

    def models_used(self) -> list[str]:
        """Models actually invoked. In orchestrated mode ``selected_model_id``
        only produced the triage signal, so it is listed only if some sub-call
        used it."""
        seen: list[str] = []
        for s in self.steps:
            for node in s.leaves():
                if node.model_id not in seen:
                    seen.append(node.model_id)
        if self.mode == "single" and self.selected_model_id and self.selected_model_id not in seen:
            seen.insert(0, self.selected_model_id)
        return seen

    def errors(self) -> list[SubCall]:
        """Sub-calls that failed (their answers are placeholders); superseded
        attempts, which no longer feed the answer, are not listed."""
        return [n for s in self.steps for n in s.walk()
                if n.error is not None and not n.meta.get("superseded")]

    def summary(self) -> str:
        if self.mode == "single":
            q = "?" if self.predicted_quality is None else f"{self.predicted_quality:.2f}"
            return f"single -> {self.selected_model_id} (q~{q}): {self.triage_reason}"
        n = sum(1 for s in self.steps for _ in s.leaves())
        return (f"orchestrated ({self.orchestrator}) -> {n} sub-call(s) across "
                f"{self.models_used()}: {self.triage_reason}")

    def debug_trace(self) -> str:
        """Full human-readable dump of this run: header, the ``steps`` tree,
        helper calls, plans, and the quality check -- everything
        ``AgenticResult`` carries for one :meth:`AgenticRouter.run` call. For
        ``run(..., debug=...)``."""
        lines: list[str] = []
        lines.append(f"prompt: {self.prompt!r}")
        lines.append(f"mode: {self.mode}")
        lines.append(f"triage_reason: {self.triage_reason}")
        lines.append(f"selected_model_id: {self.selected_model_id}")
        lines.append(f"predicted_quality: {self.predicted_quality}")
        lines.append(f"orchestrator: {self.orchestrator}")
        lines.append(f"cost: {self.cost}")
        lines.append(f"timings: {self.timings}")
        if self.triage_scores:
            ranked = sorted(self.triage_scores.items(), key=lambda kv: kv[1], reverse=True)
            lines.append("triage_scores: " + ", ".join(f"{m}={q:.3f}" for m, q in ranked))
        lines.append(f"answer: {self.answer}")
        lines.append("")
        lines.append("=== steps ===")
        for step in self.steps:
            lines.extend(_render_subcall(step))
        if self.helper_calls:
            lines.append("")
            lines.append("=== helper_calls ===")
            for i, hc in enumerate(self.helper_calls):
                lines.append(f"[{i}] role={hc.get('role')} model_id={hc.get('model_id')} "
                            f"cost={hc.get('cost')} latency={hc.get('latency')}")
                lines.append(f"    request: {hc.get('request')}")
                lines.append(f"    output: {hc.get('output')}")
        if self.plans:
            lines.append("")
            lines.append("=== plans ===")
            for p in self.plans:
                lines.append(f"[{p.get('plan_id')}] rev={p.get('revision')} n_steps={p.get('n_steps')} "
                            f"outcome={p.get('outcome')}")
                lines.append(f"    estimated: cost={p.get('estimated_cost')} "
                            f"quality={p.get('estimated_quality')} latency={p.get('estimated_latency')}")
                lines.append(f"    actual:    cost={p.get('actual_cost')} latency={p.get('actual_latency')}")
                lines.append(f"    quality_score={p.get('quality_score')} passed={p.get('quality_passed')}")
        if self.quality_check:
            lines.append("")
            lines.append("=== quality_check ===")
            qc = self.quality_check
            lines.append(f"best_round={qc.get('best_round')} passed={qc.get('passed')} "
                        f"score={qc.get('score')} status={qc.get('status')}")
            if qc.get("issues"):
                lines.append(f"issues: {qc['issues']}")
            for r in qc.get("rounds", []):
                lines.append(f"  round plan_id={r.get('plan_id')} passed={r.get('passed')} "
                            f"score={r.get('score')} escalated={r.get('escalated')} "
                            f"follow_ups={r.get('follow_ups')}")
        return "\n".join(lines)

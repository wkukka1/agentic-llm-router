"""Stage 1: is this prompt good enough to route at all?

Routing an underspecified prompt wastes the expensive path and produces an
answer the user has to reject anyway. The gate runs first and can stop the flow
to collect requirements, which is the ``follow up`` node feeding back to the
user in the design.

The gate is deliberately biased towards letting prompts through: a false
"needs clarification" is a visible annoyance on every turn, while a false
"looks fine" costs one routed call. ``min_confidence`` is that bias, made
explicit.
"""

from __future__ import annotations

import logging

from router.agent.backends import _GATE_SYSTEM, LLMBackend, _extract_json
from router.agent.contracts import GateResult

log = logging.getLogger(__name__)


class PromptGate:
    """Checks a prompt for missing requirements before any routing happens."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        min_prompt_chars: int = 8,
        max_questions: int = 3,
        enabled: bool = True,
    ) -> None:
        self.backend = backend
        self.min_prompt_chars = min_prompt_chars
        self.max_questions = max_questions
        self.enabled = enabled

    def check(self, prompt: str, *, has_history: bool = False) -> GateResult:
        """Decide whether ``prompt`` carries enough to be answered.

        ``has_history`` relaxes the gate: in a follow-up turn the missing
        referent is usually in the previous turn, not missing from the request.
        """
        text = (prompt or "").strip()
        if not text:
            return GateResult(
                is_sufficient=False,
                missing_requirements=["What would you like help with?"],
                reason="empty prompt",
            )
        if not self.enabled:
            return GateResult(is_sufficient=True, reason="gate disabled")
        if len(text) < self.min_prompt_chars and not has_history:
            return GateResult(
                is_sufficient=False,
                missing_requirements=["Could you give a bit more detail about what you need?"],
                reason=f"prompt shorter than {self.min_prompt_chars} chars",
            )

        try:
            raw = self.backend.complete(_GATE_SYSTEM, text, max_tokens=400)
        except Exception as exc:  # noqa: BLE001 - never block a turn on gate failure
            log.warning("gate backend failed (%s); passing prompt through", exc)
            return GateResult(is_sufficient=True, reason=f"gate error: {exc}", confidence=0.0)

        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return GateResult(is_sufficient=True, reason="unparseable gate response", confidence=0.0)

        missing = [str(q) for q in (parsed.get("missing") or [])][: self.max_questions]
        sufficient = bool(parsed.get("sufficient", True))
        if has_history and missing:
            # A follow-up inherits context; do not re-interrogate the user.
            log.debug("gate: suppressing %d question(s) because history is present", len(missing))
            return GateResult(is_sufficient=True, reason="follow-up turn; context inherited", confidence=0.5)

        return GateResult(
            is_sufficient=sufficient and not missing,
            missing_requirements=missing,
            reason=str(parsed.get("reason", "")),
        )

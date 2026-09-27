"""``CostModel``: predicted cost only. Observed cost lives in the dataset and
in :class:`~router.llm.client.LLMResponse`, never here -- see
``docs/architecture.md``'s "predicted vs. observed cost" note. ``RouterModel``,
``PlanForecaster`` and ``OptimizationObjective`` all take a predicted cost as
input; nothing routing-facing should read observed cost.

This is intentionally simple: a per-model output-token *prior* (a plain
lookup), not a learned regressor. The diagram's two ``predictCost`` overloads
(by token counts, and by a whole ``RoutingContext``) become two distinctly-
named methods here -- :meth:`predict_cost` and :meth:`predict_cost_for_context`
-- since Python has no signature-based overloading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..context import RoutingContext
    from .profile import LLMProfile

#: fallback output-token estimate for a model with no entry in
#: ``output_token_priors`` -- a round, conservative guess, not a fitted value.
_DEFAULT_OUTPUT_TOKENS = 256

#: characters-per-token used to turn a raw prompt string into an input-token
#: estimate for :meth:`CostModel.predict_cost_for_context` -- a rough English-text
#: average, not a real tokenizer count.
_CHARS_PER_TOKEN = 4


@dataclass
class CostModel:
    version: str = "0"
    output_token_priors: Mapping[str, float] = field(default_factory=dict)

    def predict_output_tokens(self, model: "LLMProfile", input_tokens: int) -> int:
        """A per-model prior, ignoring ``input_tokens`` -- there is no signal
        here yet relating prompt length to expected output length."""
        prior = self.output_token_priors.get(model.model_id)
        return int(prior) if prior is not None else _DEFAULT_OUTPUT_TOKENS

    def predict_cost(self, model: "LLMProfile", input_tokens: int, output_tokens: int) -> float:
        return float(input_tokens * model.input_cost_per_token
                     + output_tokens * model.output_cost_per_token)

    def predict_cost_for_context(self, model: "LLMProfile", context: "RoutingContext") -> float:
        """Estimates ``input_tokens`` from the raw prompt length (no tokenizer
        call), then delegates to :meth:`predict_output_tokens`/:meth:`predict_cost`."""
        input_tokens = max(1, len(context.request.prompt) // _CHARS_PER_TOKEN)
        output_tokens = self.predict_output_tokens(model, input_tokens)
        return self.predict_cost(model, input_tokens, output_tokens)

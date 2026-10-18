"""``CostModel``: predicted cost only. Observed cost lives in the dataset and
in :class:`~router.llm.client.LLMResponse`, never here -- see
``docs/architecture.md``'s "predicted vs. observed cost" note. ``RouterModel``,
``PlanForecaster`` and ``OptimizationObjective`` all take a predicted cost as
input; nothing routing-facing should read observed cost.

This is intentionally simple: a per-model output-token *prior* (a plain
lookup), not a learned regressor. Cost is predicted from token counts
(:meth:`predict_cost`); a caller with a prompt estimates its tokens with
:func:`estimate_tokens`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .profile import LLMProfile

#: fallback output-token estimate for a model with no entry in
#: ``output_token_priors`` -- a round, conservative guess, not a fitted value.
_DEFAULT_OUTPUT_TOKENS = 256

#: characters-per-token used to turn a raw prompt string into an input-token
#: estimate (:func:`estimate_tokens`) -- a rough English-text
#: average, not a real tokenizer count.
_CHARS_PER_TOKEN = 4


@dataclass
class CostModel:
    version: str = "0"
    output_token_priors: Mapping[str, float] = field(default_factory=dict)

    def predict_output_tokens(self, model: LLMProfile, input_tokens: int) -> int:
        """A per-model prior, ignoring ``input_tokens`` -- there is no signal
        here yet relating prompt length to expected output length."""
        prior = self.output_token_priors.get(model.model_id)
        return int(prior) if prior is not None else _DEFAULT_OUTPUT_TOKENS

    def predict_cost(self, model: LLMProfile, input_tokens: int, output_tokens: int) -> float:
        return float(input_tokens * model.input_cost_per_token
                     + output_tokens * model.output_cost_per_token)


def estimate_tokens(chars: int) -> int:
    """Rough token count for ``chars`` characters of text (no tokenizer call;
    ``_CHARS_PER_TOKEN`` characters per token, at least 1)."""
    return max(1, int(chars) // _CHARS_PER_TOKEN)


def price_tokens(input_tokens: int | None, output_tokens: int | None,
                 input_price: float | None, output_price: float | None) -> float | None:
    """``in * input_price + out * output_price`` (USD), or ``None`` when it
    can't be known: no token counts at all, or a side that was actually used
    (nonzero tokens) has no price -- an unpriced side is unknown, never free."""
    if input_tokens is None and output_tokens is None:
        return None
    if (input_tokens and input_price is None) or (output_tokens and output_price is None):
        return None
    return float((input_tokens or 0) * (input_price or 0.0)
                 + (output_tokens or 0) * (output_price or 0.0))


def estimate_call_cost(
    model: LLMProfile | None,
    input_chars: int | None = None,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_model: CostModel | None = None,
    fallback: float | None = None,
) -> tuple[float | None, str]:
    """Expected USD cost of one call *before* it runs, and where the number
    came from -- the "expected" side of an expected-vs-actual comparison, and
    the one estimate both a single call and a plan step are priced with.

    * ``"profile"``: ``model`` has both per-token prices -> priced at
      ``input_tokens`` (else estimated from ``input_chars``) and
      ``output_tokens`` (else ``cost_model``'s per-model prior,
      ``_DEFAULT_OUTPUT_TOKENS`` when it has none).
    * ``"router_prior"``: no prices, but ``fallback`` is given -- typically the
      router's per-query cost for that pool model from its training data
      (``RouterModel.cost_by_model()``), a different (and coarser) basis.
    * ``"unknown"``: neither -> ``None``.
    """
    in_price = getattr(model, "input_cost_per_token", None)
    out_price = getattr(model, "output_cost_per_token", None)
    if model is not None and in_price is not None and out_price is not None:
        cm = cost_model or CostModel()
        in_tok = input_tokens if input_tokens is not None else estimate_tokens(input_chars or 0)
        out_tok = output_tokens if output_tokens is not None else cm.predict_output_tokens(model, in_tok)
        return price_tokens(in_tok, out_tok, in_price, out_price), "profile"
    if fallback is not None:
        return float(fallback), "router_prior"
    return None, "unknown"

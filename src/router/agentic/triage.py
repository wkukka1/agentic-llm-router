"""Triage: given a prompt, decide whether the routed model can answer it directly
or whether the request should be decomposed and run through an orchestrator.

The signal comes from the **router itself**. The router scores the prompt against
the candidate pool; the best predicted quality ``q*`` is a difficulty proxy
(this is the NIRT "capacity / complexity" reading -- ``docs/nirt_capacity.md``):

* ``q*`` high  -> the pool has a model that handles this well -> **single shot**.
* ``q*`` low   -> even the best candidate is predicted to struggle -> **decompose**
  so each sub-part can be routed to a strong model.

``q*`` is the best score in the pool, independent of the router's cost weight:
with ``lam > 0`` the *selected* model may be a cheaper, weaker one, and judging
difficulty by it would flag easy prompts as hard.

A cheap lexical multi-part check (``"and then"``, enumerated lists, multiple
question marks, length) also forces decomposition. An optional LLM triage
(:class:`LLMTriage`) can override the heuristic when a chat model is supplied.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .llm_clients import message_text

__all__ = ["TriageDecision", "HeuristicTriage", "LLMTriage"]

_MULTIPART_MARKERS = (
    " and then ", " after that ", " followed by ", "step by step", "first,",
    "; then", " as well as ",
)
# enumerations only at line start: "1. foo" / "2) bar", not "1.5" or "v2.0"
_ENUMERATION_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


@dataclass
class TriageDecision:
    mode: str                      # "single" | "orchestrate"
    selected_model_id: str
    predicted_quality: float       # best predicted quality in the pool (the difficulty signal)
    reason: str
    scores: dict = field(default_factory=dict)          # model_id -> predicted quality

    @property
    def orchestrate(self) -> bool:
        return self.mode == "orchestrate"


def _fmt_q(q: float) -> str:
    return f"{q:.2f}" if math.isfinite(q) else "unknown"


class HeuristicTriage:
    """Router-quality threshold + a lexical multi-part check. No LLM."""

    def __init__(self, *, quality_threshold: float = 0.55, max_chars_single: int = 600):
        self.quality_threshold = float(quality_threshold)
        self.max_chars_single = int(max_chars_single)

    def __call__(self, prompt: str, *, model_id: str, predicted_quality: float,
                 scores: Optional[dict] = None) -> TriageDecision:
        low = prompt.lower()
        multipart = (
            prompt.count("?") >= 3
            or len(prompt) > self.max_chars_single
            or any(m in low for m in _MULTIPART_MARKERS)
            or len(_ENUMERATION_RE.findall(prompt)) >= 2
        )
        q = float(predicted_quality)
        known = math.isfinite(q)
        # an unscorable prompt (NaN) is not evidence of a strong model
        weak = (not known) or q < self.quality_threshold

        if multipart and weak:
            reason = (f"multi-part prompt and best predicted quality "
                      f"{_fmt_q(q)} < {self.quality_threshold:.2f}")
            mode = "orchestrate"
        elif multipart:
            reason = "prompt looks multi-part (decompose so each part routes independently)"
            mode = "orchestrate"
        elif not known:
            reason = f"no usable router score for this prompt (best candidate {model_id}); decompose"
            mode = "orchestrate"
        elif weak:
            reason = (f"best predicted quality {_fmt_q(q)} "
                      f"< {self.quality_threshold:.2f}; decompose to raise it")
            mode = "orchestrate"
        else:
            reason = (f"best predicted quality {_fmt_q(q)} "
                      f">= {self.quality_threshold:.2f}; {model_id} answers directly")
            mode = "single"
        return TriageDecision(mode, model_id, q, reason, scores or {})


class LLMTriage:
    """Ask a LangChain chat model for a yes/no on decomposition; fall back to the
    heuristic on any call failure or unparseable verdict. The router's scores
    are given to the model as context so its judgement is grounded in the same
    difficulty signal."""

    _PROMPT = (
        "You triage user requests for an LLM router.\n"
        "Routed model: {model_id}; best predicted quality in the pool: {q} (0-1 scale).\n"
        "Per-model predicted quality: {scores}\n\n"
        "Request:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
        "Answer with exactly one word: SINGLE if one model call can handle it well, "
        "or DECOMPOSE if it should be split into sub-tasks."
    )

    def __init__(self, chat_model, *, fallback: Optional[HeuristicTriage] = None):
        self._chat = chat_model
        self._fallback = fallback or HeuristicTriage()

    def __call__(self, prompt: str, *, model_id: str, predicted_quality: float,
                 scores: Optional[dict] = None) -> TriageDecision:
        finite = {k: v for k, v in (scores or {}).items() if math.isfinite(v)}
        top = dict(sorted(finite.items(), key=lambda kv: -kv[1])[:5])
        request = self._PROMPT.format(
            model_id=model_id, q=_fmt_q(float(predicted_quality)),
            scores=", ".join(f"{k}={v:.2f}" for k, v in top.items()) or "n/a",
            prompt=prompt[:2000],
        )
        try:  # only the provider call is guarded; formatting bugs must surface
            msg = self._chat.invoke(request)
        except Exception as exc:  # pragma: no cover - network / provider errors
            d = self._fallback(prompt, model_id=model_id,
                               predicted_quality=predicted_quality, scores=scores)
            d.reason = f"LLM triage failed ({exc!r}); heuristic: {d.reason}"
            return d
        verdict = message_text(msg).strip().upper()

        if verdict.startswith("DECOMP"):
            return TriageDecision("orchestrate", model_id, float(predicted_quality),
                                  "LLM triage: DECOMPOSE", scores or {})
        if verdict.startswith("SINGLE"):
            return TriageDecision("single", model_id, float(predicted_quality),
                                  "LLM triage: SINGLE", scores or {})
        d = self._fallback(prompt, model_id=model_id,
                           predicted_quality=predicted_quality, scores=scores)
        d.reason = f"LLM triage verdict unparseable ({verdict[:40]!r}); heuristic: {d.reason}"
        return d


def best_from_result(routing_result) -> tuple[str, float, dict]:
    """``(selected_model_id, best_predicted_quality, {model_id: quality})`` for a
    1-row :class:`~router.routing.base.RoutingResult`.

    The quality is the pool's best finite score (the triage difficulty signal),
    not the score of the -- possibly cost-selected -- model. It is NaN when the
    router had nothing selectable for the row (``RoutingResult.fallback``)."""
    row = np.asarray(routing_result.scores, dtype=np.float64)[0]
    scores = dict(zip(routing_result.model_ids, (float(x) for x in row)))
    mid = routing_result.selected_model_ids[0]
    fallback = getattr(routing_result, "fallback", None)
    if fallback is not None and bool(np.asarray(fallback)[0]):
        return mid, float("nan"), scores
    finite = row[np.isfinite(row)]
    return mid, (float(finite.max()) if finite.size else float("nan")), scores

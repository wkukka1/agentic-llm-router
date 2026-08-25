"""How the agentic steps talk to an actual model.

Two steps genuinely need generation -- deciding what a vague prompt is missing,
and decomposing a hard prompt into sub-queries. Both sit behind
:class:`LLMBackend` so that:

* the control flow is testable with no network and no keys, and
* swapping provider or model is configuration.

:class:`HeuristicBackend` is the default. It is not a stub that returns nothing:
it implements real rule-based versions of both steps, so the pipeline is
end-to-end runnable out of the box. It is, however, clearly weaker than the LLM
path -- ``requires_llm`` says so, and the pipeline records which was used.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

log = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = """You decompose a hard user request into independent sub-queries.
Return ONLY a JSON array. Each element: {"text": str, "depends_on": [int]}.
`depends_on` holds indices of earlier sub-queries whose answers this one needs.
Produce between 2 and $MAX_SUB_QUERIES sub-queries. Prefer independent ones."""

_GATE_SYSTEM = """You judge whether a user request is specified well enough to answer.
Return ONLY JSON: {"sufficient": bool, "missing": [str], "reason": str}.
`missing` holds short questions asking the user for what is genuinely absent.
Be strict about absent constraints, but do not invent requirements for a request
that is already clear."""


class LLMBackend(Protocol):
    """Minimal generation surface the agent layer depends on."""

    #: False when the backend works offline, which the pipeline records.
    requires_llm: bool
    #: True only if this backend can produce a real answer to an arbitrary
    #: prompt. Rule-based backends can classify and decompose but cannot
    #: generate, and the pipeline refuses execute mode without this.
    can_generate: bool

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str: ...


def _extract_json(text: str) -> object | None:
    """Pull the first JSON value out of a completion, tolerating prose/fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"[\[{].*[\]}]", cleaned, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("backend returned unparseable JSON: %s", cleaned[:200])
        return None


class LiteLLMBackend:
    """Provider-agnostic backend via LiteLLM.

    Requires the relevant provider key in the environment. Failures are raised,
    not swallowed -- a silently degraded router is worse than a loud one.
    """

    requires_llm = True
    can_generate = True

    def __init__(self, model: str = "claude-sonnet-5", *, temperature: float = 0.0, timeout: int = 60) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""


#: Words that signal an unbounded reference the answer would have to guess at.
_VAGUE_MARKERS = (
    "something", "somehow", "stuff", "things", "etc", "and so on",
    "make it better", "improve it", "fix it", "optimize it", "help me with",
)

#: A prompt asking for an artefact usually needs constraints to be answerable.
_ARTEFACT_MARKERS = ("write", "build", "create", "design", "implement", "generate", "make")


class HeuristicBackend:
    """Rule-based fallback so the pipeline runs with no keys configured.

    The gate rules encode the cheap, high-precision cases: a very short
    artefact request, an unresolved pronoun with no antecedent, an explicit
    vagueness marker. Decomposition splits on enumerated or conjoined asks.
    """

    requires_llm = False
    #: Rules can gate and decompose; they cannot answer. Returning the gate's
    #: JSON as if it were an answer would be worse than refusing.
    can_generate = False

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        if "decompose" in system.lower():
            return json.dumps(self._decompose(user))
        return json.dumps(self._gate(user))

    def _gate(self, prompt: str) -> dict:
        text = prompt.strip()
        lowered = text.lower()
        words = text.split()
        missing: list[str] = []

        wants_artefact = any(lowered.startswith(m) or f" {m} " in lowered for m in _ARTEFACT_MARKERS)
        if wants_artefact and len(words) < 12:
            missing.append("What are the requirements, constraints and target format?")
        if any(marker in lowered for marker in _VAGUE_MARKERS):
            missing.append("Which specific part should this cover, and what does 'better' mean here?")
        # A leading pronoun with nothing before it has no referent to resolve.
        if re.match(r"^(it|that|this|they|them)\b", lowered) and len(words) < 15:
            missing.append("What exactly does this refer to?")

        return {
            "sufficient": not missing,
            "missing": missing[:3],
            "reason": "heuristic gate" if missing else "prompt appears self-contained",
        }

    def _decompose(self, prompt: str) -> list[dict]:
        # Numbered or bulleted asks are already a decomposition.
        enumerated = re.split(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", prompt)
        parts = [p.strip() for p in enumerated if p.strip()]
        if len(parts) < 2:
            parts = [p.strip() for p in re.split(r"\band then\b|\balso\b|;", prompt) if p.strip()]
        if len(parts) < 2:
            parts = [prompt.strip()]
        return [{"text": part, "depends_on": []} for part in parts[:6]]


def default_backend(model: str | None = None) -> LLMBackend:
    """LiteLLM when a model is named, heuristics otherwise."""
    return LiteLLMBackend(model) if model else HeuristicBackend()

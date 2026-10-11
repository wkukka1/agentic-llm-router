"""Claude Code subagents: recognise a spawn, name its role, give it a tier.

Claude Code sends a subagent's requests through ``ANTHROPIC_BASE_URL`` like
the main agent's, as a conversation of their own: its own system prompt and
tools, and a first message that is the task the main agent wrote. The billing
line at the top of its system prompt says ``cc_is_subagent=true`` (seen on
Claude Code 2.1.293; the main agent's does not).

A subagent is the cheapest place to change model: its system prompt and tools
differ from the main agent's, so it has a prompt cache of its own, and moving
it never makes the main conversation rewrite its cache.

Its role, first match wins:

1. ``definition`` -- one of :data:`ROLES`, which ``aar-gateway code`` hands to
   Claude Code; their prompts carry ``aar-role: <name>``;
2. ``builtin`` -- a Claude Code built-in recognised by its system prompt
   (Explore, Plan);
3. ``task`` -- the task the main agent wrote, by keyword
   (:func:`classify_task`).

Otherwise it has no role, and the general router picks as for any turn.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional

from .pricing import canonical_model
from .routing import incompatibility, price_for

SUBAGENT_MARK = "cc_is_subagent=true"
_ROLE_MARK = re.compile(r"\baar-role:\s*([a-z-]+)")
#: Claude Code's built-in subagents, by the start of their system prompt
#: (read off Claude Code 2.1.293-2.1.296's requests)
_BUILTINS = (("explore", re.compile(r"You are a file search specialist")),
             ("plan", re.compile(r"You are a software architect and planning specialist")))

_HAIKU, _SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
#: role -> model. ``None`` keeps the model the session asked for.
DEFAULT_ROLE_MODELS: dict[str, Optional[str]] = {
    "scout": _HAIKU, "extractor": _HAIKU, "explore": _HAIKU,
    "reviewer": _SONNET, "planner": _SONNET, "tester": _SONNET, "plan": _SONNET,
    "architect": None,
}

#: Task keywords, in priority order: a task that asks for a root cause and
#: also says "find" is an architect's, not a scout's. Only the start of the
#: task is read -- that is where the main agent states what it wants.
_TASK_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("architect", re.compile(r"\broot[- ]cause|\bdebug|\barchitectur|\brace condition|\bdeadlock")),
    ("reviewer", re.compile(r"\breview|\baudit\b|\bcritique|\bfind (?:bugs|issues|problems)")),
    ("tester", re.compile(r"\b(?:write|add|generate) (?:\w+ )?tests?\b|\btest cases?\b")),
    ("planner", re.compile(r"\bplan (?:for|how|the)\b|\bdesign (?:a|an|the)\b|\bpropose\b|\boutline\b")),
    ("extractor", re.compile(r"\blabel|\bclassify|\bextract|\bsummari[sz]e|\bcommit message|\bchangelog"
                             r"|\bconvert\b|\breformat|\btranslate\b")),
    ("scout", re.compile(r"\bfind\b|\blocate\b|\bsearch\b|\bwhere (?:is|are|does|do)\b|\blist (?:all|every)\b"
                         r"|\bgrep\b|\bcount\b|\bwhich files?\b|\bexplore\b")),
)
TASK_CHARS = 400
#: A task that opens by asking for something to be made is generation, whatever
#: it is about ("Write 100 prompts ... of the task type SUMMARIZE" is not
#: extraction): only the roles ranked above planner may still claim it.
_MAKES = re.compile(r"^(?:please\s+)?(?:write|generate|create|compose|produce|build|implement)\b")
_BEFORE_MAKES = ("architect", "reviewer", "tester")


def classify_task(task: str) -> Optional[tuple[str, str]]:
    """``(role, the words that decided it)`` for a subagent's task, or ``None``."""
    head = (task or "")[:TASK_CHARS].lower()
    makes = bool(_MAKES.match(head.lstrip()))
    for role, pattern in _TASK_RULES:
        if makes and role not in _BEFORE_MAKES:
            return None
        found = pattern.search(head)
        if found:
            return role, found.group(0)
    return None


def _system_text(body: Mapping) -> str:
    system = body.get("system")
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system or [] if isinstance(b, Mapping))


def detect(body: Mapping, task: Optional[str]) -> Optional[dict]:
    """``None`` for anything but a subagent; else ``{"role", "source", "why"}``
    (``role`` may be ``None``: a subagent nothing here recognises)."""
    system = _system_text(body)
    if SUBAGENT_MARK not in system:
        return None
    found = _ROLE_MARK.search(system)
    if found:
        return {"role": found.group(1), "source": "definition", "why": found.group(0)}
    for role, pattern in _BUILTINS:
        if pattern.search(system):
            return {"role": role, "source": "builtin", "why": pattern.pattern}
    guessed = classify_task(task or "")
    if guessed:
        return {"role": guessed[0], "source": "task", "why": guessed[1]}
    return {"role": None, "source": None, "why": None}


def role_decision(agent: Mapping, body: Mapping, requested: Optional[str], settings: Mapping,
                  blocked: Mapping[str, str], est_prompt_tokens: int) -> Optional[dict]:
    """The model for a subagent's turn from its role, or ``None`` to leave it
    to the general router (no role, or the role's model can't take this
    request). Never a model pricier than the one the session asked for: a
    role tier is a ceiling on cost, not an upgrade."""
    role = agent.get("role")
    models = {**DEFAULT_ROLE_MODELS, **((settings.get("subagents") or {}).get("roles") or {})}
    if role is None or role not in models:
        return None
    label = f"{role} subagent ({agent['source']}: {agent['why']})"
    target = models[role]
    if target is None:
        return {"model": requested, "reason": f"{label}: keeps the session's model"}
    target = canonical_model(target)
    spec = settings["models"].get(target)
    if spec is None or target in blocked or incompatibility(body, spec) \
            or est_prompt_tokens > spec.get("context_window", float("inf")):
        return None
    mine, asked = price_for(settings, target), price_for(settings, requested)
    if not mine or not asked:
        return None
    if mine["output"] > asked["output"]:
        return {"model": requested, "reason": f"{label}: {target} would cost more than the session's model"}
    return {"model": target, "reason": f"{label} -> {target}"}


#: The roles ``aar-gateway code`` gives Claude Code (``--agents``). Every
#: prompt ends with an output budget: a subagent's whole answer lands in the
#: main agent's context, so a long report costs the main conversation on
#: every request after it. ``model`` pins the tier in Claude Code itself, so
#: the roles keep their tiers without the gateway too.
_BUDGET = ("Your final message is all the main agent sees, and it carries it for the rest of the "
           "conversation: keep it within the budget below. No preamble, no restating the task, no "
           "closing summary.")

ROLES: dict[str, dict] = {
    "scout": {
        "model": "haiku",
        "description": "Finds things in the codebase: where X is defined or used, which files do Y, "
                       "how many of Z. Read-only. Use for any search or lookup.",
        "prompt": "You locate code and facts in the repository. Do not change files.\n\n" + _BUDGET +
                  "\n\nOutput budget: at most 10 bullets, each `path:line` and one sentence. If nothing "
                  "matches, say so in one line.",
    },
    "extractor": {
        "model": "haiku",
        "description": "Mechanical, checkable work on given text: pull fields out of logs, label or "
                       "classify items, draft a commit message or changelog, check a lockfile.",
        "prompt": "You do mechanical extraction and drafting. Follow the requested format exactly.\n\n"
                  + _BUDGET + "\n\nOutput budget: only the structured result asked for (list, table, "
                  "JSON or draft text); write large results to the file the task names and reply with "
                  "its path and a count.",
    },
    "reviewer": {
        "model": "sonnet",
        "description": "Reviews a diff or code for bugs, risks and missing tests. Read-only.",
        "prompt": "You review code. Report only real problems, each tied to a line. Do not change files.\n\n"
                  + _BUDGET + "\n\nOutput budget: one-line verdict, then at most 10 findings, each "
                  "`severity (high/medium/low) · path:line · problem · fix` on one line.",
    },
    "planner": {
        "model": "sonnet",
        "description": "Turns a goal into an implementation plan: steps, files touched, risks. Read-only.",
        "prompt": "You plan changes. Read what you need; do not change files.\n\n" + _BUDGET +
                  "\n\nOutput budget: a numbered plan of at most 12 steps (each: what, which files), then "
                  "at most 5 risks or open questions. No prose paragraphs.",
    },
    "tester": {
        "model": "sonnet",
        "description": "Writes or extends tests for given code and runs them.",
        "prompt": "You write tests that match the project's existing test style, then run them.\n\n"
                  + _BUDGET + "\n\nOutput budget: the test files you changed, how many tests pass or "
                  "fail, and for each failure one line saying why. No test code in the reply.",
    },
    "architect": {
        "model": "inherit",
        "description": "Hard reasoning: root-cause debugging, architecture decisions, trade-offs that "
                       "need the strongest model.",
        "prompt": "You investigate hard problems: find root causes and weigh designs, with evidence "
                  "from the code.\n\n" + _BUDGET + "\n\nOutput budget: conclusion first, then at most 8 "
                  "findings with `path:line` evidence, then the recommended next step.",
    },
}


def agent_definitions() -> dict[str, dict]:
    """``ROLES`` as Claude Code's ``--agents`` JSON, each prompt marked with
    its role so the gateway can recognise its requests."""
    return {name: {"description": r["description"], "model": r["model"],
                   "prompt": f"{r['prompt']}\n\naar-role: {name}"}
            for name, r in ROLES.items()}


def agent_markdown(name: str) -> str:
    """One role as a Claude Code agent file (``~/.claude/agents/<name>.md``)."""
    d = agent_definitions()[name]
    return f"---\nname: {name}\ndescription: {d['description']}\nmodel: {d['model']}\n---\n\n{d['prompt']}\n"

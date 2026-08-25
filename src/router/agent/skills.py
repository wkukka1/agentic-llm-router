"""Skills the pool can invoke, and the rules for attaching them to a sub-query.

A skill is a capability the model does not have on its own -- a tool call, a
retrieval step, an interpreter. In the design they hang off the model pool and
feed both the answer path and the follow-up path.

Attachment is signal-driven so the orchestrator does not have to reason about
tool availability: it decomposes, and the router decorates each sub-query with
the skills its signals imply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SKILLS_PATH = Path("configs/skills.yaml")


@dataclass(slots=True)
class Skill:
    """A tool the router can attach to a routed sub-query."""

    name: str
    description: str
    #: Task types that trigger this skill.
    task_types: list[str] = field(default_factory=list)
    #: Domains that trigger this skill.
    domains: list[str] = field(default_factory=list)
    #: Attach whenever difficulty is at or above this.
    min_difficulty: float | None = None
    #: Rough USD per invocation, folded into the cost estimate.
    cost: float = 0.0


@dataclass(slots=True)
class SkillRegistry:
    skills: list[Skill]

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_SKILLS_PATH) -> SkillRegistry:
        raw = yaml.safe_load(Path(path).read_text())
        return cls([Skill(**row) for row in raw["skills"]])

    def select(self, signals, *, available: list[str] | None = None) -> list[Skill]:
        """Skills whose trigger conditions the signals satisfy.

        ``available`` restricts to what the chosen model is permitted to call,
        so a pool member without tool access never gets handed one.
        """
        chosen: list[Skill] = []
        for skill in self.skills:
            if available is not None and skill.name not in available:
                continue
            triggered = (
                signals.task_type in skill.task_types
                or signals.domain in skill.domains
                or (skill.min_difficulty is not None and signals.difficulty >= skill.min_difficulty)
            )
            if triggered:
                chosen.append(skill)
        return chosen

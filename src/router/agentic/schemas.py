"""Pydantic schemas for the agentic layer's settings.

Everything an ``AgenticRouter`` is configured with -- which router run, which
clients, how triage / decomposition / synthesis / orchestration behave, and the
text of every LLM prompt -- loads from one YAML file
(``configs/agentic.yaml``) into :class:`AgenticRouterSettings` and is validated
before anything is built::

    from router.agentic import AgenticRouter
    from router.agentic.schemas import load_settings

    settings = load_settings()                       # configs/agentic.yaml
    agent = AgenticRouter.from_settings(settings)    # or from_settings("path.yaml")

**Settings.** One model per helper (:class:`TriageSettings`,
:class:`DecomposerSettings`, :class:`SynthesizerSettings`), one per
orchestrator option (:class:`PlanSettings`, :class:`ToolAgentSettings`), and
the root :class:`AgenticRouterSettings`. All forbid unknown keys, so a
misspelled setting is an error rather than a silently ignored default.

**Prompts** are plain ``str.format`` templates, ``{name: template}`` under the
file's ``prompts`` section -- the single copy of every prompt the LLM helpers
send. A helper names the prompt it uses; a name that isn't in ``prompts`` is
a load-time error. :func:`default_prompts` is what the helpers fall back to
when built without an explicit ``prompt=``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "TriageSettings",
    "DecomposerSettings",
    "SynthesizerSettings",
    "PlanSettings",
    "ToolAgentSettings",
    "AgenticRouterSettings",
    "load_settings",
    "default_prompts",
]

#: repo-relative default settings file
DEFAULT_SETTINGS_PATH = "configs/agentic.yaml"

#: every model forbids unknown keys (a misspelled setting is an error) and is immutable
_STRICT = ConfigDict(extra="forbid", frozen=True)


class TriageSettings(BaseModel):
    model_config = _STRICT

    kind: Literal["heuristic", "llm"] = "heuristic"
    quality_threshold: float = Field(0.55, ge=0.0, le=1.0)
    max_chars_single: int = Field(600, gt=0)
    #: llm only: ask the LLM only when the router's best score is within this
    #: band of the threshold (``None`` = always ask)
    margin: Optional[float] = Field(None, ge=0.0)
    prompt: str = "triage"
    model: Optional[str] = None


class DecomposerSettings(BaseModel):
    model_config = _STRICT

    #: routed: run on the cheapest pool model whose predicted quality >= min_quality
    kind: Literal["naive", "llm", "routed"] = "naive"
    max_parts: int = Field(6, gt=0)
    min_part_chars: int = Field(8, ge=0)
    max_prompt_chars: int = Field(4000, gt=0)
    prompt: str = "decompose"
    model: Optional[str] = None
    min_quality: float = Field(0.75, ge=0.0, le=1.0)


class SynthesizerSettings(BaseModel):
    model_config = _STRICT

    #: routed: as on DecomposerSettings
    kind: Literal["concat", "llm", "routed"] = "concat"
    max_body_chars: int = Field(8000, gt=0)
    prompt: str = "synthesize"
    model: Optional[str] = None
    min_quality: float = Field(0.75, ge=0.0, le=1.0)


class QualityCheckSettings(BaseModel):
    """The plan orchestrator's final quality check (``router.agentic.quality``)."""

    model_config = _STRICT

    enabled: bool = False
    #: the judge = cheapest pool model predicted >= this for the request
    min_quality: float = Field(0.8, ge=0.0, le=1.0)
    #: a PASS verdict also needs SCORE >= this (0-10)
    pass_score: float = Field(7.0, ge=0.0, le=10.0)
    prompt: str = "verify"


class PlanSettings(BaseModel):
    """The plan orchestrator (``router.execution.routed.plan_orchestrator``)."""

    model_config = _STRICT

    budget: Optional[float] = Field(None, gt=0.0)        # USD per request; None = unlimited
    max_workers: int = Field(4, gt=0)
    sequential: bool = False
    max_replans: int = Field(1, ge=0)
    #: the prompt the plan orchestrator's planner uses (numbered steps with [after N])
    planner_prompt: str = "plan"
    #: route each step just before it runs, on its prompt plus dependency results
    route_late: bool = False
    quality_check: QualityCheckSettings = Field(default_factory=QualityCheckSettings)


class ToolAgentSettings(BaseModel):
    """The LangChain tool-calling orchestrator."""

    model_config = _STRICT

    model: Optional[str] = None
    max_iterations: int = Field(8, gt=0)
    prompt: str = "tool_agent_system"


class AgenticRouterSettings(BaseModel):
    """Root of ``configs/agentic.yaml``: the router, its helpers, and the prompts."""

    model_config = _STRICT

    #: ``{name: str.format template}`` -- every prompt the LLM helpers send
    prompts: dict[str, str]
    run: str = "nirt-2d-projected"
    runs_dir: Optional[str] = None
    clients: Literal["echo", "anthropic", "langchain"] = "echo"
    serve_map: str = "configs/serve_anthropic.yaml"
    lam: float = Field(0.0, ge=0.0)
    max_depth: int = Field(2, ge=0)
    max_calls: Optional[int] = Field(None, gt=0)
    warmup: bool = True
    result_sink: Optional[str] = None
    triage: TriageSettings = Field(default_factory=TriageSettings)
    decomposer: DecomposerSettings = Field(default_factory=DecomposerSettings)
    synthesizer: SynthesizerSettings = Field(default_factory=SynthesizerSettings)
    orchestrator: Literal["recursive", "plan", "tool_agent"] = "recursive"
    plan: PlanSettings = Field(default_factory=PlanSettings)
    tool_agent: ToolAgentSettings = Field(default_factory=ToolAgentSettings)

    @model_validator(mode="before")
    @classmethod
    def _flatten_router_section(cls, data):
        # the file is {prompts: ..., router: {...}}; the model is flat
        if isinstance(data, dict) and isinstance(data.get("router"), dict):
            data = {**data["router"], **{k: v for k, v in data.items() if k != "router"}}
        return data

    @model_validator(mode="after")
    def _references_resolve(self) -> "AgenticRouterSettings":
        """Every prompt a helper names exists, and every LLM-backed helper has a model."""
        uses = [("triage", self.triage), ("decomposer", self.decomposer),
                ("synthesizer", self.synthesizer), ("tool_agent", self.tool_agent)]
        for section, s in uses:
            if s.prompt not in self.prompts:
                raise ValueError(f"{section}.prompt: no prompt {s.prompt!r}; "
                                 f"known: {sorted(self.prompts)}")
            if getattr(s, "kind", None) == "llm" and not s.model:
                raise ValueError(f"{section}.kind is 'llm' but {section}.model is not set")
        if self.orchestrator == "tool_agent" and not self.tool_agent.model:
            raise ValueError("orchestrator is 'tool_agent' but tool_agent.model is not set")
        if self.orchestrator == "plan" and self.plan.planner_prompt not in self.prompts:
            raise ValueError(f"plan.planner_prompt: no prompt {self.plan.planner_prompt!r}; "
                             f"known: {sorted(self.prompts)}")
        q = self.plan.quality_check
        if self.orchestrator == "plan" and q.enabled and q.prompt not in self.prompts:
            raise ValueError(f"plan.quality_check.prompt: no prompt {q.prompt!r}; "
                             f"known: {sorted(self.prompts)}")
        return self

    @field_validator("run")
    @classmethod
    def _run_name(cls, v: str) -> str:
        if not v or "/" in v or "\\" in v:
            raise ValueError("run is a run name under the runs dir, not a path")
        return v


def load_settings(path: Union[str, Path, None] = None) -> AgenticRouterSettings:
    """Read and validate a settings file (default ``configs/agentic.yaml``)."""
    import yaml

    from ..config import resolve_path

    path = resolve_path(path or DEFAULT_SETTINGS_PATH)
    with open(path, encoding="utf-8") as fh:
        return AgenticRouterSettings.model_validate(yaml.safe_load(fh) or {})


@lru_cache(maxsize=1)
def default_prompts() -> dict[str, str]:
    """The prompts in ``configs/agentic.yaml`` -- what the LLM helpers use when
    no ``prompt=`` is given. Loaded once per process."""
    return dict(load_settings().prompts)

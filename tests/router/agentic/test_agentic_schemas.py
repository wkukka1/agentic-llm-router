"""Pydantic agent-settings schemas (router.agentic.schemas) and
AgenticRouter.from_settings."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from router.agentic import AgenticRouter, HeuristicTriage, LLMDecomposer, LLMSynthesizer, LLMTriage
from router.agentic.schemas import AgenticRouterSettings, default_prompts, load_settings
from router.routing import RouterModel

POOL = ["small", "big"]


class _TextRouter(RouterModel):
    kind = "schemas_text"
    can_route_text = True

    def __init__(self):
        super().__init__(POOL, name="t")

    def predict_scores(self, query_ids):
        return pd.DataFrame(0.5, index=[str(q) for q in query_ids], columns=POOL)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([[0.6, 0.9]] * len(texts), index=range(len(texts)), columns=POOL)


class _Chat:
    """Chat-model stand-in that records every request text."""

    def __init__(self, reply="SINGLE"):
        self.reply, self.requests = reply, []

    def invoke(self, text, **kw):
        self.requests.append(text)
        return self.reply


def _settings(**router):
    base = load_settings().model_dump()
    return AgenticRouterSettings.model_validate({**base, **router})


# --------------------------------------------------------------------------- #
# the shipped file                                                            #
# --------------------------------------------------------------------------- #
def test_shipped_settings_file_validates():
    s = load_settings()
    assert set(s.prompts) == {"triage", "decompose", "plan", "verify", "synthesize"}
    assert s.run == "nirt-2d-projected" and s.plan.model
    assert default_prompts() == s.prompts


# --------------------------------------------------------------------------- #
# settings validation                                                         #
# --------------------------------------------------------------------------- #
def test_unknown_setting_is_an_error():
    with pytest.raises(ValidationError, match="extra"):
        _settings(max_cals=3)
    with pytest.raises(ValidationError, match="extra"):
        _settings(triage={"kind": "heuristic", "treshold": 0.5})


def test_llm_helper_without_a_model_is_an_error():
    with pytest.raises(ValidationError, match="triage.model is not set"):
        _settings(triage={"kind": "llm"})


def test_plan_without_a_model_is_an_error():
    with pytest.raises(ValidationError):
        _settings(plan={"budget": 1.0})  # drops the shipped plan.model


def test_helper_must_reference_a_known_prompt():
    with pytest.raises(ValidationError, match="no prompt 'nope'"):
        _settings(synthesizer={"prompt": "nope"})


def test_values_are_range_checked():
    with pytest.raises(ValidationError):
        _settings(triage={"quality_threshold": 1.5})
    with pytest.raises(ValidationError):
        _settings(plan={"budget": -1})
    with pytest.raises(ValidationError, match="not a path"):
        _settings(run="../elsewhere")


def test_settings_are_immutable():
    s = load_settings()
    with pytest.raises(ValidationError):
        s.max_depth = 9


# --------------------------------------------------------------------------- #
# helpers send the file's prompts                                             #
# --------------------------------------------------------------------------- #
def test_llm_triage_sends_the_file_prompt():
    chat = _Chat("SINGLE")
    LLMTriage(chat)("What is 2+2?", model_id="big", predicted_quality=0.9, scores={"big": 0.9})
    expected = default_prompts()["triage"].format(model_id="big", q="0.90", scores="big=0.90",
                                                  prompt="What is 2+2?")
    assert chat.requests == [expected]


def test_helpers_accept_a_custom_prompt():
    chat = _Chat("first task\nsecond task")
    dec = LLMDecomposer(chat, prompt="SPLIT: {prompt}")
    assert dec("x and y") == ["first task", "second task"]
    assert chat.requests == ["SPLIT: x and y"]
    assert LLMSynthesizer(_Chat()).prompt == default_prompts()["synthesize"]


# --------------------------------------------------------------------------- #
# AgenticRouter.from_settings                                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_run(monkeypatch):
    from router.routing import NIRTRouter

    monkeypatch.setattr(NIRTRouter, "from_run", classmethod(lambda cls, run, runs_dir=None: _TextRouter()))


def test_from_settings_builds_the_default_router(fake_run, monkeypatch):
    import router.llm.adapters.langchain_adapter as lca
    from router.execution.routed import PlanOrchestratorBridge

    monkeypatch.setattr(lca, "chat_model", lambda model, **kw: _Chat("SINGLE"))
    agent = AgenticRouter.from_settings(_settings(warmup=False))
    assert isinstance(agent.triage, HeuristicTriage) and agent.triage.quality_threshold == 0.55
    assert isinstance(agent.orchestrator, PlanOrchestratorBridge)
    assert agent.run("What is 2+2?").mode == "single"


def test_from_settings_wires_llm_helpers_plan_orchestrator_and_sink(fake_run, monkeypatch, tmp_path):
    import router.llm.adapters.langchain_adapter as lca
    from router.agentic.quality import QualityChecker
    from router.agentic.router import flush_bookkeeping
    from router.execution.routed import PlanOrchestratorBridge
    from router.tracing.run_store import connect

    chats = []
    monkeypatch.setattr(lca, "chat_model", lambda model, **kw: chats.append(model) or _Chat("SINGLE"))
    db = tmp_path / "runs.sqlite"
    agent = AgenticRouter.from_settings(_settings(
        warmup=False, max_calls=5, result_sink=str(db),
        prompts={**load_settings().prompts, "terse_synth": "Merge: {original} / {body}"},
        triage={"kind": "llm", "model": "claude-haiku-4-5", "margin": 0.1},
        synthesizer={"prompt": "terse_synth"},
        plan={"model": "claude-sonnet-5", "budget": 0.5, "max_workers": 2},
    ))
    assert isinstance(agent.triage, LLMTriage) and agent.triage.margin == 0.1
    # one shared chat model drives decompose + judge + synthesize
    assert isinstance(agent.decomposer, LLMDecomposer)
    assert isinstance(agent.synthesizer, LLMSynthesizer) and agent.synthesizer.prompt == "Merge: {original} / {body}"
    assert isinstance(agent.orchestrator, PlanOrchestratorBridge) and agent.orchestrator.budget == 0.5
    assert isinstance(agent.orchestrator.checker, QualityChecker)  # always on now
    # one chat() call for triage's own model, one (shared, reused 3x) for plan.model
    assert chats == ["claude-haiku-4-5", "claude-sonnet-5"]
    assert agent.max_calls == 5
    assert agent.run("What is 2+2?").mode == "single"
    flush_bookkeeping(timeout=5)
    assert connect(db).execute("SELECT router FROM runs").fetchone() == ("nirt-2d-projected",)


def test_from_settings_accepts_a_yaml_path(fake_run, monkeypatch, tmp_path):
    import router.llm.adapters.langchain_adapter as lca
    import yaml

    from router.execution.routed import PlanOrchestratorBridge

    monkeypatch.setattr(lca, "chat_model", lambda model, **kw: _Chat("SINGLE"))
    data = yaml.safe_load(open("configs/agentic.yaml", encoding="utf-8"))
    data["router"]["warmup"] = False
    path = tmp_path / "agentic.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert isinstance(AgenticRouter.from_settings(path).orchestrator, PlanOrchestratorBridge)

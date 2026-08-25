"""End-to-end: a trained run directory serves predictions and drives the router.

Skipped unless a run has been trained with --save-model, since these assert on a
real artifact rather than a fixture.
"""

from pathlib import Path

import pytest

ARTIFACTS = Path("artifacts/domain")


def _first_saved_run() -> Path | None:
    if not ARTIFACTS.exists():
        return None
    for run in sorted(ARTIFACTS.iterdir()):
        if (run / "model").exists() and (run / "metrics.json").exists():
            return run
    return None


@pytest.fixture(scope="module")
def run_dir() -> Path:
    run = _first_saved_run()
    if run is None:
        pytest.skip("no saved model; run `router train experiments/domain --save-model`")
    return run


def test_domain_head_loads_and_predicts(run_dir):
    from router.data.taxonomy import DOMAIN_LABELS
    from router.inference import DomainHead

    head = DomainHead(run_dir)
    prediction = head.predict("Write a Python function that reverses a linked list in place.")
    assert prediction.domain in DOMAIN_LABELS
    assert 0.0 <= prediction.confidence <= 1.0
    assert sum(prediction.distribution.values()) == pytest.approx(1.0, abs=1e-6)


def test_domain_head_batch_matches_single(run_dir):
    from router.inference import DomainHead

    head = DomainHead(run_dir)
    prompts = ["Who wrote Moby Dick?", "Compute the eigenvalues of a 2x2 matrix."]
    batch = head.predict_batch(prompts)
    assert [b.domain for b in batch] == [head.predict(p).domain for p in prompts]


def test_low_confidence_sets_the_escalation_flag(run_dir):
    from router.inference import DomainHead

    never = DomainHead(run_dir, escalate_below=0.0)
    always = DomainHead(run_dir, escalate_below=1.01)
    prompt = "Tell me about it."
    assert not never.predict(prompt).should_escalate
    assert always.predict(prompt).should_escalate


def test_trained_head_drives_the_agentic_router(run_dir):
    from router.agent import Action, AgenticRouter
    from router.agent.tasktype import infer_task_type
    from router.inference import DomainHead, as_domain_fn

    router = AgenticRouter(
        domain_fn=as_domain_fn(DomainHead(run_dir)),
        task_type_fn=infer_task_type,
    )
    result = router.route("Explain how a B-tree splits a full node during insertion.")
    assert result.decision.action is not Action.CLARIFY
    assert result.decision.model is not None
    assert result.decision.signals.domain is not None

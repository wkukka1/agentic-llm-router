# AGENTICROUTER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

([router.py:180-188](../../../src/router/agentic/router.py#L180-L188), regression test `tests/router/agentic/test_agentic_router.py::test_result_predicted_quality_matches_the_model_that_answered`)

`run()`'s single-mode branch now uses `predicted_quality=root.predicted_quality` (the leaf
`SubCall`'s own value, already `_quality_of(decision)` under the hood) instead of
`decision.predicted_quality`. The orchestrated-mode branch is unchanged (there's no single
answering model to substitute in, so the pool-best triage signal is the least-wrong choice
there — the docstring already says `selected_model_id` in that mode "only produced the triage
signal"). `AgenticResult.predicted_quality` and `steps[0].predicted_quality` now agree in
single mode.

## Fix suggested when the bug was filed

In the single-mode branch use `root.predicted_quality` (or `_quality_of(decision)`). Keep the
pool-best value only in the triage reason.

## Regression tests

`tests/router/agentic/test_agentic_router.py:373` `test_result_predicted_quality_matches_the_model_that_answered`

```python
def test_result_predicted_quality_matches_the_model_that_answered():
    """AGENTICROUTER-001: with lam driving the pick toward a cheaper, weaker
    model, predicted_quality must be that model's own score, not the pool
    best -- previously the result claimed the pool-best model's quality next
    to the cheaper model that actually answered."""
    scores = pd.DataFrame([[0.90, 0.80]], index=["q"], columns=["big", "cheap"])
    router = MatrixRouter(scores, model_costs=[1.0, 0.0])
    agent = AgenticRouter(router, ClientRegistry.echo(["big", "cheap"]),
                          query_resolver=lambda p: "q", lam=0.5)
    res = agent.run("hello")
    assert res.selected_model_id == "cheap"
    assert res.predicted_quality == pytest.approx(0.80)
    assert res.predicted_quality == res.steps[0].predicted_quality
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/agentic/router.py`

```diff
--- a/src/router/agentic/router.py
+++ b/src/router/agentic/router.py
@@ -135,8 +181,10 @@ class AgenticRouter:
                 prompt=prompt, mode="single", answer=root.answer,
                 triage_reason=reason,
                 selected_model_id=root.model_id,
-                predicted_quality=decision.predicted_quality,
-                steps=[root], cost=root.cost or 0.0,
+                # the model that actually answered's own score, not the pool
+                # best decision.predicted_quality carries (AGENTICROUTER-001)
+                predicted_quality=root.predicted_quality,
+                steps=[root], cost=root.cost,
             )
         return AgenticResult(
             prompt=prompt, mode="orchestrated", answer=root.answer,
```

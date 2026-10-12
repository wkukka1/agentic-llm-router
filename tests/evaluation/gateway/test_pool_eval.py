"""Replaying routing policies on graded prompts: the margin rule, the
leave-one-out kNN, and the random-mix line a router must beat."""

from __future__ import annotations

import json

import numpy as np

from evaluation.gateway import pool_eval as pe


def test_margin_pick_takes_the_cheapest_tier_close_to_the_best():
    pred = np.array([[0.70, 0.80, 0.82], [0.50, 0.60, 0.90], [0.90, 0.90, 0.90]])
    assert list(pe.margin_pick(pred, 0.05)) == [1, 2, 0]


def test_leave_one_out_never_uses_a_prompts_own_score():
    vec = np.eye(3)
    scores = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pred = pe.loo_knn(vec, scores, k=1)
    assert not np.allclose(np.diag(pred), 1.0)


def test_a_router_on_the_random_mix_line_scores_zero_and_the_oracle_above_it(tmp_path):
    rows = []
    for i, (h, s, o) in enumerate([(0.9, 0.9, 0.9), (0.2, 0.9, 0.9), (0.2, 0.3, 0.9), (0.9, 0.9, 0.9)]):
        for tier, score, cost in zip(pe.TIERS, (h, s, o), (1.0, 5.0, 10.0)):
            rows.append({"prompt_hash": f"p{i}", "prompt": f"q{i}", "model": tier, "score": score,
                         "usage_usd": cost, "matches_best": score >= max(h, s, o)})
    path = tmp_path / "graded.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    prompts = pe.load_prompts(path)
    assert len(prompts) == 4
    picks = pe.policies(prompts, margin=0.0, k=2)
    results = {r["policy"]: r for r in pe.evaluate(prompts, picks=picks)}
    assert results["always opus"]["vs_mix"] == 0.0
    assert results["oracle"]["matches_best"] == 1.0 and results["oracle"]["vs_mix"] > 0
    assert list(picks["oracle"]) == [0, 1, 2, 0]  # the cheapest tier that matched the best

"""``training.nirt.judge_finetune``: fine-tuning a projected NIRT on judged tier scores."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from router.nirt.model import build_model
from training.nirt.judge_finetune import fit_judged, predict_judged


def _world(seed=0, n=60, q_dim=12, p_dim=7):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model({"dim": 2, "model_params": "projected", "query_hidden": 16}, n_models=3,
                        query_dim=q_dim, profile_dim=p_dim).eval()
    e_q = rng.standard_normal((n, q_dim)).astype(np.float32)
    refs = rng.standard_normal((3, p_dim)).astype(np.float32)
    # judged world: the dearest tier is clearly best, the cheapest clearly worst
    scores = np.clip(np.stack([np.full(n, 0.45), np.full(n, 0.65), np.full(n, 0.8)], axis=1)
                     + 0.05 * rng.standard_normal((n, 3)), 0.0, 1.0)
    return model, e_q, refs, scores


@pytest.mark.parametrize("variant", ["tiers", "tiers+head", "full"])
def test_fine_tuning_learns_the_tier_gap(variant):
    model, e_q, refs, scores = _world()
    fit = fit_judged(model, e_q, scores, refs, variant=variant, steps=300)
    gap = (predict_judged(fit, e_q, refs)[:, 2] - predict_judged(fit, e_q, refs)[:, 0]).mean()
    assert gap == pytest.approx(0.35, abs=0.05)


def test_none_and_the_input_model_are_untouched():
    model, e_q, refs, scores = _world()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit_judged(model, e_q, scores, refs, variant="full", steps=50)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    m, da, db = fit_judged(model, e_q, scores, refs, variant="none")
    assert not da.any() and not db.any()
    with pytest.raises(ValueError, match="variant"):
        fit_judged(model, e_q, scores, refs, variant="bogus")

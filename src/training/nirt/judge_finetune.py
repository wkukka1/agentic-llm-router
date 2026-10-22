"""Fine-tune a projected NIRT on judged gateway prompts, scored per Claude tier.

A projected NIRT scores an unseen model (a Claude 5.5 tier) from its profile
alone; with no labelled answer from that tier it gets the tiers' gap wrong (the
profile-manifold run predicted Opus - Haiku ~ +0.01 where an LLM judge measured
+0.18). Here the model is trained on graded prompts -- the judge's score for each
tier's answer -- with soft BCE, plus per-tier offsets ``(da, db)`` on that tier's
``(a_m, b_m)``. ``scripts/nirt/judge_finetune_cv.py`` measured the variants
out-of-fold; ``scripts/routing/export_judged_router.py`` folds the result into a
gateway router file (:func:`training.trainers.nirt_router_export.fold_nirt` with
``offsets``).

Variants (what trains): ``none``; ``tiers`` (offsets only); ``tiers+head`` (offsets
+ the query head's last layer); ``full`` (every weight at ``lr``, plus offsets).
"""

from __future__ import annotations

import copy

import numpy as np

VARIANTS = ("none", "tiers", "tiers+head", "full")

__all__ = ["VARIANTS", "fit_judged", "predict_judged"]


def _tier_logits(model, q, refs, da, db):
    """``[n, n_tiers]`` logits of queries ``q`` against the tier profiles ``refs``."""
    theta = model.latent_query(q)
    a, b = model.model_parameters(refs)
    a, b = a + da, b + db
    n, m = len(q), len(refs)
    return model.score(theta.repeat_interleave(m, 0), a.repeat(n, 1), b.repeat(n)).view(n, m)


def fit_judged(model, e_q: np.ndarray, scores: np.ndarray, refs: np.ndarray, *, variant: str = "full",
               steps: int = 300, lr: float = 1e-3, offset_lr: float = 0.05, offset_l2: float = 1e-3,
               seed: int = 0) -> tuple:
    """Full-batch Adam on ``scores`` ``[n, n_tiers]`` (judge scores in [0, 1]) for queries
    ``e_q`` ``[n, d]`` against tier profiles ``refs`` ``[n_tiers, p]``. Returns
    ``(model, da, db)``: a fine-tuned COPY of ``model`` (the input is untouched) and the
    per-tier offsets on ``a_m`` ``[n_tiers, K]`` and ``b_m`` ``[n_tiers]``."""
    import torch

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    torch.manual_seed(seed)
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    q = torch.from_numpy(np.ascontiguousarray(e_q, dtype=np.float32))
    R = torch.from_numpy(np.ascontiguousarray(refs, dtype=np.float32))
    Y = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    da = torch.zeros(len(refs), m.dim, requires_grad=True)
    db = torch.zeros(len(refs), requires_grad=True)
    if variant != "none":
        groups = [{"params": [da, db], "lr": offset_lr}]
        if variant == "tiers+head":
            head = list(m.query_head.parameters())[-2:]          # the last Linear's weight + bias
            for p in head:
                p.requires_grad_(True)
            groups.append({"params": head, "lr": lr})
        elif variant == "full":
            for p in m.parameters():
                p.requires_grad_(True)
            groups.append({"params": list(m.parameters()), "lr": lr})
        opt = torch.optim.Adam(groups)
        for _ in range(int(steps)):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(_tier_logits(m, q, R, da, db), Y)
            loss = loss + offset_l2 * (da.pow(2).sum() + db.pow(2).sum())   # keep the offsets small
            loss.backward()
            opt.step()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, da.detach().numpy().copy(), db.detach().numpy().copy()


def predict_judged(fit: tuple, e_q: np.ndarray, refs: np.ndarray, *, logits: bool = False) -> np.ndarray:
    """``[n, n_tiers]`` P(good answer) (or logits) from a :func:`fit_judged` result."""
    import torch

    model, da, db = fit
    with torch.no_grad():
        out = _tier_logits(model, torch.from_numpy(np.ascontiguousarray(e_q, dtype=np.float32)),
                           torch.from_numpy(np.ascontiguousarray(refs, dtype=np.float32)),
                           torch.from_numpy(da), torch.from_numpy(db))
        out = out if logits else torch.sigmoid(out)
    return out.numpy().astype(np.float64)

"""Train the NIRT baseline predictor (Phase 2, v1).

    from training.nirt.train import fit
    res = fit(yaml.safe_load(open("configs/nirt.yaml")))

``fit`` reads the training split through the Phase 1 facade
(:func:`training.data.facade.load_training_data` -> :meth:`TrainingData.nirt_dataset`),
gathers it into RAM once (``train.materialize``), runs Adam with early stopping
on the validation split, and writes ``model.pt`` + ``run.json`` under
``runs_dir/<name>/``. Loading a saved run back for inference is
:func:`router.nirt.checkpoint.load_run`, not this module -- serving code must
not import a trainer.

Only the absolute-correctness signal drives the main forward/loss -- this is
N-IRT, kept isolated from any pairwise / M-IRT path -- with one opt-in exception
(item 2 of the capacity workstream, ``docs/nirt_model.md``): `train.preference`
adds an auxiliary Bradley-Terry loss over Chatbot Arena / GPT-4-Judge battles
(``training.nirt.pairwise``) on the SAME shared query head, never folded into
the correctness matrix. Off by default.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from router.config import DEFAULT_NIRT_RUNS_DIR, require_choice
from router.determinism import seed_everything
from router.nirt.model import build_model
from router.provenance import git_sha as _git_sha

from .metrics import bias_argmax_agreement, effective_rank, marginal_baselines, prediction_metrics

__all__ = ["fit", "RunResult", "seed_everything"]


@dataclass
class RunResult:
    name: str
    config: dict
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    val_metrics: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    path: Optional[Path] = None
    model: object = None            # the fitted NIRTModel (in-memory; not serialised here)
    model_index: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# data packing                                                                #
# --------------------------------------------------------------------------- #
def _pack(dataset, model_index: dict[str, int]) -> dict:
    """Materialize a :class:`NIRTDataset` into contiguous arrays (one copy)."""
    g = dataset.gather()
    model_ids = np.asarray(dataset.model_ids)
    midx = np.array([model_index.get(str(m), -1) for m in model_ids], dtype=np.int64)
    qids = np.asarray(dataset.query_ids).astype(str)
    # dense per-query group id (0..G-1), for the P5 query-grouped sampler / losses
    _, qgroup = np.unique(qids, return_inverse=True)
    return {
        "q": np.ascontiguousarray(g["query_embedding"], dtype=np.float32),
        "m": np.ascontiguousarray(g["model_embedding"], dtype=np.float32),
        "y": np.ascontiguousarray(g["target"], dtype=np.float32),
        "midx": midx,
        "model_ids": model_ids,
        "qgroup": np.ascontiguousarray(qgroup, dtype=np.int64),
    }


# --------------------------------------------------------------------------- #
# losses  (P5: routing-aware pairwise / listwise, alongside the per-cell BCE)  #
# --------------------------------------------------------------------------- #
_POINTWISE = ("soft_bce", "hard_bce", "mse")
_GROUPED = ("pairwise", "listwise", "bce_pairwise")
_PAIRWISE_WEIGHTINGS = ("none", "abs_diff")


def _group_blocks(groups):
    """``groups`` (row-order = grouped, from :func:`_grouped_batches`) -> list of
    ``(start, size)`` per contiguous group, plus a fast-path ``(G, k)`` when every
    group is the same size ``k``."""
    import torch

    boundaries = torch.cat([
        torch.tensor([0], device=groups.device),
        (groups[1:] != groups[:-1]).nonzero(as_tuple=True)[0] + 1,
        torch.tensor([len(groups)], device=groups.device),
    ])
    sizes = (boundaries[1:] - boundaries[:-1]).tolist()
    starts = boundaries[:-1].tolist()
    equal_k = sizes[0] if sizes and all(s == sizes[0] for s in sizes) else None
    return list(zip(starts, sizes)), equal_k


def _pairwise_within_group(logit, y, groups, *, weighting: str = "none"):
    """Mean RankNet / Bradley-Terry loss over model pairs *within* each query
    group: ``BCE(z_m - z_m', 1[y_m > y_m'])``. Ties (``y_m == y_m'``) are skipped.

    ``weighting="abs_diff"`` (P7c) weights each discordant pair's loss by
    ``|y_m - y_n|`` instead of counting it equally with every other pair --
    misroute analysis shows most routing regret concentrates in the most
    extreme (near-1.0-vs-0.0) pairs, so this directs gradient budget where the
    routing cost actually is. ``weighting="none"`` (default) reproduces the
    original unweighted mean exactly (every pair gets weight 1)."""
    import torch
    from torch.nn import functional as F

    blocks, k = _group_blocks(groups)
    if k is not None and k >= 2:                      # fast path: dense (G, k)
        G = len(blocks)
        Z = logit.view(G, k)
        Y = y.view(G, k)
        dZ = Z.unsqueeze(2) - Z.unsqueeze(1)          # (G, k, k)
        dY = Y.unsqueeze(2) - Y.unsqueeze(1)
        m = torch.triu(torch.ones(k, k, dtype=torch.bool, device=logit.device), 1) & (dY != 0)
        if not m.any():
            return logit.sum() * 0.0      # graph-connected zero: backward() stays valid
        per_pair = F.binary_cross_entropy_with_logits(
            dZ[m], (dY[m] > 0).to(logit.dtype), reduction="none")
        w = dY[m].abs() if weighting == "abs_diff" else torch.ones_like(per_pair)
        return (per_pair * w).sum() / w.sum().clamp_min(1e-12)
    total = torch.zeros((), dtype=logit.dtype, device=logit.device)
    wsum = torch.zeros((), dtype=logit.dtype, device=logit.device)
    for s, sz in blocks:
        if sz < 2:
            continue
        z, t = logit[s:s + sz], y[s:s + sz]
        dz = z.unsqueeze(1) - z.unsqueeze(0)
        dt = t.unsqueeze(1) - t.unsqueeze(0)
        mm = torch.triu(torch.ones_like(dz, dtype=torch.bool), 1) & (dt != 0)
        if mm.any():
            per_pair = F.binary_cross_entropy_with_logits(
                dz[mm], (dt[mm] > 0).to(logit.dtype), reduction="none")
            w = dt[mm].abs() if weighting == "abs_diff" else torch.ones_like(per_pair)
            total = total + (per_pair * w).sum()
            wsum = wsum + w.sum()
    if not bool(wsum > 0):
        return logit.sum() * 0.0          # no discordant pair in the batch
    return total / wsum


def _listwise_within_group(logit, y, groups, *, tau: float = 0.1):
    """ListNet top-1: cross-entropy between ``softmax(z_q)`` and the soft label
    distribution ``softmax(y_q / tau)`` over the models scored on each query."""
    import torch
    from torch.nn import functional as F

    blocks, k = _group_blocks(groups)
    if k is not None and k >= 2:
        G = len(blocks)
        logp = F.log_softmax(logit.view(G, k), dim=1)
        q = F.softmax(y.view(G, k) / tau, dim=1)
        return -(q * logp).sum(1).mean()
    total, ng = torch.zeros((), dtype=logit.dtype, device=logit.device), 0
    for s, sz in blocks:
        if sz < 2:
            continue
        total = total - (F.softmax(y[s:s + sz] / tau, dim=0)
                         * F.log_softmax(logit[s:s + sz], dim=0)).sum()
        ng += 1
    if ng == 0:
        return logit.sum() * 0.0          # every group has < 2 models
    return total / ng


def _weighted_bce(logit, y, weight=None):
    """``BCE(logit, y)``, optionally per-row weighted (P7c
    ``train.zero_variance_weight``). ``weight=None`` reproduces the plain
    unweighted mean exactly."""
    from torch.nn import functional as F

    if weight is None:
        return F.binary_cross_entropy_with_logits(logit, y)
    per = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (per * weight).sum() / weight.sum().clamp_min(1e-12)


def _loss_fn(kind: str, *, aux_weight: float = 0.5, tau: float = 0.1,
             pairwise_weighting: str = "none"):
    """Return ``loss(logit, y, groups=None, weight=None)``. Pointwise losses use
    ``weight`` (per-row, e.g. ``train.zero_variance_weight``) and ignore
    ``groups``; the grouped losses (P5) require ``groups`` (query-group ids per
    row) and ignore ``weight`` (a zero-variance query already contributes ~0 to
    a pairwise/listwise loss, since it has no discordant pairs)."""
    import torch
    from torch.nn import functional as F

    if kind == "soft_bce":
        return lambda logit, y, groups=None, weight=None: _weighted_bce(logit, y, weight)
    if kind == "hard_bce":
        return lambda logit, y, groups=None, weight=None: _weighted_bce(
            logit, (y >= 0.5).float(), weight)
    if kind == "mse":
        return lambda logit, y, groups=None, weight=None: (
            F.mse_loss(torch.sigmoid(logit), y) if weight is None else
            (F.mse_loss(torch.sigmoid(logit), y, reduction="none") * weight).sum()
            / weight.sum().clamp_min(1e-12))
    if kind == "pairwise":
        return lambda logit, y, groups, weight=None: _pairwise_within_group(
            logit, y, groups, weighting=pairwise_weighting)
    if kind == "listwise":
        return lambda logit, y, groups, weight=None: _listwise_within_group(logit, y, groups, tau=tau)
    if kind == "bce_pairwise":
        return lambda logit, y, groups, weight=None: (
            _weighted_bce(logit, y, weight)
            + aux_weight * _pairwise_within_group(logit, y, groups, weighting=pairwise_weighting))
    raise ValueError(f"unknown loss {kind!r}; use {_POINTWISE + _GROUPED}")


def _group_rows(qgroup: np.ndarray) -> list:
    """Row indices per dense query group ``0..G-1`` (ascending within a group),
    computed once with one stable argsort instead of a scan per group."""
    qgroup = np.asarray(qgroup)
    if len(qgroup) == 0:
        return []
    order = np.argsort(qgroup, kind="stable")
    return np.split(order, np.cumsum(np.bincount(qgroup))[:-1])


def _grouped_batches(group_rows: list, batch_queries: int, gen):
    """Yield row-index arrays, ``batch_queries`` whole query groups per batch
    (shuffled). Every model row for a sampled query is in the same batch, so the
    pairwise / listwise losses see complete per-query model lists.
    ``group_rows`` comes from :func:`_group_rows` (built once per fit); a raw
    dense ``qgroup`` array is also accepted."""
    import torch

    if isinstance(group_rows, np.ndarray):
        group_rows = _group_rows(group_rows)
    order = torch.randperm(len(group_rows), generator=gen).numpy()
    for i in range(0, len(order), batch_queries):
        yield np.concatenate([group_rows[g] for g in order[i : i + batch_queries]])


def _val_regret(packed: dict, prob: np.ndarray) -> float:
    """Mean per-query routing regret: ``mean_q( max_m y[q,m] - y[q, argmax_m prob[q,m]] )``.
    Only queries with >= 2 models scored contribute. For ``val_metric: regret`` (P5)."""
    y = np.asarray(packed["y"], np.float64)
    g = np.asarray(packed["qgroup"], np.int64)
    p = np.asarray(prob, np.float64)
    if len(y) == 0:
        return 0.0
    counts = np.bincount(g)
    ymax = np.full(len(counts), -np.inf)
    np.maximum.at(ymax, g, y)
    # first row (in row order) holding each group's max prob == np.argmax's pick
    order = np.lexsort((np.arange(len(p)), -p, g))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    valid = counts >= 2
    picked = y[order[starts[valid]]]
    return float((ymax[valid] - picked).sum() / max(int(valid.sum()), 1))


def _query_variance_mask(y: np.ndarray, qgroup: np.ndarray) -> np.ndarray:
    """Per-row boolean: True if this row's query has ZERO label variance --
    every model scored on it shares one label, so there is no ranking
    information for it (a pairwise/listwise loss already contributes ~0), yet
    a per-cell BCE loss still spends a full batch slot fitting the shared base
    rate. Precomputed once over the whole packed split (not per-batch), so it
    is correct under both ``sampler: cell`` and ``sampler: query`` -- see
    ``train.zero_variance_weight`` (P7c)."""
    y = np.asarray(y, np.float64)
    qgroup = np.asarray(qgroup, np.int64)
    if len(y) == 0:
        return np.zeros(0, dtype=bool)
    G = int(qgroup.max()) + 1
    lo, hi = np.full(G, np.inf), np.full(G, -np.inf)
    np.minimum.at(lo, qgroup, y)
    np.maximum.at(hi, qgroup, y)
    return (hi - lo <= 1e-12)[qgroup]


def _theta_cov_penalty(theta) -> "torch.Tensor":
    """VICReg-style off-diagonal covariance penalty on ``theta_q``: mean squared
    off-diagonal entry of ``Cov(theta)`` over the batch, pushing the query
    head's output dimensions toward decorrelation -- direct counter-pressure
    against the query head collapsing onto a low-rank subspace (see
    ``model.query_head_batchnorm``, which forces this structurally;
    ``train.theta_cov_weight`` is a softer, additive alternative / complement).
    No-op (returns 0) when ``K < 2`` (nothing off-diagonal to penalise) or the
    batch has fewer than 2 rows."""
    import torch

    B, K = theta.shape
    if K < 2 or B < 2:
        return torch.zeros((), dtype=theta.dtype, device=theta.device)
    tc = theta - theta.mean(0, keepdim=True)
    cov = (tc.T @ tc) / (B - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return (off ** 2).sum() / (K * (K - 1))


def _collapse_diagnostics(model, theta_q: np.ndarray) -> dict:
    """Cheap, per-epoch health check for the P7 monotone-collapse failure mode:
    ``theta_q``'s effective rank, and (query_latent / model_params=free /
    difficulty=scalar only -- the full model pool's a_m/b_m are cheaply
    available in that configuration alone) how often the full router's argmax
    agrees with the query-independent constant ``argmax_m(-b_m)``. Logged into
    ``run.json`` history every epoch so the failure mode is visible during
    training, not only via a post-hoc diagnostic run.

    ``theta_q`` should hold one row per *query* (not per observation), so both
    numbers are per-query rather than weighted by model coverage."""
    out = {"theta_effective_rank": effective_rank(theta_q)}
    # the agreement is computed on the bilinear logit only -- exact only when the
    # model has no interaction residual, so it is skipped otherwise
    if (getattr(model, "orientation", None) == "query_latent"
            and getattr(model, "model_params", None) == "free"
            and getattr(model, "difficulty", None) == "scalar"
            and not getattr(model, "interaction", False)):
        import torch

        with torch.no_grad():
            a, b = model.model_parameters(torch.arange(model.n_models, device=_device_of(model)))
        a, b = a.cpu().numpy(), b.cpu().numpy()
        full_logit = theta_q @ a.T - b[None, :]
        out["bias_argmax_agreement"] = bias_argmax_agreement(full_logit, b)
    return out


def _device_of(model):
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameter-free model
        return "cpu"


def _predict_packed(model, packed: dict, batch_size: int = 8192) -> np.ndarray:
    import torch

    projected = model.model_params == "projected"
    dev = _device_of(model)
    q = torch.from_numpy(packed["q"])
    ref_src = torch.from_numpy(packed["m"]) if projected else torch.from_numpy(packed["midx"])
    out = np.empty(len(packed["y"]), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(out), batch_size):
            sl = slice(i, i + batch_size)
            out[sl] = torch.sigmoid(model(q[sl].to(dev), ref_src[sl].to(dev))).cpu().numpy()
    return out


# val_metric keys: lower is better / higher is better (the latter are negated
# for early stopping and ReduceLROnPlateau, which both minimise)
_VAL_LOWER = ("bce", "log_loss", "mse", "brier", "mae", "ece", "regret")
_VAL_HIGHER = ("auc", "spearman_r", "pearson_r", "accuracy", "acc@0.5")


def _drop_rows(packed: dict, keep: np.ndarray) -> dict:
    """``packed`` restricted to ``keep`` rows (query groups re-densified)."""
    out = {k: v[keep] for k, v in packed.items()}
    _, out["qgroup"] = np.unique(out["qgroup"], return_inverse=True)
    out["qgroup"] = np.ascontiguousarray(out["qgroup"], dtype=np.int64)
    return out


def _json_safe(obj):
    """NaN / inf -> ``None`` so ``run.json`` is strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# fit                                                                         #
# --------------------------------------------------------------------------- #
def fit(
    nirt_cfg: dict,
    *,
    phase0_cfg=None,
    datasets: Optional[tuple] = None,
    pairwise_arrays=None,
    name: Optional[str] = None,
    runs_dir: Optional[os.PathLike] = None,
    save: bool = True,
    verbose: bool = True,
) -> RunResult:
    import torch

    nirt_cfg = json.loads(json.dumps(nirt_cfg))  # deep copy, plain types
    seed = int(nirt_cfg.get("seed", 42))
    seed_everything(seed)

    mcfg = nirt_cfg.get("model", {}) or {}
    tcfg = nirt_cfg.get("train", {}) or {}
    dcfg = nirt_cfg.get("data", {}) or {}
    pathway = dcfg.get("pathway", "irt")
    query_pathway = dcfg.get("query_pathway") or None
    query_features = dcfg.get("query_features") or None

    # -- data ---------------------------------------------------------------
    # NOTE: `train.preference` (this block, Arena/Judge battles) is a distinct
    # concept from `train.loss: pairwise` (P5's RankNet/BT loss on model pairs
    # *within* a correctness query) -- named differently to avoid confusion.
    pcfg = nirt_cfg.get("train", {}).get("preference", {}) or {}
    pairwise_on = bool(pcfg.get("enabled", False))
    d = None
    if datasets is not None:
        train_ds, val_ds = datasets
    else:
        from router.config import load_config

        from ..data.facade import load_training_data

        p0 = phase0_cfg or load_config()
        d = load_training_data(p0)
        train_ds = d.nirt_dataset(split="train", pathway=pathway, query_pathway=query_pathway,
                                  query_features=query_features)
        val_ds = d.nirt_dataset(split="validation", pathway=pathway, query_pathway=query_pathway,
                                query_features=query_features)

    if pairwise_on and pairwise_arrays is None:
        if d is None:
            raise ValueError(
                "train.preference.enabled requires either loading data from scratch "
                "(datasets=None) or passing pairwise_arrays= explicitly"
            )
        from .pairwise import build_pairwise_arrays

        # NOTE: `preference.query_pathway` is deliberately its OWN key, not
        # `data.query_pathway` -- the battles reference a different query set
        # (e.g. an Arena sample) than whatever the correctness task's query
        # store points at, so the two must be free to differ.
        pairwise_arrays = build_pairwise_arrays(
            d, source=pcfg.get("source"), split="train", pathway=pathway,
            query_pathway=pcfg.get("query_pathway") or None,
            # same concat as the correctness dataset, or the shared query head
            # sees a narrower e_q than it was sized for
            query_features=query_features,
        )

    if len(val_ds) == 0:
        raise ValueError(
            "fit: the validation split is empty -- early stopping and val_metrics need it "
            "(set split.validation_fraction > 0 in the phase-0 config and rebuild splits)"
        )

    # `free` mode: a model seen only in validation would get an embedding row
    # that never receives gradient, yet be scored, checkpointed and offered as a
    # routing candidate -- index train models only and drop such val rows.
    # `projected` mode scores any model from its profile, so val-only models stay.
    model_params_cfg = mcfg.get("model_params", "projected")
    train_ids = set(map(str, train_ds.model_ids))
    if model_params_cfg == "free":
        all_ids = sorted(train_ids)
    else:
        all_ids = sorted(train_ids | set(map(str, val_ds.model_ids)))
    model_index = {m: i for i, m in enumerate(all_ids)}

    tr = _pack(train_ds, model_index)
    va = _pack(val_ds, model_index)
    if (tr["midx"] < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("training observation references a model id not in the index")
    if (va["midx"] < 0).any():
        unseen = sorted(set(va["model_ids"][va["midx"] < 0].astype(str)))
        if verbose:
            print(f"[nirt] dropping {int((va['midx'] < 0).sum()):,} validation rows for "
                  f"{len(unseen)} model(s) absent from train: {unseen}")
        va = _drop_rows(va, va["midx"] >= 0)
        if len(va["y"]) == 0:
            raise ValueError("fit: no validation rows left for models seen in training")

    device = str(tcfg.get("device", "cpu") or "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- model ------------------------------------------------------------
    model = build_model(
        mcfg,
        n_models=len(model_index),
        query_dim=int(train_ds.query_dim),
        profile_dim=int(train_ds.model_dim),
    ).to(device)
    projected = model.model_params == "projected"
    if pairwise_on and not projected:
        raise ValueError(
            "train.preference.enabled needs model.model_params='projected' -- a "
            "'free' model has no row for battle participants outside the "
            "correctness training pool"
        )
    if pairwise_on and (pairwise_arrays is None or len(pairwise_arrays) == 0):
        n_drop = getattr(pairwise_arrays, "n_dropped", 0)
        raise ValueError(
            f"train.preference.enabled but zero battles survived the embedding join "
            f"({n_drop:,} dropped: query or profile embedding missing). A bert query store "
            f"built by phase0 holds only correctness queries -- rebuild it with "
            f"`--bert-pairwise`, or point train.preference.query_pathway at a store that "
            f"covers the battle queries."
        )
    if pairwise_on and pairwise_arrays.e_q.shape[1] != model.query_dim:
        raise ValueError(
            f"train.preference battle query width {pairwise_arrays.e_q.shape[1]} != model "
            f"query_dim {model.query_dim} (data.query_features concat mismatch?)"
        )

    lr = float(tcfg.get("lr", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-5))
    batch_size = int(tcfg.get("batch_size", 4096))
    epochs = int(tcfg.get("epochs", 50))
    patience = int(tcfg.get("patience", 6))
    val_key = str(tcfg.get("val_metric", "bce") or "bce")
    require_choice(val_key, _VAL_LOWER + _VAL_HIGHER, field="train.val_metric")
    val_sign = -1.0 if val_key in _VAL_HIGHER else 1.0   # minimise sign * metric
    grad_clip = float(tcfg.get("grad_clip", 0.0) or 0.0)          # 0 -> off (legacy)
    lr_schedule = str(tcfg.get("lr_schedule", "none") or "none")  # {none, cosine, plateau}
    loss_kind = tcfg.get("loss", "soft_bce")
    sampler = str(tcfg.get("sampler", "cell") or "cell")          # {cell, query} (P5)
    if loss_kind in _GROUPED and sampler != "query":
        sampler = "query"   # a within-query loss needs whole query groups per batch
    pairwise_weighting = str(tcfg.get("pairwise_weighting", "none") or "none")   # P7c
    require_choice(pairwise_weighting, _PAIRWISE_WEIGHTINGS, field="train.pairwise_weighting")
    loss_fn = _loss_fn(loss_kind, aux_weight=float(tcfg.get("loss_aux_weight", 0.5)),
                       tau=float(tcfg.get("listwise_tau", 0.1)),
                       pairwise_weighting=pairwise_weighting)
    batch_queries = int(tcfg.get("batch_queries", 512))
    zero_variance_weight = float(tcfg.get("zero_variance_weight", 1.0)
                                 if tcfg.get("zero_variance_weight") is not None else 1.0)  # P7c
    theta_cov_weight = float(tcfg.get("theta_cov_weight", 0.0) or 0.0)          # P7b
    abort_on_collapse = bool(tcfg.get("abort_on_collapse", False))              # P7d
    collapse_rank_threshold = float(tcfg.get("collapse_rank_threshold", 1.3))
    collapse_agreement_threshold = float(tcfg.get("collapse_agreement_threshold", 0.95))
    collapse_patience = int(tcfg.get("collapse_patience", 3))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    elif lr_schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(1, patience // 2)
        )
    elif lr_schedule != "none":
        raise ValueError(f"lr_schedule must be one of none|cosine|plateau, got {lr_schedule!r}")

    pw_weight = float(pcfg.get("weight", 0.3))
    pw_batch = int(pcfg.get("batch_size", 4096))
    pw = None
    if pairwise_on:
        from .pairwise import pairwise_loss

        pw = {
            "e_q": torch.from_numpy(pairwise_arrays.e_q).to(device),
            "e_a": torch.from_numpy(pairwise_arrays.e_a).to(device),
            "e_b": torch.from_numpy(pairwise_arrays.e_b).to(device),
            "y": torch.from_numpy(pairwise_arrays.y).to(device),
        }
        if verbose:
            print(f"[nirt] pairwise auxiliary: {len(pairwise_arrays):,} battles "
                 f"({pairwise_arrays.n_dropped:,} dropped, no embedding), weight={pw_weight}")
    pw_cursor = {"perm": None, "pos": 0}

    def _next_battles():
        """Next ``pw_batch`` battle indices, reshuffling when the pool is exhausted."""
        npw = len(pw["y"])
        if pw_cursor["perm"] is None or pw_cursor["pos"] >= npw:
            pw_cursor["perm"], pw_cursor["pos"] = torch.randperm(npw, generator=gen), 0
        s = pw_cursor["pos"]
        pw_cursor["pos"] = s + pw_batch
        return pw_cursor["perm"][s : s + pw_batch]

    qt = torch.from_numpy(tr["q"]).to(device)
    mt = torch.from_numpy(tr["m"]).to(device)
    yt = torch.from_numpy(tr["y"]).to(device)
    it = torch.from_numpy(tr["midx"]).to(device)
    gt = torch.from_numpy(tr["qgroup"]).to(device)
    n = len(yt)
    gen = torch.Generator().manual_seed(seed)
    tr_zero_var = _query_variance_mask(tr["y"], tr["qgroup"]) if zero_variance_weight != 1.0 else None
    tr_group_rows = _group_rows(tr["qgroup"]) if sampler == "query" else None
    theta_once = theta_cov_weight > 0 and hasattr(model, "latent_query")
    # one query per validation group for the collapse diagnostics (per-query, not
    # weighted by how many models scored it)
    va_first_row = np.unique(va["qgroup"], return_index=True)[1]

    baselines = marginal_baselines(tr["y"], tr["model_ids"], va["y"], va["model_ids"])

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    since_improved = 0
    collapse_streak = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        if sampler == "query":
            batches = [torch.from_numpy(b) for b in
                       _grouped_batches(tr_group_rows, batch_queries, gen)]
        else:
            perm = torch.randperm(n, generator=gen)
            batches = [perm[i : i + batch_size] for i in range(0, n, batch_size)]
        running = seen = 0.0
        pw_running = pw_seen = 0.0
        for idx in batches:
            ref = mt[idx] if projected else it[idx]
            if theta_once:
                # run the query head ONCE: the loss and the covariance penalty share
                # one BatchNorm update and one dropout mask
                theta = model.latent_query(qt[idx])
                logit = model(qt[idx], ref, theta=theta)
            else:
                logit = model(qt[idx], ref)
            weight_idx = None
            if tr_zero_var is not None:
                w = np.where(tr_zero_var[idx.numpy()], zero_variance_weight, 1.0).astype(np.float32)
                weight_idx = torch.from_numpy(w).to(device)
            loss = loss_fn(logit, yt[idx], gt[idx], weight_idx)
            if theta_once:
                loss = loss + theta_cov_weight * _theta_cov_penalty(theta)
            total = loss
            if pw is not None:
                # a genuine loss weight: the battle term joins the SAME step
                b = _next_battles()
                pw_loss = pairwise_loss(model, pw["e_q"][b], pw["e_a"][b], pw["e_b"][b], pw["y"][b])
                total = loss + pw_weight * pw_loss
                pw_running += float(pw_loss.item()) * len(b)
                pw_seen += len(b)
            opt.zero_grad()
            if total.requires_grad:        # defensive: nothing to learn from this batch
                total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            running += float(loss.item()) * len(idx)
            seen += len(idx)
        train_loss = running / max(seen, 1)
        pw_loss_epoch = pw_running / max(pw_seen, 1) if pw is not None else None

        val_prob = _predict_packed(model, va, batch_size=max(batch_size, 8192))
        vm = prediction_metrics(va["y"], val_prob)
        if val_key == "regret":
            vm["regret"] = _val_regret(va, val_prob)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in vm.items()}}
        if pw_loss_epoch is not None:
            row["pairwise_loss"] = pw_loss_epoch

        # -- P7d: per-epoch monotone-collapse alarm --------------------------
        collapse_diag = None
        if hasattr(model, "latent_query"):
            with torch.no_grad():
                theta_val = model.latent_query(
                    torch.from_numpy(va["q"][va_first_row]).to(device)).cpu().numpy()
            collapse_diag = _collapse_diagnostics(model, theta_val)
            row.update(collapse_diag)
        history.append(row)
        if verbose:
            print(
                f"[nirt] epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                f"val_bce {vm['bce']:.4f}  val_spearman {vm['spearman_r']:.3f}"
                + (f"  val_regret {vm['regret']:.4f}" if "regret" in vm else "")
                + (f"  pairwise_loss {pw_loss_epoch:.4f}" if pw_loss_epoch is not None else "")
            )

        # -- model selection first, so an aborting epoch is still considered --
        score = val_sign * float(vm[val_key])
        improved = bool(np.isfinite(score)) and score < best_val - 1e-5   # NaN never improves
        if sched is not None:
            if lr_schedule == "plateau":
                if np.isfinite(score):
                    sched.step(score)
            else:
                sched.step()
        if improved:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        if collapse_diag is not None:
            # effective rank is identically 1 when K == 1, so the rank criterion
            # only means something for K >= 2
            rank = collapse_diag.get("theta_effective_rank", float("nan"))
            rank_collapsed = model.dim >= 2 and rank < collapse_rank_threshold
            is_collapsed = (
                rank_collapsed
                or collapse_diag.get("bias_argmax_agreement", 0.0) > collapse_agreement_threshold
            )
            if is_collapsed:
                collapse_streak += 1
                if verbose:
                    print(f"[nirt] WARNING: possible routing collapse at epoch {epoch} "
                         f"(theta_effective_rank={collapse_diag.get('theta_effective_rank')}, "
                         f"bias_argmax_agreement={collapse_diag.get('bias_argmax_agreement')})")
                if abort_on_collapse and collapse_streak >= collapse_patience:
                    if verbose:
                        print(f"[nirt] aborting: collapse sustained for {collapse_streak} epochs")
                    break
            else:
                collapse_streak = 0

        if since_improved >= patience:
            if verbose:
                print(f"[nirt] early stop at epoch {epoch} (best {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    val_prob = _predict_packed(model, va, batch_size=max(batch_size, 8192))
    val_metrics = prediction_metrics(va["y"], val_prob)
    val_metrics["delta_bce_vs_model_mean"] = val_metrics["bce"] - baselines["model_mean"]["bce"]
    val_metrics["regret"] = _val_regret(va, val_prob)

    tag = "irt" if model.orientation == "model_latent" else "nirt"
    name = name or f"{tag}-{model.dim}d-{model.model_params}"
    result = RunResult(
        name=name,
        config=nirt_cfg,
        history=history,
        best_epoch=best_epoch,
        val_metrics=val_metrics,
        baselines=baselines,
        model=model,
        model_index=model_index,
    )

    if save:
        base = Path(runs_dir) if runs_dir is not None else _resolve_runs_dir(nirt_cfg, phase0_cfg)
        out = base / name
        out.mkdir(parents=True, exist_ok=True)
        # record which phase-0 config (paths, stores, pool) the run was trained
        # against, so NIRTRouter.from_run can reload the same data
        src = getattr(phase0_cfg, "source_path", None)
        if src is not None:
            from router.config import REPO_ROOT

            src = Path(src).resolve()
            try:
                src = src.relative_to(REPO_ROOT)
            except ValueError:
                pass
            nirt_cfg.setdefault("data", {})["phase0_config"] = src.as_posix()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": nirt_cfg,
                "n_models": len(model_index),
                "model_index": model_index,
                "query_dim": int(train_ds.query_dim),
                "profile_dim": int(train_ds.model_dim),
            },
            out / "model.pt",
        )
        (out / "run.json").write_text(
            json.dumps(
                _json_safe({
                    "name": name,
                    "git_sha": _git_sha(),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "config": nirt_cfg,
                    "n_models": len(model_index),
                    "model_index": model_index,
                    "best_epoch": best_epoch,
                    "history": history,
                    "val_metrics": val_metrics,
                    "baselines": baselines,
                }),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        result.path = out
        if verbose:
            print(f"[nirt] wrote {out}")

    return result


def _resolve_runs_dir(nirt_cfg: dict, phase0_cfg) -> Path:
    rel = nirt_cfg.get("runs_dir", DEFAULT_NIRT_RUNS_DIR)
    p = Path(rel)
    if p.is_absolute():
        return p
    if phase0_cfg is not None and hasattr(phase0_cfg, "resolve"):
        return phase0_cfg.resolve(rel)
    from router.config import REPO_ROOT

    return REPO_ROOT / rel

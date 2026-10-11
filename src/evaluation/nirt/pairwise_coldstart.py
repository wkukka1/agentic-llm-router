"""E0-E6: pairwise-battle generalization eval for NIRT's profile-only
cold-start pathway.

The existing zero-shot cold-start check (:func:`evaluation.nirt.evaluate.cold_start_eval`)
scores a held-out model's own pointwise correctness. This module asks a
different, pairwise question: given a model's profile embedding alone (no
pointwise training rows, no pairwise training exposure), can NIRT's
Bradley-Terry scoring (:func:`training.nirt.pairwise.pairwise_logit_diff`)
predict how it fares in actual Chatbot Arena / GPT-4-Judge battles against
models it DID train on?

**Headline claim** this module is built around: can publicly available
capability information, excluding Arena performance itself, initialise NIRT
well enough to predict human Arena preferences for models completely absent
from training? E2 is that claim; E4 is the Arena-Elo-excluded version that
isolates "excluding Arena performance itself"; E3 explains why E2 lands
where it does.

Evaluation-only (same layering rule the sibling modules already follow: may
import ``training``, never imported by it or by ``router`` -- see
``docs/architecture.md``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from training.models.profiles import variant_benchmarks
from training.nirt.metrics import prediction_metrics
from training.nirt.pairwise import pairwise_logit_diff

from .evaluate import nearest_profile_model

DEFAULT_SOURCES = ("chatbot_arena", "gpt4_judge")
_N_BOOTSTRAP = 1000
_BOOTSTRAP_PERCENTILES = (2.5, 97.5)


# --------------------------------------------------------------------------- #
# leakage guards                                                              #
# --------------------------------------------------------------------------- #
def training_pairwise_model_ids(data, split: str = "train") -> set:
    """Every model id that WOULD reach the pairwise auxiliary training loss
    if ``train.preference.enabled`` were on, regardless of whether it
    actually is (:func:`training.nirt.pairwise.build_pairwise_arrays` feeds
    it exactly this: ``data.pairwise(split=split, models="all")``). The
    model-ID-level leakage guard checks target identities against this set
    directly -- not just the config flag -- for defense in depth."""
    df = data.pairwise(split=split, models="all")
    if df is None or df.empty:
        return set()
    return set(df["model_a"]) | set(df["model_b"])


def _check_leakage_guards(run_cfg: dict, data, target_models: Sequence[str], *,
                          require_warm_opponent: bool = True) -> None:
    """Two independent checks, both enforced:

    1. **Config-level**: ``train.preference.enabled`` must be falsy for the
       loaded run -- if it isn't, EVERY model's battles (cold or warm) may
       have reached training (``build_pairwise_arrays`` hardcodes
       ``models="all"``, there is no partial/per-model scoping), so the
       cold-start claim would be void regardless of which models are
       targeted.
    2. **Model-identity**: when ``require_warm_opponent`` is true (the E2+
       cold-start claim), every target must actually BE a registered
       cold-start model (``data.cold_start_model_ids()``) -- catches
       accidentally testing a warm model under a zero-shot framing, which
       the config-flag check alone cannot catch. Note a cold model's own
       battles are *expected* to appear in the general pairwise pool --
       that presence is exactly what E2 scores against, never itself the
       violation; only a mislabeled identity is. E1 (``require_warm_opponent
       =False``) targets warm models on purpose and skips this check.
    """
    pref_enabled = bool(((run_cfg or {}).get("train") or {}).get("preference", {}).get("enabled"))
    if pref_enabled:
        raise ValueError(
            "pairwise_cold_start_eval refuses to run: train.preference.enabled is true "
            "for this run, so cold models' own battles may have reached training via the "
            "auxiliary pairwise loss -- the cold-start claim would be void."
        )
    if not require_warm_opponent:
        return
    cold_ids = set(data.cold_start_model_ids())
    mislabeled = [t for t in target_models if t not in cold_ids]
    if mislabeled:
        raise ValueError(
            f"pairwise_cold_start_eval refuses to run: target model(s) {sorted(mislabeled)} "
            "are not registered cold-start models (data.cold_start_model_ids()) -- scoring "
            "them under require_warm_opponent=True would silently claim a zero-shot result "
            "for a model that was actually trained on."
        )


# --------------------------------------------------------------------------- #
# battle reorientation                                                       #
# --------------------------------------------------------------------------- #
def _reorient_to_target(raw: pd.DataFrame, target: str) -> pd.DataFrame:
    """Every row touching ``target`` -> reoriented so ``target`` is always
    ``model_a`` (flipping ``model_b``-side rows and their ``score_a``).
    Rows touching neither side, or both sides (target-vs-itself), are
    dropped."""
    a_side = raw[(raw["model_a"] == target) & (raw["model_b"] != target)].copy()
    b_side = raw[(raw["model_b"] == target) & (raw["model_a"] != target)].copy()
    if not b_side.empty:
        swapped = b_side[["model_b", "model_a"]].to_numpy()
        b_side[["model_a", "model_b"]] = swapped
        b_side["score_a"] = 1.0 - b_side["score_a"].astype(float)
    return pd.concat([a_side, b_side], ignore_index=True)


# --------------------------------------------------------------------------- #
# bootstrap CI                                                                #
# --------------------------------------------------------------------------- #
def _bootstrap_ci(y: np.ndarray, p: np.ndarray, *, metric: str = "bce",
                   n_boot: int = _N_BOOTSTRAP, seed: int = 0) -> Optional[tuple]:
    n = len(y)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([prediction_metrics(y[row], p[row])[metric] for row in idx])
    lo, hi = np.percentile(vals, _BOOTSTRAP_PERCENTILES)
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# core eval                                                                   #
# --------------------------------------------------------------------------- #
def pairwise_cold_start_eval(
    model,
    run_cfg: dict,
    data,
    *,
    target_models: Optional[Sequence[str]] = None,
    sources: Sequence[str] = DEFAULT_SOURCES,
    pathway: str = "irt",
    profile_override: Optional[dict] = None,
    profile_store_override=None,
    require_warm_opponent: bool = True,
) -> dict:
    """Score ``target_models``' real pairwise battles via NIRT's
    Bradley-Terry scoring, from their profile embedding alone.

    The opponent side is ALWAYS restricted to warm models (there is no
    principled reason to score against a non-warm opponent -- the baselines
    assume a known reference). ``require_warm_opponent`` (default ``True``,
    the E2/E4/E5/E6 setting) instead controls the leakage guard's identity
    check: every target must be a *registered* cold-start model. Set
    ``False`` for E1, where targets are deliberately warm (a
    reference/upper-bound condition, not a cold-start claim) -- the guard
    skips the cold-registration check accordingly.

    ``profile_override`` (target model_id -> a *different* model_id's
    profile to substitute) powers E5's permutation sanity check.

    ``profile_store_override`` (a whole :class:`EmbeddingStore`, used
    instead of ``data.profile_embeddings(pathway)``) powers E3's ablation
    sweep -- each P0-P6 variant lives in its own store
    (:func:`training.models.profiles.build_variant_profile_embeddings`),
    not the pathway's default one.

    ``sources`` results are always keyed separately in the returned dict;
    there is no code path that pools ``chatbot_arena`` and ``gpt4_judge``
    together (E6 must never be averaged into E2/E4).
    """
    if getattr(model, "model_params", None) != "projected":
        raise ValueError(
            "pairwise_cold_start_eval needs a 'projected' run -- a 'free' model has no "
            "row for an unseen model."
        )
    target_models = list(target_models) if target_models is not None else data.cold_start_model_ids()
    sources = tuple(sources)
    _check_leakage_guards(run_cfg, data, target_models, require_warm_opponent=require_warm_opponent)

    warm_ids = set(data.warm_model_ids())
    q_store = data.query_embeddings(pathway)
    p_store = profile_store_override if profile_store_override is not None else data.profile_embeddings(pathway)

    raw = pd.concat(
        [data.pairwise(source=src, models="all") for src in sources],
        ignore_index=True,
    ) if sources else pd.DataFrame(columns=["query_id", "model_a", "model_b", "score_a"])

    per_model: dict = {}
    for target in target_models:
        battles = _reorient_to_target(raw, target)
        battles = battles[battles["model_b"].isin(warm_ids)]
        # drop battles whose query was never embedded (the pairwise and
        # correctness query spaces are built separately -- see
        # training.nirt.pairwise.build_pairwise_arrays's own n_dropped), not
        # a crash.
        n_before = len(battles)
        battles = battles[battles["query_id"].isin(q_store._index)]
        n_dropped = n_before - len(battles)
        per_model[target] = _score_one_model(
            model, data, q_store, p_store, target, battles, warm_ids,
            pathway=pathway, profile_id=(profile_override or {}).get(target, target),
        )
        per_model[target]["n_dropped"] = n_dropped

    return {
        "sources": sources,
        "cold_models": [m for m in target_models if m in set(data.cold_start_model_ids())],
        "warm_models": sorted(warm_ids),
        "per_model": per_model,
        "notes": (
            "gpt4_judge and chatbot_arena are never pooled together; gpt4_judge covers "
            "exactly one historical matchup and is reported as a named case study, not a "
            "generalization claim across models."
        ),
    }


def _score_one_model(model, data, q_store, p_store, target: str, battles: pd.DataFrame,
                     warm_ids: set, *, pathway: str, profile_id: str) -> dict:
    import torch

    if battles.empty:
        return {"prediction": {"n": 0}, "baselines": {}}

    e_q = q_store.gather(battles["query_id"].tolist())
    e_target = np.repeat(np.asarray(p_store.get(profile_id), np.float32)[None], len(battles), axis=0)
    e_opp = p_store.gather(battles["model_b"].tolist())
    y = battles["score_a"].to_numpy(np.float64)

    with torch.no_grad():
        diff = pairwise_logit_diff(
            model,
            torch.from_numpy(np.ascontiguousarray(e_q, np.float32)),
            torch.from_numpy(np.ascontiguousarray(e_target, np.float32)),
            torch.from_numpy(np.ascontiguousarray(e_opp, np.float32)),
        )
        p = torch.sigmoid(diff).numpy().astype(np.float64)

    metrics = prediction_metrics(y, p)
    ci = _bootstrap_ci(y, p, metric="bce")
    if ci is not None:
        metrics["bce_ci"] = ci

    coin_flip = prediction_metrics(y, np.full_like(y, 0.5))
    warm_pool_mean = _warm_pool_mean_baseline(data, battles, target)

    nearest = nearest_profile_model(p_store, profile_id, [w for w in sorted(warm_ids) if w != target])
    e_nearest = np.repeat(np.asarray(p_store.get(nearest), np.float32)[None], len(battles), axis=0)
    with torch.no_grad():
        nearest_diff = pairwise_logit_diff(
            model,
            torch.from_numpy(np.ascontiguousarray(e_q, np.float32)),
            torch.from_numpy(np.ascontiguousarray(e_nearest, np.float32)),
            torch.from_numpy(np.ascontiguousarray(e_opp, np.float32)),
        )
        p_nearest = torch.sigmoid(nearest_diff).numpy().astype(np.float64)
    nearest_metrics = prediction_metrics(y, p_nearest)

    return {
        "prediction": metrics,
        "baselines": {
            "coin_flip": coin_flip,
            "warm_pool_mean": warm_pool_mean,
            "nearest_warm_model": {"model_id": nearest, **nearest_metrics},
        },
    }


def assert_arena_blind(structured_benchmarks: dict, variant: str = "P4") -> None:
    """E4's explicit runtime guard: ``variant`` must not actually render
    ``arena_elo`` into the profile text used to build its embedding store --
    checked, not just assumed from ``variant_benchmarks``'s filter logic.
    Raise, don't warn: an E4 run that silently includes Arena Elo isn't
    testing "excluding Arena performance itself" at all."""
    rendered = variant_benchmarks(structured_benchmarks, variant)
    if "arena_elo" in rendered:
        raise ValueError(
            f"E4 (Arena-blind) guard failed: variant {variant!r} still renders 'arena_elo' "
            "into its profile text -- this run would not isolate 'excluding Arena "
            "performance itself', which is the entire point of E4."
        )


def _warm_pool_mean_baseline(data, battles: pd.DataFrame, target: str) -> dict:
    """Mean ``score_a`` over all-warm-vs-this-opponent-pool battles (the
    target's own rows excluded, so this never leaks the target's own
    outcomes into its own baseline)."""
    opponents = battles["model_b"].unique().tolist()
    warm_vs_opponents = pd.concat(
        [data.pairwise(source=src, models="all") for src in ("chatbot_arena", "gpt4_judge")],
        ignore_index=True,
    )
    mask = (
        warm_vs_opponents["model_b"].isin(opponents)
        & (warm_vs_opponents["model_a"] != target)
        & (warm_vs_opponents["model_a"].isin(data.warm_model_ids()))
    )
    pool = warm_vs_opponents[mask]
    if pool.empty:
        return {"n": 0}
    y_true = battles["score_a"].to_numpy(np.float64)
    y_pred = np.full(len(battles), float(pool["score_a"].astype(float).mean()))
    return {"n": int(len(pool)), **prediction_metrics(y_true, y_pred)}

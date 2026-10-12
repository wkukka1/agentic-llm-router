"""How well does EVERY trained NIRT variant's zero-shot (profile-only)
placement generalize to LLMs released after it was trained -- not a
split-forced-cold model (still has correctness/profile rows, just held out
of training), but a model with zero rows anywhere in this project's data?

Scored against real human battles between frontier models
(``training.data.modern_pool_battles``, restricted to battles where both
sides are one of the "brand-new" targets, so the test is self-contained:
no warm reference pool, no opponent profile beyond the ones already built).
Evaluation-only (imports ``training``, never imported by it or by
``router``), same layering as the sibling ``evaluation.nirt`` modules.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["score_run_on_modern_pool", "sweep_projected_runs"]


def _knn_impute_profile_store(p_store, ref_store, warm_ids: Sequence[str], *,
                              k: int, weighting: str):
    """Every id in ``p_store`` pulled toward its ``k`` nearest ``warm_ids``
    neighbours in ``ref_store`` (cosine-similarity-weighted mean) -> a new,
    same-shape :class:`~router.embeddings.EmbeddingStore`. Built on
    :func:`training.retrieval.model_bank.build_model_bank`/``knn_impute_model``
    -- the same primitive the live ad hoc routing path
    (``router.routing.model_profiles.impute_unknown_profile``) uses, rather
    than a second, brute-force reimplementation of
    ``QueryBank.neighbor_weighted_mean``'s weighting rule."""
    from router.embeddings import EmbeddingStore
    from training.retrieval.model_bank import build_model_bank, knn_impute_model

    bank = build_model_bank(ref_store, warm_ids)
    imputed = np.stack([
        knn_impute_model(bank, p_store.get(mid), ref_store, k=k, weighting=weighting)[0]
        for mid in p_store.ids
    ])
    manifest = dict(p_store.manifest, knn_k=int(k), knn_weighting=weighting,
                    knn_reference_count=len(warm_ids))
    return EmbeddingStore(p_store.ids, imputed, manifest, id_field=p_store.id_field)


def score_run_on_modern_pool(run_name: str, battles: pd.DataFrame, q_store, p_store,
                             *, runs_dir=None, cfg=None,
                             knn_k: Optional[int] = None, knn_weighting: str = "similarity") -> dict:
    """One ``projected`` NIRT run's zero-shot Bradley-Terry prediction
    against ``battles`` (``query_id, model_a, model_b, score_a`` -- see
    :func:`training.data.modern_pool_battles.filter_modern_pool_battles``),
    plus a coin-flip baseline.

    ``q_store``/``p_store`` are the battle queries' and the modern-pool
    models' RAW embeddings under this run's own pathway (built once by the
    caller and shared across every run of that pathway).

    ``knn_k`` (``None`` -> raw ``p_store``, unchanged): before scoring,
    replace each modern-pool model's profile vector with the
    similarity-weighted mean of its ``knn_k`` nearest models in THIS run's
    own trained ``model_index`` (its warm pool, read straight from the
    checkpoint) -- "closer to something the model has seen," per-run,
    since different runs train on different model pools even within the
    same pathway."""
    import torch

    from router.nirt.checkpoint import load_run
    from training.nirt.metrics import prediction_metrics
    from training.nirt.pairwise import pairwise_logit_diff

    model, run_cfg, model_index = load_run(run_name, runs_dir=runs_dir)
    if model.model_params != "projected":
        raise ValueError(
            f"{run_name}: needs model_params='projected' -- a 'free' model has no "
            "row for a model absent from its own training"
        )
    if battles.empty:
        return {"run": run_name, "n": 0}

    pathway = run_cfg.get("data", {}).get("pathway", "irt")
    if knn_k:
        from router.config import load_config
        from router.embeddings import EmbeddingStore, default_store_dir

        cfg = cfg or load_config()
        ref_dir = default_store_dir(cfg, "model_profile", pathway)
        warm_ids: list[str] = []
        if EmbeddingStore.exists(ref_dir):
            ref_store = EmbeddingStore.load(ref_dir)
            warm_ids = sorted(set(model_index) & set(ref_store._index))
        if warm_ids:
            p_store = _knn_impute_profile_store(
                p_store, ref_store, warm_ids, k=knn_k, weighting=knn_weighting,
            )
        # else: this run's own warm pool isn't in the persistent profile store
        # under this pathway -- keep the raw p_store rather than silently
        # scoring against zero neighbours.

    e_q = q_store.gather(battles["query_id"].tolist())
    e_a = p_store.gather(battles["model_a"].tolist())
    e_b = p_store.gather(battles["model_b"].tolist())
    y = battles["score_a"].to_numpy(np.float64)

    with torch.no_grad():
        diff = pairwise_logit_diff(
            model,
            torch.from_numpy(np.ascontiguousarray(e_q, np.float32)),
            torch.from_numpy(np.ascontiguousarray(e_a, np.float32)),
            torch.from_numpy(np.ascontiguousarray(e_b, np.float32)),
        )
        p = torch.sigmoid(diff).numpy().astype(np.float64)

    metrics = prediction_metrics(y, p)
    coin_flip_bce = prediction_metrics(y, np.full_like(y, 0.5))["bce"]
    dim = run_cfg.get("model", {}).get("dim")
    return {
        "run": run_name, "pathway": pathway, "dim": dim, "knn_k": knn_k,
        "coin_flip_bce": coin_flip_bce, **metrics,
    }


def sweep_projected_runs(run_names: list[str], battles: pd.DataFrame, profile_texts: dict[str, str],
                         *, cfg=None, runs_dir=None,
                         knn_k: Optional[int] = None, knn_weighting: str = "similarity") -> pd.DataFrame:
    """:func:`score_run_on_modern_pool` for every ``run_names`` entry,
    grouped by its own embedding pathway so each pathway's query/profile
    encoder only runs once (shared across every run that uses it).

    Runs whose own ``run.json`` turns out to be ``model_params != "projected"``
    (checked via ``load_run`` itself, not duplicated here) or whose pathway
    has no encoder configured are skipped with an ``error`` column instead
    of aborting the whole sweep -- one bad run must not lose every other
    run's result."""
    from router.config import load_config
    from router.embeddings import EmbeddingStore, load_encoder
    from router.nirt.checkpoint import load_run

    cfg = cfg or load_config()
    by_pathway: dict[str, list[str]] = {}
    for name in run_names:
        try:
            model, run_cfg, _ = load_run(name, runs_dir=runs_dir)
        except Exception as exc:  # pragma: no cover -- surfaced as a row, not a crash
            by_pathway.setdefault("__error__", []).append((name, str(exc)))
            continue
        if model.model_params != "projected":
            continue
        pathway = run_cfg.get("data", {}).get("pathway", "irt")
        by_pathway.setdefault(pathway, []).append(name)

    rows: list[dict] = []
    for name, err in by_pathway.pop("__error__", []):
        rows.append({"run": name, "error": err})

    for pathway, names in by_pathway.items():
        encoder = load_encoder(cfg, pathway=pathway)
        q_store = EmbeddingStore.build(battles["query_id"].tolist(), battles["query_text"].tolist(),
                                       encoder, id_field="query_id", show_progress=False)
        model_ids = sorted(profile_texts)
        p_store = EmbeddingStore.build(model_ids, [profile_texts[m] for m in model_ids], encoder,
                                       id_field="model_id", show_progress=False)
        for name in names:
            try:
                rows.append(score_run_on_modern_pool(
                    name, battles, q_store, p_store, runs_dir=runs_dir, cfg=cfg,
                    knn_k=knn_k, knn_weighting=knn_weighting,
                ))
            except Exception as exc:  # pragma: no cover -- surfaced as a row, not a crash
                rows.append({"run": name, "error": str(exc)})

    return pd.DataFrame(rows)

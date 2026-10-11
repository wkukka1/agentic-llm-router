"""Serving-time bridge to the model-profile schema (``configs/model_profiles.yaml``).

:class:`~router.routing.routers.NIRTRouter` in ``model_params="projected"`` mode
already scores any model from its profile embedding -- ``a_m``/``b_m`` are a
linear/MLP head on ``e_m``, not a free per-model parameter -- so a model with no
training observations can be placed correctly without a new checkpoint. These
helpers are what let :meth:`NIRTRouter.resolve_pool_profile_matrix` do that for a
model *outside* the already-built profile embedding store too:

* :func:`embed_ad_hoc_profile` renders + embeds a caller-supplied (or curated)
  profile entry on the fly, reusing the exact schema/rendering
  ``training.models.profiles`` already uses to build the persistent store.
* :func:`list_profiled_model_ids` / :func:`lookup_curated_entry` read
  ``configs/model_profiles.yaml`` directly -- "every model registered in the
  pool" per that file, independent of training/live-client coverage.
* :func:`impute_unknown_profile` additionally kNN-blends an ad hoc embedding
  toward the TRAINED pool's nearest profiles
  (:mod:`training.retrieval.model_bank`) -- an explicit opt-in, not a default;
  see that module's docstring for the caveat this carries over from the
  query-side version of the same idea.

Every ``training.*`` import below is function-local -- this is the one
serving-side module allowed to cross into ``training`` (see the
``ignore_imports`` entries in ``pyproject.toml``'s import-linter contract),
and keeping the crossing confined to functions means a router that never
touches ad hoc/cold-start profiles pays nothing extra to import this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from router.embeddings.encoder import EmbeddingStore, TextEncoder


def embed_ad_hoc_profile(model_id: str, entry: Optional[dict], encoder: "TextEncoder", *,
                         cfg=None) -> np.ndarray:
    """Render + embed one ad hoc model profile with ``encoder`` (the caller's
    own pathway :class:`~router.embeddings.encoder.TextEncoder` -- the same
    one scoring queries, since profile and query text share a pathway here).

    ``entry`` is a dict in the ``configs/model_profiles.yaml`` schema
    (``feature``, ``benchmark_scores``/``hf_repo_id``/``arena_name``/
    ``benchlm_slug``, ``strengths``, ``weaknesses``, ...) -- the same shape
    :func:`training.models.profiles.build_model_profiles` reads one entry as.
    ``None`` still works (:func:`~training.models.profiles.render_profile_text`
    falls back to a generic templated sentence), but a real entry places the
    model far better than a bare id can. No project response rows exist for
    an ad hoc model, so this never renders the empirical addendum (shared
    with :func:`training.data.modern_pool_battles.modern_pool_profile_texts`
    via :func:`training.models.profiles.render_zero_shot_profile_text`).
    """
    from training.models.profiles import render_zero_shot_profile_text

    full_text = render_zero_shot_profile_text(cfg, model_id, entry)
    vec = encoder.encode([full_text])[0]
    return np.asarray(vec, dtype=np.float32)


def list_profiled_model_ids(cfg) -> list[str]:
    """Every model id curated in ``configs/model_profiles.yaml`` (its
    ``profiles:`` keys) -- "every model registered in the pool", independent
    of training coverage or a live ``ClientRegistry`` adapter."""
    from training.models.profiles import load_profile_entries

    return sorted(load_profile_entries(cfg))


def lookup_curated_entry(cfg, model_id: str) -> Optional[dict]:
    """The ``configs/model_profiles.yaml`` entry for ``model_id``, or ``None``
    if it has no curated entry there."""
    from training.models.profiles import load_profile_entries

    return load_profile_entries(cfg).get(model_id)


def impute_unknown_profile(raw_vec: np.ndarray, store: "EmbeddingStore",
                           trained_model_ids: Sequence[str], *, k: int = 10,
                           weighting: str = "similarity") -> tuple[np.ndarray, int]:
    """Blend ``raw_vec`` toward its ``k`` nearest TRAINED-pool profile
    neighbours (cosine similarity in ``store``'s embedding space) -- the
    model-side analog of the query kNN-impute pathway
    (:mod:`training.retrieval.knn_impute`). Falls back to ``raw_vec`` itself
    when the bank has no usable neighbour (``found == 0``), exactly as that
    module's query-side builder does.

    Returns ``(blended_or_raw_vec, n_neighbours_found)``.

    Caveat (``docs/nirt_model.md``, kNN-imputed queries): the query-side
    version of this barely moved ID accuracy and hurt OOD calibration -- the
    model-side analog risks the same failure mode (blending an unknown
    model's embedding toward trained neighbours may distort its
    discrimination/difficulty more than it helps placement). Treat this as an
    explicit opt-in, not a default, until it's been evaluated on its own
    terms.
    """
    from training.retrieval.model_bank import build_model_bank, knn_impute_model

    bank = build_model_bank(store, trained_model_ids)
    blended, found = knn_impute_model(bank, raw_vec, store, k=k, weighting=weighting)
    if found <= 0:
        return np.asarray(raw_vec, dtype=np.float32), found
    return blended, found

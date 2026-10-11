"""Concrete :class:`~router.routing.base.RouterModel` implementations.

Every class here is a thin adapter over machinery that already exists elsewhere
in the package -- the point is the *shared interface*, not new modelling:

* :class:`MatrixRouter`  -- wrap a precomputed ``[query x model]`` score frame
  (a ZOIB / Bernoulli checkpoint matrix, a distilled table, a hand-made oracle).
* :class:`NIRTRouter`    -- a trained NIRT / IRT-Router run
  (:func:`router.nirt.checkpoint.load_run` + :func:`router.nirt.predict.predict_dataset`).
* :class:`KNNRouter`     -- the RouterBench-style k-NN router
  (:func:`router.nirt.baselines_infer.knn_router_matrix`).
* :class:`MLPRouter`     -- the IRT-free ``e_q -> R^M`` head. Fit it with
  :func:`training.trainers.mlp_router.fit_mlp_router_as_router` -- a serving
  class has no ``.train()`` of its own.
* :class:`RandomRouter`  -- uniform-random pick; a sanity floor for evaluation.

Torch / scikit-learn are imported lazily inside the methods that need them, so
``import router.routing`` stays cheap.
"""

from __future__ import annotations

import hashlib
import itertools
import warnings
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from .base import RouterModel, _resolve_costs
from .registry import register

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..llm.cost import CostModel
    from ..models.artifacts import RouterModelArtifact

__all__ = [
    "MatrixRouter",
    "NIRTRouter",
    "KNNRouter",
    "MLPRouter",
    "RandomRouter",
    "load_training_data_for_router",
]


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #
_OBS_COLUMNS = dict(
    target=0.0, score_raw=0.0, metric_type="accuracy", source="route",
    split=pd.NA, cost=np.nan, is_multiple_choice=False, n_choices=pd.NA,
)


def _cross_obs(query_ids: Sequence[str], model_ids: Sequence[str]) -> pd.DataFrame:
    """A synthetic NIRT observation frame: the full ``query x model`` grid with a
    dummy target, so :class:`~router.data.nirt.NIRTDataset` will score every cell."""
    pairs = list(itertools.product([str(q) for q in query_ids], [str(m) for m in model_ids]))
    df = pd.DataFrame(pairs, columns=["query_id", "model_id"])
    for col, val in _OBS_COLUMNS.items():
        df[col] = val
    return df


_UNSET = object()


def _mean_train_costs(data, model_ids: Sequence[str]) -> Optional[np.ndarray]:
    """Per-model mean cost on the train split (USD/query), in ``model_ids`` order.

    ``None`` if the observation table carries no usable cost signal.
    """
    obs = data.nirt_observations()
    obs = obs[(obs["split"] == "train") & obs["model_id"].isin(set(map(str, model_ids)))]
    cost = pd.to_numeric(obs["cost"], errors="coerce")
    if cost.notna().sum() == 0:
        return None
    by_model = cost.groupby(obs["model_id"].astype(str)).mean()
    vec = np.array([by_model.get(str(m), np.nan) for m in model_ids], dtype=np.float64)
    if not np.isfinite(vec).any():
        return None
    return np.where(np.isfinite(vec), vec, np.nanmax(vec))


def _lazy_model_costs(router) -> Optional[np.ndarray]:
    """Shared ``default_model_costs`` body for a data-backed router. When the
    router carries an injected ``cost_model`` (``RouterModel.cost_model``,
    ROUTERMODEL-002), read one USD/query prior per pool model from its
    ``output_token_priors`` -- the same per-model-id, USD-scale shape this
    property already returns, so this doesn't change what any caller compares
    it against. Falls back to the lazily-cached train-split mean cost
    (``router._cost_cache``, set to ``_UNSET`` in ``__init__``) when there is
    no cost_model, exactly as before."""
    cost_model = getattr(router, "cost_model", None)
    if cost_model is not None:
        priors = cost_model.output_token_priors
        return np.array([float(priors.get(m, 0.0)) for m in router._model_ids], dtype=np.float64)
    if router._cost_cache is _UNSET:
        router._cost_cache = _mean_train_costs(router._data, router._model_ids)
    return router._cost_cache


# NOTE -- tracked, deliberate exception to "router must never import training":
# NIRTRouter / KNNRouter / MLPRouter read pre-built embedding stores and a
# train-split cost table off `TrainingData`, the same way NIRTRouter.from_run
# reads a pre-built model checkpoint. Splitting a serving-safe "artifact
# reader" out of `TrainingData` would be new abstraction the reorg explicitly
# scoped out; the import-linter contract allowlists this one crossing.
def load_training_data_for_router(data=None, config_path=None, *, serving: bool = False):
    """``data`` as given, else ``TrainingData`` for ``config_path`` (default:
    ``configs/phase0.yaml``). ``serving=True`` loads the ``cfg``-only view
    (no training tables) -- enough for scoring, stores and the cost vector."""
    if data is not None:
        return data
    from router.config import load_config
    from training.data.facade import load_training_data

    return load_training_data(load_config(config_path), serving=serving)


def _lazy_training_data(router):
    """Shared ``_data`` body for NIRTRouter / MLPRouter: the injected
    ``TrainingData``, else the one for ``router._phase0_config``, loaded on first
    use. Constructing a router (e.g. ``RouterModelFactory.create``) must not
    read the full ``data/processed`` tables -- a fresh checkout has none -- and
    the lazy load is the ``serving=True`` view, which never reads them either.
    Pass ``data=`` to give a router the full tables."""
    if router._data_cache is None:
        # serving view: routing reads only cfg + stores + nirt_observations, and
        # the full table read was most of the first request's latency
        router._data_cache = load_training_data_for_router(
            None, router._phase0_config, serving=True,
        )
    return router._data_cache


def _encode_texts(data, texts: Sequence[str], pathway: str, *,
                  fallback_pathway: Optional[str] = None, encoder=None,
                  cache: Optional[dict] = None) -> np.ndarray:
    """Embed raw prompt strings with the project's encoder for ``pathway``.

    Falls back to ``fallback_pathway`` only when ``pathway`` has no encoder
    config of its own (a synthetic store name, e.g. a kNN-imputed query
    pathway). Any other error loading the encoder (download failure, CUDA OOM)
    propagates instead of silently switching embedding space.

    ``cache`` (typically a router instance's ``_encoder_cache`` dict, keyed by
    ``pathway``) avoids reloading the encoder from disk on every call when the
    caller doesn't pass its own ``encoder`` -- important on the live/agentic
    serving path, which calls this once per prompt.
    """
    if encoder is None:
        encoder = _resolve_encoder(data, pathway, fallback_pathway=fallback_pathway, cache=cache)
    return np.asarray(encoder.encode(list(texts)), dtype=np.float64)


def _resolve_encoder(data, pathway: str, *, fallback_pathway: Optional[str] = None,
                     cache: Optional[dict] = None):
    """The :class:`~router.embeddings.encoder.TextEncoder` for ``pathway``
    (``cache[pathway]`` when present, else loaded and cached) -- see
    :func:`_encode_texts` for the fallback rule."""
    if cache is not None and pathway in cache:
        return cache[pathway]
    from ..embeddings.encoder import available_pathways, load_encoder

    synthetic = pathway not in available_pathways(data.cfg)
    if synthetic and fallback_pathway and fallback_pathway != pathway:
        enc = load_encoder(data.cfg, fallback_pathway)
    else:
        enc = load_encoder(data.cfg, pathway)
    if cache is not None:
        cache[pathway] = enc
    return enc


# --------------------------------------------------------------------------- #
# MatrixRouter -- wrap a precomputed score frame                              #
# --------------------------------------------------------------------------- #
@register
class MatrixRouter(RouterModel):
    """A router backed by an already-computed ``[query_id x model_id]`` frame of
    predicted quality. Handy for checkpoints whose matrices are produced
    elsewhere (ZOIB ``E[Y]``, the Bernoulli baseline) and for tests."""

    kind = "matrix"

    def __init__(self, scores: pd.DataFrame, *, name: Optional[str] = None,
                 model_costs: Optional[Sequence[float]] = None,
                 cost_matrix: Optional[pd.DataFrame] = None,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None):
        super().__init__(list(scores.columns), name=name, cost_model=cost_model, artifact=artifact)
        # the pool is stringified by RouterModel; the frame's labels must match or
        # every reindex misses (int-labelled frames would route on all-NaN)
        scores = scores.astype(np.float64)
        scores.index = [str(q) for q in scores.index]
        scores.columns = [str(m) for m in scores.columns]
        self._scores = scores
        self._costs = _resolve_costs(model_costs, self._model_ids)
        # optional predicted per-cell expected cost (e.g. a ZOIB cost probe's
        # checkpoint_matrix(field="cost")), same label convention as `scores`
        self._cost_matrix = None
        if cost_matrix is not None:
            cm = cost_matrix.astype(np.float64)
            cm.index = [str(q) for q in cm.index]
            cm.columns = [str(m) for m in cm.columns]
            self._cost_matrix = cm

    @property
    def default_model_costs(self) -> Optional[np.ndarray]:
        return self._costs

    def predict_costs(self, query_ids: Sequence[str]) -> Optional[pd.DataFrame]:
        return self._cost_matrix

    def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        return self._scores.reindex(index=[str(q) for q in query_ids])


# --------------------------------------------------------------------------- #
# NIRTRouter -- a trained NIRT / IRT-Router run                               #
# --------------------------------------------------------------------------- #
@register
class NIRTRouter(RouterModel):
    """Route with a trained NIRT / IRT-Router response model.

    Build it from a saved run directory with :meth:`from_run`, or pass an
    in-memory ``(model, model_index)`` pair directly.
    """

    kind = "nirt"
    can_route_text = True

    def __init__(
        self,
        model,
        model_index: dict,
        *,
        data=None,
        pathway: str = "irt",
        query_pathway: Optional[str] = None,
        query_features: Optional[str] = None,
        name: Optional[str] = None,
        cost_model: Optional["CostModel"] = None,
        artifact: Optional["RouterModelArtifact"] = None,
        phase0_config: Optional[str] = None,
    ):
        super().__init__(sorted(model_index, key=model_index.get), name=name,
                         cost_model=cost_model, artifact=artifact)
        self._model = model
        self._model_index = dict(model_index)
        self._data_cache = data
        self._phase0_config = phase0_config
        self._pathway = pathway
        self._query_pathway = query_pathway or pathway
        self._query_features = query_features
        self._cost_cache = _UNSET
        self._encoder_cache: dict = {}
        self._store_cache: dict = {}
        self._pool_ref_cache: Optional[tuple] = None    # see _pool_refs()
        self._warned_zero_features = False

    @classmethod
    def from_run(cls, run_name: str, *, data=None, runs_dir=None, name: Optional[str] = None):
        """Load ``runs_dir/<run_name>/model.pt`` and read its data pathways from
        the stored config. With ``data=None`` the phase-0 config the run was
        trained against (``data.phase0_config``, recorded by ``fit``) is loaded;
        runs saved before that was recorded fall back to ``configs/phase0.yaml``."""
        from ..nirt.checkpoint import load_run

        model, cfg, model_index = load_run(run_name, runs_dir=runs_dir)
        dcfg = cfg.get("data", {}) or {}
        return cls(
            model,
            model_index,
            data=data,
            phase0_config=dcfg.get("phase0_config"),
            pathway=dcfg.get("pathway", "irt"),
            query_pathway=dcfg.get("query_pathway") or None,
            query_features=dcfg.get("query_features") or None,
            name=name or f"nirt:{run_name}",
        )

    @classmethod
    def from_artifact(cls, artifact, cost_model=None, *, data=None, name: Optional[str] = None):
        """The artifact-driven construction path
        :class:`~router.models.registry.RouterModelFactory` describes:
        rebuild the model in-memory from ``artifact.payload`` rather than
        reading a checkpoint path -- otherwise identical to :meth:`from_run`.
        """
        from ..models.artifacts import NIRTArtifactPayload
        from ..nirt.model import build_model

        payload = artifact.payload
        if not isinstance(payload, NIRTArtifactPayload):
            raise TypeError(
                f"NIRTRouter.from_artifact needs a NIRTArtifactPayload, "
                f"got {type(payload).__name__}"
            )
        model = build_model(
            payload.config.get("model", {}),
            n_models=payload.n_models,
            query_dim=payload.query_dim,
            profile_dim=payload.profile_dim,
        )
        model.load_state_dict(payload.state_dict)
        model.eval()
        dcfg = payload.config.get("data", {}) or {}
        return cls(
            model,
            payload.model_index,
            data=data,
            phase0_config=dcfg.get("phase0_config"),
            pathway=dcfg.get("pathway", "irt"),
            query_pathway=dcfg.get("query_pathway") or None,
            query_features=dcfg.get("query_features") or None,
            name=name or f"nirt:{artifact.artifact_id}",
            cost_model=cost_model,
            artifact=artifact,
        )

    @property
    def _data(self):
        return _lazy_training_data(self)

    @property
    def default_model_costs(self) -> Optional[np.ndarray]:
        return _lazy_model_costs(self)

    def warmup(self) -> dict[str, float]:
        """:meth:`RouterModel.warmup`, with the NIRT-specific first-use loads
        timed separately: the (serving-view) ``TrainingData``, the query
        encoder, and the cached pool reference (profile store)."""
        import time

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        data = self._data
        timings["data"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = _resolve_encoder(data, self._query_pathway, fallback_pathway=self._pathway,
                             cache=self._encoder_cache).model
        timings["encoder"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._pool_refs()
        timings["pool_refs"] = time.perf_counter() - t0
        timings.update(super().warmup())
        return timings

    def _store(self, kind: str, name: Optional[str]):
        """Embedding / feature store, loaded once per router instance (every
        routing call used to reopen them from disk)."""
        key = (kind, name)
        if key not in self._store_cache:
            loader = getattr(self._data, {
                "query": "query_embeddings",
                "profile": "profile_embeddings",
                "features": "query_features",
            }[kind])
            store = loader(name)
            if store is None:          # not built yet -- don't cache the miss
                return None
            self._store_cache[key] = store
        return self._store_cache[key]

    def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        from ..data.nirt import NIRTDataset
        from ..nirt.frames import pivot_qm
        from ..nirt.predict import predict_dataset

        ids = [str(q) for q in query_ids]
        q_store = self._store("query", self._query_pathway)
        m_store = self._store("profile", self._pathway)
        if q_store is None or m_store is None:
            raise FileNotFoundError(
                f"NIRTRouter: embeddings not built for pathway={self._pathway!r} "
                f"query_pathway={self._query_pathway!r}"
            )
        feat = self._store("features", self._query_features) if self._query_features else None

        ds = NIRTDataset(_cross_obs(ids, self._model_ids), q_store, m_store, feature_store=feat)
        _, prob = predict_dataset(self._model, ds, self._model_index)
        long = pd.DataFrame({"query_id": ds.query_ids, "model_id": ds.model_ids, "pred": prob})
        return pivot_qm(long, "pred")

    def predict_scores_text(self, texts: Sequence[str], *, encoder=None) -> pd.DataFrame:
        e_q = _encode_texts(
            self._data, texts, self._query_pathway,
            fallback_pathway=self._pathway, encoder=encoder, cache=self._encoder_cache,
        )
        return self._score_embeddings(e_q)

    def _score_embeddings(self, e_q: np.ndarray) -> pd.DataFrame:
        """Forward the NIRT model over a raw ``[N, d]`` query-embedding batch ->
        ``[N x model_id]`` predicted ``P(correct)``, against this run's own
        trained pool (:attr:`model_ids`). See :meth:`_forward_pool` for the
        pure forward-pass math this wraps."""
        e_q = np.ascontiguousarray(e_q, dtype=np.float32)
        want = int(getattr(self._model, "query_dim", e_q.shape[1]))
        if e_q.shape[1] != want:
            e_q = self._pad_query_features(e_q, want)
        refs, pool_ref = self._pool_refs()
        prob = self._forward_pool(e_q, refs, pool_ref)
        return pd.DataFrame(prob, columns=self._model_ids)

    def _forward_pool(self, e_q: np.ndarray, refs: np.ndarray,
                      pool_ref: Optional[np.ndarray]) -> np.ndarray:
        """Forward an already-padded ``[N, d]`` query-embedding batch against
        every row of ``refs`` -> ``[N, M]`` predicted ``P(correct)``.

        ``refs`` is the ``[M, p]`` profile matrix for a projected run (not
        necessarily this run's own trained pool -- see
        :meth:`resolve_pool_profile_matrix`) or the ``[M]`` model-index vector
        for a free one; ``pool_ref`` is the projected-mode centring reference
        (``None`` for free mode).

        One forward pass over the ``N x M`` grid (model-major: every query
        against model 0, then model 1, ...), not one pass per pool model -- the
        model scores each row independently, and projected-mode centring uses
        a fixed pool reference, so this matches a per-model loop exactly."""
        import torch

        from ..nirt.predict import pool_centering

        model = self._model
        n, m_count = len(e_q), len(refs)
        q = torch.from_numpy(np.tile(e_q, (m_count, 1)))             # [M*N, d]
        if model.model_params == "projected":
            refs = np.ascontiguousarray(refs, dtype=np.float32)
            ref = torch.from_numpy(np.repeat(refs, n, axis=0))      # [M*N, p]
        else:
            ref = torch.from_numpy(np.repeat(refs, n))              # [M*N] model index
        model.eval()
        with torch.no_grad(), pool_centering(model, pool_ref):
            prob = torch.sigmoid(model(q, ref)).cpu().numpy().astype(np.float64)
        return prob.reshape(m_count, n).T

    def resolve_pool_profile_matrix(
        self, pool_ids: Sequence[str], *, profiles: Optional[dict[str, dict]] = None,
        impute_unknown: bool = False, knn_k: int = 10, knn_weighting: str = "similarity",
        profile_encoder=None,
    ) -> np.ndarray:
        """``[len(pool_ids), p]`` profile-embedding matrix for an ARBITRARY
        pool -- not just this run's trained :attr:`model_ids`. Each row comes
        from, in order:

        1. the already-built profile embedding store, if ``mid`` is in it;
        2. else a caller-supplied (``profiles[mid]``) or curated
           (``configs/model_profiles.yaml``) entry, rendered + embedded on the
           fly (:mod:`router.routing.model_profiles`) -- ``None`` if neither
           exists, which still renders a generic templated profile;
        3. if ``impute_unknown`` and ``mid`` is not part of this run's own
           trained pool (no training observations either way), that vector is
           additionally kNN-blended toward the trained pool's nearest
           profiles (see :func:`router.routing.model_profiles.impute_unknown_profile`
           for the caveat this carries).

        Projected mode only: a 'free' run's model-side params are a
        transductive per-model table, not a function of ``e_m``, so there is
        nothing to project an unseen model onto -- this generalizes the same
        guard :meth:`_pool_refs` raises for a single missing id.

        ``profile_encoder`` overrides the encoder used to embed an ad hoc
        profile's rendered text (default: this run's own ``pathway`` encoder,
        lazily resolved/cached like :meth:`predict_scores_text`'s ``encoder``
        -- pass an explicit one in tests to avoid loading a real model).
        """
        if self._model.model_params != "projected":
            raise ValueError(
                f"{self.name}: resolve_pool_profile_matrix needs a 'projected' NIRT "
                f"run (cold-start capable); this run is model_params='free'"
            )
        from . import model_profiles

        m_store = self._store("profile", self._pathway)
        if m_store is None:
            raise FileNotFoundError(
                f"NIRTRouter: profile embeddings for pathway {self._pathway!r} not built"
            )
        cfg = getattr(self._data, "cfg", None)
        trained = set(self._model_ids)
        encoder = profile_encoder
        rows = []
        for mid in pool_ids:
            mid = str(mid)
            if mid in m_store:
                vec = np.asarray(m_store.get(mid), dtype=np.float32)
            else:
                if encoder is None:
                    encoder = _resolve_encoder(self._data, self._pathway, cache=self._encoder_cache)
                entry = (profiles or {}).get(mid)
                if entry is None and cfg is not None:
                    entry = model_profiles.lookup_curated_entry(cfg, mid)
                vec = model_profiles.embed_ad_hoc_profile(mid, entry, encoder, cfg=cfg)
            if impute_unknown and mid not in trained:
                vec, _found = model_profiles.impute_unknown_profile(
                    vec, m_store, self._model_ids, k=knn_k, weighting=knn_weighting,
                )
            rows.append(vec)
        return np.stack(rows).astype(np.float32)

    def predict_pool_scores_text(
        self, texts: Sequence[str], pool_ids: Sequence[str], *,
        profiles: Optional[dict[str, dict]] = None, impute_unknown: bool = False,
        knn_k: int = 10, knn_weighting: str = "similarity", encoder=None,
        profile_encoder=None,
    ) -> pd.DataFrame:
        """:meth:`predict_scores_text`, but against ``pool_ids`` instead of
        this run's own trained :attr:`model_ids` -- see
        :meth:`resolve_pool_profile_matrix` for how each pool id's profile is
        resolved (``profile_encoder`` is forwarded there). The broadened pool
        is centred on itself (``pool_ref == refs``), matching
        :func:`router.nirt.predict.predict_dataset`'s own projected-mode
        centring semantics (the dataset's whole model pool, not a fixed
        trained-only reference)."""
        e_q = _encode_texts(
            self._data, texts, self._query_pathway,
            fallback_pathway=self._pathway, encoder=encoder, cache=self._encoder_cache,
        )
        e_q = np.ascontiguousarray(e_q, dtype=np.float32)
        want = int(getattr(self._model, "query_dim", e_q.shape[1]))
        if e_q.shape[1] != want:
            e_q = self._pad_query_features(e_q, want)
        mat = self.resolve_pool_profile_matrix(
            pool_ids, profiles=profiles, impute_unknown=impute_unknown,
            knn_k=knn_k, knn_weighting=knn_weighting, profile_encoder=profile_encoder,
        )
        prob = self._forward_pool(e_q, mat, mat)
        return pd.DataFrame(prob, columns=[str(m) for m in pool_ids])

    def _pool_refs(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """``(per-model reference, pool centring reference)``, built once per
        router: the ``[M, p]`` profile matrix for a projected run (both
        values), or the ``[M]`` model-index vector for a free one (no pool
        reference). Read from the profile store on the first call only."""
        if self._pool_ref_cache is None:
            if self._model.model_params == "projected":
                m_store = self._store("profile", self._pathway)
                if m_store is None:
                    raise FileNotFoundError(
                        f"NIRTRouter: profile embeddings for pathway {self._pathway!r} not built"
                    )
                mat = np.stack([np.asarray(m_store.get(m), np.float32) for m in self._model_ids])
                self._pool_ref_cache = (mat, mat)
            else:
                idx = []
                for m in self._model_ids:
                    j = self._model_index.get(str(m))
                    if j is None:
                        raise ValueError(f"model {m!r} absent from a 'free' NIRT run's index")
                    idx.append(int(j))
                self._pool_ref_cache = (np.asarray(idx, dtype=np.int64), None)
        return self._pool_ref_cache

    def _pad_query_features(self, e_q: np.ndarray, want: int) -> np.ndarray:
        """A features run expects ``e_q (+) features``. Live text has no feature
        row, so the feature block is zero-filled -- but only when the shortfall
        is *exactly* the run's feature dim. Any other mismatch (e.g. a 384-d
        encoder against a 768-d run) is an error, not something to pad."""
        feat_dim = 0
        if self._query_features:
            feat = self._store("features", self._query_features)
            feat_dim = int(feat.dim) if feat is not None else 0
        if not feat_dim or e_q.shape[1] + feat_dim != want:
            raise ValueError(
                f"encoded query dim {e_q.shape[1]} does not match model query_dim {want} "
                f"(query_features={self._query_features!r}, feature dim {feat_dim}); "
                f"wrong encoder for pathway {self._query_pathway!r}?"
            )
        if not self._warned_zero_features:
            warnings.warn(
                f"NIRTRouter: run uses query_features={self._query_features!r}; raw-text "
                "scoring zero-fills them, so text-path scores are off the training "
                "distribution and can differ from route(query_ids)",
                stacklevel=3,
            )
            self._warned_zero_features = True
        return np.hstack([e_q, np.zeros((len(e_q), feat_dim), np.float32)])


# --------------------------------------------------------------------------- #
# KNNRouter -- RouterBench-style nearest-neighbour router                     #
# --------------------------------------------------------------------------- #
@register
class KNNRouter(RouterModel):
    """Predict a query's per-model quality as the mean over its ``k`` nearest
    training queries (cosine, retrieval embeddings). No training step."""

    kind = "knn"
    can_route_text = True

    def __init__(
        self,
        data=None,
        *,
        k: int = 5,
        pathway: str = "retrieval",
        model_ids: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        cost_model: Optional["CostModel"] = None,
        artifact: Optional["RouterModelArtifact"] = None,
    ):
        d = load_training_data_for_router(data)
        if model_ids is None:
            obs = d.nirt_observations()
            model_ids = sorted(obs.loc[obs["split"] == "train", "model_id"].astype(str).unique())
        super().__init__(model_ids, name=name or f"knn(k={k})",
                         cost_model=cost_model, artifact=artifact)
        self._data = d
        self._k = int(k)
        self._pathway = pathway
        self._cost_cache = _UNSET
        self._encoder_cache: dict = {}
        self._fit_cache = _UNSET

    @property
    def default_model_costs(self) -> Optional[np.ndarray]:
        return _lazy_model_costs(self)

    def _fitted(self) -> tuple:
        """The ``(nn, C_imp, model_ids)`` kNN index, fit once and cached -- the
        train-split query embeddings and correctness matrix don't change across
        calls, so refitting per request (as every routing call previously did)
        is wasted work."""
        if self._fit_cache is _UNSET:
            from ..nirt.baselines_infer import _fit_knn

            self._fit_cache = _fit_knn(self._data, k=self._k, pathway=self._pathway)
        return self._fit_cache

    def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        from ..nirt.baselines_infer import knn_router_matrix

        return knn_router_matrix(
            self._data, [str(q) for q in query_ids], k=self._k, pathway=self._pathway,
            fitted=self._fitted(),
        )

    def predict_scores_text(self, texts: Sequence[str], *, encoder=None) -> pd.DataFrame:
        e_q = _encode_texts(self._data, texts, self._pathway, encoder=encoder,
                            cache=self._encoder_cache)
        return self._score_embeddings(e_q)

    def _score_embeddings(self, e_q: np.ndarray) -> pd.DataFrame:
        """k-NN over the train-query embeddings for a raw ``[N, d]`` batch."""
        from ..nirt.baselines_infer import knn_router_matrix_from_embeddings

        return knn_router_matrix_from_embeddings(
            self._data, e_q, k=self._k, pathway=self._pathway, model_ids=self._model_ids,
            fitted=self._fitted(),
        )


# --------------------------------------------------------------------------- #
# MLPRouter -- the IRT-free e_q -> R^M head                                   #
# --------------------------------------------------------------------------- #
@register
class MLPRouter(RouterModel):
    """A plain MLP mapping the query embedding to a per-model correctness vector
    (masked BCE). The "NIRT minus the bilinear form" ablation, exposed as a
    router. Fit it with
    :func:`training.trainers.mlp_router.fit_mlp_router_as_router` -- a serving
    class has no ``.train()`` of its own."""

    kind = "mlp"

    def __init__(self, model, model_ids: Sequence[str], *, data=None,
                 pathway: str = "irt", query_pathway: Optional[str] = None,
                 query_features: Optional[str] = None, name: Optional[str] = None,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None,
                 phase0_config: Optional[str] = None):
        super().__init__(model_ids, name=name or "mlp", cost_model=cost_model, artifact=artifact)
        self._model = model
        self._data_cache = data
        self._phase0_config = phase0_config
        self._pathway = pathway
        self._query_pathway = query_pathway or pathway
        self._query_features = query_features
        self._cost_cache = _UNSET

    @classmethod
    def from_artifact(cls, artifact, cost_model=None, *, data=None, name: Optional[str] = None):
        """The artifact-driven construction path
        :class:`~router.models.registry.RouterModelFactory` describes: rebuild
        the model in-memory from ``artifact.payload`` via
        :func:`router.nirt.baselines_infer.build_mlp_router` + ``load_state_dict``,
        mirroring :meth:`NIRTRouter.from_artifact`.
        """
        from ..models.artifacts import MLPArtifactPayload
        from ..nirt.baselines_infer import build_mlp_router

        payload = artifact.payload
        if not isinstance(payload, MLPArtifactPayload):
            raise TypeError(
                f"MLPRouter.from_artifact needs an MLPArtifactPayload, "
                f"got {type(payload).__name__}"
            )
        model = build_mlp_router(payload.in_dim, len(payload.model_ids), payload.hidden, payload.dropout)
        model.load_state_dict(payload.state_dict)
        model.eval()
        return cls(
            model, payload.model_ids, data=data, name=name or f"mlp:{artifact.artifact_id}",
            pathway=payload.pathway, query_pathway=payload.query_pathway,
            query_features=payload.query_features,
            cost_model=cost_model, artifact=artifact,
        )

    @property
    def _data(self):
        return _lazy_training_data(self)

    @property
    def default_model_costs(self) -> Optional[np.ndarray]:
        return _lazy_model_costs(self)

    def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        from ..nirt.baselines_infer import mlp_router_matrix

        return mlp_router_matrix(
            self._model, self._model_ids, self._data,
            [str(q) for q in query_ids], pathway=self._pathway,
            query_pathway=self._query_pathway, query_features=self._query_features,
        )


# --------------------------------------------------------------------------- #
# RandomRouter -- evaluation floor                                            #
# --------------------------------------------------------------------------- #
@register
class RandomRouter(RouterModel):
    """Uniform-random model per query (deterministic given ``seed``). Not a real
    strategy -- a floor to check that a learned router beats chance.

    Each query's scores are seeded from ``(seed, hash(query_id))``, so they
    depend on the query, not its position in the batch."""

    kind = "random"

    def __init__(self, model_ids: Sequence[str], *, seed: int = 0, name: Optional[str] = None,
                 cost_model: Optional["CostModel"] = None,
                 artifact: Optional["RouterModelArtifact"] = None):
        super().__init__(model_ids, name=name or "random", cost_model=cost_model, artifact=artifact)
        self._seed = int(seed)

    def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
        ids = [str(q) for q in query_ids]
        m = len(self._model_ids)
        rows = [
            np.random.default_rng(
                [self._seed, int.from_bytes(hashlib.sha1(q.encode("utf-8")).digest()[:8], "little")]
            ).random(m)
            for q in ids
        ]
        return pd.DataFrame(
            np.array(rows).reshape(len(ids), m),
            index=ids, columns=self._model_ids,
        )

"""Classifier implementations and the registry that names them.

Every model exposes the same four methods (``fit`` / ``predict_proba`` /
``save`` / ``load``) and registers itself with ``@register("name")``, so adding
a model never touches the experiment runner and an experiment is just a YAML
file naming one.

``predict_proba`` -- not ``predict`` -- is the contract. Downstream consumers
threshold on confidence, and a threshold on a miscalibrated score is
meaningless, which is why every run is temperature-scaled and reports ECE.

The fine-tuned transformer lives in ``router.finetune``; it carries a training
loop and does not belong in the same file as three sklearn wrappers.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from router.embeddings import EmbeddingEncoder


class DomainClassifier(ABC):
    """Predicts a distribution over :data:`~router.taxonomy.DOMAIN_LABELS`.

    Probabilities, not argmax, are the contract: the router needs calibrated
    scores to decide when a prompt is ambiguous enough to warrant a stronger
    model or a clarifying follow-up.
    """

    #: Populated by :meth:`fit`; index position defines the column order of
    #: :meth:`predict_proba`.
    labels: list[str]

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.labels = []

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fit(self, train_texts: list[str], train_labels: list[str],
            val_texts: list[str] | None = None, val_labels: list[str] | None = None) -> None:
        """Train in place. ``val_*`` is for early stopping/model selection only."""

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n_texts, n_labels)`` array of probabilities."""

    def predict(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.labels[i] for i in proba.argmax(axis=1)]

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist to a directory."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore from a directory written by :meth:`save`."""

    def size_bytes(self) -> int:
        """Approximate on-disk footprint; reported alongside accuracy."""
        return 0


_REGISTRY: dict[str, Callable[..., DomainClassifier]] = {}


def register(name: str) -> Callable[[type[DomainClassifier]], type[DomainClassifier]]:
    def decorator(cls: type[DomainClassifier]) -> type[DomainClassifier]:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build(name: str, **params: Any) -> DomainClassifier:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def available() -> list[str]:
    return sorted(_REGISTRY)


def _vectorizer(word_ngrams: tuple[int, int], char_ngrams: tuple[int, int] | None,
                min_df: int, max_features: int) -> Any:
    """Word n-grams, optionally unioned with char n-grams.

    Char n-grams buy robustness to code identifiers, LaTeX and non-English
    fragments, which word tokenisation shreds.
    """
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngrams,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    if char_ngrams is None:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngrams,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


class _SklearnClassifier(DomainClassifier):
    """Shared fit/predict/persist plumbing for sklearn pipelines."""

    def _build_pipeline(self) -> Pipeline:
        raise NotImplementedError

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None) -> None:
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(train_texts, train_labels)
        self.labels = list(self.pipeline.classes_)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as fh:
            pickle.dump({"pipeline": self.pipeline, "labels": self.labels}, fh)

    def load(self, path: Path) -> None:
        with (path / "model.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self.pipeline = state["pipeline"]
        self.labels = state["labels"]

    def size_bytes(self) -> int:
        return len(pickle.dumps(self.pipeline))


@register("tfidf_logreg")
class TfidfLogisticRegression(_SklearnClassifier):
    def __init__(self, *, word_ngrams=(1, 2), char_ngrams=None, min_df=2,
                 max_features=200_000, C=4.0, class_weight="balanced", max_iter=2000, **kw):
        super().__init__(word_ngrams=word_ngrams, char_ngrams=char_ngrams, min_df=min_df,
                         max_features=max_features, C=C, class_weight=class_weight,
                         max_iter=max_iter, **kw)

    def _build_pipeline(self) -> Pipeline:
        p = self.params
        return make_pipeline(
            _vectorizer(tuple(p["word_ngrams"]),
                        tuple(p["char_ngrams"]) if p["char_ngrams"] else None,
                        p["min_df"], p["max_features"]),
            LogisticRegression(
                C=p["C"],
                class_weight=p["class_weight"],
                max_iter=p["max_iter"],
            ),
        )


@register("tfidf_linear_svm")
class TfidfLinearSVM(_SklearnClassifier):
    """LinearSVC wrapped in Platt scaling so it still exposes probabilities."""

    def __init__(self, *, word_ngrams=(1, 2), char_ngrams=(3, 5), min_df=2,
                 max_features=200_000, C=1.0, class_weight="balanced", cv=3, **kw):
        super().__init__(word_ngrams=word_ngrams, char_ngrams=char_ngrams, min_df=min_df,
                         max_features=max_features, C=C, class_weight=class_weight, cv=cv, **kw)

    def _build_pipeline(self) -> Pipeline:
        p = self.params
        return make_pipeline(
            _vectorizer(tuple(p["word_ngrams"]),
                        tuple(p["char_ngrams"]) if p["char_ngrams"] else None,
                        p["min_df"], p["max_features"]),
            CalibratedClassifierCV(
                LinearSVC(C=p["C"], class_weight=p["class_weight"]),
                cv=p["cv"],
            ),
        )


class _FrozenEncoderModel(DomainClassifier):
    """Encodes once, then delegates to an sklearn head."""

    def __init__(self, *, encoder_model: str, pooling: str = "mean", max_length: int = 256,
                 batch_size: int = 64, device: str | None = None, cache_tag: str | None = None, **kw):
        super().__init__(encoder_model=encoder_model, pooling=pooling, max_length=max_length,
                         batch_size=batch_size, device=device, cache_tag=cache_tag, **kw)
        self.encoder = EmbeddingEncoder(
            encoder_model, pooling=pooling, max_length=max_length,
            batch_size=batch_size, device=device,
        )
        self.head = None

    def _build_head(self):
        raise NotImplementedError

    def _encode(self, texts: list[str], split: str | None) -> np.ndarray:
        tag = self.params.get("cache_tag")
        if tag and split:
            return self.encoder.encode_cached(texts, tag=f"{tag}/{split}")
        return self.encoder.encode(texts)

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None) -> None:
        features = self._encode(list(train_texts), "train")
        self.head = self._build_head()
        self.head.fit(features, train_labels)
        self.labels = list(self.head.classes_)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        # Prediction-time encoding is deliberately uncached: at serving time
        # there is no cache, so this is also the honest latency path.
        return self.head.predict_proba(self.encoder.encode(list(texts)))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "head.pkl").open("wb") as fh:
            pickle.dump({"head": self.head, "labels": self.labels, "params": self.params}, fh)

    def load(self, path: Path) -> None:
        with (path / "head.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self.head = state["head"]
        self.labels = state["labels"]

    def size_bytes(self) -> int:
        return len(pickle.dumps(self.head))


@register("embed_logreg")
class EmbeddingLogisticRegression(_FrozenEncoderModel):
    def __init__(self, *, C=8.0, class_weight="balanced", max_iter=3000, **kw):
        super().__init__(C=C, class_weight=class_weight, max_iter=max_iter, **kw)

    def _build_head(self):
        p = self.params
        return LogisticRegression(
            C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"]
        )


@register("embed_mlp")
class EmbeddingMLP(_FrozenEncoderModel):
    """Non-linear head: tests whether domain structure is linearly separable
    in the frozen embedding space or needs a bit of curvature."""

    def __init__(self, *, hidden_sizes=(256,), alpha=1e-4, max_iter=400,
                 learning_rate_init=1e-3, seed=20260824, **kw):
        super().__init__(hidden_sizes=hidden_sizes, alpha=alpha, max_iter=max_iter,
                         learning_rate_init=learning_rate_init, seed=seed, **kw)

    def _build_head(self):
        p = self.params
        return MLPClassifier(
            hidden_layer_sizes=tuple(p["hidden_sizes"]),
            alpha=p["alpha"],
            max_iter=p["max_iter"],
            learning_rate_init=p["learning_rate_init"],
            early_stopping=True,
            n_iter_no_change=15,
            random_state=p["seed"],
        )


@register("embed_reduced_logreg")
class EmbeddingReducedLogisticRegression(_FrozenEncoderModel):
    """Frozen encoder -> decorrelation (VIF / PCA) -> linear head.

    Embedding dimensions are heavily collinear, so a linear head spends capacity
    modelling redundancy. This measures whether removing it helps, hurts, or does
    nothing -- the answer is not obvious a priori, since L2 regularisation
    already handles collinearity to a degree.
    """

    def __init__(self, *, order="vif_then_pca", n_components=0.95, vif_threshold=10.0,
                 max_vif_drop=200, standardize=True, C=8.0, class_weight="balanced",
                 max_iter=3000, **kw):
        super().__init__(order=order, n_components=n_components, vif_threshold=vif_threshold,
                         max_vif_drop=max_vif_drop, standardize=standardize, C=C,
                         class_weight=class_weight, max_iter=max_iter, **kw)

    def _build_head(self):
        from sklearn.pipeline import Pipeline

        from router.reduction import DenseReducer

        p = self.params
        reducer = DenseReducer(
            order=p["order"], n_components=p["n_components"],
            vif_threshold=p["vif_threshold"], max_vif_drop=p["max_vif_drop"],
            standardize=p["standardize"],
        )
        head = LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"])
        # Pipeline delegates `classes_` to the final estimator, so the shared
        # _FrozenEncoderModel.fit reads it without any extra plumbing.
        return Pipeline([("reduce", reducer), ("clf", head)])


# Imported for its side effect: FineTunedTransformer self-registers on import.
# This sits at the bottom, after DomainClassifier and register are defined, so
# that router.finetune's import of them resolves against this partially
# initialised module. Without it, `build("finetune_transformer")` raises even
# though the class exists.
from router import finetune as _finetune  # noqa: E402,F401

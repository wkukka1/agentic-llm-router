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


#: Human-readable descriptions of each domain, used as the "anchor" text a
#: zero-shot classifier compares prompts against. Bare slugs ("cs_general")
#: embed poorly; the encoder needs real words to place a label in its space.
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "cs_general": "computer science, programming, algorithms, software, data structures, information systems",
    "technology": "technology, engineering, medicine, health, applied sciences, management",
    "science": "science, mathematics, physics, chemistry, biology, earth science, geology",
    "history": "history, historical events, civilisations, wars, historical figures",
    "literature": "literature, novels, poetry, fiction, literary criticism, rhetoric",
    "language": "language, linguistics, grammar, translation, vocabulary, word meaning",
    "arts_recreation": "arts, music, sports, games, entertainment, film, recreation",
    "social_science": "social science, economics, law, politics, sociology, business",
    "philosophy_psychology": "philosophy, psychology, ethics, logic, reasoning, the mind",
}


@register("zeroshot_similarity")
class ZeroShotSimilarity(DomainClassifier):
    """Nearest-label classification with no training at all.

    Embed a description of each label, embed the prompt, and pick the closest
    label by cosine similarity. Nothing is fitted -- ``fit`` only records the
    label set -- so this is the honest floor for "what does the base encoder
    already know", and the only approach here that cannot overfit the training
    distribution, because it never sees it.

    Scores are softmaxed similarities. They are comparable within a row but are
    not probabilities in any calibrated sense, which is why the temperature step
    matters more here than elsewhere.
    """

    def __init__(self, *, encoder_model: str = "BAAI/bge-small-en-v1.5", pooling: str = "cls",
                 max_length: int = 256, batch_size: int = 64, temperature: float = 0.05,
                 descriptions: dict[str, str] | None = None, device: str | None = None, **kw):
        super().__init__(encoder_model=encoder_model, pooling=pooling, max_length=max_length,
                         batch_size=batch_size, temperature=temperature,
                         descriptions=descriptions, device=device, **kw)
        self.encoder = EmbeddingEncoder(
            encoder_model, pooling=pooling, max_length=max_length,
            batch_size=batch_size, normalize=True, device=device,
        )
        self._anchors = None

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None) -> None:
        # No training. The label set is all that is taken from the data, and the
        # anchors come from the descriptions, not from any example.
        self.labels = sorted(set(train_labels))
        descriptions = self.params.get("descriptions") or DOMAIN_DESCRIPTIONS
        missing = [label for label in self.labels if label not in descriptions]
        if missing:
            raise ValueError(f"no description for label(s): {missing}")
        self._anchors = self.encoder.encode([descriptions[label] for label in self.labels])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._anchors is None:
            raise RuntimeError("call fit() first to establish the label set")
        # Both sides are L2-normalised, so a dot product is cosine similarity.
        similarity = self.encoder.encode(list(texts)) @ self._anchors.T
        scaled = similarity / max(self.params["temperature"], 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "zeroshot.pkl").open("wb") as fh:
            pickle.dump({"labels": self.labels, "anchors": self._anchors, "params": self.params}, fh)

    def load(self, path: Path) -> None:
        with (path / "zeroshot.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self.labels, self._anchors = state["labels"], state["anchors"]

    def size_bytes(self) -> int:
        return 0 if self._anchors is None else self._anchors.nbytes



@register("knn_embed")
class EmbeddingKNN(DomainClassifier):
    """Frozen encoder + k-nearest-neighbour vote.

    Worth testing rather than assuming: with few examples per class, kNN can
    beat a fine-tune, because it makes no attempt to learn a decision boundary
    from data too sparse to define one. It also degrades gracefully, exposes
    *which* training prompts drove a prediction, and absorbs a new class by
    appending rows rather than retraining.

    Probabilities come from distance-weighted neighbour votes, so they are
    genuinely soft rather than a hard vote rounded off.
    """

    def __init__(self, *, encoder_model: str = "BAAI/bge-small-en-v1.5", pooling: str = "cls",
                 max_length: int = 256, batch_size: int = 64, k: int = 15,
                 weights: str = "distance", temperature: float = 0.05,
                 device: str | None = None, **kw):
        super().__init__(encoder_model=encoder_model, pooling=pooling, max_length=max_length,
                         batch_size=batch_size, k=k, weights=weights,
                         temperature=temperature, device=device, **kw)
        self.encoder = EmbeddingEncoder(encoder_model, pooling=pooling, max_length=max_length,
                                        batch_size=batch_size, normalize=True, device=device)
        self._bank = None
        self._bank_labels = None

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None) -> None:
        self.labels = sorted(set(train_labels))
        self._bank = self.encoder.encode(list(train_texts))
        index = {label: i for i, label in enumerate(self.labels)}
        self._bank_labels = np.array([index[label] for label in train_labels])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._bank is None:
            raise RuntimeError("call fit() first")
        p = self.params
        query = self.encoder.encode(list(texts))
        # Both sides L2-normalised, so a dot product is cosine similarity.
        similarity = query @ self._bank.T
        k = min(p["k"], similarity.shape[1])
        top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]

        out = np.zeros((len(texts), len(self.labels)))
        for row, neighbours in enumerate(top):
            sims = similarity[row, neighbours]
            if p["weights"] == "distance":
                # Softmax over similarity: near neighbours dominate, but a
                # tie among the top few stays genuinely uncertain.
                w = np.exp((sims - sims.max()) / max(p["temperature"], 1e-6))
            else:
                w = np.ones_like(sims)
            for weight, label_idx in zip(w, self._bank_labels[neighbours], strict=True):
                out[row, label_idx] += weight
        return out / out.sum(axis=1, keepdims=True).clip(min=1e-12)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "knn.pkl").open("wb") as fh:
            pickle.dump({"bank": self._bank, "bank_labels": self._bank_labels,
                         "labels": self.labels, "params": self.params}, fh)

    def load(self, path: Path) -> None:
        with (path / "knn.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self._bank, self._bank_labels = state["bank"], state["bank_labels"]
        self.labels = state["labels"]

    def size_bytes(self) -> int:
        return 0 if self._bank is None else self._bank.nbytes



@register("ensemble")
class EnsembleClassifier(DomainClassifier):
    """Averages the probabilities of several member classifiers.

    Fine-tuning on ~1.3k examples turned out to be high-variance: five seeds of
    the same config spanned 63.5-67.8% (sd 1.75, 95% CI +/-3.4 points), which
    is wider than most of the effects being measured. Frozen encoders with a
    linear head are deterministic and individually stronger, and averaging
    several *different* encoders adds the diversity that seed-averaging could
    not.

    Members must be chosen on validation, never on test: searching member
    combinations against the test set inflated the score by ~2 points in this
    project before the protocol was fixed.
    """

    def __init__(self, *, members: list[dict], temperature: float = 1.0, **kw):
        super().__init__(members=members, temperature=temperature, **kw)
        self._members: list[DomainClassifier] = []

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None) -> None:
        self._members = []
        for spec in self.params["members"]:
            m = build(spec["name"], **(spec.get("params") or {}))
            m.fit(train_texts, train_labels, val_texts, val_labels)
            self._members.append(m)
        self.labels = self._members[0].labels
        if any(m.labels != self.labels for m in self._members):
            raise ValueError("ensemble members disagree on the label ordering")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self._members:
            raise RuntimeError("call fit() or load() first")
        stacked = np.mean([m.predict_proba(list(texts)) for m in self._members], axis=0)
        temperature = self.params.get("temperature", 1.0)
        if temperature == 1.0:
            return stacked
        # Temperature is fitted on validation by the experiment runner; applying
        # it here keeps the served probabilities calibrated (ECE 0.25 -> 0.05).
        logits = np.log(np.clip(stacked, 1e-12, None)) / max(temperature, 1e-12)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self._members):
            m.save(path / f"member_{i}")
        with (path / "ensemble.pkl").open("wb") as fh:
            pickle.dump({"labels": self.labels, "params": self.params}, fh)

    def load(self, path: Path) -> None:
        with (path / "ensemble.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self.labels = state["labels"]
        self.params = state["params"]
        self._members = []
        for i, spec in enumerate(self.params["members"]):
            m = build(spec["name"], **(spec.get("params") or {}))
            m.load(path / f"member_{i}")
            self._members.append(m)

    def size_bytes(self) -> int:
        return sum(m.size_bytes() for m in self._members)


# Imported for its side effect: FineTunedTransformer self-registers on import.
# This sits at the bottom, after DomainClassifier and register are defined, so
# that router.finetune's import of them resolves against this partially
# initialised module. Without it, `build("finetune_transformer")` raises even
# though the class exists.
from router import finetune as _finetune  # noqa: E402,F401

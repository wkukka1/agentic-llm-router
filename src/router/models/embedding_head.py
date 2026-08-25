"""Frozen encoder + light head.

The interesting middle ground for a router: one forward pass through a small
encoder (cacheable, batchable, shareable with other heads such as difficulty)
and a head that costs nothing to retrain when the label space changes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from router.features.embeddings import EmbeddingEncoder
from router.models.base import DomainClassifier
from router.models.registry import register


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

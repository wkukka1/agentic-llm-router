"""Sparse lexical baselines.

These matter beyond being a baseline: a TF-IDF + linear model runs in tens of
microseconds on CPU, which is the latency budget an inline router actually has.
A transformer has to beat it by enough to justify the extra milliseconds.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from router.models.base import DomainClassifier
from router.models.registry import register


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

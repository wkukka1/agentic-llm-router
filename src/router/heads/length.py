"""Serving head: how long will the answer be, and how sure is that.

Mirrors :class:`~router.heads.domain.DomainHead` in
shape -- construct from a run directory, call ``predict`` or ``predict_batch``
-- but the output is a length distribution rather than a label distribution,
because the routing decision it feeds is "how much work is this" rather than
"what kind of work is this".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from router.features import build_features
from router.heads.length_model import LengthModel

#: Bucket names for the quartile view, shortest first.
BUCKETS = ("short", "medium", "long", "very_long")


@dataclass(slots=True)
class LengthPrediction:
    """One prompt's expected answer size."""

    expected_tokens: float
    #: The model's estimate in the space it was fitted in.
    log_tokens: float
    #: Which training quartile the estimate falls in.
    bucket: str
    #: Spread of the residuals, shared by every prediction from this head. A
    #: point estimate without it invites the caller to over-trust a number that
    #: is typically off by a factor of `exp(sigma)`.
    sigma: float

    def probability_over(self, tokens: float) -> float:
        """``P(the answer exceeds `tokens`)`` under the fitted spread."""
        from scipy.stats import norm

        if self.sigma <= 0:
            return float(self.expected_tokens > tokens)
        return float(norm.sf(np.log1p(tokens), loc=self.log_tokens, scale=self.sigma))


class LengthHead:
    """Loads a trained length run and serves per-prompt estimates."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.model = LengthModel.load(self.run_dir)
        meta = json.loads((self.run_dir / "length.json").read_text(encoding="utf-8"))
        self.feature_set: str = meta["feature_set"]
        self.encoder_model: str | None = meta.get("encoder_model")
        self._encoder = None

    @property
    def encoder(self):
        """Built on first use, and only when the feature set needs one.

        The surface-only head is the deployable one until an encoder pass is
        already being paid for by the domain head; keeping the import in here
        means it costs nothing to load.
        """
        if self._encoder is None:
            from router.embeddings.encoder import EmbeddingEncoder

            self._encoder = EmbeddingEncoder(self.encoder_model, max_length=256)
        return self._encoder

    def _features(self, prompts: list[str], embeddings: np.ndarray | None = None) -> np.ndarray:
        if "embedding" in self.feature_set and embeddings is None:
            embeddings = self.encoder.encode(prompts)
        return build_features(prompts, self.feature_set, embeddings)

    def predict(self, prompt: str) -> LengthPrediction:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str],
                      embeddings: np.ndarray | None = None) -> list[LengthPrediction]:
        """``embeddings`` lets a caller that has already encoded these prompts
        hand the vectors in rather than paying for a second pass. It must come
        from :attr:`encoder_model`; feeding another encoder's output produces
        confident nonsense, which is why the composite groups heads by encoder
        name rather than assuming."""
        prompts = list(prompts)
        if not prompts:
            return []
        log_tokens = self.model.predict(self._features(prompts, embeddings))
        edges = self.model.edges
        return [
            LengthPrediction(
                expected_tokens=float(np.expm1(value)),
                log_tokens=float(value),
                bucket=BUCKETS[int(np.searchsorted(edges, value))],
                sigma=self.model.sigma,
            )
            for value in log_tokens
        ]

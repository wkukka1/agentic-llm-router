"""Serving head: every free-label attribute of one prompt, calibrated.

Returns probabilities, never decisions. A gate's threshold is a deployment
question -- the cost of wrongly believing a prompt needs code execution differs
by an order of magnitude between setups -- so the caller picks it and this
returns the number to pick against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from router.heads.attributes_model import AttributeModel
from router.heads.attributes_taxonomy import CRITERIA, GATES


@dataclass(slots=True)
class AttributePrediction:
    """One prompt's attributes as calibrated probabilities."""

    probabilities: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, name: str) -> float:
        return self.probabilities[name]

    @property
    def gates(self) -> dict[str, float]:
        """The preconditions: code, maths, creative writing, constraints."""
        return {k: v for k, v in self.probabilities.items() if k in GATES}

    @property
    def criteria(self) -> dict[str, float]:
        """The seven judged difficulty dimensions.

        Inputs to a downstream model, not a difficulty score. The arena's own
        rubric over these predicts whether the stronger model was needed at
        chance (AUC 0.487); treating them as an answer repeats a mistake this
        project has already made three times.
        """
        return {k: v for k, v in self.probabilities.items() if k in CRITERIA}

    def above(self, threshold: float) -> list[str]:
        """Attributes whose probability clears ``threshold``, strongest first."""
        hits = [(k, v) for k, v in self.probabilities.items() if v >= threshold]
        return [k for k, _ in sorted(hits, key=lambda kv: -kv[1])]

    def vector(self, names: tuple[str, ...]) -> np.ndarray:
        """Fixed-order float vector, for the handoff to a downstream model."""
        return np.array([self.probabilities.get(n, np.nan) for n in names], dtype=float)


class AttributeHead:
    """Loads a trained attribute run and serves calibrated probabilities."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.model = AttributeModel.load(self.run_dir)
        meta = json.loads((self.run_dir / "attributes.json").read_text(encoding="utf-8"))
        self.encoder_model: str = meta["encoder_model"]
        self.feature_set: str = meta.get("feature_set", "embedding")
        self._encoder = None

    @property
    def names(self) -> tuple[str, ...]:
        """The attributes this run actually fitted, in taxonomy order."""
        return tuple(n for n in self.model.names if n in self.model.heads)

    @property
    def encoder(self):
        if self._encoder is None:
            from router.embeddings.encoder import EmbeddingEncoder

            self._encoder = EmbeddingEncoder(self.encoder_model, max_length=256)
        return self._encoder

    def _features(self, prompts: list[str], embeddings: np.ndarray | None = None) -> np.ndarray:
        from router.features import build_features

        if "embedding" in self.feature_set and embeddings is None:
            embeddings = self.encoder.encode(prompts)
        return build_features(prompts, self.feature_set, embeddings)

    def predict(self, prompt: str) -> AttributePrediction:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str],
                      embeddings: np.ndarray | None = None) -> list[AttributePrediction]:
        """``embeddings`` must come from :attr:`encoder_model` -- see the note
        on :meth:`LengthHead.predict_batch`."""
        prompts = list(prompts)
        if not prompts:
            return []
        per_attribute = self.model.predict_proba(self._features(prompts, embeddings))
        return [AttributePrediction({name: float(values[i]) for name, values in per_attribute.items()})
                for i in range(len(prompts))]

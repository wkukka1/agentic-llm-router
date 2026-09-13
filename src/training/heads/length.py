"""Train the length head over each feature set and report which one earns its cost.

Every number here comes from the same protocol as the rest of the project:
alpha is chosen on validation, the residual spread is fitted on validation, and
test is read exactly once per configuration. The comparison that matters is not
"does it beat zero" -- prompt length alone beats zero -- but "does the encoder
beat the free features by enough to pay for an encoder pass at serving time".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from router.features import FEATURE_SETS, build_features
from router.heads.length_model import LengthModel, bootstrap_ci, evaluate
from router.settings import settings
from training.data.arena_corpus import load_arena_splits

log = logging.getLogger(__name__)

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def artifacts_dir() -> Path:
    return settings().artifacts_dir / "length"


def encode_splits(splits: dict[str, pd.DataFrame], encoder_model: str) -> dict[str, np.ndarray]:
    """Embed every split, cached on disk by content -- reruns are free."""
    from router.embeddings.encoder import EmbeddingEncoder

    encoder = EmbeddingEncoder(encoder_model, max_length=256, batch_size=128)
    short = encoder_model.split("/")[-1]
    return {name: encoder.encode_cached(frame["prompt"].tolist(), tag=f"length/{short}/{name}")
            for name, frame in splits.items()}


def run_length_sweep(*, encoder_model: str = DEFAULT_ENCODER,
                     feature_sets: tuple[str, ...] = FEATURE_SETS,
                     out_dir: Path | None = None) -> pd.DataFrame:
    """Fit every feature set and save the best by validation Spearman.

    Training produces models; judging them is `evaluation`'s job and lives in
    `evaluation.length_audit`, which is why the audit is no longer run from
    here. The tier rule forced the separation and the separation is right: a
    sweep that scores itself is a sweep that can quietly grade on a curve.
    """
    out_dir = out_dir or artifacts_dir()
    splits = load_arena_splits()
    y = {name: frame["y"].to_numpy() for name, frame in splits.items()}
    long_edge = float(np.quantile(y["train"], 0.75))

    embeddings = ({} if all("embedding" not in f for f in feature_sets)
                  else encode_splits(splits, encoder_model))

    rows: list[dict] = []
    best: tuple[float, LengthModel, str] | None = None

    # The floor every feature set is measured against: predict the training
    # mean for everything. Its R-squared is zero by construction, and its
    # typical error is the spread of the target itself.
    rows.append({"features": "constant", "n_features": 0, "alpha": None,
                 **evaluate(y["test"], np.full(len(y["test"]), y["train"].mean()))})

    for kind in feature_sets:
        X = {name: build_features(frame["prompt"].tolist(), kind, embeddings.get(name))
             for name, frame in splits.items()}

        alpha = max(ALPHAS, key=lambda a: evaluate(
            y["val"], LengthModel(alpha=a).fit(X["train"], y["train"]).predict(X["val"]))["spearman"])
        model = (LengthModel(alpha=alpha, feature_set=kind)
                 .fit(X["train"], y["train"])
                 .calibrate(X["val"], y["val"]))
        val_score = evaluate(y["val"], model.predict(X["val"]))["spearman"]
        predictions = model.predict(X["test"])
        metrics = evaluate(y["test"], predictions, sigma=model.sigma, long_threshold=long_edge)
        low, high = bootstrap_ci(y["test"], predictions)
        rows.append({"features": kind, "n_features": X["train"].shape[1], "alpha": alpha,
                     "val_spearman": val_score, "spearman_lo": low, "spearman_hi": high,
                     **metrics})

        if best is None or val_score > best[0]:
            best = (val_score, model, kind)

    if best is not None:
        _save(best[1], best[2], encoder_model, out_dir)

    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "leaderboard.csv", index=False)
    return frame


def _save(model: LengthModel, feature_set: str, encoder_model: str, out_dir: Path) -> None:
    """Write the serving artifact: the model plus what it expects to be fed.

    The directory name carries the encoder, not just the feature set. Two
    encoders over the same feature set are two different heads that cannot be
    served each other's weights, and naming them alike meant the second sweep
    silently overwrote the first -- which is exactly the mismatch
    `_ENCODER_PARAMS` guards against in the classifier models.
    """
    name = feature_set.replace("+", "_")
    if "embedding" in feature_set:
        name = f"{encoder_model.split('/')[-1]}__{name}"
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir)
    (run_dir / "length.json").write_text(json.dumps({
        "feature_set": feature_set,
        "encoder_model": encoder_model if "embedding" in feature_set else None,
        "alpha": model.alpha,
        "sigma": model.sigma,
        "edges": model.edges,
    }, indent=2), encoding="utf-8")
    log.info("saved length head to %s", run_dir)

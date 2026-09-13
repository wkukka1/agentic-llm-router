"""Train the attribute heads and report what each one is worth.

Reported per attribute, not averaged. A mean AUC over eleven heads with base
rates from 0.075 to 0.819 is a number with no referent -- maths at 0.91 and
complexity at 0.77 are different products and the average is neither.

Every head is scored against its own base rate, because AUC hides prevalence:
a head at 0.82 on an attribute that is true 82% of the time may still be
useless at the threshold anyone would actually use, which is why precision at a
fixed recall is reported alongside.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from router.features import build_features
from router.heads.attributes_model import AttributeModel
from router.heads.attributes_taxonomy import ATTRIBUTE_NAMES, describe
from router.settings import settings
from training.data.arena_corpus import load_arena_splits
from training.data.arena_labels import label_for, load_arena_labels
from training.metrics import expected_calibration_error

log = logging.getLogger(__name__)

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"


def artifacts_dir() -> Path:
    return settings().artifacts_dir / "attributes"


def evaluate_head(y_true: np.ndarray, proba: np.ndarray, *, recall_target: float = 0.5) -> dict:
    """AUC, average precision, calibration, and a usable operating point."""
    known = ~np.isnan(y_true)
    y_true, proba = y_true[known].astype(int), proba[known]
    out = {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        # Calibration over the predicted probability, not the top-1 confidence:
        # a binary head is thresholded directly, so what matters is whether 0.8
        # means 0.8.
        "ece": expected_calibration_error(proba, y_true.astype(float)),
    }
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    usable = recall[:-1] >= recall_target
    if usable.any():
        best = int(np.argmax(precision[:-1] * usable))
        out[f"precision@recall{int(recall_target * 100)}"] = float(precision[best])
        out["threshold"] = float(thresholds[best])
    return out


def run_attribute_sweep(*, encoder_model: str = DEFAULT_ENCODER,
                        feature_set: str = "surface+embedding",
                        names: tuple[str, ...] = ATTRIBUTE_NAMES,
                        out_dir: Path | None = None) -> pd.DataFrame:
    """Fit every attribute head, score on test, save the run."""
    out_dir = out_dir or artifacts_dir()
    splits = load_arena_splits()
    labels = load_arena_labels()

    embeddings = {}
    if "embedding" in feature_set:
        from training.heads.length import encode_splits

        embeddings = encode_splits(splits, encoder_model)
    X = {name: build_features(frame["prompt"].tolist(), feature_set, embeddings.get(name))
         for name, frame in splits.items()}
    y = {split: {n: label_for(labels, frame["arena_id"], n) for n in names}
         for split, frame in splits.items()}

    model = (AttributeModel(names=names)
             .fit(X["train"], y["train"])
             .calibrate(X["val"], y["val"]))

    proba = model.predict_proba(X["test"])
    rows = []
    for name in model.heads:
        attribute = describe(name)
        rows.append({"attribute": name, "kind": attribute.kind,
                     **evaluate_head(y["test"][name], proba[name])})
    frame = pd.DataFrame(rows).sort_values(["kind", "auc"], ascending=[True, False])

    _save(model, encoder_model, feature_set, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "attributes.csv", index=False)
    return frame


def _save(model: AttributeModel, encoder_model: str, feature_set: str, out_dir: Path) -> None:
    name = feature_set.replace("+", "_")
    if "embedding" in feature_set:
        name = f"{encoder_model.split('/')[-1]}__{name}"
    run_dir = out_dir / name
    model.save(run_dir)
    (run_dir / "attributes.json").write_text(json.dumps({
        "encoder_model": encoder_model if "embedding" in feature_set else None,
        "feature_set": feature_set,
        "attributes": sorted(model.heads),
        "calibrated": sorted(model.calibrators),
        "base_rates": model.base_rates,
    }, indent=2), encoding="utf-8")
    log.info("saved attribute heads to %s", run_dir)

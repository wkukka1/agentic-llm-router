"""Does the length signal change a routing decision?

A head that correlates with something is not the same as a head that is worth
serving. This module asks the only question that settles it: does knowing how
long the answer will be help choose *which model should write it*?

The answer here is no, and it is worth having the code that says so, because
"long answer therefore big model" is the intuition everyone arrives with.

The test set is the arena preference data, where the label is the decision
itself -- `strong_needed` when the stronger model won the battle,
`weak_sufficient` when the cheap one won or drew. Rows join to the length
corpus on `arena_id`, not on prompt text, because the two pipelines extract
that text differently.

Read the oracle row first. It uses the *true* token counts, so it bounds what
any length predictor could ever buy: if perfect knowledge of the answer's
length does not separate the classes, no improvement to the estimator will.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from prompt_decomposition.core.settings import settings

log = logging.getLogger(__name__)


def load_routing_labels(splits: tuple[str, ...] = ("train", "val", "test")) -> pd.DataFrame:
    """Arena battles with the routing decision attached, keyed by `arena_id`."""
    directory = settings().processed_dir / "preference"
    frames = [pd.read_parquet(directory / f"{name}.parquet") for name in splits
              if (directory / f"{name}.parquet").exists()]
    if not frames:
        raise FileNotFoundError(f"no preference splits under {directory}")
    frame = pd.concat(frames, ignore_index=True)
    frame["arena_id"] = [
        json.loads(m).get("arena_id") if isinstance(m, str) else (m or {}).get("arena_id")
        for m in frame["meta"]
    ]
    return frame[["arena_id", "routing_label", "hardness_score"]].dropna(
        subset=["arena_id"]).drop_duplicates("arena_id")


def routing_value(test: pd.DataFrame, predictions: np.ndarray | None = None) -> dict:
    """Score length -- true and predicted -- against the routing decision.

    ``test`` is a split of the length corpus, carrying `arena_id`, `y` and the
    two token counts. ``predictions`` is the head's output for those rows, or
    None to run the oracle rows only.
    """
    joined = test.assign(**({"pred": predictions} if predictions is not None else {})).merge(
        load_routing_labels(), on="arena_id", how="inner")
    if len(joined) < 200:
        raise ValueError(f"only {len(joined)} rows join to the routing labels; too few to read")

    needed = (joined["routing_label"] == "strong_needed").astype(int).to_numpy()
    hardness = joined["hardness_score"].to_numpy()
    log_a, log_b = np.log1p(joined["tokens_a"]), np.log1p(joined["tokens_b"])

    scores = {
        # The oracle: true length, which bounds every estimator of it.
        "true_length": joined["y"].to_numpy(),
        # Available only after generating both answers, so not a routing signal
        # -- included because it is the one length-derived quantity that is not
        # at chance, which shows the target is not pure noise.
        "length_disagreement": np.abs(log_a - log_b).to_numpy(),
        "prompt_length": joined["prompt"].str.len().to_numpy(),
        # Arena's own LLM-judged difficulty rubric: the strongest available
        # statement of "how hard is this prompt", and the reference for whether
        # hardness in any form moves this decision.
        "hardness_rubric": hardness,
    }
    if predictions is not None:
        scores["predicted_length"] = joined["pred"].to_numpy()

    out = {"n": int(len(joined)), "base_rate": float(needed.mean()), "auc": {}, "hardness_rho": {}}
    for name, value in scores.items():
        out["auc"][name] = float(roc_auc_score(needed, value))
        out["hardness_rho"][name] = float(spearmanr(value, hardness).statistic)
    return out


def render(result: dict) -> str:
    """A table, and the conclusion it supports."""
    lines = [
        f"Routing value, {result['n']} held-out prompts "
        f"(base rate {result['base_rate']:.3f} 'strong model needed')",
        "",
        f"{'signal':24} {'AUC: strong needed':>20} {'rho: hardness rubric':>22}",
    ]
    for name in result["auc"]:
        lines.append(f"{name:24} {result['auc'][name]:>20.4f} {result['hardness_rho'][name]:>22.4f}")
    return "\n".join(lines)


def report(test: pd.DataFrame, predictions: np.ndarray | None = None,
           out_path: Path | None = None) -> dict:
    result = routing_value(test, predictions)
    text = render(result)
    log.info("\n%s", text)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return result

"""Battles between a weak and a strong model, with the weak model's answer.

The cascade asks a different question from every other head in this project.
Those ask "what will happen"; this asks "given what just happened, was it good
enough". The prompt-side question is closed -- four independent measurements put
it at chance, and the best anyone reached predicting the decision directly was
AUC 0.567. This corpus is what lets the other question be asked honestly.

One row is one arena battle where a weak model met a strong one, carrying the
**weak model's own response** and whether the strong model actually won. A
policy fitted here answers: read the cheap answer, decide whether to pay for the
expensive one.

Tiers are derived from observed win rates, never hand-assigned. Labelling fifty
models by reputation would bake the author's priors into the ground truth, and
the win rate is the thing the label is about anyway.
"""

from __future__ import annotations

import glob
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from decompose.classifiers.prompt_decomposition.settings import settings
from training.prompt_decomposition.data.arena_corpus import ARENA_GLOB, _content_text

log = logging.getLogger(__name__)

#: A model needs this many battles before its win rate means anything.
MIN_BATTLES = 200

#: Tier edges as quantiles of win rate. The middle third is dropped: a battle
#: between two mid-tier models says little about whether paying more helps.
WEAK_QUANTILE, STRONG_QUANTILE = 1 / 3, 2 / 3


def _assistant_text(conversation) -> str:
    """The first assistant turn -- the answer the user actually saw."""
    for message in conversation:
        if message.get("role") == "assistant":
            return _content_text(message.get("content"))
    return ""


def _user_text(conversation) -> str:
    for message in conversation:
        if message.get("role") == "user":
            return _content_text(message.get("content"))
    return ""


def model_tiers(frame: pd.DataFrame) -> tuple[set[str], set[str], pd.Series]:
    """Split models into weak and strong by observed win rate.

    Ties count as a loss for both sides here, which is deliberate: a model that
    draws a lot is not demonstrating strength, and the routing label treats a
    draw as "the cheap one was sufficient".
    """
    stacked = pd.concat([
        pd.DataFrame({"model": frame["model_a"], "won": frame["winner"] == "model_a"}),
        pd.DataFrame({"model": frame["model_b"], "won": frame["winner"] == "model_b"}),
    ])
    rates = stacked.groupby("model")["won"].agg(["size", "mean"])
    rates = rates[rates["size"] >= MIN_BATTLES]["mean"].sort_values()
    weak = set(rates[rates <= rates.quantile(WEAK_QUANTILE)].index)
    strong = set(rates[rates >= rates.quantile(STRONG_QUANTILE)].index)
    return weak, strong, rates


def build_cascade_corpus(pattern: str = ARENA_GLOB) -> pd.DataFrame:
    """One row per weak-vs-strong battle, carrying the weak model's answer."""
    paths = sorted(glob.glob(pattern.replace("~", str(Path.home()))))
    if not paths:
        raise FileNotFoundError(f"no arena parquet files under {pattern}")

    columns = ["id", "model_a", "model_b", "winner", "conversation_a",
               "conversation_b", "conv_metadata"]
    raw = pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True)
    meta = pd.json_normalize(raw["conv_metadata"])
    raw = raw[(meta["turns"] == 1).to_numpy()].reset_index(drop=True)

    weak, strong, _ = model_tiers(raw)
    a_weak = raw["model_a"].isin(weak) & raw["model_b"].isin(strong)
    b_weak = raw["model_b"].isin(weak) & raw["model_a"].isin(strong)
    mixed = raw[a_weak | b_weak].reset_index(drop=True)
    weak_is_a = a_weak[a_weak | b_weak].to_numpy()

    weak_conv = np.where(weak_is_a, mixed["conversation_a"], mixed["conversation_b"])
    # "Was the strong model needed?" -- it won. A draw, a weak win, or both
    # being bad all mean paying more would not have helped.
    strong_won = np.where(weak_is_a, mixed["winner"] == "model_b", mixed["winner"] == "model_a")

    frame = pd.DataFrame({
        "arena_id": mixed["id"],
        "prompt": [_user_text(c) for c in weak_conv],
        "weak_response": [_assistant_text(c) for c in weak_conv],
        "weak_model": np.where(weak_is_a, mixed["model_a"], mixed["model_b"]),
        "strong_model": np.where(weak_is_a, mixed["model_b"], mixed["model_a"]),
        "escalate": strong_won.astype(int),
    })
    frame = frame[(frame["prompt"].str.len() >= 10) & (frame["weak_response"].str.len() >= 1)]
    frame = frame[~frame["prompt"].str.lower().str.replace(r"\s+", " ", regex=True).duplicated()]
    frame = frame.reset_index(drop=True)
    frame["split"] = [_split_of(p) for p in frame["prompt"]]
    log.info("cascade corpus: %d battles, escalate rate %.3f, %s",
             len(frame), frame["escalate"].mean(), frame["split"].value_counts().to_dict())
    return frame


def _split_of(prompt: str) -> str:
    """Hashed, like the arena corpus, so a rebuild never moves a row."""
    position = int(hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if position < 0.7 else ("val" if position < 0.85 else "test")


def corpus_dir() -> Path:
    return settings().processed_dir / "cascade"


def save_cascade_corpus(frame: pd.DataFrame) -> None:
    out = corpus_dir()
    out.mkdir(parents=True, exist_ok=True)
    for split, part in frame.groupby("split"):
        part.reset_index(drop=True).to_parquet(out / f"{split}.parquet", index=False)


def load_cascade_splits() -> dict[str, pd.DataFrame]:
    out = corpus_dir()
    if not (out / "train.parquet").exists():
        raise FileNotFoundError(f"no cascade corpus at {out}; run build_cascade_corpus first")
    return {s: pd.read_parquet(out / f"{s}.parquet") for s in ("train", "val", "test")}

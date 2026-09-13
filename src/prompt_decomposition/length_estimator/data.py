"""The length corpus: prompts paired with how long two models actually answered.

The target is free. Every LMArena row carries both models' responses and the
token count of each, so there is nothing to annotate -- 109,335 single-turn
prompts with an exact tokenizer count on each side.

Two facts about this target shape everything downstream:

**It is model-dependent.** Given the *same* prompt, the two models agree on
length at Spearman 0.590. That is not measurement noise to be cleaned up, it is
the task: "how long is the answer" is partly a property of who answers. A
prompt-only predictor cannot beat that agreement against a single model's
length, which is why the target averages the two -- averaging removes part of
the model-specific variance and leaves more of what a prompt can explain. The
average is correspondingly more predictable than either side: Spearman-Brown
puts its reliability at ~0.742, and that is the ceiling for scores against it.

**It is log-scaled.** Raw token counts span 1 to 88,300. A squared-error fit on
raw counts is a fit to the top 1%, and the routing decision ("is this a 200
token answer or a 2,000 token one") is multiplicative anyway.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import logging

import numpy as np
import pandas as pd

from prompt_decomposition.core.settings import settings

log = logging.getLogger(__name__)

#: The arena dump, as `huggingface_hub` lays it out on disk.
ARENA_GLOB = (
    "~/.cache/huggingface/hub/datasets--lmarena-ai--arena-human-preference-140k"
    "/snapshots/*/data/*.parquet"
)

#: Prompts shorter than this are fragments ("hi", "?") whose answer length is
#: whatever the model feels like; they carry no signal and 0.4% of rows.
MIN_PROMPT_CHARS = 10

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

#: Above this share of serialised-looking prompts, the extractor is broken
#: rather than the users being unusual. The real failure hit 100% of rows.
SERIALISED_PROMPT_LIMIT = 0.01


def _first_user_text(conversation) -> str:
    """The opening user turn, whatever shape this dump stores it in.

    The shapes seen in the wild: a plain string; a list of content parts; a
    *numpy array* of content parts, because pyarrow hands nested lists back
    that way; and a string holding the repr of any of those.

    The numpy case is why this is careful. `isinstance(x, list)` is False for
    an ndarray, so an earlier version fell through to `str(content)` and stored
    `"[{'type': 'text', 'text': 'the real prompt', ...}]"` -- every prompt in
    the corpus wrapped in its own repr, silently, with no parse error to notice.
    Sequence-ness is tested by iteration, not by exact type.
    """
    for message in conversation:
        if message.get("role") != "user":
            continue
        return _content_text(message.get("content"))
    return ""


def _content_text(content) -> str:
    """Pull the text out of one message's content, whatever it is wrapped in."""
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped.startswith(("[", "{")):
            return content
        try:
            content = ast.literal_eval(stripped)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, (list, tuple, np.ndarray)):
        return " ".join(_content_text(part) for part in content).strip()
    return str(content)


def _split_of(prompt: str) -> str:
    """Assign a split by hashing the prompt, not by shuffling rows.

    Hashing means a rebuild puts every prompt back where it was, so a model
    trained before the rebuild is still being scored on rows it never saw. A
    seeded shuffle does not survive the corpus growing by one row.
    """
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    position = int(digest[:8], 16) / 0xFFFFFFFF
    if position < SPLIT_FRACTIONS["train"]:
        return "train"
    if position < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]:
        return "val"
    return "test"


def build_length_dataset(*, max_rows: int | None = None) -> pd.DataFrame:
    """Read the arena dump and return one row per distinct prompt.

    Single-turn rows only: in a multi-turn conversation the response length is
    driven by the turn it answers, and the router sees the first prompt.
    """
    frames = []
    for path in sorted(glob.glob(ARENA_GLOB.replace("~", str(__import__("pathlib").Path.home())))):
        raw = pd.read_parquet(path, columns=["id", "conversation_a", "conv_metadata", "language"])
        meta = pd.json_normalize(raw["conv_metadata"])
        keep = (
            (meta["turns"] == 1)
            & (meta["sum_assistant_a_tokens"] > 0)
            & (meta["sum_assistant_b_tokens"] > 0)
        ).to_numpy()
        raw, meta = raw[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
        frames.append(pd.DataFrame({
            "arena_id": raw["id"],
            "prompt": [_first_user_text(c) for c in raw["conversation_a"]],
            "language": raw["language"],
            "tokens_a": meta["sum_assistant_a_tokens"].astype(int),
            "tokens_b": meta["sum_assistant_b_tokens"].astype(int),
        }))
    if not frames:
        raise FileNotFoundError(f"no arena parquet files under {ARENA_GLOB}")

    frame = pd.concat(frames, ignore_index=True)
    frame["prompt"] = frame["prompt"].fillna("").str.strip()
    frame = frame[frame["prompt"].str.len() >= MIN_PROMPT_CHARS]

    # Deduplicate on normalised text. The same prompt is put to the arena many
    # times; leaving those in would place near-identical rows on both sides of
    # the split and report memorisation as generalisation.
    normalised = frame["prompt"].str.lower().str.replace(r"\s+", " ", regex=True)
    frame = frame[~normalised.duplicated()].reset_index(drop=True)

    # Guard against the extractor failing quietly. A prompt that still looks
    # like serialised content means it met a shape it did not understand, and a
    # corpus of reprs trains and scores without ever raising.
    #
    # Rate-based, not absolute: a shape bug hits *every* row, while a user
    # pasting JSON whose first key is "type" is one row in a hundred thousand
    # and is a real prompt. The threshold is what separates those.
    looks_serialised = frame["prompt"].str.match(r"""^\s*\[\s*\{\s*['"]type['"]\s*:""")
    share = float(looks_serialised.mean())
    if share > SERIALISED_PROMPT_LIMIT:
        raise ValueError(
            f"{looks_serialised.sum()} of {len(frame)} prompts ({share:.1%}) still look "
            f"like serialised content, e.g. {frame['prompt'][looks_serialised].iloc[0][:120]!r}. "
            f"The content extractor has met a shape it does not handle."
        )
    if looks_serialised.any():
        log.info("%d prompts look like pasted JSON; kept as real prompts",
                 int(looks_serialised.sum()))

    frame["y"] = (np.log1p(frame["tokens_a"]) + np.log1p(frame["tokens_b"])) / 2.0
    frame["split"] = [_split_of(p) for p in frame["prompt"]]
    frame["uid"] = [hashlib.sha1(p.encode()).hexdigest()[:16] for p in frame["prompt"]]

    if max_rows is not None and max_rows < len(frame):
        # Sample within split so the proportions survive the cap, and keep the
        # hash assignment: a capped corpus is a subset of the full one, not a
        # different partition of it.
        rng = np.random.default_rng(settings().seed)
        share = max_rows / len(frame)
        keep: list[int] = []
        for positions in frame.groupby("split").indices.values():
            n = max(1, round(len(positions) * share))
            keep.extend(rng.choice(positions, size=n, replace=False).tolist())
        frame = frame.iloc[sorted(keep)].reset_index(drop=True)

    log.info("length corpus: %d prompts (%s)", len(frame),
             frame["split"].value_counts().to_dict())
    return frame


def dataset_dir():
    return settings().processed_dir / "length"


def save_length_dataset(frame: pd.DataFrame) -> None:
    out = dataset_dir()
    out.mkdir(parents=True, exist_ok=True)
    for split, part in frame.groupby("split"):
        part.reset_index(drop=True).to_parquet(out / f"{split}.parquet", index=False)


def load_length_splits() -> dict[str, pd.DataFrame]:
    out = dataset_dir()
    if not (out / "train.parquet").exists():
        raise FileNotFoundError(f"no length corpus at {out}; run build_length_dataset first")
    return {s: pd.read_parquet(out / f"{s}.parquet") for s in ("train", "val", "test")}

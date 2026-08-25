"""LMArena human preference battles -> the routing decision, supervised directly.

This is the alternative to the domain path. Instead of
``prompt -> domain -> capability -> model``, it learns ``prompt -> model tier``
from data where humans already made that call: each row is a head-to-head
between two models on a real user prompt, with a recorded winner.

The framing follows RouteLLM. For every battle between a *strong* and a *weak*
model, the label is whether the strong model was actually needed:

* strong won                -> ``strong_needed``
* weak won, tie, both bad   -> ``weak_sufficient``

Ties count as weak-sufficient on purpose: if a cheap model draws with an
expensive one, routing to the cheap one is the correct decision. That makes the
label answer the question the router actually asks -- "will spending more get a
better answer?" -- rather than "which model is better in general".

Tiers are *derived from the data*, not assumed. Hand-labelling 50+ models by
reputation would bake this author's priors into the ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from router.data.schema import Example
from router.data.sources.arena import REPO_ID, _first_user_text, _hardness_score
from router.data.taxonomy import task_type_from_arena_tags

log = logging.getLogger(__name__)

SOURCE = "lmarena_preference"

STRONG_NEEDED = "strong_needed"
WEAK_SUFFICIENT = "weak_sufficient"
ROUTING_LABELS = [WEAK_SUFFICIENT, STRONG_NEEDED]

_COLUMNS = [
    "id", "conversation_a", "conv_metadata", "category_tag",
    "language", "is_code", "model_a", "model_b", "winner",
]


@dataclass(slots=True)
class TierAssignment:
    """Which models count as strong, derived from observed win rates."""

    strong: set[str]
    weak: set[str]
    win_rates: pd.Series
    threshold: float

    def gap(self, model_a: str, model_b: str) -> float | None:
        """Absolute win-rate difference between two models, or None if untiered."""
        if model_a not in self.win_rates or model_b not in self.win_rates:
            return None
        return abs(float(self.win_rates[model_a]) - float(self.win_rates[model_b]))

    def tier_of(self, model: str) -> str | None:
        if model in self.strong:
            return "strong"
        if model in self.weak:
            return "weak"
        return None


def _battles_frame(shards: list[str], language: str | None) -> pd.DataFrame:
    frames = []
    for shard in shards:
        path = hf_hub_download(REPO_ID, shard, repo_type="dataset")
        frame = pq.read_table(path, columns=["model_a", "model_b", "winner", "language"]).to_pandas()
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if language is not None:
        frame = frame[frame["language"] == language]
    return frame


def compute_tiers(
    battles: pd.DataFrame,
    *,
    min_battles: int = 40,
    quantile: float = 0.5,
) -> TierAssignment:
    """Split models into strong/weak by observed win rate.

    A tie counts as half a win for both sides. ``min_battles`` drops models with
    too little evidence to place, and ``quantile`` sets where the cut falls --
    raising it makes "strong" a smaller, more elite group and shifts the class
    balance of the resulting labels.
    """
    records: list[tuple[str, float]] = []
    for model_a, model_b, winner in battles[["model_a", "model_b", "winner"]].itertuples(index=False):
        if winner == "model_a":
            records += [(model_a, 1.0), (model_b, 0.0)]
        elif winner == "model_b":
            records += [(model_a, 0.0), (model_b, 1.0)]
        else:
            records += [(model_a, 0.5), (model_b, 0.5)]

    stats = pd.DataFrame(records, columns=["model", "win"]).groupby("model")["win"].agg(["mean", "size"])
    eligible = stats[stats["size"] >= min_battles]
    if eligible.empty:
        raise ValueError(f"no model has >= {min_battles} battles; lower min_battles")

    threshold = float(eligible["mean"].quantile(quantile))
    win_rates = eligible["mean"].sort_values(ascending=False)
    strong = set(win_rates[win_rates >= threshold].index)
    weak = set(win_rates[win_rates < threshold].index)

    log.info(
        "tiers from %d models (>=%d battles): %d strong / %d weak at win-rate %.3f",
        len(eligible), min_battles, len(strong), len(weak), threshold,
    )
    return TierAssignment(strong=strong, weak=weak, win_rates=win_rates, threshold=threshold)


def load(
    *,
    max_shards: int | None = None,
    language: str | None = "en",
    min_battles: int = 40,
    tier_quantile: float = 0.5,
    min_tier_gap: float = 0.0,
    max_prompt_chars: int = 20_000,
) -> list[Example]:
    """Build routing-labelled examples from strong-vs-weak battles.

    Only *mixed-tier* battles are usable: a strong-vs-strong pairing says
    nothing about whether the cheap path would have sufficed.

    Args:
        min_tier_gap: Drop battles whose two models are closer than this in win
            rate. This matters more than it looks. The label mixes battles with
            very different strong/weak gaps -- in the full set, the widest-gap
            third is 57% ``strong_needed`` while the narrowest third is 42% --
            but the gap is a property of the *pairing*, which the classifier
            never sees. From its point of view that variance is pure label
            noise. Requiring a wide gap trades volume for a crisper label, which
            is why RouteLLM fixes a single strong/weak pair outright.
    """
    shards = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet"))
    if max_shards is not None:
        shards = shards[:max_shards]

    tiers = compute_tiers(
        _battles_frame(shards, language), min_battles=min_battles, quantile=tier_quantile
    )

    examples: list[Example] = []
    skipped_same_tier = 0
    skipped_unranked = 0
    skipped_narrow_gap = 0

    for shard in shards:
        path = hf_hub_download(REPO_ID, shard, repo_type="dataset")
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=_COLUMNS):
            for row in batch.to_pylist():
                if language is not None and row.get("language") != language:
                    continue

                tier_a = tiers.tier_of(row.get("model_a"))
                tier_b = tiers.tier_of(row.get("model_b"))
                if tier_a is None or tier_b is None:
                    skipped_unranked += 1
                    continue
                if tier_a == tier_b:
                    skipped_same_tier += 1
                    continue

                gap = tiers.gap(row.get("model_a"), row.get("model_b"))
                if min_tier_gap > 0.0 and (gap is None or gap < min_tier_gap):
                    skipped_narrow_gap += 1
                    continue

                prompt = _first_user_text(row.get("conversation_a"))
                if not prompt:
                    continue

                winner = row.get("winner")
                strong_side = "model_a" if tier_a == "strong" else "model_b"
                label = STRONG_NEEDED if winner == strong_side else WEAK_SUFFICIENT

                tag = row.get("category_tag") or {}
                meta = row.get("conv_metadata") or {}
                examples.append(
                    Example(
                        prompt=prompt[:max_prompt_chars],
                        source=SOURCE,
                        subset=shard.rsplit("/", 1)[-1],
                        routing_label=label,
                        task_type=task_type_from_arena_tags(
                            is_code=bool(row.get("is_code")),
                            is_math=bool((tag.get("math_v0.1") or {}).get("math")),
                            is_creative_writing=bool(
                                (tag.get("creative_writing_v0.1") or {}).get("creative_writing")
                            ),
                        ).value,
                        hardness_score=_hardness_score(tag),
                        meta={
                            "arena_id": row.get("id"),
                            "model_a": row.get("model_a"),
                            "model_b": row.get("model_b"),
                            "winner": winner,
                            "strong_model": row.get(strong_side),
                            "tier_gap": gap,
                            "sum_user_tokens": meta.get("sum_user_tokens"),
                            "turns": meta.get("turns"),
                        },
                    )
                )

    log.info(
        "preference: kept %d, skipped %d same-tier + %d unranked + %d narrow-gap",
        len(examples), skipped_same_tier, skipped_unranked, skipped_narrow_gap,
    )
    return examples

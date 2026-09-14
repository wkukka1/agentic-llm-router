"""Pick the anchor model for the anchor-topology judge pipeline (docs/anchor_judge.md).

Anchoring on the strongest model in the pool floors most judged gains and
compresses the signal the whole experiment depends on. This picks the
MEDIAN-quality model (by mean training accuracy) instead -- a computed,
reproducible mid-pool choice rather than a guess. Writes
``data/processed/anchor_judge/anchor_selection.json`` so a human can skim the
pick before copying it into ``configs/phase0.yaml``'s ``anchor_judge.anchor_model_id``.

Requires ``data/processed/nirt_observations.parquet`` to already exist
(``python scripts/data/build_nirt_dataset.py`` if not).

    python scripts/data/select_anchor_model.py
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info, write_json

ROUTERBENCH_11 = [
    "gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-v1", "claude-v2",
    "claude-instant-v1", "llama-2-70b-chat", "code-llama-34b-instruct",
    "mixtral-8x7b-instruct", "mistral-7b-instruct", "wizardlm-13b-v1.2", "yi-34b-chat",
]


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)

    from training.data.facade import load_training_data
    from evaluation.nirt.routing import train_quality

    d = load_training_data(cfg)
    quality = train_quality(d, ROUTERBENCH_11, metric="accuracy")
    missing = [m for m in ROUTERBENCH_11 if m not in quality]
    if missing:
        raise ValueError(f"no training observations for pool models: {missing}")

    # sort by (quality, model_id) -- stable, reproducible tiebreak even though
    # 11 is odd and a true tie at the median is unlikely.
    ranked = sorted(quality.items(), key=lambda kv: (kv[1], kv[0]))
    median_idx = len(ranked) // 2
    anchor = ranked[median_idx][0]

    payload = {
        "pool": ROUTERBENCH_11,
        "train_quality_accuracy": dict(ranked),
        "ranked_low_to_high": [m for m, _ in ranked],
        "anchor_model_id": anchor,
        "anchor_rank": median_idx + 1,
        "n_pool": len(ranked),
    }
    write_json("data/processed/anchor_judge/anchor_selection.json", payload, root=cfg.root)
    info(f"anchor = {anchor}  (rank {median_idx + 1} of {len(ranked)} by train accuracy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download the Chatbot Arena human-preference data.

Uses ``lmarena-ai/arena-human-preference-55k``
(https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k),
saved with ``datasets.save_to_disk`` into ``sources.arena.local_dir``.

Fields: id, model_a, model_b, prompt, response_a, response_b,
winner_model_a, winner_model_b, winner_tie. This release does NOT include
per-vote timestamps or language tags; those columns are therefore absent from
the normalized output (represented as missing, not fabricated).
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)
    dest = cfg.resolve(cfg.sources.arena.local_dir)
    repo = cfg.sources.arena.hf_repo

    if dest.exists():
        info(f"Arena data already present at {dest}")
        return 0

    from datasets import load_dataset

    info(f"downloading {repo}")
    ds = load_dataset(repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(dest))
    info(f"saved -> {dest}")

    # GPT-4 Judge battles (same schema, LLM-judged) -> separate pairwise source
    judge = cfg.get("sources.gpt4_judge")
    if judge is not None:
        jdest = cfg.resolve(judge.local_dir)
        if jdest.exists():
            info(f"GPT-4 Judge data already present at {jdest}")
        else:
            info(f"downloading {judge.hf_repo}")
            load_dataset(judge.hf_repo).save_to_disk(str(jdest))
            info(f"saved -> {jdest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

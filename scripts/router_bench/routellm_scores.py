"""RouteLLM's own decision inputs for the router bake-off. Runs in RouteLLM's venv (its pinned
dependencies clash with this repo's), never imported by the repo.

    <routellm venv>/python -I routellm_scores.py PROMPTS.jsonl OUT.json

Reads ``{"prompt_hash", "prompt"}`` rows. It writes the ``bert`` router's strong-model win rate per
prompt (``Controller.batch_calculate_win_rate``, the native API), the calibration thresholds for
strong shares of 0..100% from RouteLLM's public Arena data (``calibrate_threshold``'s procedure:
the ``1 - share`` quantile), and a check that ``Controller.route`` agrees with ``win_rate >= threshold``.
No prompt text is written.
"""

import json
import sys

import pandas as pd
from datasets import load_dataset
from routellm.controller import Controller


def main(src: str, out: str) -> int:
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    prompts = pd.Series([r["prompt"] for r in rows])
    controller = Controller(routers=["bert"], strong_model="claude-opus-5-5", weak_model="claude-haiku-5-5")
    rates = controller.batch_calculate_win_rate(prompts, "bert")
    arena = load_dataset("routellm/lmsys-arena-human-preference-55k-thresholds", split="train").to_pandas()
    thresholds = {f"{pct:.2f}": float(arena["bert"].quantile(q=1 - pct)) for pct in [i / 100 for i in range(101)]}
    checks = []
    for prompt, rate in list(zip(prompts, rates))[:5]:
        for t in (0.0, float(rate), 1.0):
            routed = controller.route(prompt, "bert", t)
            checks.append(routed == ("claude-opus-5-5" if rate >= t else "claude-haiku-5-5"))
    json.dump({"win_rate": {r["prompt_hash"]: float(x) for r, x in zip(rows, rates)},
               "arena_thresholds": thresholds, "route_agrees": all(checks),
               "checkpoint": "routellm/bert_gpt4_augmented"}, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

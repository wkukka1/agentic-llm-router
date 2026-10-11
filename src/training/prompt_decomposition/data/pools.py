"""Model pools: prompts x models correctness, from any source, plugged in by name.

A routing analysis is only as current as its model pool. RouterBench is 2023
(GPT-4-1106 against Llama-2), and a conclusion drawn there -- "one model wins
almost every category" -- may be an artefact of one model being far ahead of
the field at that moment rather than a fact about routing. The only way to know
is to run the same analysis on a newer pool, which means the pool has to be a
parameter rather than a hard-coded loader.

So a pool is a name mapped to a function returning a :class:`CorrectnessTable`.
Adding one is a `register_pool` call::

    register_pool("my_eval", lambda: CorrectnessTable(prompts, correct, cost, models))

Two ship here:

* ``routerbench`` -- 36,497 prompts, 11 models (2023), with per-call cost.
* ``sprout``      -- 6,637 prompts, 13 models (2024-25: Claude 3.5 Sonnet,
  GPT-4o, Llama-3.3-70B, Granite, Mixtral). Carries per-call token counts, so
  cost follows once a price table is supplied.

**Cost is optional and never invented.** A pool whose source has no prices gets
``NaN`` costs, and the analyses that need cost say so rather than quietly
ranking models by a number somebody guessed. Supply prices explicitly, or point
at the project's own table (``configs/model_prices.yaml``, which
``training.data.pricing`` already parses).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROUTERBENCH_META = ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")
SPROUT_META = ("key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer")


@dataclass(slots=True)
class CorrectnessTable:
    """Prompts x models: correctness in [0, 1], and the cost of each call.

    ``cost`` may be all-NaN when the source does not price its models; callers
    that need cost check :attr:`has_cost` rather than discovering it in a plot.
    """

    prompts: pd.DataFrame          # one row per prompt; at least `prompt`, `split`
    correct: np.ndarray            # (n_prompts, n_models)
    cost: np.ndarray               # (n_prompts, n_models), dollars per call
    models: list[str] = field(default_factory=list)
    name: str = "pool"

    def __post_init__(self) -> None:
        if self.correct.shape != self.cost.shape or self.correct.shape != (
                len(self.prompts), len(self.models)):
            raise ValueError(
                f"correct {self.correct.shape}, cost {self.cost.shape}, "
                f"{len(self.prompts)} prompts x {len(self.models)} models -- must agree")

    @property
    def has_cost(self) -> bool:
        return bool(np.isfinite(self.cost).all())

    def subset(self, mask: np.ndarray) -> CorrectnessTable:
        return CorrectnessTable(self.prompts[mask].reset_index(drop=True),
                                self.correct[mask], self.cost[mask], list(self.models), self.name)


def hashed_split(keys, train_share: float = 0.7) -> list[str]:
    """Split by hashing the key, so a rebuild never moves a prompt across it."""
    import hashlib

    return ["train" if int(hashlib.sha1(str(k).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            < train_share else "test" for k in keys]


def benchmark_family(eval_name: str) -> str:
    """`mmlu-professional-law` -> `mmlu`; the tiny Chinese-language sets ->
    `chinese`; everything else keeps its name.

    Not a split on the first hyphen, which turns `grade-school-math` into
    `grade` and `arc-challenge` into `arc`.
    """
    name = str(eval_name).split(".")[0]
    for prefix in ("mmlu", "mtbench"):
        if name.startswith(prefix):
            return prefix
    if name.lower().startswith("chinese"):
        return "chinese"
    return name


def load_routerbench(path: str | Path = "routerbench/routerbench_0shot.pkl") -> CorrectnessTable:
    """RouterBench's wide table. 2023 pool, ships its own per-call cost.

    Rows missing any model's score are dropped rather than imputed: an imputed
    correctness is a guess about exactly the quantity being measured.
    """
    raw = pd.read_pickle(path)
    models = [c for c in raw.columns if "|" not in c and c not in ROUTERBENCH_META]
    cost_cols = [f"{m}|total_cost" for m in models]
    complete = raw[models + cost_cols].notna().all(axis=1)
    raw = raw[complete].reset_index(drop=True)
    prompts = pd.DataFrame({
        "sample_id": raw["sample_id"].astype(str),
        "prompt": raw["prompt"].astype(str),
        "eval_name": raw["eval_name"].astype(str),
        "benchmark": raw["eval_name"].map(benchmark_family),
    })
    prompts["split"] = hashed_split(prompts["sample_id"])
    log.info("routerbench: %d prompts x %d models (%d incomplete rows dropped)",
             len(prompts), len(models), int((~complete).sum()))
    return CorrectnessTable(prompts, raw[models].to_numpy(float),
                            raw[cost_cols].to_numpy(float), models, name="routerbench")


def load_sprout(split: str = "test", prices: dict[str, tuple[float, float]] | None = None
                ) -> CorrectnessTable:
    """CARROT/SPROUT. 2024-25 pool, judge scores plus per-call token counts.

    ``prices`` maps model -> (USD per 1M input tokens, USD per 1M output
    tokens). Without it the cost column is NaN: the source prices nothing, and
    a made-up price would produce a cost-quality frontier that looks
    authoritative and means nothing.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("CARROT-LLM-Routing/SPROUT",
                           f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
    raw = pd.read_parquet(path)
    models = [c for c in raw.columns if c not in SPROUT_META]

    correct = np.column_stack([[cell["score"] for cell in raw[m]] for m in models]).astype(float)
    cost = np.full(correct.shape, np.nan)
    for j, model in enumerate(models):
        rate = (prices or {}).get(model)
        if rate is None:
            continue
        per_in, per_out = rate
        cost[:, j] = [c["num_input_tokens"] * per_in / 1e6 + c["num_output_tokens"] * per_out / 1e6
                      for c in raw[model]]
    if prices is None:
        log.info("sprout: no price table given, cost is NaN "
                 "(token counts are in the source; pass `prices=` to enable cost analysis)")

    prompts = pd.DataFrame({
        "sample_id": raw["key"].astype(str),
        "prompt": raw["prompt"].astype(str),
        "eval_name": raw["dataset"].astype(str),
        # `lighteval/MATH/all/test` -> `MATH`, `TIGER-Lab/MMLU-Pro` -> `MMLU-Pro`
        "benchmark": raw["dataset"].astype(str).str.split("/").str[1],
    })
    prompts["split"] = hashed_split(prompts["sample_id"])
    log.info("sprout[%s]: %d prompts x %d models", split, len(prompts), len(models))
    return CorrectnessTable(prompts, correct, cost, models, name=f"sprout-{split}")


POOLS: dict[str, Callable[..., CorrectnessTable]] = {
    "routerbench": load_routerbench,
    "sprout": load_sprout,
}


def register_pool(name: str, loader: Callable[..., CorrectnessTable]) -> None:
    """Add a pool. This is the whole extension point for a new model set."""
    if name in POOLS:
        raise ValueError(f"pool {name!r} already registered")
    POOLS[name] = loader


def load_pool(name: str, **kwargs) -> CorrectnessTable:
    if name not in POOLS:
        raise KeyError(f"unknown pool {name!r}; registered: {sorted(POOLS)}")
    return POOLS[name](**kwargs)


def load_pool_from_frame(frame: pd.DataFrame, models: list[str], *,
                         cost_columns: list[str] | None = None,
                         category: str = "benchmark", name: str = "custom") -> CorrectnessTable:
    """Wrap an arbitrary wide table -- one row per prompt, one column per model.

    The path for a pool this project evaluates itself: score N models on M
    prompts, put the scores in columns named after the models, and this reads
    it. No loader to write.
    """
    missing = [m for m in models if m not in frame]
    if missing:
        raise KeyError(f"columns missing for models: {missing}")
    prompts = frame[[c for c in ("sample_id", "prompt", "eval_name", category) if c in frame]].copy()
    if "sample_id" not in prompts:
        prompts["sample_id"] = [str(i) for i in range(len(frame))]
    if category not in prompts:
        raise KeyError(f"no category column {category!r} to group by")
    prompts["split"] = hashed_split(prompts["sample_id"])
    cost = (frame[cost_columns].to_numpy(float) if cost_columns
            else np.full((len(frame), len(models)), np.nan))
    return CorrectnessTable(prompts, frame[models].to_numpy(float), cost, list(models), name=name)


def prices_from_config(path: str | Path = "configs/model_prices.yaml",
                       ) -> dict[str, tuple[float, float]]:
    """Read the project's price table into ``model -> (in, out)`` per 1M tokens.

    Uses the repo's own snapshot rather than a second copy of the numbers. Its
    header is explicit that the prices are approximate and must be re-verified
    before anything published leans on them; that warning travels with them.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, tuple[float, float]] = {}
    for model, entry in (raw.get("prices") or {}).items():
        if isinstance(entry, dict) and "input_per_1m" in entry and "output_per_1m" in entry:
            out[model] = (float(entry["input_per_1m"]), float(entry["output_per_1m"]))
    log.info("prices: %d models from %s (snapshot %s)", len(out), path, raw.get("snapshot_date"))
    return out


def summarise_pool(table: CorrectnessTable) -> str:
    """One line per model: accuracy, and cost when the pool has it."""
    lines = [f"{table.name}: {len(table.prompts)} prompts x {len(table.models)} models, "
             f"cost {'present' if table.has_cost else 'NOT priced'}"]
    order = np.argsort(-table.correct.mean(axis=0))
    for j in order:
        cost = f"${table.cost[:, j].mean():.6f}" if np.isfinite(table.cost[:, j]).all() else "  --"
        lines.append(f"  {table.models[j]:48} {table.correct[:, j].mean():.3f}  {cost}")
    return "\n".join(lines)


def write_pool_json(table: CorrectnessTable, path: str | Path) -> Path:
    """Persist the per-model summary, so a pool's numbers outlive the run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pool": table.name, "n_prompts": len(table.prompts), "models": table.models,
        "accuracy": dict(zip(table.models, table.correct.mean(axis=0).round(4).tolist(),
                             strict=True)),
        "has_cost": table.has_cost,
    }, indent=2), encoding="utf-8")
    return path

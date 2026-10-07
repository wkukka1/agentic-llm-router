"""Prompt-decomposition training-side repros. Run from the repo root.

(0) build_real_only_dataset / build_task_dataset are broken as shipped (ImportError).
Sections 1-3 need them working, so they shim the two defects IN MEMORY ONLY:
  * alias the stale `decompose.classifiers.prompt_decomposition.sources` to the real module
  * re-add `load_handlabelled` (deleted in ea4d209) as a plain read of HANDLABELLED_PATH
"""
import logging, sys, tempfile, warnings
from pathlib import Path
import pandas as pd
warnings.simplefilter("ignore"); logging.disable(logging.CRITICAL)

from training.prompt_decomposition.data import dataset, sources
import decompose.classifiers.prompt_decomposition as dcp

# (0) as shipped
for fn in (dataset.build_real_only_dataset, dataset.build_task_dataset):
    try:
        fn(out_dir=Path(tempfile.mkdtemp())); print("(0)", fn.__name__, "OK")
    except Exception as e:
        print("(0)", fn.__name__, "->", type(e).__name__ + ":", str(e)[:100])

# shims
dcp.sources = sources
sys.modules["decompose.classifiers.prompt_decomposition.sources"] = sources
def load_handlabelled(path=sources.HANDLABELLED_PATH):
    f = pd.read_parquet(path)
    return [dataset.Example(prompt=r.prompt, source="handlabelled", capability=r.domain) for r in f.itertuples()]
sources.load_handlabelled = load_handlabelled

# (1) frozen eval vs the pool actually used
frozen = pd.read_parquet(dataset.FROZEN_EVAL_PATH)
hand = dataset.dedupe(dataset.to_frame(sources.load_handlabelled()))
key = lambda s: s.map(dataset.normalize_prompt)
fn_ = frozen.assign(_k=key(frozen["prompt"])).drop_duplicates("_k")
hn_ = hand.assign(_k=key(hand["prompt"]))
m = fn_.merge(hn_[["_k", "domain"]], on="_k", how="left", suffixes=("_frozen", "_pool"))
print(f"(1) frozen rows {len(fn_)} | in hand pool {int(m['domain_pool'].notna().sum())} "
      f"| missing {int(m['domain_pool'].isna().sum())}")
if "domain" in frozen.columns:
    both = m.dropna(subset=["domain_pool"])
    print("    label disagreements frozen vs pool:", int((both["domain_frozen"] != both["domain_pool"]).sum()))
sp = dataset.build_real_only_dataset(out_dir=Path(tempfile.mkdtemp()))
n_eval = int((sp["test"]["source"] == "handlabelled_eval").sum())
print(f"    test rows tagged handlabelled_eval: {n_eval} (frozen set size {len(fn_)})")

# (2) run_all with every config failing
from training.prompt_decomposition.config import ExperimentConfig, ModelConfig, DataConfig
from training.prompt_decomposition.experiment import run_all
out = Path(tempfile.mkdtemp())
run_all([ExperimentConfig(name="bad", model=ModelConfig("nope"), data=DataConfig(variant="does_not_exist"))], out_dir=out)
print("(2) run_all, one failing config -> files written:", sorted(p.name for p in out.iterdir()) or "NONE")

# (3) missing synthetic file: documented as warn-and-continue
try:
    sources.load_synthetic_tasks(["not_a_task"]); print("(3) missing synthetic file: warned and continued")
except Exception as e:
    print("(3) missing synthetic file raised", type(e).__name__)

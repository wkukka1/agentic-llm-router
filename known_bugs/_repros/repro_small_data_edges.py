"""Small edge-case repros: stratify (graded unanimity), chance correction (n_choices<2), render_prompt."""
import warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")

# (a) stratify: a row where every model scores 0.5 is unanimous, so it has no routing signal
from evaluation.data.stratify import sample_stratified_queries
R = pd.DataFrame({"m1": [0.5, 1.0, 0.0, 1.0], "m2": [0.5, 1.0, 0.0, 0.0], "m3": [0.5, 1.0, 0.0, 1.0]},
                 index=["unanimous_half", "all_right", "all_wrong", "mixed"])
s = sample_stratified_queries(R, n_discriminative=10, n_natural=0, seed=0)
print("(a) discriminative stratum:", sorted(s.index[s.stratum == "discriminative"]),
      "-> 'unanimous_half' has row_std", float(R.loc['unanimous_half'].std()))

# (b) chance correction with n_choices in {1, 0}
from router.config import load_config
from training.data.chance_correction import apply_chance_correction
cfg = load_config()
df = pd.DataFrame({"score": [1.0, 0.0, 1.0], "is_multiple_choice": [True] * 3,
                   "n_choices": pd.array([1, 0, 4], dtype="Int64"), "dataset": ["x"] * 3})
with np.errstate(all="ignore"):
    out = apply_chance_correction(df, cfg)
print("(b) corrected_score for n_choices=[1,0,4]:", out["corrected_score"].tolist(),
      "| chance_corrected:", out["chance_corrected"].tolist())

# (c) a prompt that is itself a list of strings
from training.data.normalize import render_prompt
p = "['b', 'a', 'c']"
print("(c) render_prompt(%r) ->" % p, repr(render_prompt(p)))

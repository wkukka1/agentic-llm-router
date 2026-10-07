"""fill_costs: priced models with NO token counts get cost 0.0 instead of NaN."""
import numpy as np
import pandas as pd
from training.data.pricing import fill_costs, load_prices
from router.config import load_config

cfg = load_config()
prices = load_prices(cfg)
mid = next(iter(k for k in prices))
df = pd.DataFrame({
    "source": ["lm_harness"] * 3, "model_id": [mid] * 3,
    "cost": [np.nan] * 3,
    "input_tokens": [np.nan, 100.0, np.nan],
    "output_tokens": [np.nan, np.nan, 50.0],
})
print(f"model {mid!r}: in={prices[mid]['input_per_1m']}/1M out={prices[mid]['output_per_1m']}/1M")
print(fill_costs(df, cfg)[["input_tokens", "output_tokens", "cost"]])

# and on the real table
r = pd.read_parquet(cfg.path("processed") / "responses.parquet",
                    columns=["source", "model_id", "cost", "input_tokens", "output_tokens"])
nrb = r[r["source"].astype(str) != "routerbench"]
no_tok = nrb["input_tokens"].isna() & nrb["output_tokens"].isna()
print("\nnon-routerbench rows:", len(nrb))
print("  with NO token counts:", int(no_tok.sum()),
      "| of those cost==0:", int((nrb.loc[no_tok, "cost"] == 0).sum()),
      "| cost NaN:", int(nrb.loc[no_tok, "cost"].isna().sum()))
print("  only one of in/out tokens known:", int((nrb["input_tokens"].isna() ^ nrb["output_tokens"].isna()).sum()))
print(nrb.assign(no_tok=no_tok).groupby(["source"]).agg(rows=("cost", "size"), no_tok=("no_tok", "sum"),
      cost_zero=("cost", lambda s: int((s == 0).sum())), cost_nan=("cost", lambda s: int(s.isna().sum()))))

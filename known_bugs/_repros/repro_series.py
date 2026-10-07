import sys
sys.path.insert(0, "src")
import pandas as pd
from router.routing.base import _resolve_costs
ids = ["big", "small"]
by_label = pd.Series({"small": 0.001, "big": 0.010})   # labelled, different order
print("dict  :", _resolve_costs({"small": 0.001, "big": 0.010}, ids))
print("series:", _resolve_costs(by_label, ids), "<- expected [0.01, 0.001]")

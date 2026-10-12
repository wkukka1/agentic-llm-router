"""Repro: kNN router with query ids absent from the store. Run from repo root."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from router.nirt.baselines_infer import knn_router_matrix

class Store:
    def __init__(self, ids, dim=4):
        self._index = {q: i for i, q in enumerate(ids)}
        self._m = np.random.default_rng(0).normal(size=(len(ids), dim))
    def gather(self, ids):
        return self._m[[self._index[q] for q in ids]] if len(ids) else np.zeros((0, self._m.shape[1]))

class Data:
    def __init__(self):
        rows = [(q, m, float((i + j) % 2), "train") for i, q in enumerate(["q0","q1","q2","q3","q4","q5"])
                for j, m in enumerate(["a", "b"])]
        self._obs = pd.DataFrame(rows, columns=["query_id", "model_id", "target", "split"])
        self._store = Store(["q0","q1","q2","q3","q4","q5"])
    def nirt_observations(self): return self._obs
    def query_embeddings(self, pathway): return self._store

d = Data()
print("known ids  :", knn_router_matrix(d, ["q0"], k=2).shape)
try:
    print("unknown id :", knn_router_matrix(d, ["not-in-store"], k=2).shape)
except Exception as e:
    print("unknown id :", type(e).__name__, "-", str(e).splitlines()[0])

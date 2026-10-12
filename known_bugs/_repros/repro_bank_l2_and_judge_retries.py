"""(1) QueryBank 'similarity' weighting on a flat_l2 index; (2) JudgeClient(max_retries=0);
(3) held-out OOD families inside the taxonomy clusters."""
import numpy as np, faiss

# (1) two stored vectors: 'near' is identical to the query, 'far' is distant
from router.embeddings.encoder import EmbeddingStore
from training.retrieval.query_bank import QueryBank
X = np.array([[1, 0], [10, 0]], np.float32)          # near, far
idx = faiss.IndexFlatL2(2); idx.add(X)
bank = QueryBank(idx, ["near", "far"], {"normalized": False, "index_type": "flat_l2", "default_k": 2})
store = EmbeddingStore(["near", "far"], np.array([[1, 0], [0, 1]], np.float32), {"id_field": "query_id"})
vec, found = bank.neighbor_weighted_mean(np.array([[1, 0]], np.float32), store, k=2, weighting="similarity")
print("(1) flat_l2 similarity-weighted mean =", vec[0].round(3), "(want ~[1,0] = the near neighbour; got weight on FAR)")

# (2) max_retries=0 means 'no retries', i.e. one attempt
from evaluation.judge.client import JudgeClient
calls = []
c = JudgeClient("dummy", "m", max_retries=0)
c._call_once = lambda m: calls.append(1) or '{"margin": 0}'
try:
    c._call_with_retry([])
except RuntimeError as e:
    print("(2) max_retries=0 ->", e, "| attempts actually made:", len(calls))

# (3) OOD families in the clusters
from router.config import load_config
from training.data.facade import load_training_data
from training.taxonomy.clustering import load_clusters
from evaluation.nirt.ood import family_of_query, DEFAULT_HOLDOUT
d = load_training_data(load_config())
fam = family_of_query(d)
cl = load_clusters(load_config())
in_ood = cl["query_id"].map(lambda q: fam.get(str(q)) in DEFAULT_HOLDOUT)
print(f"(3) clustered queries in held-out families {DEFAULT_HOLDOUT}: {int(in_ood.sum())} of {len(cl)}")

"""Do the taxonomy clusters / centroids include validation, test and OOD queries?"""
from router.config import load_config
from training.taxonomy.clustering import load_clusters, load_meta
from training.data.splits import load_splits

cfg = load_config()
cl = load_clusters(cfg)
sp = load_splits(cfg)
ids = set(cl["query_id"])
print("clustered queries:", len(ids), "| query_source:", load_meta(cfg).get("query_source"))
for name in ("train", "validation", "test", "ood"):
    e = sp.get(name)
    if not e:
        continue
    s = set(e["query_ids"])
    print(f"  {name:10s} split={len(s):7d}  clustered={len(ids & s):7d}")

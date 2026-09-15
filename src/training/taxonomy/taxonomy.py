"""Readable labels for the ability-taxonomy clusters -- no LLM.

Per cluster: dominant benchmark ``family`` (``profiles.task_families``), top
TF-IDF ``top_terms``, ``top_datasets``, and ``label = "<family>: t, t, t"``. A
candidate taxonomy; an opt-in LLM pass (``labeling`` config) can refine it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from router.config import Config
from ..data.families import family_of as _family_of
from ..data.families import task_family_map as _family_map
from .clustering import load_clusters, load_meta

TAXONOMY_FILE = "taxonomy.json"


def _top_terms(texts: list, k: int = 8) -> list:
    from sklearn.feature_extraction.text import TfidfVectorizer

    tf = TfidfVectorizer(stop_words="english", max_features=4000, min_df=2,
                         token_pattern=r"[a-z][a-z0-9\-']+")
    try:
        m = tf.fit_transform(texts)
    except ValueError:            # empty vocabulary after pruning
        return []
    vocab = tf.get_feature_names_out()
    return [str(vocab[i]) for i in m.mean(axis=0).A1.argsort()[::-1][:k]]


def build_taxonomy(cfg: Config, *, save: bool = True, max_texts_per_cluster: int = 400) -> dict:
    import pandas as pd

    meta = load_meta(cfg)
    df = load_clusters(cfg).merge(
        pd.read_parquet(cfg.path("processed") / "queries.parquet",
                        columns=["query_id", "query", "dataset"]),
        on="query_id", how="left")
    fam_map = _family_map(cfg)

    entries = []
    for cid, g in df[df["cluster_id"] >= 0].groupby("cluster_id"):
        texts = g["query"].dropna().astype(str).tolist()[:max_texts_per_cluster]
        fams = Counter(f for f in (_family_of(d, fam_map) for d in g["dataset"].fillna("")) if f)
        family = fams.most_common(1)[0][0] if fams else "mixed"
        terms = _top_terms(texts) if len(texts) >= 2 else []
        entries.append({
            "cluster_id": int(cid), "size": int(len(g)), "family": family,
            "family_purity": round(fams.most_common(1)[0][1] / len(g), 3) if fams else 0.0,
            "top_terms": terms,
            "top_datasets": [d for d, _ in Counter(g["dataset"].fillna("unknown")).most_common(5)],
            "label": (f"{family}: " + ", ".join(terms[:3])) if terms else family,
        })

    taxonomy = {
        "version": int(cfg.get("taxonomy.version", 1)), "n_clusters": len(entries),
        "noise_fraction": meta.get("noise_fraction"), "pathway": meta.get("pathway"),
        "clusters": sorted(entries, key=lambda e: -e["size"]),
    }
    if save:
        out = cfg.path("taxonomy")
        out.mkdir(parents=True, exist_ok=True)
        (out / TAXONOMY_FILE).write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    return taxonomy


def load_taxonomy(cfg: Config) -> Optional[dict]:
    p: Path = cfg.path("taxonomy") / TAXONOMY_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

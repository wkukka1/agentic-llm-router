"""Deterministic train / validation / test / cold-start splits.

Leakage rules
-------------
* Splitting is done over **query groups**, where a group is all query_ids that
  share a normalized-content hash (:func:`router.data.normalize.content_hash`).
  A group is assigned wholesale to one split, so semantically identical queries
  from different source datasets can never straddle train/test.
* Assignment is by hashing the group key with the configured seed -- fully
  deterministic and independent of row order or dataset size.

Cold-start models
-----------------
A fraction of models (``split.cold_start_model_fraction``) is held out entirely.
Their observations appear in NONE of train/val/test; they exist only in
``cold_start_models.json`` for evaluating profile-based ``theta_m``
initialization later. The held-out set is chosen deterministically from the
seed as well.

Outputs (under ``paths.splits``): train.json, validation.json, test.json,
cold_start_models.json -- each a JSON object with explicit id lists and the
parameters used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from router.config import Config


def _bucket(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def make_splits(responses: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    seed = int(cfg.seed)
    sp = cfg.split
    f_train = float(sp.train_fraction)
    f_val = float(sp.validation_fraction)
    f_cold = float(sp.get("cold_start_model_fraction", 0.0) or 0.0)

    # -- cold-start models -------------------------------------------------
    # Only models with enough observations are eligible to be *held out* -- a
    # model with a handful of rows cannot be meaningfully cold-start-evaluated.
    # Every model below the threshold stays in the warm set.
    min_obs = int(sp.get("cold_start_min_observations", 0) or 0)
    counts = responses["model_id"].value_counts()
    models = sorted(responses["model_id"].dropna().unique().tolist())

    # Explicit overrides: forced ids skip the cold lottery entirely. Used by the
    # pool-expansion workstream to bring specific held-out models warm (or pin a
    # documented cold arm) without perturbing every other model's assignment.
    force_warm = {str(m) for m in (sp.get("force_warm") or [])}
    force_cold = {str(m) for m in (sp.get("force_cold") or [])}
    both = force_warm & force_cold
    if both:
        raise ValueError(f"models in both split.force_warm and split.force_cold: {sorted(both)}")

    # Stratify by source so every source keeps ~(1 - f_cold) of its models warm
    # (otherwise the hash lottery can gut a single dense source like RouterBench).
    # The lottery runs over the *unmodified* eligible set -- forced ids are then
    # masked in/out afterwards, so an override never shifts any other model's
    # assignment (it only frees or fills its own slot).
    cold_set: set[str] = set()
    for src, g in responses.groupby("source"):
        src_models = sorted(g["model_id"].dropna().unique().tolist())
        eligible = [m for m in src_models if counts.get(m, 0) >= min_obs]
        k = int(round(f_cold * len(eligible)))
        ranked = sorted(eligible, key=lambda m: _bucket(f"model::{m}", seed))
        cold_set.update(ranked[:k])

    cold_set -= force_warm
    cold_set |= {m for m in force_cold if m in set(models)}

    cold_models = sorted(cold_set)
    warm_models = [m for m in models if m not in cold_set]

    warm = responses[responses["model_id"].isin(warm_models)]

    # -- query groups by content hash -----------------------------------
    if "content_hash" in warm.columns:
        group_key = warm.groupby("query_id")["content_hash"].first()
    else:
        from .normalize import content_hash

        group_key = warm.groupby("query_id")["query"].first().map(content_hash)

    split_of_qid: dict[str, str] = {}
    for qid, gkey in group_key.items():
        b = _bucket(f"query::{gkey}", seed)
        if b < f_train:
            split_of_qid[qid] = "train"
        elif b < f_train + f_val:
            split_of_qid[qid] = "validation"
        else:
            split_of_qid[qid] = "test"

    buckets: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for qid, s in split_of_qid.items():
        buckets[s].append(qid)
    for s in buckets:
        buckets[s].sort()

    params = {
        "seed": seed,
        "method": "content-hash group split",
        "train_fraction": f_train,
        "validation_fraction": f_val,
        "test_fraction": float(sp.test_fraction),
        "cold_start_model_fraction": f_cold,
        "dedup_on": sp.get("dedup_on", "normalized_query"),
        "force_warm": sorted(force_warm),
        "force_cold": sorted(force_cold),
    }
    return {
        "train": {"query_ids": buckets["train"], "params": params},
        "validation": {"query_ids": buckets["validation"], "params": params},
        "test": {"query_ids": buckets["test"], "params": params},
        "cold_start_models": {
            "model_ids": cold_models,
            "warm_model_ids": warm_models,
            "params": params,
        },
    }


def make_presplit(responses: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Honour a source's pre-partitioned split instead of the hash lottery.

    Used by the IRT-Router suite (:mod:`router.data.loaders.load_irt_router`),
    which ships ``train.csv`` / ``test1.csv`` / ``test2.csv``. Each response row
    carries ``metadata['origin'] in {train, test1, test2}``:

    * ``train``  -> ``train`` + ``validation`` (split by content hash, using
      ``split.train_fraction`` / ``validation_fraction``),
    * ``test1``  -> ``test`` (in-distribution),
    * ``test2``  -> ``ood`` (out-of-distribution held-out datasets).

    All models are warm (``cold_start_models`` is empty).
    """
    seed = int(cfg.seed)
    sp = cfg.split
    f_train = float(sp.train_fraction)
    f_val = float(sp.get("validation_fraction", 0.0) or 0.0)
    denom = f_train + f_val or 1.0

    origin = responses["metadata"].map(
        lambda m: (m or {}).get("origin") if isinstance(m, dict) else None
    )
    if origin.isna().all():
        raise ValueError(
            "make_presplit: no response row carries metadata['origin']; this "
            "config's split.presplit is only valid for a pre-partitioned source."
        )

    if "content_hash" in responses.columns:
        chash = responses.groupby("query_id")["content_hash"].first()
    else:
        from .normalize import content_hash

        chash = responses.groupby("query_id")["query"].first().map(content_hash)
    origin_of = origin.groupby(responses["query_id"]).first()

    buckets: dict[str, list[str]] = {"train": [], "validation": [], "test": [], "ood": []}
    for qid, org in origin_of.items():
        if org == "test1":
            buckets["test"].append(qid)
        elif org == "test2":
            buckets["ood"].append(qid)
        else:  # train.csv -> train / validation
            b = _bucket(f"query::{chash.get(qid)}", seed) * denom
            buckets["validation" if b >= f_train else "train"].append(qid)
    for s in buckets:
        buckets[s].sort()

    models = sorted(responses["model_id"].dropna().unique().tolist())
    params = {
        "seed": seed,
        "method": "pre-partitioned source split (train.csv / test1.csv / test2.csv)",
        "train_fraction": f_train,
        "validation_fraction": f_val,
        "test_fraction": 0.0,
    }
    return {
        "train": {"query_ids": buckets["train"], "params": params},
        "validation": {"query_ids": buckets["validation"], "params": params},
        "test": {"query_ids": buckets["test"], "params": params},
        "ood": {"query_ids": buckets["ood"], "params": params},
        "cold_start_models": {
            "model_ids": [],
            "warm_model_ids": models,
            "params": params,
        },
    }


_SPLIT_FILES = {
    "train": "train.json",
    "validation": "validation.json",
    "test": "test.json",
    "ood": "ood.json",
    "cold_start_models": "cold_start_models.json",
}


def write_splits(splits: dict[str, Any], cfg: Config) -> dict[str, Path]:
    out_dir = cfg.path("splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    name_map = {k: v for k, v in _SPLIT_FILES.items() if k in splits}
    paths = {}
    for key, fname in name_map.items():
        p = out_dir / fname
        p.write_text(json.dumps(splits[key], indent=2), encoding="utf-8")
        paths[key] = p
    return paths


def load_splits(cfg: Config) -> dict[str, Any]:
    out_dir = cfg.path("splits")
    result: dict[str, Any] = {}
    for key, fname in _SPLIT_FILES.items():
        p = out_dir / fname
        if key == "ood" and not p.exists():
            continue  # ood is optional (only pre-partitioned sources have it)
        result[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return result


def check_leakage(splits: dict[str, Any]) -> list[str]:
    problems = []
    names = [k for k in ("train", "validation", "test", "ood")
             if isinstance(splits.get(k), dict)]
    sets = {k: set(splits[k]["query_ids"]) for k in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                problems.append(f"{len(inter)} query_ids in both {a} and {b}")
    return problems

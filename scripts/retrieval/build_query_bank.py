"""Build the FAISS query bank over the TRAIN query embeddings.

    python scripts/retrieval/build_query_bank.py
    python scripts/retrieval/build_query_bank.py --pathway retrieval --index-type flat_ip

    # leakage-safe bank for imputing OOD queries: drop the held-out families
    python scripts/retrieval/build_query_bank.py --exclude-ood-families

Writes indexes/query_bank/{index.faiss, ids.parquet, manifest.json} (or
indexes/query_bank__ood/ with --exclude-ood-families). Train queries only --
never rebuilt during Phase 1 / Phase 2 model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from router.config import load_config
from router.retrieval.query_bank import build_query_bank, query_bank_dir


def _ood_holdout_ids(cfg, nirt_config: str) -> tuple[list[str], list[str]]:
    """(query_ids to exclude, holdout family names) from configs/nirt.yaml."""
    from router.data.phase1 import load_phase1
    from router.nirt.ood import family_of_query, ood_families

    p = Path(nirt_config)
    p = p if p.is_absolute() else Path(cfg.root) / p
    nirt_cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    fams = list(ood_families(nirt_cfg))
    d = load_phase1(cfg)
    fam_of = family_of_query(d)
    ids = [q for q, f in fam_of.items() if f in set(fams)]
    return ids, fams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pathway", default=None)
    ap.add_argument("--index-type", default=None, choices=["flat_ip", "flat_l2"])
    ap.add_argument("--exclude-ood-families", action="store_true",
                    help="drop configs/nirt.yaml evaluation.ood_holdout_families from the "
                         "bank and write it to indexes/query_bank__ood/")
    ap.add_argument("--nirt-config", default="configs/nirt.yaml")
    ap.add_argument("--out-dir", default=None, help="override the bank output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)

    exclude_ids: list[str] = []
    fams = None
    out_dir = args.out_dir
    if args.exclude_ood_families:
        exclude_ids, fams = _ood_holdout_ids(cfg, args.nirt_config)
        if out_dir is None:
            out_dir = str(query_bank_dir(cfg).parent / "query_bank__ood")
        print(f"[query-bank] excluding {len(exclude_ids):,} queries from families {fams}")

    bank = build_query_bank(cfg, split=args.split, pathway=args.pathway,
                            index_type=args.index_type, exclude_query_ids=exclude_ids or None,
                            holdout_families=fams, out_dir=out_dir)
    where = out_dir or query_bank_dir(cfg)
    print(f"[query-bank] {len(bank):,} {args.split} queries indexed -> {where}")
    print(json.dumps(bank.manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

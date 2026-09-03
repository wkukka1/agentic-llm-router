"""Build LLM profiles -> data/processed/model_profiles.parquet.

Combines curated descriptions (configs/model_profiles.yaml) with an empirical
behaviour summary computed from data/processed/responses.parquet. Use
`profiles.include_empirical: false` in the config for description-only profiles.
"""

from __future__ import annotations

import sys

from router.cli import base_parser, get_config, info
from router.models.profiles import build_model_profiles, write_model_profiles


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--print", action="store_true", help="print each profile")
    args = parser.parse_args()
    cfg = get_config(args)

    df = build_model_profiles(cfg)
    path = write_model_profiles(df, cfg)
    info(f"wrote {len(df)} profiles -> {path}")
    info(f"curated: {int(df['has_curated'].sum())} / {len(df)}")

    if args.print:
        for _, r in df.sort_values("n_observations", ascending=False).iterrows():
            print(f"\n### {r['model_id']}  (curated={r['has_curated']}, n_obs={r['n_observations']})")
            print(r["profile_text"])
    else:
        top = df.sort_values("n_observations", ascending=False).head(3)
        for _, r in top.iterrows():
            print(f"\n### {r['model_id']}\n{r['profile_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

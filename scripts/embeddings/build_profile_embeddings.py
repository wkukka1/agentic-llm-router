"""Embed LLM profile texts (data/processed/model_profiles.parquet) -> store.

    python scripts/embeddings/build_profile_embeddings.py --pathway irt

This is the NIRT-paper model pathway: the same encoder that embeds queries
embeds each model's profile, giving a `model_id -> vector` map that Phase 1 uses
to initialise theta_m (and to place cold-start models from description alone).

Output: <embedding.cache_dir>/model_profile__<pathway>/.

The actual build logic is :func:`training.models.profiles.build_profile_embeddings`
-- this script is a thin CLI wrapper over it.
"""

from __future__ import annotations

import sys

from training.cli import base_parser, get_config, info
from router.embeddings import available_pathways
from training.models.profiles import build_profile_embeddings


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--pathway", default=None)
    parser.add_argument("--all-pathways", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--text-column", default="profile_text",
                        choices=["profile_text", "feature"],
                        help="'feature' = curated description only; "
                             "'profile_text' = description + empirical addendum")
    args = parser.parse_args()
    cfg = get_config(args)

    if args.all_pathways:
        pathways = available_pathways(cfg)
    else:
        pathways = [args.pathway or cfg.get("embedding.default_pathway")
                    or available_pathways(cfg)[0]]

    for pw in pathways:
        store = build_profile_embeddings(
            cfg, pw, restart=args.restart, text_column=args.text_column,
        )
        info(f"[{pw}] wrote {len(store)} profile embeddings (dim={store.dim}) ({args.text_column})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

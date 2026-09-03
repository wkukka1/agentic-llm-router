"""Embed LLM profile texts (data/processed/model_profiles.parquet) -> store.

    python scripts/embeddings/build_profile_embeddings.py --pathway irt

This is the NIRT-paper model pathway: the same encoder that embeds queries
embeds each model's profile, giving a `model_id -> vector` map that Phase 1 uses
to initialise theta_m (and to place cold-start models from description alone).

Output: <embedding.cache_dir>/model_profile__<pathway>/.
"""

from __future__ import annotations

import sys

from router.cli import base_parser, get_config, info
from router.embeddings import available_pathways, build_store, default_store_dir, load_encoder
from router.models.profiles import build_model_profiles, read_model_profiles


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

    try:
        profiles = read_model_profiles(cfg)
    except FileNotFoundError:
        info("model_profiles.parquet missing; building it now")
        from router.models.profiles import write_model_profiles

        profiles = build_model_profiles(cfg)
        write_model_profiles(profiles, cfg)

    profiles = profiles.sort_values("model_id").reset_index(drop=True)

    if args.all_pathways:
        pathways = available_pathways(cfg)
    else:
        pathways = [args.pathway or cfg.get("embedding.default_pathway")
                    or available_pathways(cfg)[0]]

    for pw in pathways:
        encoder = load_encoder(cfg, pathway=pw)
        out_dir = default_store_dir(cfg, "model_profile", pw)
        info(f"[{pw}] {encoder.cfg.backend}:{encoder.cfg.model_name} -> "
             f"{len(profiles)} profiles ({args.text_column})")
        store = build_store(
            out_dir, profiles["model_id"].tolist(),
            profiles[args.text_column].tolist(), encoder,
            id_field="model_id", resume=not args.restart, force=args.restart,
            manifest_extra={"text_column": args.text_column},
        )
        info(f"[{pw}] wrote {len(store)} profile embeddings (dim={store.dim}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

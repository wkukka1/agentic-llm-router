"""Obtain the RouterBench raw files.

RouterBench is published as the Hugging Face dataset ``withmartian/routerbench``
(https://huggingface.co/datasets/withmartian/routerbench). It ships three
pickled pandas DataFrames:

    routerbench_0shot.pkl   wide: sample_id, prompt, eval_name, <model> score
                            columns, <model>|model_response, <model>|total_cost,
                            oracle_model_to_route_to
    routerbench_5shot.pkl   same schema, 5-shot generations
    routerbench_raw.pkl     long form of the 0-shot table

This script snapshots the repo (git-LFS) into ``sources.routerbench.local_dir``
so the raw files are preserved verbatim. If you already cloned it manually,
just point the config at that directory.
"""

from __future__ import annotations

import subprocess
import sys

from router.cli import base_parser, get_config, info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = get_config(args)
    dest = cfg.resolve(cfg.sources.routerbench.local_dir)
    repo = cfg.sources.routerbench.hf_repo

    if (dest / "routerbench_0shot.pkl").exists():
        info(f"RouterBench already present at {dest}")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}"
    info(f"cloning {url} -> {dest}")
    try:
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        info(f"git clone failed ({exc}); falling back to `datasets` download")
        from datasets import load_dataset

        ds = load_dataset(repo)
        (dest).mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(dest / "hf_dataset"))
        info("saved HF dataset form; note the loader expects the .pkl files -- "
             "clone the repo with git-LFS for the canonical format.")
        return 1
    info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

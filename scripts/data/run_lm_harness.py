"""Run (or emit) lm-evaluation-harness for the pool-expansion E3 model set,
matched to the alignable RouterBench standard-benchmark subset.

Model list / task set / device come from ``configs/phase0.yaml -> sources.lm_harness``.
Per-model ``load`` (``fp16`` | ``4bit``) picks the HF loader args for a small GPU.

    python scripts/data/run_lm_harness.py                       # print commands
    python scripts/data/run_lm_harness.py --execute              # run them (needs a GPU env)
    python scripts/data/run_lm_harness.py --model qwen2.5-3b-instruct --execute --limit 20

The ingestion pipeline consumes any ``*samples*.jsonl`` under
``sources.lm_harness.results_dir`` (this script always passes ``--log_samples``).
"""

from __future__ import annotations

import shlex
import subprocess
import sys

from training.cli import base_parser, get_config, info

DEFAULT_TASKS = ["mmlu", "hellaswag", "winogrande", "arc_challenge"]


def _models(cfg) -> list[dict]:
    raw = cfg.get("sources.lm_harness.models") or []
    return [m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in raw]


def _model_args(hf_repo: str, revision: str, load: str) -> str:
    parts = [f"pretrained={hf_repo}", f"revision={revision}", "trust_remote_code=True"]
    if load == "4bit":
        parts += ["load_in_4bit=True", "bnb_4bit_compute_dtype=bfloat16"]
    else:
        parts += ["dtype=float16"]
    return ",".join(parts)


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--model", default=None, help="restrict to one config model id")
    parser.add_argument("--tasks", default=None, help="override config task list (comma sep)")
    parser.add_argument("--limit", type=int, default=None, help="cap docs per task (smoke)")
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--device", default=None, help="override sources.lm_harness.device")
    parser.add_argument("--execute", action="store_true", help="actually run lm_eval")
    args = parser.parse_args()
    cfg = get_config(args)

    out_root = cfg.resolve(cfg.sources.lm_harness.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    models = _models(cfg)
    if args.model:
        models = [m for m in models if m.get("id") == args.model]
        if not models:
            parser.error(f"model {args.model!r} not in sources.lm_harness.models")
    if not models:
        parser.error("no models in sources.lm_harness.models")

    tasks = args.tasks or ",".join(cfg.get("sources.lm_harness.tasks") or DEFAULT_TASKS)
    device = args.device or cfg.get("sources.lm_harness.device") or "cuda:0"

    info(f"{len(models)} model(s) | tasks={tasks} | device={device} | batch={args.batch_size}")
    print()

    rc = 0
    for m in models:
        mid, repo, rev = m["id"], m["hf_repo"], m.get("revision", "main")
        load = m.get("load", "fp16")
        out_dir = out_root / mid
        cmd = [
            "lm_eval", "--model", "hf",
            "--model_args", _model_args(repo, rev, load),
            "--tasks", tasks,
            "--device", device,
            "--batch_size", str(args.batch_size),
            "--log_samples",
            "--output_path", str(out_dir),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]

        print(f"# {mid}  ({repo}@{rev}, {load})")
        print(" ".join(shlex.quote(c) for c in cmd))
        print()

        if args.execute:
            out_dir.mkdir(parents=True, exist_ok=True)
            info(f"running {mid} ...")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                info(f"  {mid} FAILED (exit {r.returncode})")
                rc = r.returncode
            else:
                info(f"  {mid} done -> {out_dir}")

    if not args.execute:
        info("nothing executed (pass --execute). After logs land:")
        info("  python scripts/data/build_response_matrix.py")
    return rc


if __name__ == "__main__":
    sys.exit(main())

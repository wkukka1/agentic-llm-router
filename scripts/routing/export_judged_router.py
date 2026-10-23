"""Fine-tune projected NIRT runs on judged gateway prompts and export them as ONE
gateway router file over the Claude tiers.

Each run (a seed) is fine-tuned on every judged prompt (``training.nirt.judge_finetune``,
the ``full`` variant at a high lr: the best out-of-fold in
``scripts/nirt/judge_finetune_cv.py``), folded over the tiers' own profiles with its
learned per-tier offsets, and the seeds are combined by averaging their logits --
exact in the ``mlp`` format (``ensemble_fold``). The file is checked against the
fine-tuned models in memory and again after reloading through the gateway's loader,
and the fine-tuned checkpoints are kept under ``<runs_dir>/<run>-judged/``.

    python scripts/routing/export_judged_router.py \\
        --graded ~/.cache/aar/routing-pool/multiturn/graded.jsonl \\
        --runs pm-ft-d1024-s42,pm-ft-d1024-s43,pm-ft-d1024-s44 \\
        --out aar-gateway/src/aar_gateway/routers/manifold-nirt-claude-judged.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from evaluation.gateway.pool_eval import TIERS, load_prompts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graded", action="append", required=True)
    ap.add_argument("--runs", default="pm-ft-d1024-s42,pm-ft-d1024-s43,pm-ft-d1024-s44")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch

    from router.config import DEFAULT_NIRT_RUNS_DIR, REPO_ROOT, load_config
    from router.embeddings.encoder import load_encoder
    from router.models.artifacts import MLPArtifactPayload
    from router.routing.routers import NIRTRouter, _encode_texts
    from training.data.facade import load_training_data
    from training.nirt.judge_finetune import fit_judged, predict_judged
    from training.trainers.nirt_router_export import (
        check_foldable,
        check_servable_encoder,
        ensemble_fold,
        fold_nirt,
        gateway_encoder_spec,
        parity_embeddings,
        parity_max_diff,
        parity_max_diff_file,
    )

    seen, prompts = set(), []
    for g in args.graded:
        for p in load_prompts(Path(g).expanduser()):
            if p["prompt_hash"] not in seen:
                seen.add(p["prompt_hash"])
                prompts.append(p)
    texts = [p["route_text"] for p in prompts]
    S = np.stack([p["score"] for p in prompts])
    print(f"{len(prompts)} judged prompts; fine-tuning {args.variant} at lr {args.lr:g} for {args.steps} steps")

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    states, fits, encoder, pathway, parity_q = [], [], None, None, None
    data = load_training_data(load_config())   # the full view: parity needs the split lists
    runs_dir = REPO_ROOT / DEFAULT_NIRT_RUNS_DIR
    for i, run in enumerate(runs):
        router = NIRTRouter.from_run(run, data=data)
        check_foldable(router._model, {"data": {"pathway": router._pathway}})
        if pathway is None:
            pathway = router._pathway
            encoder = gateway_encoder_spec(load_encoder(data.cfg, pathway).cfg, dim=int(router._model.query_dim))
            check_servable_encoder(encoder)
            parity_q = parity_embeddings(data, pathway, n=300)
        elif router._pathway != pathway:
            raise ValueError(f"{run} uses pathway {router._pathway!r}, the others {pathway!r}: one file has one encoder")
        e_q = np.ascontiguousarray(_encode_texts(data, texts, router._query_pathway, fallback_pathway=pathway,
                                                 cache=router._encoder_cache), dtype=np.float32)
        refs = router.resolve_pool_profile_matrix(list(TIERS))
        fit = fit_judged(router._model, e_q, S, refs, variant=args.variant, steps=args.steps, lr=args.lr, seed=i)
        model, da, db = fit
        state = fold_nirt(model, refs, None, offsets=(da, db))
        # each fold exact against its own fine-tuned model, on held-out dataset queries and the judged prompts
        for q in (parity_q, e_q):
            parity_max_diff(state, predict_judged(fit, q, refs), q)
        states.append(state)
        fits.append((fit, refs))
        out_dir = runs_dir / f"{run}-judged"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "nirt-judged", "source_run": run, "state_dict": model.state_dict(),
                    "tier_ids": list(TIERS), "da": da, "db": db, "refs": refs, "variant": args.variant,
                    "steps": args.steps, "lr": args.lr, "n_prompts": len(prompts),
                    "graded": [str(Path(g).expanduser()) for g in args.graded]}, out_dir / "finetuned.pt")
        print(f"  {run}: fine-tuned, folded, kept in {out_dir}")

    state = ensemble_fold(states)
    want = 1 / (1 + np.exp(-np.mean([predict_judged(f, parity_q, r, logits=True) for f, r in fits], axis=0)))
    diff = parity_max_diff(state, want, parity_q)

    out = Path(args.out).expanduser()
    payload = MLPArtifactPayload(
        state_dict={"0.weight": state["w1"], "0.bias": state["b1"], "3.weight": state["w2"], "3.bias": state["b2"]},
        model_ids=list(TIERS), in_dim=int(state["w1"].shape[1]), hidden=int(state["w1"].shape[0]),
        dropout=0.0, pathway=pathway)
    header = {"artifact_id": f"nirt-judged-{out.stem}", "dataset": "judged gateway prompts",
              "source_runs": runs, "judged_prompts": len(prompts), "variant": args.variant,
              "steps": args.steps, "lr": args.lr, "parity_max_abs_diff": diff, "parity_n": int(len(parity_q)),
              "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    payload.save(out, encoder=encoder, model_costs=None, meta=header)
    saved = parity_max_diff_file(out, list(TIERS), parity_q, want)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {len(runs)} fine-tuned seeds averaged, "
          f"hidden={payload.hidden}; parity max |dP| {diff:.2g} in memory, {saved:.2g} after reload")
    print(json.dumps({"encoder": encoder}, indent=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How well routing picks between the Claude tiers, measured on graded prompts.

Every prompt in a graded file (``aar-gateway-grade --out``) has a real score
and a real plan-usage cost for each tier, so a routing policy can be replayed
offline: for each prompt, which tier would it pick, and what did that tier
score and cost?

    python -m evaluation.gateway.pool_eval --graded ~/.cache/aar/routing-pool/graded.jsonl \\
        --router-file ~/.aar-gateway/router.npz

Policies, each scored on the same prompts:

* ``always <tier>`` -- the fixed baselines.
* ``benchmark router`` -- what the gateway does today without your grades:
  the kNN router's predictions for each tier's stand-in (``proxy`` in the
  gateway config), then the cheapest tier within ``--margin`` of the best.
* ``pool kNN`` -- each tier's predicted score is the similarity-weighted mean
  of its scores on the ``k`` most similar *other* graded prompts
  (leave-one-out), then the same margin rule. What routing on graded Claude
  data would do.
* ``oracle`` -- the cheapest tier whose answer matched the best (the judge's
  ``matches_best``): the most any router could save at no loss.

"vs mix" is quality minus what randomly mixing two fixed tiers achieves at the
same cost: a router sitting on that line has bought nothing with its
predictions. "Quality" is the mean judge score of the picked answers; "matches best" is how
often the picked answer was as good as the best of the three; "usage" is the
picked answers' summed plan usage at API prices.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from router.gateway.config import read_jsonl

TIERS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")  # cheapest first


def load_prompts(graded_file: Path, tiers=TIERS) -> list[dict]:
    """Prompts with a score, a cost and a matches-best verdict for every tier."""
    by: dict[str, dict] = defaultdict(dict)
    for r in read_jsonl(graded_file):
        if r.get("score") is None or r.get("model") not in tiers:
            continue
        by[r["prompt_hash"]][r["model"]] = r
    out = []
    for h, rows in by.items():
        if all(t in rows and rows[t].get("usage_usd") is not None for t in tiers):
            first = rows[tiers[0]]
            out.append({"prompt_hash": h, "prompt": first["prompt"], "source": first.get("source") or "own",
                        "score": np.array([rows[t]["score"] for t in tiers], dtype=float),
                        "cost": np.array([rows[t]["usage_usd"] for t in tiers], dtype=float),
                        "matches": np.array([bool(rows[t].get("matches_best")) for t in tiers])})
    return out


def margin_pick(pred: np.ndarray, margin: float) -> np.ndarray:
    """Per row, the cheapest tier (lowest index) predicted within ``margin`` of the best."""
    ok = pred >= pred.max(axis=1, keepdims=True) - margin
    return ok.argmax(axis=1)


def loo_knn(vectors: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out kNN: each row's predicted tier scores from its ``k`` most
    similar other rows, weighted by (clipped) cosine similarity."""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    k = min(k, len(vectors) - 1)
    pred = np.empty_like(scores)
    for i in range(len(vectors)):
        nn = np.argpartition(-sims[i], k - 1)[:k]
        w = np.clip(sims[i, nn], 1e-3, None)
        pred[i] = (scores[nn] * w[:, None]).sum(axis=0) / w.sum()
    return pred


def evaluate(prompts: list[dict], *, picks: dict[str, np.ndarray]) -> list[dict]:
    S = np.stack([p["score"] for p in prompts])
    C = np.stack([p["cost"] for p in prompts])
    M = np.stack([p["matches"] for p in prompts])
    rows_idx = np.arange(len(prompts))
    base_cost = C[:, -1].sum()
    # the line through the fixed tiers: what randomly splitting traffic between
    # two of them buys at a given cost. A router is only worth something above it.
    fixed = sorted((C[:, j].sum(), S[:, j].mean()) for j in range(S.shape[1]))
    def mixture(cost: float) -> float:
        xs, ys = zip(*fixed, strict=True)
        return float(np.interp(cost, xs, ys))
    out = []
    for name, pick in picks.items():
        cost = C[rows_idx, pick].sum()
        out.append({"policy": name, "quality": float(S[rows_idx, pick].mean()),
                    "matches_best": float(M[rows_idx, pick].mean()), "usage_usd": float(cost),
                    "vs_opus": float(1 - cost / base_cost) if base_cost else 0.0,
                    "vs_mix": float(S[rows_idx, pick].mean() - mixture(cost)),
                    "mix": {t.split("-")[1]: int((pick == j).sum()) for j, t in enumerate(TIERS)}})
    return out


def policies(prompts: list[dict], *, margin: float, k: int, router=None, proxies=None,
             encode=None) -> dict[str, np.ndarray]:
    n = len(prompts)
    S = np.stack([p["score"] for p in prompts])
    M = np.stack([p["matches"] for p in prompts])
    picks = {f"always {t.split('-')[1]}": np.full(n, j) for j, t in enumerate(TIERS)}
    if router is not None and proxies:
        frame = router.predict_scores_text([p["prompt"] for p in prompts])
        pred = np.stack([frame[proxies[t]].to_numpy(float) for t in TIERS], axis=1)
        picks["benchmark router"] = margin_pick(pred, margin)
    if encode is not None and n > k:
        vec = np.asarray(encode([p["prompt"] for p in prompts]), dtype=float)
        vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        picks["pool kNN (leave-one-out)"] = margin_pick(loo_knn(vec, S, k), margin)
    oracle = np.where(M.any(axis=1), M.argmax(axis=1), S.argmax(axis=1))  # cheapest that matched the best
    picks["oracle"] = oracle
    return picks


def report(results: list[dict], title: str) -> str:
    lines = [title, f"{'policy':28s} {'quality':>8s} {'vs mix':>7s} {'matches best':>13s} {'usage $':>9s} "
                    f"{'vs Opus':>8s}  mix (H/S/O)"]
    for r in results:
        mix = r["mix"]
        lines.append(f"{r['policy']:28s} {r['quality']:8.3f} {r['vs_mix']:+7.3f} {r['matches_best']:13.0%} {r['usage_usd']:9.2f} "
                     f"{r['vs_opus']:8.0%}  {mix.get('haiku', 0)}/{mix.get('sonnet', 0)}/{mix.get('opus', 0)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.gateway.pool_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--graded", required=True)
    ap.add_argument("--router-file", help="a kNN router file, for the benchmark-router policy")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10, help="neighbours for the pool kNN")
    ap.add_argument("--by-source", action="store_true", help="also one table per prompt source")
    args = ap.parse_args(argv)

    prompts = load_prompts(Path(args.graded).expanduser())
    if not prompts:
        print("no prompt has a score and cost for every tier yet", file=sys.stderr)
        return 1
    router = proxies = None
    if args.router_file:
        from router.gateway.config import DEFAULT_MODELS
        from router.gateway.routing import load_router_model

        router = load_router_model({"artifact": args.router_file})
        proxies = {t: DEFAULT_MODELS[t]["proxy"] for t in TIERS}
    from router.embeddings.encoder import EncoderConfig, TextEncoder, light_backend

    encoder = TextEncoder(EncoderConfig(name="pool-eval", backend=light_backend()))
    groups = {"all": prompts}
    if args.by_source:
        for p in prompts:
            groups.setdefault(p["source"], []).append(p)
    for name, group in groups.items():
        picks = policies(group, margin=args.margin, k=args.k, router=router, proxies=proxies, encode=encoder.encode)
        print(report(evaluate(group, picks=picks), f"\n== {name}: {len(group)} prompts, margin {args.margin}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

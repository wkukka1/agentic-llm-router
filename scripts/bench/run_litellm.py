"""LiteLLM's query-aware routers for the router bake-off, scored offline.

Runs in ``.venv-routellm`` (litellm is a RouteLLM dependency there); never imported
by the repo. LiteLLM's other strategies (simple-shuffle, least-busy, latency-,
usage-, cost-based) balance load across deployments of one model and never read
the query, so they are not routers in this sense.

    .venv-routellm/bin/python -I scripts/bench/run_litellm.py \\
        --bundle data/processed/router_bakeoff/primary --texts <bundle>/texts.parquet

2-model arms (strong = gpt_4o, weak = gpt_4o_mini) -> ``<bundle>/litellm/<arm>.parquet``
(query_id, strong_score). Higher means "send to gpt_4o":
* ``complexity (heuristic)``: ``complexity_router`` default weighted rule scorer
  (``ComplexityRouter.classify``). Score = its complexity score. No training.
* ``complexity heuristic_v2 (bundled UltraFeedback)``: the calibrated success
  predictor with LiteLLM's shipped artifact. Tier 1 = gpt_4o_mini. The router
  escalates when P(success | tier 1) < threshold, so score = -P(tier 1).
* ``complexity heuristic_v2 (retrained)``: the same predictor and priors with an
  artifact built from our train split: tier 1 = gpt_4o_mini, tiers 2-4 = gpt_4o,
  successes = summed graded score, statistics per request type and cohort as
  LiteLLM defines them (``classify_prompt``, ``similarity_cohort``).
* ``adaptive (bandit, trained)``: ``adaptive_router``. Beta cells per
  (request_type, model) with LiteLLM's cold-start prior (README prefs: gpt_4o
  tier 3, gpt_4o_mini tier 2) and ``apply_delta`` updates of (y, 1-y) from a
  shuffled stream of train queries, under its hard SAMPLE_CAP. Score = posterior
  mean(gpt_4o) - mean(gpt_4o_mini). Posterior means, not Thompson draws, so the
  ranking is deterministic.

20-model arm -> ``<bundle>/litellm/adaptive/seed<k>/lam<λ>.npz`` (same format as
``run_llmrouter.py``): the adaptive bandit over all 20 models (quality_tier =
train-mean-score tertile), scores = posterior mean - λ·C_train, i.e. the shared
utility in place of LiteLLM's 0.7/0.3 weights.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

os.environ.setdefault("OPENAI_API_KEY", "unused-no-api-calls")   # litellm import-time client

import numpy as np
import pandas as pd

from litellm.router_strategy.adaptive_router.bandit import apply_delta, initial_cell
from litellm.router_strategy.adaptive_router.classifier import classify_prompt
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.tier_predictor import (
    TierSuccessPredictor, TrainedTierArtifact, resolve_tier_artifact, similarity_cohort)
from litellm.types.router import AdaptiveRouterPreferences

STRONG, WEAK = "gpt_4o", "gpt_4o_mini"
LAMBDAS = (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
TIERS = {"tiers": {"SIMPLE": "weak", "MEDIUM": "strong", "COMPLEX": "strong", "REASONING": "strong"}}


def v2_artifact(texts: list, types: list, y_tier: dict[int, np.ndarray]) -> TrainedTierArtifact:
    """LiteLLM's artifact schema filled from our train split (graded successes)."""
    coh = pd.Series([similarity_cohort(t, rt) for t, rt in zip(texts, types)])
    tval = pd.Series([rt.value for rt in types])
    g, dom, cohs = [], [], []
    for tier, y in y_tier.items():
        g.append({"tier": tier, "successes": float(y.sum()), "observations": float(len(y))})
        for rt, ix in tval.groupby(tval).groups.items():
            ix = np.asarray(list(ix))
            dom.append({"tier": tier, "request_type": rt, "successes": float(y[ix].sum()),
                        "observations": float(len(ix))})
        for c, ix in coh.groupby(coh).groups.items():
            ix = np.asarray(list(ix))
            cohs.append({"tier": tier, "cohort": c, "successes": float(y[ix].sum()),
                         "observations": float(len(ix))})
    return TrainedTierArtifact.model_validate({
        "global_statistics": g, "domain_statistics": dom, "cohort_statistics": cohs,
        "datasets": [{"name": "irt_router train split", "url": "local", "license": "see source",
                      "rows": int(len(texts))}],
        "success_definition": "graded score (fractional successes)"})


def train_cells(types: list, Y: np.ndarray, models: list[str], prefs: dict, seed: int) -> dict:
    """LiteLLM adaptive bandit: cold-start prior, then apply_delta over a shuffled stream."""
    cells = {}
    for rt in set(types):
        for j, m in enumerate(models):
            cells[(rt, m)] = initial_cell(prefs[m], rt)
    order = list(range(len(types)))
    random.Random(seed).shuffle(order)
    for i in order:
        for j, m in enumerate(models):
            y = Y[i, j]
            if np.isfinite(y):
                cells[(types[i], m)] = apply_delta(cells[(types[i], m)], float(y), 1.0 - float(y))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b = args.bundle
    q = pd.read_parquet(os.path.join(b, "queries.parquet"))
    obs = pd.read_parquet(os.path.join(b, "observations.parquet"))
    models_df = pd.read_parquet(os.path.join(b, "models.parquet"))
    text = pd.read_parquet(args.texts).set_index("query_id")["text"]
    models = list(models_df["model_id"])
    tr = q[q["split"] == "train"].reset_index(drop=True)
    ev = q[q["split"].isin(args.splits.split(","))].reset_index(drop=True)
    Y = obs.pivot(index="query_id", columns="model_id", values="target")
    Ytr = Y.reindex(index=tr["query_id"], columns=models).to_numpy(np.float64)
    tr_text, ev_text = text.reindex(tr["query_id"]).reset_index(drop=True), text.reindex(ev["query_id"]).reset_index(drop=True)
    out = os.path.join(b, "litellm")
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    tr_types = [classify_prompt(t) for t in tr_text]
    ev_types = [classify_prompt(t) for t in ev_text]
    print(f"request types (train): {pd.Series([r.value for r in tr_types]).value_counts().to_dict()} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    def save(arm: str, name: str, s) -> None:
        pd.DataFrame({"query_id": ev["query_id"], "strong_score": np.asarray(s, float),
                      "router": name}).to_parquet(os.path.join(out, f"{arm}.parquet"), index=False)
        print(f"[{arm}] -> {name}", flush=True)

    heur = ComplexityRouter("bench", None, TIERS)
    save("complexity", "complexity (heuristic)", [heur.classify(t)[1] for t in ev_text])

    bundled = TierSuccessPredictor(resolve_tier_artifact("ultrafeedback"))
    save("v2_bundled", "complexity heuristic_v2 (bundled UltraFeedback)",
         [-bundled.predict(t, rt).probabilities[1] for t, rt in zip(ev_text, ev_types)])

    js, jw = models.index(STRONG), models.index(WEAK)
    keep = np.isfinite(Ytr[:, js]) & np.isfinite(Ytr[:, jw])
    art = v2_artifact([t for t, k in zip(tr_text, keep) if k], [r for r, k in zip(tr_types, keep) if k],
                      {1: Ytr[keep, jw], 2: Ytr[keep, js], 3: Ytr[keep, js], 4: Ytr[keep, js]})
    retrained = TierSuccessPredictor(art)
    save("v2_retrained", "complexity heuristic_v2 (retrained)",
         [-retrained.predict(t, rt).probabilities[1] for t, rt in zip(ev_text, ev_types)])

    two = [WEAK, STRONG]
    prefs2 = {STRONG: AdaptiveRouterPreferences(quality_tier=3), WEAK: AdaptiveRouterPreferences(quality_tier=2)}
    cells2 = train_cells(tr_types, Ytr[:, [jw, js]], two, prefs2, args.seed)
    save("adaptive2", "adaptive (bandit, trained)",
         [cells2[(rt, STRONG)].mean - cells2[(rt, WEAK)].mean for rt in ev_types])

    # 20-model adaptive bandit, swept over the shared utility
    strength = np.nanmean(Ytr, axis=0)
    cut = np.quantile(strength, [1 / 3, 2 / 3])
    prefs = {m: AdaptiveRouterPreferences(quality_tier=int(1 + (s > cut[0]) + (s > cut[1])))
             for m, s in zip(models, strength)}
    cells = train_cells(tr_types, Ytr, models, prefs, args.seed)
    P = np.array([[cells[(rt, m)].mean for m in models] for rt in ev_types])
    C = models_df["train_mean_cost"].to_numpy(np.float64)
    d = os.path.join(out, "adaptive", f"seed{args.seed}")
    os.makedirs(d, exist_ok=True)
    for lam in LAMBDAS:
        np.savez(os.path.join(d, f"lam{lam:g}.npz"), query_ids=ev["query_id"].to_numpy(str),
                 model_ids=np.array(models), scores=P - lam * C[None, :], seconds=0.0, hparams="{}")
    n_full = sum(c.alpha + c.beta >= 199 for c in cells.values())
    print(f"[adaptive20] {len(cells)} cells, {n_full} hit SAMPLE_CAP -> {d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

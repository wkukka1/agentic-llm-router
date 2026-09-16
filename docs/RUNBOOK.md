# Runbook: how to run everything, end to end

A copy-paste guide to actually running this repo's pipelines, in order. For
*why* each step exists and what calls what, see [workflows.md](workflows.md)
— this doc is the fast path, that one is the reference.

Run everything from the repo root, with the venv active.

---

## 0. One-time setup

```bash
python -m venv .venv
.venv/Scripts/activate                # Windows; `source .venv/bin/activate` on POSIX
pip install -e ".[dev]"
```

**GPU check** (this machine has an RTX 3050 Ti — worth 11x on embedding
encode speed):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, the CPU-only torch wheel got installed. Fix:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps torch==2.13.0
```

**Optional extras** — only if you're touching that part of the repo:

```bash
pip install -e ".[agentic]"    # LangChain orchestrator
pip install -e ".[labeling]"   # anchor-judge LLM calls (openai/anthropic SDKs)
pip install lm-eval            # run_lm_harness.py
```

**Anchor-judge only**: copy `.env.example` → `.env` and fill in
`JUDGE_API_KEY` / `JUDGE_BASE_URL`. Not needed for `--dry-run` or anything
else in this repo.

---

## 1. Data pipeline (do this once; re-run only if raw sources change)

The whole thing in three commands:

```bash
python -m training.phase0                          # deterministic tables + splits
python -m training.phase0 --embeddings              # + query/profile embeddings (slow, GPU helps)
python -m training.phase0 --taxonomy --query-bank   # + clustering/relevance + FAISS bank
```

Or stage by stage, if you need to re-run just one step:

```bash
python scripts/data/download_routerbench.py
python scripts/data/download_arena.py                        # also pulls GPT-4-Judge battles
python scripts/data/build_response_matrix.py                  # -> data/processed/{responses,queries,models}.parquet
python scripts/data/build_splits.py                           # -> data/splits/*.json
python scripts/models/build_model_profiles.py                 # -> model_profiles.parquet
python scripts/data/build_nirt_dataset.py                     # -> nirt_observations.parquet
python scripts/embeddings/build_query_embeddings.py   --pathway retrieval
python scripts/embeddings/build_query_embeddings.py   --pathway irt --from-nirt
python scripts/embeddings/build_profile_embeddings.py --all-pathways
python scripts/embeddings/build_query_features.py              # 0-/5-shot indicator + length features
python scripts/taxonomy/cluster_queries.py                     # -> data/taxonomy/{clusters.parquet,centroids.npy}
python scripts/taxonomy/generate_taxonomy.py                   # -> data/taxonomy/taxonomy.json (human-readable)
python scripts/taxonomy/build_relevance.py                     # -> query_relevance__<pathway>/
python scripts/retrieval/build_query_bank.py                   # -> data/indexes/query_bank/ (FAISS, train-only)
python scripts/retrieval/build_warmup.py                       # -> query_warmup__<pathway>/
```

**Sanity check** it worked:

```python
from router.config import load_config
from training.data.facade import load_training_data
d = load_training_data(load_config())
d.correctness_matrix(split="train").shape
```

Expect `734,469 observations · 196,791 queries · 69 models · 88 datasets`
(current checked-in raw data). Full numbers: [README](../README.md#data-pipeline-foundation).

---

## 2. Train the NIRT model

This is the main model. Defaults live in [`configs/nirt.yaml`](../configs/nirt.yaml)
(`dim: 1`, `model_params: projected`, `orientation: query_latent`) —
override on the CLI:

```bash
python scripts/nirt/train_nirt.py \
  --config configs/nirt.yaml \
  --dim 2 --model-params projected \
  --name nirt-2d-projected
```

Writes `data/processed/nirt_runs/nirt-2d-projected/{model.pt, run.json}`.

Useful overrides: `--orientation model_latent` (IRT-Router paper form
instead of ours), `--loss pairwise|listwise|bce_pairwise` (routing-aware),
`--query-features default` (fixes the 0-/5-shot confound — biggest single
lever found so far), `--ood` (also eval held-out families).

**Sanity-check the checkpoint immediately:**

```bash
python scripts/nirt/eval_nirt.py --run nirt-2d-projected --split test
```

---

## 3. Evaluate / diagnose a run

Pick the script for the question you're asking — all load an existing
checkpoint, none of them retrain (except `ab_orientation.py` and
`complexity_search.py`, which train fresh comparison runs):

| question | command |
|---|---|
| Prediction quality on a split | `python scripts/nirt/eval_nirt.py --run nirt-2d-projected --split test` |
| Full leaderboard: prediction + ranking + routing + AIQ vs every baseline | `python scripts/nirt/compare.py --run nirt-2d-projected --split test` |
| `query_latent` vs `model_latent`, head-to-head | `python scripts/nirt/ab_orientation.py --dim 2 --model-params projected` |
| Does cold-start work from profile text alone? | `python scripts/nirt/coldstart.py --run nirt-2d-projected --split test` |
| Has `theta_q` collapsed (ignoring the query)? | `python scripts/nirt/theta_collapse.py --run nirt-2d-projected --split test` |
| Overfitting or underfitting? | `python scripts/nirt/train_test_gap.py` |
| Where does underfitting come from? | `python scripts/nirt/capacity_diagnostics.py --nirt-run nirt-2d-projected` |
| Best k for kNN-imputed query embeddings? | `python scripts/nirt/knn_impute_sweep.py` |
| Does novelty-weighted shrinkage fix OOD calibration? | `python scripts/nirt/shrinkage_eval.py --nirt-run nirt-2d-projected --splits ood` |

All write JSON reports under `data/processed/nirt_runs/<run>/` or
`artifacts/phase2/`.

---

## 4. Routing evaluation

"Is the prediction good" (§3) → "does it produce good *routing decisions*."

```bash
# per-(query,model) oracle labels + regret/hit metrics for one run
python scripts/nirt/route_eval.py --run nirt-2d-projected --split test --lam 0.1

# leaderboard across POLICIES (NIRT, IRT-Router, Bernoulli, ZOIB E[Y]/LCB) + λ-sweep AIQ
python scripts/route_compare.py --nirt-run nirt-2d-projected --lam 0.1

# candidate-pool expansion battery (only when adding a new model to the pool)
python scripts/pool_expansion/run_phase.py --phase E3 --added-models <model_id>
```

`route_compare.py` is the one to run for a "how are we doing overall"
snapshot — it writes `artifacts/{phase2,irt_router}/routing_comparison.json`
and regenerates the pareto/AIQ tables.

---

## 5. Baseline control arms (Phase 1 / Phase 2)

Frozen comparison points — the plain Bernoulli/BCE arm and the continuous
(Normal/Beta/ZOIB) response heads. Rarely need retraining; retrain only if
you've deliberately changed `training.nirt.baseline`.

```bash
python scripts/nirt/baseline/train_baseline.py    --config configs/phase1.yaml            # Bernoulli/BCE
python scripts/nirt/baseline/train_continuous.py  --response zoib --config configs/phase2.yaml
python scripts/nirt/baseline/evaluate.py          --checkpoint artifacts/phase2/zoib --split test
python scripts/nirt/baseline/compare_response_models.py                                    # all 4 heads side by side
```

---

## 6. IRT-Router paper replication (isolated track)

Own config (`configs/irt_router.yaml`), own data track — doesn't touch
RouterBench:

```bash
python scripts/data/download_irt_router.py
python scripts/data/import_irt_router_embeddings.py     # paper's own BERT vectors
python scripts/data/encode_irt_router_pathway.py --pathway irt   # OR: re-encode with our encoder
```

Then train/eval exactly as in §2–4, pointing `--config configs/irt_router.yaml`.

---

## 7. Anchor-judge (spends real money — gated)

Tests whether Arena/Judge preference and RouterBench correctness are one
utility or two. Only run this deliberately.

```bash
python scripts/data/select_anchor_model.py
python scripts/data/select_anchor_judge_queries.py
python scripts/data/collect_anchor_judgments.py --dry-run              # ALWAYS dry-run first
python scripts/data/collect_anchor_judgments.py --provider openai --confirm-budget
python scripts/nirt/anchor_judge_analysis.py
```

---

## 8. Use a trained router (no scripts — library calls)

```python
from router.routing import NIRTRouter
from router.agentic import AgenticRouter

router = NIRTRouter.from_run("nirt-2d-projected")
result = router.route_text(["Prove that sqrt(2) is irrational."])
result.selected_model_ids, result.model_mix()

agent = AgenticRouter(router)
outcome = agent.run("Plan a 3-day trip to Kyoto and translate the itinerary to Japanese.")
outcome.mode, outcome.answer, outcome.cost
```

---

## 9. Tests (run before/after any change)

```bash
pytest -q            # 417 tests
lint-imports          # package-layering contract (router / training / evaluation)
```

---

## Cheat sheet: minimum path from zero to a trained, evaluated router

```bash
pip install -e ".[dev]"
python -m training.phase0 --embeddings --taxonomy --query-bank
python scripts/nirt/train_nirt.py --dim 2 --model-params projected --name nirt-2d-projected
python scripts/nirt/eval_nirt.py --run nirt-2d-projected --split test
python scripts/route_compare.py --nirt-run nirt-2d-projected --lam 0.1
```

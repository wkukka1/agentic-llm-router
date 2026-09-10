# IRT-Router benchmark suite

Evaluates the router on the **IRT-Router paper's own benchmark**
([arXiv 2506.01048](https://arxiv.org/abs/2506.01048), ACL'25): 20 LLMs over 12
datasets, split into in-distribution (ID) and out-of-distribution (OOD). This is
a fully isolated parallel pipeline — nothing shares state with the RouterBench
pipeline (`configs/phase0.yaml`). It **supersedes the E3 pool-expansion
workstream** (lm-eval cheap models) as the pool/eval-expansion path; the E3
scaffolding (`src/router/data/benchmark_id.py`, `scripts/data/run_lm_harness.py`)
is left in place, parked.

## Data provenance

`github.com/Mercidaiha/IRT-Router` @ commit `e8f258ced4ec3c40d795403603acd8c1cdfb994d`,
fetched from the Git-LFS media endpoint by `scripts/data/download_irt_router.py`
into `data/raw/irt_router/`.

| file | role |
|---|---|
| `data/train.csv` (450 MB) | ID training rows |
| `data/test1.csv` (192 MB) | ID test rows |
| `data/test2.csv` (57 MB) | OOD test rows |
| `utils/map/query.csv` | `index, task, split, id, question` (38 060 rows) |
| `utils/map/llm.csv` | `index, name, profile` (25 rows) |
| `utils/bert_embeddings/query_embeddings.pkl` | bert-base mean-pool, 768-d, per query index |
| `utils/bert_embeddings/llm_embeddings.pkl` | bert-base, 768-d, per model index |
| `utils/relevance/relevance_vectors_cluster_{train,test}_bert.pkl` | 25-d relevance vectors |
| `utils/cold/test_avg_embeddings_bert.pkl` | cold-start avg query embedding (not used yet) |

Row schema: `id, question, ground_truth, completion, input_tokens, output_tokens,
cost, performance, task, llm`. `performance` ∈ [0, 1] (binary for MC tasks, graded
F1/pass-rate for `squad` / `hotpot_qa` / `mbpp`); `cost` is the token-weighted USD
cost. Both are used verbatim — no price table.

Query identity is the exact `question` text (`make_query_id("irt_router", task,
render_prompt(question))`), matching the repo's own `query_id_map[question]`
join. 100 % of CSV queries join to a shipped embedding.

## Datasets & pool

**ID (8, 70/30 train/test):** `mmlu, cmmlu, aclue, arc_c, hotpot_qa, squad, math,
mbpp`.
**OOD (4, held out entirely):** `ceval, commonsense_qa, gsm8k, humaneval`.

Multiple-choice (data-level chance correction): `mmlu, cmmlu, aclue, arc_c,
ceval` (4 choices), `commonsense_qa` (5).

**Routing pool (20):** the exact `llms` list from the repo's `test_router.py` —
`glm_4_air, glm_4_flash, glm_4_plus, gpt_4o, gpt_4o_mini, gpt_4o_mini_cot,
deepseek_coder, deepseek_chat, qwen25_32b_int4, qwen25_7b_instruct,
qwen25_72b_instruct, qwq_32b_preview, qwen25_math_7b_instruct, llama31_8b_instruct,
llama31_70b_instruct, llama31_405b_instruct, mixtral_8x7b_instruct,
mistral_7b_instruct_v02, ministral_8b_instruct_2410, gemini15_flash`. Native ids
are kept verbatim as `model_id` (no registry aliases; the quality check emits an
"uncanonicalized_model" warning, which is expected here). CSV rows for any other
`llm` (`moonshot_v1_search`, `claude35_haiku20241022`, …) are dropped at load.

Model *profile text* is not rebuilt for v1 — profile embeddings are imported
directly from `llm_embeddings.pkl`, so `model_profiles.parquet` is intentionally
absent.

## Split mechanism

`split.presplit: true` in `configs/irt_router.yaml` makes
`scripts/data/build_splits.py` call `router.data.splits.make_presplit` instead of
the content-hash lottery: `train.csv` → `train` + `validation` (90/10 by content
hash), `test1.csv` → `test`, `test2.csv` → `ood`. `ood` is a first-class split
(`data/splits/irt_router/ood.json`) understood by `phase1.split_query_ids`,
`nirt.build_nirt_observations`, `routing.eval_matrices`, and `route_compare.py`.
All 20 models are warm.

Built counts: train 22 011 / validation 2 403 / test 10 458 / ood 3 159 queries;
760 620 observations.

## Pipeline

```bash
PY=.venv/Scripts/python.exe
CFG=configs/irt_router.yaml

$PY scripts/data/download_irt_router.py           --config $CFG   # ~720 MB
$PY scripts/data/build_response_matrix.py         --config $CFG
$PY scripts/data/build_splits.py                  --config $CFG
$PY scripts/data/import_irt_router_embeddings.py  --config $CFG
$PY scripts/data/build_nirt_dataset.py            --config $CFG

# --- train the four policy families (checkpoints under artifacts/irt_router/ and
#     data/processed/nirt_runs/irtrouter-*) ---
$PY scripts/nirt/baseline/train_continuous.py --response bernoulli --phase0-config $CFG \
    --checkpoint artifacts/irt_router/bernoulli
$PY scripts/nirt/baseline/train_continuous.py --response zoib --phase0-config $CFG \
    --checkpoint artifacts/irt_router/zoib
$PY scripts/nirt/train_nirt.py --phase0-config $CFG --dim 2 --model-params projected \
    --name irtrouter-nirt-2d-projected
$PY scripts/nirt/train_nirt.py --phase0-config $CFG --orientation model_latent --dim 25 \
    --model-params projected --query-hidden none --lr 0.002 --batch-size 512 \
    --weight-decay 0 --name irtrouter-irt-25d-projected

# --- ID + OOD routing report ---
$PY scripts/route_compare.py --config $CFG --reference gpt_4o --splits test,ood --lam 0.3 \
    --zoib artifacts/irt_router/zoib --baseline artifacts/irt_router/bernoulli \
    --nirt-run irtrouter-nirt-2d-projected --irt-run irtrouter-irt-25d-projected
```

Output: an ID table and an OOD table (oracle, 3 fixed-model baselines, 4 learned
policies) plus the ZOIB cost/quality frontier for each, written to
`artifacts/irt_router/routing_comparison.json`.

## Results

Trained 2026-09-02 (CPU, `--phase0-config configs/irt_router.yaml`). Validation
(2 403 ID queries) — all four beat the per-model-mean BCE baseline (0.573):

| family | checkpoint | val BCE | val acc | Δbce vs model-mean |
|---|---|---|---|---|
| Bernoulli | `artifacts/irt_router/bernoulli` | 0.494 | 0.760 | — |
| ZOIB | `artifacts/irt_router/zoib` (NLL 0.515) | 0.497 | 0.759 | — |
| NIRT (ours, `query_latent` K=2) | `nirt_runs/irtrouter-nirt-2d-projected` | 0.513 | 0.754 | −0.061 |
| IRT-Router (paper, `model_latent` K=25) | `nirt_runs/irtrouter-irt-25d-projected` | 0.501 | 0.759 | −0.072 |

Routing (`--reference gpt_4o --lam 0.3`), accuracy / cost-saving-vs-`gpt_4o`:

| policy | ID acc | ID save | OOD acc | OOD save |
|---|---|---|---|---|
| oracle (cheapest-best per query) | 0.954 | 98.4 % | 0.983 | 99.4 % |
| fixed `deepseek_coder` (best on train) | 0.807 | 96.4 % | 0.863 | 97.2 % |
| fixed `gpt_4o` (reference) | 0.772 | 0 % | 0.849 | 0 % |
| route: Bernoulli | **0.825** | 76.8 % | **0.864** | 90.9 % |
| route: ZOIB E[Y] | 0.823 | 73.2 % | 0.860 | 91.3 % |
| route: IRT-Router (paper) | 0.820 | 71.8 % | 0.859 | 92.2 % |
| route: NIRT (ours) | 0.807 | 96.3 % | 0.862 | 97.2 % |

ZOIB AIQ (area under the cost/quality frontier): ID absolute 0.819 / +0.016 over
linear; OOD 0.871 / +0.002.

### Oracle-routing evaluation

`scripts/nirt/route_eval.py` (see `docs/routing_evaluation.md`) scores routing
against the tie-aware oracle, kept separate from predictive NLL:

```bash
python scripts/nirt/route_eval.py --run irtrouter-irt-25d-projected --split test \
    --config configs/irt_router.yaml --lam 0.3 --tolerance 0.02 \
    --zoib artifacts/irt_router/zoib --bernoulli artifacts/irt_router/bernoulli \
    --oracle-classifier
```

K=25 model, test split: `oracle_hit_rate` 2 % but `oracle_hit_rate_any_best`
85 %, median regret 0 — the many-way ties in binary MC correctness make "exact
oracle model" almost unreachable while "picked *a* best model" is the real
signal. Hard-oracle upper bound: quality 0.953 at $0.020/1k. Artifacts land in
`data/processed/nirt_runs/<run>/route_eval_<split>.{json,_per_query.csv}` +
`oracle_labels_<split>.parquet`.

**Takeaway.** Unlike the RouterBench 9-model pool (GPT-4-dominated → quality-only
routing collapses to "always GPT-4"), this 20-model pool is *not* dominated: every
learned router **beats fixed `gpt_4o` on accuracy while cutting cost 70–97 %**, on
both the in-distribution test set and the four held-out OOD datasets. The
`model_latent` (paper) and `E[Y]` heads land within ~0.5 pt of the Bernoulli
control; the `ZOIB LCB` (pessimistic) policy is too conservative and
under-escalates. Full numbers in `artifacts/irt_router/routing_comparison.json`.

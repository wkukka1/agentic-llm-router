# Router bake-off: RouteLLM 2-model sub-study (2026-11-12, seed 0)

Setting: RouteLLM's own setup with strong = gpt_4o and weak = gpt_4o_mini (about 17× cheaper).
gpt_4o is better on 13.0 % of ID test queries, mini on 6.9 %; the rest are ties.
OOD: 8.5 % / 4.5 %.
Each router ranks queries by a "strong score". The top k % go to gpt_4o, k = 0…100 %.
Metric: RouteLLM's **APGR**, the average fraction of the mini→4o quality gap recovered
(random ≈ 0.5, higher is better). CPT(x) is the share of gpt_4o calls needed to recover x of the gap.
Δ CI = paired bootstrap (500 resamples, within dataset) of APGR minus RouteLLM's pretrained BERT.

Scripts: `scripts/bench/run_routellm.py` (RouteLLM 0.2.0 venv), `scripts/bench/routellm_substudy.py`.
Full table: `routellm_summary_seed0.csv`. Curves: `routellm_curves_seed0.csv`.

| router | ID APGR | ID Δ vs RouteLLM-BERT | ID CPT50 | OOD APGR | OOD Δ vs RouteLLM-BERT | OOD CPT50 |
|---|---|---|---|---|---|---|
| ours: bernoulli | **0.747** | [+0.157, +0.246] | 26 % | 0.722 | [+0.112, +0.349] | 26 % |
| ours: zoib | 0.744 | [+0.155, +0.247] | 26 % | 0.732 | [+0.126, +0.372] | 21 % |
| ours: irt-router K=25 | 0.731 | [+0.146, +0.232] | 28 % | **0.757** | [+0.134, +0.410] | 22 % |
| ours: nirt elo-manifold + Δ | 0.686 | [+0.103, +0.182] | 27 % | 0.742 | [+0.143, +0.377] | 22 % |
| RouteLLM mf, retrained (4o/mini pairs) | 0.671 | [+0.083, +0.178] | 31 % | 0.737 | [+0.092, +0.415] | 25 % |
| RouteLLM mf, retrained (20-model pairs) | 0.656 | [+0.063, +0.167] | 36 % | 0.473 | [−0.175, +0.143] | 54 % |
| ours: nirt K=2 | 0.644 | [+0.060, +0.142] | 32 % | 0.444 | [−0.151, +0.021] | 55 % |
| ours: knn | 0.642 | [+0.053, +0.154] | 30 % | 0.619 | [−0.003, +0.240] | 33 % |
| ours: mlp | 0.632 | [+0.045, +0.133] | 34 % | 0.720 | [+0.114, +0.355] | 17 % |
| **RouteLLM bert, pretrained (Arena)** | 0.544 | — | 43 % | 0.503 | — | 50 % |
| random | 0.523 | [−0.064, +0.022] | 46 % | 0.490 | [−0.132, +0.142] | 51 % |
| llmrouter-lib routers (best: routerdc / elo) | 0.43–0.53 | ≤ +0.04 | 44–64 % | 0.24–0.60 | | 28–82 % |
| oracle | 1.644 | | 4 % | 1.974 | | 3 % |

## Reading

* **RouteLLM's released router (BERT, trained on Chatbot Arena) is barely better than random
  here**: APGR 0.54 on ID and 0.50 on OOD. Its CPT50 of 43 % means it needs 43 % gpt_4o calls to
  recover half the gap, against 26 % for Bernoulli. This is the off-the-shelf use case, and
  Arena preferences don't transfer to these benchmark tasks.
* **Retrained on our data, RouteLLM's MF is competitive.** ID 0.66–0.67 beats NIRT K=2, kNN
  and MLP. Our three best are 0.06–0.08 APGR higher on ID; the bootstrap here is only
  against BERT, so that gap has no direct CI yet. On OOD the 4o/mini-pair MF matches our best;
  the 20-model-pair MF drops to random.
* **Best for this 2-model decision: Bernoulli / ZOIB / IRT-Router K=25**, about 0.73–0.76 APGR on ID and OOD.
  They recover half the gap with 21–28 % strong calls.
* **NIRT K=2 is weak at this specific decision** (OOD 0.44, below random) though it was tied
  for best in the 20-model bake-off. A 2-d latent hardly separates two models from the
  same family; it ranks gpt_4o vs gpt_4o_mini poorly even though it picks well among 20.
  The Elo-manifold variant fixes most of that (OOD 0.74).
* llmrouter-lib's classifiers stay at or below random, as in the main bake-off. They were trained
  on 20-model "best model" labels, where gpt_4o and gpt_4o_mini are rarely the label.
* OOD CIs are wide (3 159 queries, only about 13 % of them decisive between the two models).

## Not run (with reasons)

* RouteLLM pretrained `mf` and `sw_ranking`: they embed prompts with OpenAI's embedding API
  (no key in this environment). The retrained MF uses the same model class on our BERT embeddings.
* RouteLLM `causal_llm`: an 8B Llama-3 classifier; no GPU.
* Fine-tuning RouteLLM's BERT on our data: hours on CPU; possible as a follow-up.

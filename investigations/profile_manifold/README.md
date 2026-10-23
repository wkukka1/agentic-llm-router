# Profile-manifold NIRT (2026-10-21)

Hypothesis: encode the model profile e_m (768-d BERT) into a learned manifold z_m
(D-dim, autoencoder-shaped), pretrain the whole NIRT on Arena-Elo pseudo-labels over
150k queries, then fine-tune on the real 29-model data at a lower lr.

Pipeline: `scripts/nirt/profile_manifold.py {build,pretrain,finetune,sweep}`,
config `configs/nirt_profile_manifold.yaml`, labels `training.nirt.elo_pseudo`.

## Pseudo-labels
22/28 pool models Arena-rated. Elo->quality fit (train-split mean target, logit scale):
code r2 0.81, chat 0.88, reasoning 0.69, math 0.62, language_zh 0.44, knowledge 0.13.
1.2M rows (150k queries x 8 models). Pretraining fits them: val Spearman ~0.96.

## Results (`sweep.csv`, `irt_router_test_regret.csv`; 3 seeds each)
Val BCE, 29-pool (reference nirt-2d-projected = 0.5576):

| D    | fine-tuned      | from scratch (ctl) |
|------|-----------------|--------------------|
| 16   | 0.5579 ± 0.0007 | 0.5573 ± 0.0003    |
| 64   | 0.5576 ± 0.0005 | 0.5587 ± 0.0007    |
| 256  | 0.5588 ± 0.0023 | 0.5580 ± 0.0015    |
| 1024 | 0.5587 ± 0.0007 | 0.5571 ± 0.0009    |

Test regret, IRT-Router dense pool (reference 0.1469, one seed):

| D    | fine-tuned      | from scratch    |
|------|-----------------|-----------------|
| 16   | 0.1402 ± 0.0035 | 0.1382 ± 0.0017 |
| 64   | 0.1382 ± 0.0011 | 0.1380 ± 0.0023 |
| 256  | 0.1396 ± 0.0022 | 0.1384 ± 0.0025 |
| 1024 | 0.1366 ± 0.0023 | 0.1367 ± 0.0012 |

RouterBench test regret: unchanged (~0.123-0.124 for all).

## Verdict
- Elo pretraining adds nothing: paired fine-tuned minus control = +0.0008 regret, wins 6/12.
- The encoder itself (either training path) cuts IRT-Router test regret ~0.147 -> ~0.137-0.140,
  but the reference is a single seed trained on CPU -- confirm with a 3-seed reference
  before claiming it.
- Not run: cold-start (scripts/nirt/coldstart.py is not source-aware on the 29 pool),
  which is where an Elo-pretrained manifold would most plausibly help.

## Lower fine-tuning lr (`lr_sweep.csv`, D=64 only; stopped early by request)
Warm-started from the same pm-pre-d64 runs; encoder at 0.1x unless frozen. 3 seeds
except lr3e-6 / frozen (2; the sweep was stopped mid-run). `pseudo_spearman` = fit to
the Elo pseudo-labels after fine-tuning (how much pretraining survives).

| fine-tune        | val BCE         | IRT-Router test regret | RouterBench | pseudo Spearman |
|------------------|-----------------|------------------------|-------------|-----------------|
| control (scratch)| 0.5587 ± 0.0007 | 0.1379 ± 0.0023        | 0.1233      | 0.16            |
| lr 1e-4          | 0.5576 ± 0.0005 | 0.1382 ± 0.0011        | 0.1239      | 0.42            |
| lr 3e-5          | 0.5580 ± 0.0007 | 0.1384 ± 0.0005        | 0.1238      | 0.44            |
| lr 1e-5          | 0.5581 ± 0.0006 | 0.1375 ± 0.0008        | 0.1241      | 0.44            |
| lr 3e-6          | 0.5601 ± 0.0004 | 0.1378 ± 0.0017        | 0.1233      | 0.51            |
| lr 1e-5, frozen  | 0.5731 ± 0.0008 | 0.1595 ± 0.0015        | 0.1228      | 0.48            |

Verdict: a lower lr keeps more of the Elo structure (Spearman 0.16 -> 0.51) but routing
does not improve -- every paired difference vs control is within +/-0.0023 regret.
Freezing the encoder hurts (regret +0.02). Elo pretraining does not help at any fine-tune
lr. D=1024 low-lr runs were not done.

## Against the other routers (`routing_benchmark_test.json`, test split)
`scripts/nirt/baseline/routing_benchmark.py --nirt-runs ...`; manifold rows are 3-seed
means routed as P(correct) - lam*C(m). Quality at lam=0 (always-GPT-4: 0.783 combined):

| router              | combined | RouterBench | IRT-Router |
|---------------------|----------|-------------|------------|
| ZOIB + length cost  | 0.800    | 0.791       | 0.818      |
| IRT-Router (25d)    | 0.795    | 0.790       | 0.803      |
| manifold ft d1024   | 0.794    | 0.788       | 0.806      |
| manifold ctl d1024  | 0.794    | 0.788       | 0.806      |
| NIRT ref (2d)       | 0.790    | 0.787       | 0.796      |

The manifold beats the plain NIRT and ties IRT-Router; ZOIB stays ahead. Cost-aware
(lam 20-100) on RouterBench, routers with the per-query cost head (ZOIB, IRT-Router+C_hat)
win by ~0.03-0.05 quality at similar cost -- a cost-model gap, not a quality one. On the
IRT-Router pool at lam 100 the manifold (0.803 @ $0.06/1k) matches ZOIB (0.801 @ $0.06).

## aar-gateway graded prompts (`gateway_replay.json`, 65 prompts, Claude 5.5 tiers)
`scripts/nirt/gateway_replay.py`: tiers scored from their OWN curated profiles
(`predict_pool_scores_text`), gateway margin rule; seeds averaged. Always-Opus 0.703,
always-Haiku 0.645 (88% cheaper). Split-half margin pick, held out:

| router                       | vs mix | saved | dQ vs Opus |
|------------------------------|--------|-------|------------|
| manifold ft d1024 (Elo pre)  | +0.020 | 22%   | -0.010     |
| llmrouterbench-nirt (proxy)  | +0.011 | 47%   | -0.029     |
| MLP default (proxy)          | -0.000 | 26%   | -0.021     |
| manifold ctl d1024           | -0.003 | 69%   | -0.054     |
| IRT-Router 25d               | -0.002 | 66%   | -0.051     |
| NIRT 2d / manifold d64 lr1e-5| ~0     | 82-88%| -0.058     |

Only the Elo-pretrained d1024 ranks Opus > Haiku from the profiles (median predicted gap
+0.012; the others ~0, so they send nearly everything to Haiku). Its per-prompt gap is
uncorrelated with the real one (r -0.01), so the gain is a tier-level ordering on 65
prompts, not per-prompt discrimination -- treat as a lead, not a result.

## Served in aar-gateway (2026-10-22): lambda + BERT encoder + tier-named file
- MiniLM retrain (`sweep_mini.csv`, `-mini` runs) ranks Haiku ABOVE Opus from the tier
  profiles (median gap -0.079): unusable, everything goes to Haiku.
- BERT served instead: `scripts/routing/export_bert_encoder.py` writes bert-base-uncased
  `last_hidden_state` ONNX to ~/.aar-gateway/encoders/bert-base-uncased-mean; fastembed
  reproduces the training encoder to 2.4e-6 (incl. 256-token truncation).
- `aar_gateway/routers/manifold-nirt-claude.npz` = pm-ft-d1024-s44 (best val BCE of the 3
  seeds) folded over the three tiers' own profiles; parity 9.2e-7; replayed through the
  gateway loader it matches the run exactly.
- Per seed, lambda split-half on the 65 graded prompts: s42 +0.007 / 25% saved, s43 -0.003,
  s44 -0.004 / 48% saved. The +0.010 seen earlier was the 3-seed AVERAGE; seed spread is
  as large as the effect, so no single run beats the shipped llmrouterbench-nirt (+0.003
  lambda, +0.011 margin). λ presets left provisional (0.02/0.05/0.1).

## LLM-judge multi-turn pool (99 turns, Sonnet judge; `judge_replay_*.json`)
Haiku is 0.18 below Opus here (vs 0.058 on the 65-prompt pool). The manifold predicts tiny
tier gaps (Opus-Haiku median +0.003..+0.024), so it routes 80-90% to Haiku and loses
0.15-0.18 quality: split-half -0.009..0.000 vs mix (lambda), every seed and the 3-seed
average. MLP default (proxy) is best: +0.006 lambda (52% saved, -0.078) / +0.010 margin
(12% saved, +0.003). Per-prompt predicted gap is uncorrelated with the real one for all.
Verdict: don't serve manifold-nirt-claude; the profile-only tier ordering is too flat.

## Fine-tuning on the judged turns, cross-validated (`judge_finetune_cv_{group,source}.json`)
`scripts/nirt/judge_finetune_cv.py`: fine-tune on k-1 folds of the 99 judged turns, predict the
held-out fold; lambda rule + split-half on the out-of-fold predictions. Seeds s42/s43/s44.

| variant | folds by conversation (5) | leave-one-source-out (4) |
|---|---|---|
| none (served file) | -0.006 / 0.000 / -0.009 | same |
| tiers only (a_m,b_m offsets) | +0.004 / +0.013 / +0.020 | -0.011 / -0.011 / -0.018 |
| tiers + last query layer | +0.060 / +0.052 / +0.030 | +0.038 / -0.011 / -0.009 |
| full, lr 1e-3 (high) | +0.052 / +0.062 / +0.066 (~76% saved, dQ -0.055..-0.087) | +0.035 / +0.003 / +0.024 (64-75% saved) |
| MLP default (no fit) | +0.006 | +0.006 |
| MLP + tier bias | +0.011 | +0.001 |

Held-out vs mix. Fine-tuning learns the real tier gap (+0.18) AND some per-turn signal
(corr of predicted vs real Opus-Haiku gap 0 -> +0.1..+0.3). Gains hold across conversations
(~+0.06) and shrink but stay positive on average for an unseen source (full: +0.021 mean),
i.e. part of what it learns is source-specific. n=99; the split-half lambda lands at 0.2-0.5.

## Built: `aar_gateway/routers/manifold-nirt-claude-judged.npz` (2026-10-22)
`scripts/routing/export_judged_router.py`: pm-ft-d1024-s{42,43,44} each fine-tuned (`full`,
lr 1e-3, 300 steps, per-tier offsets; `training.nirt.judge_finetune`) on all 99 judged turns,
folded over the tiers' own profiles (`fold_nirt(..., offsets=)`), logits averaged into one MLP
(`ensemble_fold`, hidden 192). Parity 3.6e-7 in memory and after reload. Fine-tuned
checkpoints: `data/processed/nirt_runs/<run>-judged/finetuned.pt`.

Out-of-fold estimate of exactly this (3-seed logit mean, `judge_finetune_cv_*.json`):
folds by conversation +0.069 vs mix, 75% saved, dQ -0.055; leave-one-source-out +0.018,
69% saved, dQ -0.092. lambda presets set from the conversation split: quality 0.02
(24% saved, -0.005), balanced 0.2 (72%, -0.046), save_more 0.3 (76%, -0.055).
On the separately graded 65-prompt pool it does NOT beat the mix line (-0.007): it learned
the judge pool's large Haiku gap (+0.26); that pool's real gap is 0.058.

## Behind the Aptic plugin: contract + latency (`plugin_route_check.json`)
`scripts/routing/plugin_route_check.py`: isolated gateway (own home, port 8797, no upstream),
`POST /api/route` with the plugin's exact body, 120 warm calls per file (40 judged turns x
lambda off / 0.02 / 0.3; 48 of them > 2,000 chars). CPU, onnxruntime.

| file | startup | cold call | warm p50 | short p50 | long p50 | p95 | max |
|---|---|---|---|---|---|---|---|
| manifold-nirt-claude-judged (BERT) | 2.9 s | 43 ms | 46 ms | 28 ms | 151 ms | 169 ms | 182 ms |
| router.npz MLP default (MiniLM) | 2.3 s | 23 ms | 26 ms | 24 ms | 32 ms | 47 ms | 51 ms |

0 contract violations; every answer a tier or null. Lambda moves the picks
(manifold: 0.02 -> 2 H / 19 S / 19 O; 0.3 -> 18 H / 22 S / 0 O).

## Potential savings on the user's own sessions (`session_savings_{7,30}d.json`)
`scripts/routing/session_savings.py`: own Claude Code transcripts (VS Code; Windows temp
grading/bench folders excluded -- `_TEMP_PROJECTS` only knows macOS/Linux temp paths), typed
turns replayed through the gateway picker with session context, every request's REAL usage
repriced on the picked tier, a switch's first request pays the cache rewrite (1h write). No
upgrades. Baseline = the model each turn actually ran on (API-equivalent $). Predicted quality
is the router's own estimate, not measured.

| window | baseline | margin 0.05 | lambda 0.02 | lambda 0.2 | lambda 0.3 |
|---|---|---|---|---|---|
| 7 days (13 sessions, 122 priced turns, 1,715 requests) | $127.06 | $88.71 (-30%) | $125.53 (-1%) | $50.00 (-61%) | $41.89 (-67%) |
| 30 days (794 sessions, 1,113 turns, 8,967 requests) | $682.84 | $411.64 (-40%) | $526.35 (-23%) | $243.88 (-64%) | $190.66 (-72%) |

Predicted quality change: margin -0.014/-0.007, lambda 0.2 -0.042/-0.031, lambda 0.3 -0.064/-0.048.

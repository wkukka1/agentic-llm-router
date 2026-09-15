"""Collect anchor-topology judge margins (docs/anchor_judge.md).

Phase A: for each sampled query x each of the 10 non-anchor pool models,
judge the candidate's RouterBench answer against the anchor's, with a
deterministic (seed, query_id, anchor, candidate)-keyed A/B position
assignment (so position isn't confounded with model identity, and reruns are
reproducible regardless of iteration order). Phase B: re-judge a
margin-stratified ~10% subsample with position swapped, for the noise floor
(``JudgeClient.judge(..., swap=True)`` already negates back into the caller's
original frame -- see ``src/router/judge/client.py``).

Any REAL (non-dummy) provider requires ``--confirm-budget`` AND explicit
``--price-per-1k-input``/``--price-per-1k-output`` (or the config
equivalents) -- the script builds every planned prompt up front (the
response texts are already fully known) and prints the total projected cost
before placing a single paid call. Token counts are an approximation
(``tiktoken`` if installed, else chars/4) since exact provider tokenization
isn't available offline -- treat the dollar figure as a budget ceiling, not
an invoice.

    python scripts/data/collect_anchor_judgments.py --dry-run
    python scripts/data/collect_anchor_judgments.py --dry-run --dummy-mode random
    python scripts/data/collect_anchor_judgments.py --confirm-budget \\
        --provider openai --model gpt-4o-mini \\
        --price-per-1k-input 0.15 --price-per-1k-output 0.60

Prerequisites: scripts/data/select_anchor_model.py and
scripts/data/select_anchor_judge_queries.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from training.cli import get_config, info, write_json

# Kept in sync with scripts/data/select_anchor_model.py's ROUTERBENCH_11 --
# scripts/ isn't a package, so this is duplicated rather than cross-imported.
ROUTERBENCH_11 = [
    "gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-v1", "claude-v2",
    "claude-instant-v1", "llama-2-70b-chat", "code-llama-34b-instruct",
    "mixtral-8x7b-instruct", "mistral-7b-instruct", "wizardlm-13b-v1.2", "yi-34b-chat",
]

_MARGIN_BUCKETS = (-3, -2, -1, 0, 1, 2, 3)


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _anchor_is_a(seed: int, query_id: str, anchor: str, candidate: str) -> bool:
    """Deterministic A/B position assignment, keyed on (seed, query, anchor, candidate)
    rather than iteration order -- resuming/reordering collection doesn't reshuffle it."""
    h = hashlib.sha1(f"{seed}:{query_id}:{anchor}:{candidate}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2 == 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="path to phase0.yaml")
    ap.add_argument("--dry-run", action="store_true",
                     help="force the dummy provider, cap at anchor_judge.dry_run_n, no network/cost")
    ap.add_argument("--dummy-mode", default="correctness-proxy",
                     choices=["correctness-proxy", "random"],
                     help="dummy provider behavior (dry runs / tests only)")
    ap.add_argument("--confirm-budget", action="store_true",
                     help="required for any real (non-dummy) provider")
    ap.add_argument("--provider", default=None, choices=["openai", "anthropic", "dummy"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--price-per-1k-input", type=float, default=None)
    ap.add_argument("--price-per-1k-output", type=float, default=None)
    ap.add_argument("--phase-a-only", action="store_true",
                     help="stop after Phase A so the anchor-degeneracy / parse-failure "
                          "report can be reviewed before spending the Phase B budget")
    ap.add_argument("--seed", type=int, default=None)
    return ap


def _projected_cost(prompts_input_chars_or_tokens, n_swap, price_in, price_out,
                     *, output_tokens_per_call: int = 60):
    n_phase_a = len(prompts_input_chars_or_tokens)
    total_input_tokens = sum(prompts_input_chars_or_tokens)
    avg_input_tokens = (total_input_tokens / n_phase_a) if n_phase_a else 0.0
    total_input_tokens += avg_input_tokens * n_swap  # Phase B re-judges the same pairs
    n_calls = n_phase_a + n_swap
    total_output_tokens = n_calls * output_tokens_per_call
    cost = (total_input_tokens / 1000) * price_in + (total_output_tokens / 1000) * price_out
    return {
        "n_calls_phase_a": n_phase_a,
        "n_calls_phase_b_planned": n_swap,
        "total_input_tokens_est": int(total_input_tokens),
        "total_output_tokens_est": int(total_output_tokens),
        "projected_cost_usd": round(cost, 2),
    }


def main() -> int:
    args = _build_parser().parse_args()
    cfg = get_config(args)
    acfg = cfg.get("anchor_judge")
    if acfg is None:
        raise ValueError("configs/phase0.yaml is missing an `anchor_judge:` block")

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    out_dir = cfg.resolve(acfg.get("output_dir", "data/processed/anchor_judge"))
    sampled_path = out_dir / "sampled_queries.parquet"
    if not sampled_path.exists():
        raise FileNotFoundError(
            f"{sampled_path} missing -- run scripts/data/select_anchor_judge_queries.py first"
        )
    sampled = pd.read_parquet(sampled_path)

    anchor_model = acfg.get("anchor_model_id")
    if not anchor_model:
        sel_path = out_dir / "anchor_selection.json"
        if not sel_path.exists():
            raise ValueError(
                "anchor_judge.anchor_model_id is not set and "
                f"{sel_path} doesn't exist -- run scripts/data/select_anchor_model.py "
                "and copy anchor_model_id into configs/phase0.yaml"
            )
        anchor_model = json.loads(sel_path.read_text())["anchor_model_id"]
        info(f"anchor_judge.anchor_model_id unset in config; using {anchor_model!r} from {sel_path}")

    pool = list(acfg.get("pool", ROUTERBENCH_11))
    candidates = [m for m in pool if m != anchor_model]

    dummy_mode = args.dummy_mode
    if args.dry_run:
        provider, model = "dummy", f"dummy-{dummy_mode}"
        n = min(int(acfg.get("dry_run_n", 30)), len(sampled))
        # Keep both strata represented even in a tiny dry run -- the analysis
        # script's void-check needs discriminative-stratum informative pairs,
        # which a plain sequential head() could starve entirely.
        per_stratum = max(1, n // 2)
        parts = [sampled[sampled["stratum"] == s].iloc[:per_stratum]
                 for s in ("discriminative", "natural")]
        sampled = pd.concat(parts).iloc[:n] if any(len(p) for p in parts) else sampled.iloc[:n]
        price_in = price_out = 0.0
    else:
        provider = args.provider or acfg.get("provider")
        model = args.model or acfg.get("model")
        if provider is None or model is None:
            raise ValueError("--provider/--model (or config anchor_judge.provider/model) required")
        price_in = args.price_per_1k_input if args.price_per_1k_input is not None \
            else acfg.get("price_per_1k_input_tokens")
        price_out = args.price_per_1k_output if args.price_per_1k_output is not None \
            else acfg.get("price_per_1k_output_tokens")
        if provider != "dummy":
            if price_in is None or price_out is None:
                raise ValueError(
                    "real provider requires --price-per-1k-input/--price-per-1k-output "
                    "(or anchor_judge.price_per_1k_*_tokens in config)"
                )

    from evaluation.data.judge_responses import response_text_lookup
    from training.data.facade import load_training_data
    from evaluation.judge.client import JudgeClient
    from evaluation.judge.prompts import build_prompt

    d = load_training_data(cfg)
    R = d.correctness_matrix(
        metric="accuracy", models="warm", combine_metrics=["accuracy", "mc_accuracy"],
    ).reindex(columns=pool)
    queries_text = d.query_texts()

    query_ids = sampled.index.tolist()
    texts = response_text_lookup(cfg, query_ids=query_ids, shot=acfg.get("shot", "0shot"))
    missing_text = [(q, m) for q in query_ids for m in pool if (q, m) not in texts]
    if missing_text:
        raise ValueError(
            f"{len(missing_text)} (query, model) pairs missing response text, "
            f"e.g. {missing_text[:5]}"
        )

    swap_fraction = float(acfg.get("swap_fraction", 0.10))
    n_planned_a = len(query_ids) * len(candidates)
    n_planned_b = int(round(n_planned_a * swap_fraction))

    if provider not in ("dummy",):
        prompt_token_counts = []
        for q in query_ids:
            q_text = queries_text.get(q, "")
            anchor_text = texts[(q, anchor_model)]
            for m in candidates:
                messages = build_prompt(q_text, anchor_text, texts[(q, m)])
                prompt_token_counts.append(
                    _estimate_tokens("".join(msg["content"] for msg in messages))
                )
        cost = _projected_cost(prompt_token_counts, n_planned_b, price_in, price_out)
        info(f"projected cost: ${cost['projected_cost_usd']:.2f} "
             f"({cost['n_calls_phase_a']} + {cost['n_calls_phase_b_planned']} calls, "
             f"~{cost['total_input_tokens_est']:,} input / "
             f"~{cost['total_output_tokens_est']:,} output tokens, approximate)")
        if not args.confirm_budget:
            raise SystemExit(
                "refusing to place real API calls without --confirm-budget "
                "(see the projected cost printed above)"
            )
    else:
        cost = {"n_calls_phase_a": n_planned_a, "n_calls_phase_b_planned": n_planned_b,
                 "total_input_tokens_est": 0, "total_output_tokens_est": 0,
                 "projected_cost_usd": 0.0}

    import os

    api_key = os.environ.get(acfg.get("api_key_env", "JUDGE_API_KEY")) if provider != "dummy" else None
    base_url = os.environ.get(acfg.get("base_url_env", "JUDGE_BASE_URL")) or None
    client = JudgeClient(
        provider=provider, model=model, api_key=api_key, base_url=base_url,
        temperature=float(acfg.get("temperature", 0.0)),
        max_retries=int(acfg.get("max_retries", 5)),
        requests_per_minute=int(acfg.get("requests_per_minute", 60)),
        dummy_mode=dummy_mode, seed=seed,
    )

    # ---- Phase A: primary judgments ------------------------------------------
    rows = []
    for q in query_ids:
        q_text = queries_text.get(q, "")
        anchor_text = texts[(q, anchor_model)]
        correct_anchor = float(R.loc[q, anchor_model]) if q in R.index else None
        for m in candidates:
            cand_text = texts[(q, m)]
            correct_cand = float(R.loc[q, m]) if q in R.index else None
            anchor_is_a = _anchor_is_a(seed, q, anchor_model, m)
            if anchor_is_a:
                a_text, b_text, ca, cb = anchor_text, cand_text, correct_anchor, correct_cand
            else:
                a_text, b_text, ca, cb = cand_text, anchor_text, correct_cand, correct_anchor
            verdict = client.judge(q_text, a_text, b_text, correct_a=ca, correct_b=cb)
            gain = verdict.margin if anchor_is_a else -verdict.margin
            rows.append({
                "query_id": q, "anchor_model_id": anchor_model, "candidate_model_id": m,
                "gain": gain, "anchor_is_a": anchor_is_a, "reason": verdict.reason,
                "raw": verdict.raw, "parsed_ok": verdict.parsed_ok,
                "judge_provider": provider, "judge_model": model, "swapped": False,
            })
    judgments = pd.DataFrame(rows)

    parse_failure_rate = float((~judgments["parsed_ok"]).mean()) if len(judgments) else 0.0
    disc_ids = set(sampled.index[sampled["stratum"] == "discriminative"])
    # Unparsed verdicts carry gain 0 and would pull the mean toward a tie.
    parsed = judgments[judgments["parsed_ok"]] if len(judgments) else judgments
    anchor_mean_gain_disc = float(
        parsed.loc[parsed["query_id"].isin(disc_ids), "gain"].mean()
    ) if disc_ids else float("nan")
    info(f"Phase A done: {len(judgments)} judgments, "
         f"parse_failure_rate={parse_failure_rate:.3f}, "
         f"mean gain (discriminative stratum)={anchor_mean_gain_disc:.3f}")
    if parse_failure_rate > float(acfg.get("max_unparseable_rate", 0.05)):
        info(f"WARNING: parse_failure_rate {parse_failure_rate:.3f} exceeds "
             f"max_unparseable_rate {acfg.get('max_unparseable_rate', 0.05)} -- "
             "scripts/nirt/anchor_judge_analysis.py will treat this run as VOID")
    if abs(anchor_mean_gain_disc) >= 2.5:
        info(f"WARNING: anchor mean gain {anchor_mean_gain_disc:.3f} on the discriminative "
             "stratum is near the +/-3 extreme -- the anchor may be degenerate "
             "(wins/loses almost everything), compressing the signal")

    out_dir.mkdir(parents=True, exist_ok=True)
    judgments.to_parquet(out_dir / "judgments.parquet")

    if args.phase_a_only:
        info("--phase-a-only: stopping before Phase B. Review the report above, then "
             "re-run without --phase-a-only to collect the swap/noise-floor subsample.")
        write_json(out_dir / "collection_summary.json", {
            **cost, "phase": "a_only", "parse_failure_rate": parse_failure_rate,
            "anchor_mean_gain_discriminative": anchor_mean_gain_disc,
            "provider": provider, "model": model, "anchor_model_id": anchor_model,
        }, root=cfg.root)
        return 0

    # ---- Phase B: position-swapped re-judgments, stratified by Phase-A gain --
    rng = np.random.default_rng(seed)
    per_bucket = max(1, n_planned_b // len(_MARGIN_BUCKETS))
    swap_idx = []
    for bucket in _MARGIN_BUCKETS:
        in_bucket = judgments.index[judgments["gain"] == bucket].to_numpy()
        take = min(per_bucket, len(in_bucket))
        if take:
            swap_idx.extend(rng.choice(in_bucket, size=take, replace=False).tolist())

    swap_rows = []
    for idx in swap_idx:
        row = judgments.loc[idx]
        q, m = row["query_id"], row["candidate_model_id"]
        anchor_is_a = bool(row["anchor_is_a"])
        anchor_text = texts[(q, anchor_model)]
        cand_text = texts[(q, m)]
        q_text = queries_text.get(q, "")
        correct_anchor = float(R.loc[q, anchor_model]) if q in R.index else None
        correct_cand = float(R.loc[q, m]) if q in R.index else None
        if anchor_is_a:
            a_text, b_text, ca, cb = anchor_text, cand_text, correct_anchor, correct_cand
        else:
            a_text, b_text, ca, cb = cand_text, anchor_text, correct_cand, correct_anchor
        verdict = client.judge(q_text, a_text, b_text, swap=True, correct_a=ca, correct_b=cb)
        gain = verdict.margin if anchor_is_a else -verdict.margin
        swap_rows.append({
            "query_id": q, "anchor_model_id": anchor_model, "candidate_model_id": m,
            "gain": gain, "anchor_is_a": anchor_is_a, "reason": verdict.reason,
            "raw": verdict.raw, "parsed_ok": verdict.parsed_ok,
            "judge_provider": provider, "judge_model": model, "swapped": True,
        })

    all_judgments = pd.concat([judgments, pd.DataFrame(swap_rows)], ignore_index=True)
    all_judgments.to_parquet(out_dir / "judgments.parquet")

    usage = client.usage
    write_json(out_dir / "collection_summary.json", {
        "n_calls_phase_a": len(judgments), "n_calls_phase_b": len(swap_rows),
        "parse_failure_rate": parse_failure_rate,
        "anchor_mean_gain_discriminative": anchor_mean_gain_disc,
        "projected_cost_usd": cost.get("projected_cost_usd", 0.0),
        "client_usage": usage,
        "provider": provider, "model": model, "anchor_model_id": anchor_model,
    }, root=cfg.root)
    info(f"Phase B done: {len(swap_rows)} swapped re-judgments -> "
         f"{out_dir / 'judgments.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Anchor-judge decision-rule analysis (docs/anchor_judge.md).

Reads ``data/processed/anchor_judge/{sampled_queries,judgments}.parquet``
and applies the pre-registered decision rule:

  1. Validity gates (computed on the LIVE judge run, not just the dummy
     self-test): parse-failure rate <= ``max_unparseable_rate``, and the
     correctness-classifier void-check -- on the discriminative stratum's
     informative pairs (gold correctness differs between anchor and
     candidate), the judge must prefer the gold-INCORRECT response more than
     ``void_check_min_incorrect_preferred_rate`` of the time. A judge that
     never does this is acting as a correctness classifier, not a preference
     judge, and the run is void regardless of the fork below.
  2. Noise floor: for queries touched by a Phase-B (position-swapped)
     re-judgment, recompute that query's routing decision / regret with the
     swapped gain substituted for the touched candidate, restricted to just
     those queries (not diluted across the full sample) -- the shift from
     the original values on that same restricted set is the floor.
  3. Primary fork: build the [Q, M] judge gain matrix (anchor column = 0),
     score it against gold correctness via the EXISTING
     ``evaluation.routing.oracle.routing_evaluation`` (regret directly
     comparable to the 0.1181 baseline in docs/nirt_capacity.md) and the
     query-level decision-disagreement rate, both on the discriminative and
     natural strata separately, each judged against the Step-2 noise floor.
  4. Secondary diagnostic (not the fork): informative-pair agreement rate.
  5. Verdict, printed verbatim.

    python scripts/nirt/anchor_judge_analysis.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from training.cli import get_config, info, raw_parser, write_json

ROUTERBENCH_11 = [
    "gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-v1", "claude-v2",
    "claude-instant-v1", "llama-2-70b-chat", "code-llama-34b-instruct",
    "mixtral-8x7b-instruct", "mistral-7b-instruct", "wizardlm-13b-v1.2", "yi-34b-chat",
]

_TOL = 1e-9


def _gain_matrix(unswapped: pd.DataFrame, anchor: str, pool: list, query_ids: list) -> pd.DataFrame:
    mat = unswapped.pivot(index="query_id", columns="candidate_model_id", values="gain")
    mat = mat.reindex(index=query_ids, columns=[m for m in pool if m != anchor])
    if mat.isna().any().any():
        missing = int(mat.isna().sum().sum())
        raise ValueError(f"{missing} (query, candidate) gain cells missing from judgments.parquet")
    mat[anchor] = 0.0
    return mat[pool]


def _decision_metrics(gain: pd.DataFrame, R: pd.DataFrame) -> dict:
    from evaluation.routing.oracle import routing_evaluation

    result = routing_evaluation(
        gain.to_numpy(np.float64), R.to_numpy(np.float64), None, list(gain.columns),
    )
    return {
        "mean_regret": result["mean_regret"],
        "decision_disagreement_rate": 1.0 - result["oracle_hit_rate"],
    }


def _void_check_and_secondary(unswapped: pd.DataFrame, R: pd.DataFrame, anchor: str,
                               disc_ids: set) -> dict:
    rows = unswapped[unswapped["query_id"].isin(disc_ids)].copy()
    correct_anchor = rows["query_id"].map(lambda q: R.loc[q, anchor] >= 0.5)
    correct_cand = [
        R.loc[q, m] >= 0.5 for q, m in zip(rows["query_id"], rows["candidate_model_id"])
    ]
    rows["correct_anchor"] = correct_anchor.to_numpy()
    rows["correct_candidate"] = correct_cand
    informative = rows[rows["correct_anchor"] != rows["correct_candidate"]]

    def _prefers_incorrect(r) -> float:
        candidate_is_correct = r["correct_candidate"]
        if r["gain"] == 0:
            return 0.5
        prefers_candidate = r["gain"] > 0
        prefers_incorrect = (prefers_candidate and not candidate_is_correct) or \
            ((not prefers_candidate) and candidate_is_correct)
        return 1.0 if prefers_incorrect else 0.0

    if informative.empty:
        return {"n_informative_pairs": 0, "prefers_incorrect_rate": float("nan"),
                "informative_pair_agreement_rate": float("nan"),
                "contingency_table": {}}

    prefers_incorrect_rate = float(informative.apply(_prefers_incorrect, axis=1).mean())
    contingency = (
        informative.assign(prefers_candidate=lambda d: np.sign(d["gain"]))
        .groupby(["correct_anchor", "correct_candidate"])["prefers_candidate"]
        .mean().to_dict()
    )
    return {
        "n_informative_pairs": int(len(informative)),
        "prefers_incorrect_rate": prefers_incorrect_rate,
        "informative_pair_agreement_rate": 1.0 - prefers_incorrect_rate,
        "contingency_table": {str(k): float(v) for k, v in contingency.items()},
    }


def _noise_floor(unswapped: pd.DataFrame, swapped: pd.DataFrame, anchor: str, pool: list,
                  R: pd.DataFrame) -> dict:
    if swapped.empty:
        return {"n_swap_queries": 0, "regret_shift": float("nan"),
                "decision_disagreement_shift": float("nan")}

    touched_queries = swapped["query_id"].unique().tolist()
    gain_orig_full = _gain_matrix(unswapped, anchor, pool, unswapped["query_id"].unique().tolist())
    gain_pert_full = gain_orig_full.copy()
    for _, row in swapped.iterrows():
        gain_pert_full.loc[row["query_id"], row["candidate_model_id"]] = row["gain"]

    R_touched = R.reindex(index=touched_queries)
    orig = _decision_metrics(gain_orig_full.loc[touched_queries], R_touched)
    pert = _decision_metrics(gain_pert_full.loc[touched_queries], R_touched)
    return {
        "n_swap_queries": len(touched_queries),
        "regret_shift": abs(pert["mean_regret"] - orig["mean_regret"]),
        "decision_disagreement_shift": abs(
            pert["decision_disagreement_rate"] - orig["decision_disagreement_rate"]
        ),
    }


def main() -> int:
    ap = raw_parser(__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = get_config(args)
    acfg = cfg.get("anchor_judge")
    out_dir = cfg.resolve(acfg.get("output_dir", "data/processed/anchor_judge"))

    sampled = pd.read_parquet(out_dir / "sampled_queries.parquet")
    judgments = pd.read_parquet(out_dir / "judgments.parquet")
    unswapped = judgments[~judgments["swapped"]].copy()
    swapped = judgments[judgments["swapped"]].copy()

    # Restrict to queries actually present in judgments.parquet -- a --dry-run
    # (or any partial collection) covers only a subset of sampled_queries.parquet,
    # and the analysis must be self-consistent with what was actually judged.
    judged_ids = set(unswapped["query_id"].unique())
    sampled = sampled.loc[sampled.index.isin(judged_ids)]

    anchor = str(judgments["anchor_model_id"].iloc[0])
    pool = list(acfg.get("pool", ROUTERBENCH_11))

    from training.data.facade import load_training_data

    d = load_training_data(cfg)
    R = d.correctness_matrix(
        metric="accuracy", models="warm", combine_metrics=["accuracy", "mc_accuracy"],
    ).reindex(columns=pool, index=sampled.index)
    if R.isna().any().any():
        raise ValueError("sampled queries missing correctness data for one or more pool models")

    disc_ids = set(sampled.index[sampled["stratum"] == "discriminative"])
    natural_ids = set(sampled.index[sampled["stratum"] == "natural"])

    # ---- gate 1: parse quality -------------------------------------------------
    parse_failure_rate = float((~judgments["parsed_ok"]).mean())
    max_unparseable = float(acfg.get("max_unparseable_rate", 0.05))
    gate_parse_ok = parse_failure_rate <= max_unparseable

    # ---- gate 2: correctness-classifier void-check -----------------------------
    void = _void_check_and_secondary(unswapped, R, anchor, disc_ids)
    min_incorrect_rate = float(acfg.get("void_check_min_incorrect_preferred_rate", 0.10))
    gate_void_ok = (
        void["n_informative_pairs"] > 0
        and void["prefers_incorrect_rate"] > min_incorrect_rate
    )

    report: dict = {
        "anchor_model_id": anchor,
        "gate_parse_quality": {"parse_failure_rate": parse_failure_rate,
                                "threshold": max_unparseable, "passed": gate_parse_ok},
        "gate_void_check": {**void, "threshold": min_incorrect_rate, "passed": gate_void_ok},
    }

    if not gate_parse_ok:
        report["verdict"] = "VOID: parse_failure_rate exceeds max_unparseable_rate"
    elif not gate_void_ok:
        report["verdict"] = "VOID: judge appears to act as a correctness classifier"
    else:
        noise = _noise_floor(unswapped, swapped, anchor, pool, R)
        report["noise_floor"] = noise

        # Raw decision-disagreement/regret against a correctness oracle is large
        # for ANY uninformative signal -- a judge that carries zero information
        # about correctness will disagree with the oracle about as often as
        # random routing does, which looks identical to "genuine conflict" if
        # compared to the (tiny) position-swap noise floor alone. So before
        # applying the two-utilities-vs-domain-shift fork, require the judge's
        # implied routing to beat a fixed-seed random-routing baseline by more
        # than that noise floor -- otherwise there is no signal here to
        # adjudicate the hypothesis either way. This gate is what
        # dummy-random-mode verification actually exercises (see docs/anchor_judge.md's
        # verification plan): without it, pure noise falsely triggers "two utilities".
        strata = {}
        for name, ids in (("discriminative", disc_ids), ("natural", natural_ids)):
            if not ids:
                continue
            qids = [q for q in ids]
            gain = _gain_matrix(unswapped, anchor, pool, qids)
            metrics = _decision_metrics(gain, R.loc[qids])
            # Structure-matched baseline: the anchor column in `gain` is ALWAYS
            # exactly 0 by construction, so a fair "is this judge informative"
            # baseline must hold the anchor column at 0 too and only randomize
            # the candidate columns (over the same -3..3 integer margin scale)
            # -- randomizing all 11 columns independently is NOT apples-to-apples
            # and creates a spurious "beats random" signal purely from that
            # structural difference (verified empirically: a genuinely random
            # dummy judge falsely "beat" an all-columns-random baseline at both
            # n=30 and n=400 until this was fixed).
            rng = np.random.default_rng(0)
            random_gain = pd.DataFrame(
                rng.integers(-3, 4, size=(len(qids), len(pool))).astype(float),
                index=qids, columns=pool,
            )
            random_gain[anchor] = 0.0
            random_metrics = _decision_metrics(random_gain[pool], R.loc[qids])
            beats_random = (
                random_metrics["mean_regret"] - metrics["mean_regret"]
            ) > noise.get("regret_shift", 0.0)
            strata[name] = {
                **metrics,
                "random_baseline_regret": random_metrics["mean_regret"],
                "random_baseline_decision_disagreement": random_metrics["decision_disagreement_rate"],
                "beats_random_baseline": beats_random,
                "excess_regret": metrics["mean_regret"] - noise.get("regret_shift", 0.0),
                "excess_decision_disagreement": metrics["decision_disagreement_rate"]
                - noise.get("decision_disagreement_shift", 0.0),
            }
        report["strata"] = strata

        informative_strata = {n: s for n, s in strata.items() if s["beats_random_baseline"]}
        if not informative_strata:
            report["verdict"] = (
                "INCONCLUSIVE: judge-implied routing is statistically indistinguishable from "
                "random routing on every stratum (does not beat the random-routing baseline "
                "beyond the swap noise floor) -- there is no detectable correctness-relevant "
                "signal in this judge run to evaluate the two-utilities hypothesis. This is "
                "NOT a domain-shift retraction and NOT evidence of two utilities; it means the "
                "judge itself needs fixing (rubric, model, or scale) before this run can answer "
                "the question."
            )
        else:
            material = any(
                s["excess_decision_disagreement"] > 2 * noise.get("decision_disagreement_shift", 0.0)
                or s["excess_regret"] > 2 * noise.get("regret_shift", 0.0)
                for s in informative_strata.values()
            )
            if material:
                report["verdict"] = (
                    "TWO UTILITIES: excess regret/decision-disagreement is stable and material "
                    "beyond the swap noise floor, on a stratum where the judge demonstrably beats "
                    "random routing. Build the minimal per-channel-bias model (shared low-rank A, "
                    "model embeddings e_m, per-channel bias beta_{m,channel}; channel observed via "
                    "`source`, no gate, no latent mixture, no Psi matrix) and test it against the "
                    "pooled train.preference.weight baseline on this set."
                )
            else:
                report["verdict"] = (
                    "DOMAIN SHIFT: excess regret/decision-disagreement is indistinguishable from "
                    "the swap noise floor, on stratum/strata where the judge demonstrably carries "
                    "correctness-relevant signal. RETRACT (not just caveat) docs/nirt_capacity.md "
                    "Item 2's 'different notions of quality' conclusion -- the original "
                    "train.preference.weight sweep measured domain shift (160k OOD training "
                    "queries), not incompatible utilities. Close the mixture/regime-model "
                    "research line."
                )

    write_json(out_dir / "anchor_judge_analysis.json", report, root=cfg.root)
    info(report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

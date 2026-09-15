"""Source dataset loaders.

Each ``load_*`` returns a DataFrame already in canonical response-observation
form (see :mod:`router.data.schemas`). Loaders do **not** write files and do not
mutate anything on disk except when explicitly downloading raw inputs.

Sources
-------
RouterBench      local git-LFS clone of ``withmartian/routerbench`` (pickled
                 DataFrames). Preserves the graded ``performance`` score, the
                 native ``eval_name`` task, per-model cost, and the dev/test
                 marker embedded in ``sample_id``.
Chatbot Arena    ``lmarena-ai/arena-human-preference-55k`` saved via
                 ``datasets.save_to_disk``. Each pairwise battle becomes two
                 observations (one per participating model) with a *pairwise*
                 preference score in {0.0, 0.5, 1.0}. This is NOT an absolute
                 correctness score -- see docs/data_sources.md.
lm-eval-harness  a directory of harness result files. The per-sample logs
                 (``--log_samples``) are consumed at query granularity; bare
                 aggregate results cannot enter the response matrix and are
                 skipped with a warning.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from router.config import Config, section
from . import schemas
from .model_registry import canonical_model_id
from .normalize import make_query_id, render_prompt

_SPLIT_TOKENS = {"dev", "test", "train", "val", "valid", "validation"}


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _mc_choices(eval_name: str, cfg: Config) -> Optional[int]:
    """Number of answer choices for an eval task, or None if not MC."""
    mc = section(cfg, "multiple_choice")
    by_task = mc.get("by_task", {}) or {}
    if eval_name in by_task:
        return int(by_task[eval_name])
    name = eval_name.lower()
    for prefix, n in (mc.get("by_prefix", {}) or {}).items():
        if name.startswith(prefix.lower()):
            return int(n)
    return None


def _parse_routerbench_split(sample_id: str) -> Optional[str]:
    parts = str(sample_id).rsplit(".", 2)
    if len(parts) == 3 and parts[1].lower() in _SPLIT_TOKENS:
        tok = parts[1].lower()
        return {"valid": "validation", "val": "validation"}.get(tok, tok)
    return None


# --------------------------------------------------------------------------- #
# RouterBench                                                                 #
# --------------------------------------------------------------------------- #
def load_routerbench(cfg: Config) -> pd.DataFrame:
    src_cfg = cfg.sources.routerbench
    local_dir = cfg.resolve(src_cfg.local_dir)
    shots: list[str] = list(src_cfg.shots)

    # Item-identity policy for multi-shot RouterBench (see docs/pool_expansion.md,
    # phase E2): 0-shot and 5-shot of one question are **separate items** with
    # distinct query_ids (``:5shot`` suffix), but BOTH carry the underlying
    # question text (the 0-shot prompt). Consequences:
    #   * they share a content hash -> the same split group -> a question's
    #     0-/5-shot observations never straddle train/test;
    #   * 0-shot query_ids are byte-identical to a 0-shot-only build;
    #   * the shot is recorded in metadata but NOT featurised (the model has no
    #     shot input) -- the predictor sees 2x observations of each question's
    #     difficulty, while the routing matrices carry the true per-shot scores.
    zero_prompts: dict[str, str] = {}
    zero_pkl = local_dir / "routerbench_0shot.pkl"
    zero_wide = None
    if zero_pkl.exists() and shots != ["0shot"]:
        zero_wide = pd.read_pickle(zero_pkl)
        zero_prompts = {
            str(s): render_prompt(p)
            for s, p in zip(zero_wide["sample_id"], zero_wide["prompt"])
        }

    frames: list[pd.DataFrame] = []
    for shot in shots:
        pkl = local_dir / f"routerbench_{shot}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"RouterBench file not found: {pkl}\n"
                f"Run: python scripts/data/download_routerbench.py"
            )
        wide = zero_wide if (shot == "0shot" and zero_wide is not None) else pd.read_pickle(pkl)
        frames.append(_melt_routerbench(wide, shot, cfg, question_text=zero_prompts))
    out = pd.concat(frames, ignore_index=True)
    return schemas.coerce_response_frame(out)


def _melt_routerbench(wide: pd.DataFrame, shot: str, cfg: Config,
                      question_text: Optional[dict[str, str]] = None) -> pd.DataFrame:
    meta_cols = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    model_cols = [
        c for c in wide.columns
        if c not in meta_cols
        and "|" not in c
        and f"{c}|total_cost" in wide.columns
    ]

    native_prompt = wide["prompt"].map(render_prompt)
    qtext = question_text or {}
    is_multishot = shot != "0shot"
    # the item's text is the underlying question (0-shot prompt); the few-shot
    # exemplars are not featurised. Fall back to the native prompt if the 0-shot
    # sibling is somehow absent.
    query = [
        qtext.get(sid, np) if is_multishot else np
        for sid, np in zip(wide["sample_id"].astype(str), native_prompt)
    ]
    base = pd.DataFrame(
        {
            "native_sample_id": wide["sample_id"].astype(str),
            "eval_name": wide["eval_name"].astype(str),
            "query": query,
            "native_prompt": native_prompt.to_numpy(),
            "oracle_model_to_route_to": wide.get("oracle_model_to_route_to"),
        }
    )
    suffix = "" if not is_multishot else f":{shot}"
    base["query_id"] = [
        make_query_id(schemas.Source.ROUTERBENCH, e, q) + suffix
        for e, q in zip(base["eval_name"], base["query"])
    ]
    base["split"] = base["native_sample_id"].map(_parse_routerbench_split)

    choices_of = {e: _mc_choices(e, cfg) for e in base["eval_name"].unique()}
    n_choices_by_task = base["eval_name"].map(choices_of)
    is_mc = n_choices_by_task.notna()

    frames: list[pd.DataFrame] = []
    dropped_na = 0
    for m in model_cols:
        score = pd.to_numeric(wide[m], errors="coerce")
        cost = pd.to_numeric(wide[f"{m}|total_cost"], errors="coerce")
        keep = score.notna()
        dropped_na += int((~keep).sum())
        part = pd.DataFrame(
            {
                "query_id": base["query_id"],
                "model_id": canonical_model_id(m),
                "query": base["query"],
                "score": score,
                "metric_type": is_mc.map(
                    lambda x: schemas.MetricType.MC_ACCURACY
                    if x
                    else schemas.MetricType.ACCURACY
                ),
                "dataset": base["eval_name"],
                "split": base["split"],
                "source": schemas.Source.ROUTERBENCH,
                "is_multiple_choice": is_mc,
                "n_choices": n_choices_by_task,
                "input_tokens": None,     # RouterBench reports cost, not tokens
                "output_tokens": None,
                "latency": None,
                "cost": cost,
            }
        )
        part["metadata"] = [
            {
                "native_sample_id": sid,
                "native_model_name": m,
                "shots": shot,
                "eval_name": e,
                "oracle_model_to_route_to": o,
                "score_is_graded": bool(s not in (0.0, 1.0)),
                "native_prompt_differs_from_0shot": bool(is_multishot and qq != npr),
            }
            for sid, e, o, s, qq, npr in zip(
                base["native_sample_id"], base["eval_name"],
                base["oracle_model_to_route_to"], score,
                base["query"], base["native_prompt"],
            )
        ]
        frames.append(part[keep].reset_index(drop=True))

    if dropped_na:
        warnings.warn(
            f"RouterBench[{shot}]: dropped {dropped_na} (query,model) cells with "
            f"missing scores (model not run on that item)."
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Chatbot Arena                                                               #
# --------------------------------------------------------------------------- #
def _load_pairwise(
    local_dir: Path,
    source: str,
    metric_type: str,
    download_hint: str,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Shared reader for battle-format datasets (Arena, GPT-4 Judge).

    Expected columns: id, model_a, model_b, prompt, winner_model_a,
    winner_model_b, winner_tie. Each battle -> two observations (one per
    participant) with a pairwise-preference score in {1.0, 0.5, 0.0}. These are
    NOT absolute correctness scores; see docs/data_sources.md.
    """
    from datasets import load_from_disk

    if not local_dir.exists():
        raise FileNotFoundError(f"{source} data not found: {local_dir}\n{download_hint}")

    ds = load_from_disk(str(local_dir))
    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    dataset_name = local_dir.name
    records: list[dict] = []
    for i, ex in enumerate(ds):
        prompt = render_prompt(ex["prompt"])
        qid = make_query_id(source, dataset_name, prompt)
        ma = canonical_model_id(ex["model_a"])
        mb = canonical_model_id(ex["model_b"])
        if ex.get("winner_model_a"):
            outcome_a, winner = 1.0, "a"
        elif ex.get("winner_model_b"):
            outcome_a, winner = 0.0, "b"
        else:
            outcome_a, winner = 0.5, "tie"
        battle_id = str(ex.get("id", i))
        pair_id = f"{qid}|{ma}|{mb}|{battle_id}"
        for model_id, opp_id, role, score in (
            (ma, mb, "a", outcome_a),
            (mb, ma, "b", 1.0 - outcome_a),
        ):
            records.append(
                {
                    "query_id": qid,
                    "model_id": model_id,
                    "query": prompt,
                    "score": score,
                    "metric_type": metric_type,
                    "dataset": dataset_name,
                    "split": None,
                    "source": source,
                    "is_multiple_choice": False,
                    "n_choices": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "latency": None,
                    "cost": None,
                    "metadata": {
                        "battle_id": battle_id,
                        "pair_id": pair_id,
                        "opponent_model_id": opp_id,
                        "role": role,
                        "winner": winner,
                        "interpretation": "pairwise_preference",
                    },
                }
            )
    return schemas.coerce_response_frame(pd.DataFrame.from_records(records))


def load_arena(cfg: Config, max_rows: Optional[int] = None) -> pd.DataFrame:
    return _load_pairwise(
        cfg.resolve(cfg.sources.arena.local_dir),
        source=schemas.Source.ARENA,
        metric_type=schemas.MetricType.ARENA_PREFERENCE,
        download_hint="Run: python scripts/data/download_arena.py",
        max_rows=max_rows,
    )


def load_gpt4_judge(cfg: Config, max_rows: Optional[int] = None) -> pd.DataFrame:
    return _load_pairwise(
        cfg.resolve(cfg.sources.gpt4_judge.local_dir),
        source=schemas.Source.GPT4_JUDGE,
        metric_type=schemas.MetricType.JUDGE_PREFERENCE,
        download_hint="Run: python scripts/data/download_arena.py  (also fetches judge battles)",
        max_rows=max_rows,
    )


# --------------------------------------------------------------------------- #
# lm-evaluation-harness                                                       #
# --------------------------------------------------------------------------- #
def load_lm_harness(cfg: Config) -> pd.DataFrame:
    src_cfg = cfg.sources.lm_harness
    results_dir = cfg.resolve(src_cfg.results_dir)
    if not results_dir.exists():
        warnings.warn(
            f"lm-harness results dir {results_dir} does not exist; "
            f"returning no observations. See scripts/data/run_lm_harness.py."
        )
        return schemas.coerce_response_frame(schemas.empty_response_frame())

    sample_files = sorted(results_dir.rglob("*samples*.jsonl")) + sorted(
        results_dir.rglob("*samples*.json")
    )
    records: list[dict] = []
    for path in sample_files:
        records.extend(_parse_lm_harness_samples(path, cfg))

    if not records:
        agg = sorted(results_dir.rglob("results*.json"))
        if agg:
            warnings.warn(
                f"lm-harness: found {len(agg)} aggregate result file(s) but no "
                f"per-sample logs. Aggregates cannot enter the query-level "
                f"response matrix; re-run with --log_samples."
            )
        return schemas.coerce_response_frame(schemas.empty_response_frame())

    return schemas.coerce_response_frame(pd.DataFrame.from_records(records))


_LM_EVAL_TS = re.compile(r"_\d{4}-\d{2}-\d{2}T[\d\-:.]+$")


def _lm_harness_task_and_model(path: Path) -> tuple[str, str]:
    """``(task, native model name)`` from a per-sample log path.

    * lm-eval >= 0.4: ``<output_path>/<model_dir>/samples_<task>_<timestamp>.jsonl``
      -> task from the filename, model from the parent directory.
    * legacy: ``<model>__<task>_samples_<ts>.jsonl``.
    """
    stem = path.stem
    if stem.startswith("samples_"):
        task = _LM_EVAL_TS.sub("", stem[len("samples_"):])
        return task, path.parent.name or "unknown"
    task = stem.split("_samples")[0].split("__")[-1]
    return task, (stem.split("__")[0] if "__" in stem else "unknown")


def _lm_harness_prompt(arguments) -> Optional[str]:
    """First request argument (the prompt), for both the legacy list form
    ``[[prompt, ...], ...]`` and the >= 0.4 dict form ``{"gen_args_0": {"arg_0": prompt}}``."""
    if not arguments:
        return None
    first = next(iter(arguments.values())) if isinstance(arguments, dict) else arguments[0]
    if isinstance(first, dict):
        first = first.get("arg_0", next(iter(first.values()), None))
    elif isinstance(first, (list, tuple)):
        first = first[0] if first else None
    return None if first is None else str(first)


def _parse_lm_harness_samples(path: Path, cfg: Config) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get("samples", [])

    task, model_native = _lm_harness_task_and_model(path)
    model_id = canonical_model_id(model_native)
    n_choices = _mc_choices(task, cfg)

    out: list[dict] = []
    for row in rows:
        doc = row.get("doc", {}) or {}
        doc_text = _lm_harness_prompt(row.get("arguments"))
        if doc_text is None:
            doc_text = doc.get("question") or doc.get("query") or json.dumps(doc)
        doc_text = render_prompt(doc_text)
        qid = make_query_id(schemas.Source.LM_HARNESS, task, doc_text)
        for metric_key, metric_type in (
            ("acc", schemas.MetricType.LIKELIHOOD_ACC),
            ("acc_norm", schemas.MetricType.LIKELIHOOD_ACC_NORM),
            ("exact_match", schemas.MetricType.EXACT_MATCH),
            ("f1", schemas.MetricType.F1),
            ("pass@1", schemas.MetricType.PASS_AT_1),
        ):
            if metric_key not in row:
                continue
            is_mc = metric_type in (
                schemas.MetricType.LIKELIHOOD_ACC,
                schemas.MetricType.LIKELIHOOD_ACC_NORM,
            ) and n_choices is not None
            out.append(
                {
                    "query_id": qid,
                    "model_id": model_id,
                    "query": doc_text,
                    "score": float(row[metric_key]),
                    "metric_type": metric_type,
                    "dataset": task,
                    "split": row.get("split") or doc.get("split"),
                    "source": schemas.Source.LM_HARNESS,
                    "is_multiple_choice": bool(is_mc),
                    "n_choices": n_choices if is_mc else None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "latency": None,
                    "cost": None,
                    "metadata": {
                        "native_task": task,
                        "native_model_name": model_native,
                        "doc_id": row.get("doc_id"),
                        "result_file": path.name,
                    },
                }
            )
    return out


# --------------------------------------------------------------------------- #
# IRT-Router benchmark (arXiv 2506.01048)                                      #
# --------------------------------------------------------------------------- #
_IRT_ROUTER_FILES = {"train": "train.csv", "test1": "test1.csv", "test2": "test2.csv"}


def load_irt_router(cfg: Config) -> pd.DataFrame:
    """The IRT-Router paper's own suite: 20 LLMs x 12 datasets.

    Reads the three precomputed CSVs
    (``id, question, ground_truth, completion, input_tokens, output_tokens,
    cost, performance, task, llm``) from ``sources.irt_router.local_dir/data``.
    ``performance`` is a graded [0, 1] correctness score; ``cost`` is the
    token-weighted USD cost -- both used verbatim (no price table needed).

    Query identity is the exact ``question`` text (matches the repo's own
    ``query_id_map[question]`` join). The origin file (train / test1 / test2) is
    recorded in ``metadata['origin']``; :func:`router.data.splits.make_presplit`
    turns it into the train / validation / test / ood split.
    """
    src = section(cfg, "sources").get("irt_router", {})
    data_dir = cfg.resolve(src.get("local_dir", "data/raw/irt_router")) / "data"
    pool = {str(m) for m in (src.get("pool") or [])}

    frames: list[pd.DataFrame] = []
    for origin, fname in _IRT_ROUTER_FILES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"IRT-Router file not found: {path}\n"
                f"Run: python scripts/data/download_irt_router.py --config <this config>"
            )
        df = pd.read_csv(path)
        # honour both multiple_choice.by_task and by_prefix, like every other loader
        mc_by_task = {t: n for t in df["task"].astype(str).unique()
                      if (n := _mc_choices(t, cfg)) is not None}
        frames.append(_melt_irt_router(df, origin, pool, mc_by_task))
    out = pd.concat(frames, ignore_index=True)
    return schemas.coerce_response_frame(out)


def _melt_irt_router(df: pd.DataFrame, origin: str, pool: set[str],
                     mc_by_task: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["llm"] = df["llm"].astype(str)
    if pool:
        keep = df["llm"].isin(pool)
        dropped = sorted(set(df.loc[~keep, "llm"].unique()))
        if dropped:
            warnings.warn(
                f"IRT-Router[{origin}]: dropped {int((~keep).sum())} rows for "
                f"{len(dropped)} out-of-pool models: {dropped}"
            )
        df = df[keep]

    question = df["question"].map(render_prompt)
    task = df["task"].astype(str)
    n_choices = task.map(lambda t: mc_by_task.get(t))
    is_mc = n_choices.notna()
    score = pd.to_numeric(df["performance"], errors="coerce")

    part = pd.DataFrame(
        {
            "query_id": [
                make_query_id(schemas.Source.IRT_ROUTER, t, q)
                for t, q in zip(task, question)
            ],
            "model_id": [canonical_model_id(m) for m in df["llm"]],
            "query": question.to_numpy(),
            "score": score.to_numpy(),
            "metric_type": is_mc.map(
                lambda x: schemas.MetricType.MC_ACCURACY if x
                else schemas.MetricType.ACCURACY
            ).to_numpy(),
            "dataset": task.to_numpy(),
            "split": None,
            "source": schemas.Source.IRT_ROUTER,
            "is_multiple_choice": is_mc.to_numpy(),
            "n_choices": n_choices.to_numpy(),
            # some rows average token counts over repeats -> round to int for the
            # canonical schema (cost is precomputed and used verbatim regardless).
            "input_tokens": pd.to_numeric(df["input_tokens"], errors="coerce").round().to_numpy(),
            "output_tokens": pd.to_numeric(df["output_tokens"], errors="coerce").round().to_numpy(),
            "latency": None,
            "cost": pd.to_numeric(df["cost"], errors="coerce").to_numpy(),
        }
    )
    part["metadata"] = [
        {
            "native_model_name": m,
            "task": t,
            "origin": origin,
            "ground_truth": (None if pd.isna(gt) else str(gt)),
        }
        for m, t, gt in zip(df["llm"], task, df.get("ground_truth", pd.Series([None] * len(df))))
    ]
    keep = part["score"].notna()
    if (~keep).any():
        warnings.warn(
            f"IRT-Router[{origin}]: dropped {int((~keep).sum())} rows with "
            f"missing `performance`."
        )
    return part[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Anchor-topology judge (docs/anchor_judge.md)                                #
# --------------------------------------------------------------------------- #
def load_anchor_judge(cfg: Config) -> pd.DataFrame:
    """Manufactured anchor-topology judge preferences on EXISTING ``routerbench:``
    query_ids (docs/anchor_judge.md) -- produced by
    ``scripts/data/collect_anchor_judgments.py``, NOT downloaded.

    Each judged (anchor, candidate) pair -> two battle-shape rows, matching
    ``_load_pairwise``'s contract exactly, so ``TrainingData.pairwise()`` needs
    no changes. The judge's graded ``gain`` (-3..+3, positive favors the
    candidate) is collapsed to the canonical ``{0.0, 0.5, 1.0}`` pairwise
    scale for schema compatibility with the existing Arena/Judge path; the
    raw graded value is kept in ``metadata.margin`` for anything (like the
    primary anchor-judge analysis) that wants the ungraded signal instead of
    reconstructing it from this lossy 3-way collapse.
    """
    acfg = cfg.get("anchor_judge")
    out_dir = cfg.resolve(acfg.get("output_dir", "data/processed/anchor_judge") if acfg
                           else "data/processed/anchor_judge")
    path = out_dir / "judgments.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found\n"
            "Run: python scripts/data/select_anchor_model.py && "
            "python scripts/data/select_anchor_judge_queries.py && "
            "python scripts/data/collect_anchor_judgments.py"
        )
    judgments = pd.read_parquet(path)
    judgments = judgments[~judgments["swapped"]]  # swaps are a noise-floor diagnostic only
    if "parsed_ok" in judgments.columns:
        # An unparsed verdict carries margin 0; kept, it would train as a tie.
        unparsed = ~judgments["parsed_ok"].astype(bool)
        if unparsed.any():
            warnings.warn(
                f"anchor_judge: dropped {int(unparsed.sum())} judgments the judge "
                f"client could not parse (parsed_ok=False)."
            )
            judgments = judgments[~unparsed]

    # ``response_matrix.build_tables`` overwrites the text with the CURRENT
    # build's in-memory query text; a previous build's queries.parquet is only a
    # best-effort fallback for standalone use, and its absence is not an error
    # (a fresh build used to skip this source entirely).
    queries_path = cfg.path("processed") / "queries.parquet"
    query_text = (
        pd.read_parquet(queries_path, columns=["query_id", "query"]).set_index("query_id")["query"]
        if queries_path.exists() else pd.Series(dtype=object)
    )

    records: list[dict] = []
    for row in judgments.itertuples(index=False):
        qid = row.query_id
        prompt = query_text.get(qid, "")
        margin = int(row.gain)
        outcome_anchor = 0.5 if margin == 0 else (1.0 if margin < 0 else 0.0)
        battle_id = f"{qid}|{row.anchor_model_id}|{row.candidate_model_id}"
        pair_id = f"{battle_id}|anchor_judge"
        for model_id, opp_id, role, score in (
            (row.anchor_model_id, row.candidate_model_id, "a", outcome_anchor),
            (row.candidate_model_id, row.anchor_model_id, "b", 1.0 - outcome_anchor),
        ):
            records.append({
                "query_id": qid,
                "model_id": model_id,
                "query": prompt,
                "score": score,
                "metric_type": schemas.MetricType.JUDGE_PREFERENCE,
                "dataset": "anchor_judge",
                "split": None,
                "source": schemas.Source.ANCHOR_JUDGE,
                "is_multiple_choice": False,
                "n_choices": None,
                "input_tokens": None,
                "output_tokens": None,
                "latency": None,
                "cost": None,
                "metadata": {
                    "battle_id": battle_id,
                    "pair_id": pair_id,
                    "opponent_model_id": opp_id,
                    "role": role,
                    "winner": "tie" if margin == 0 else ("a" if margin < 0 else "b"),
                    "margin": margin,
                    "interpretation": "anchor_relative_judge_preference",
                },
            })
    return schemas.coerce_response_frame(pd.DataFrame.from_records(records))


# --------------------------------------------------------------------------- #
# combined                                                                    #
# --------------------------------------------------------------------------- #
LOADERS = {
    schemas.Source.ROUTERBENCH: load_routerbench,
    schemas.Source.ARENA: load_arena,
    schemas.Source.GPT4_JUDGE: load_gpt4_judge,
    schemas.Source.LM_HARNESS: load_lm_harness,
    schemas.Source.IRT_ROUTER: load_irt_router,
    schemas.Source.ANCHOR_JUDGE: load_anchor_judge,
}


def load_all(cfg: Config, sources: Optional[Iterable[str]] = None) -> pd.DataFrame:
    sources = list(sources) if sources else list(LOADERS)
    frames = []
    for s in sources:
        try:
            frames.append(LOADERS[s](cfg))
        except FileNotFoundError as exc:
            warnings.warn(f"skipping source '{s}': {exc}")
    if not frames:
        return schemas.coerce_response_frame(schemas.empty_response_frame())
    return pd.concat(frames, ignore_index=True)

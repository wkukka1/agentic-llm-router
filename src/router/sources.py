"""Loaders. Every source yields ``Example`` rows carrying a :class:`Domain`.

The central problem this file solves is that **no single source has both real
prompts and reliable domain labels**:

===================  ====================  ==========================
source               prompts               labels
===================  ====================  ==========================
hand-labelled        real user traffic     read and assigned by hand
LMArena flags        real user traffic     3 domains only
MMLU-Pro             exam questions        14 clean categories
RouterArena          exam questions        Dewey *sub*class
BIG-bench            synthetic tasks       task name
===================  ====================  ==========================

Training on benchmarks alone produced a classifier that scored 91% on exams and
47% on real prompts. Training on real prompts alone cannot cover the rarer
domains. So the mix is deliberate, every row records its ``source``, and
evaluation is always reported per source.
"""

from __future__ import annotations

import logging

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from router.dataset import Example
from router.taxonomy import (
    domain_from_arena_flags,
    domain_from_bigbench_task,
    domain_from_mmlu_pro,
    domain_from_routerarena_category,
)

log = logging.getLogger(__name__)

ARENA_REPO = "lmarena-ai/arena-human-preference-140k"
HANDLABELLED_PATH = "data/handlabelled/real_prompts.parquet"
REAL_TASKS_PATH = "data/handlabelled/real_tasks.parquet"
MINED_TASKS_PATH = "data/handlabelled/real_tasks_mined.parquet"
GENERATED_TASKS_PATH = "data/synthetic/generated_tasks.parquet"


def _first_user_text(conversation) -> str:
    for turn in conversation if conversation is not None else []:
        if turn.get("role") != "user":
            continue
        parts = turn.get("content") or []
        texts = [p.get("text") or "" for p in parts if (p.get("type") or "text") == "text"]
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined
    return ""


def load_handlabelled(path: str = HANDLABELLED_PATH) -> list[Example]:
    """Real prompts read and labelled by hand -- the anchor of the whole set.

    Small but irreplaceable: it is the only source that pairs genuine user
    traffic with a label from every domain, including `personal_life` and
    `meta_other`, which no benchmark contains at all.
    """
    frame = pd.read_parquet(path)
    return [
        Example(prompt=r.prompt, source="handlabelled", subset="lmarena",
                capability=r.domain, meta={"arena_id": r.arena_id})
        for r in frame.itertuples()
    ]


def load_arena_flagged(*, max_shards: int = 3, language: str = "en",
                       max_chars: int = 4000) -> list[Example]:
    """Real prompts whose domain LMArena's own flags can supply.

    Only flagged rows are kept. An unflagged prompt could be about anything, so
    guessing would poison the set -- those rows are exactly what the hand
    labelling covers.
    """
    shards = sorted(f for f in list_repo_files(ARENA_REPO, repo_type="dataset")
                    if f.endswith(".parquet"))[:max_shards]
    cols = ["id", "conversation_a", "category_tag", "language", "is_code"]
    out: list[Example] = []
    for shard in shards:
        path = hf_hub_download(ARENA_REPO, shard, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=cols):
            for row in batch.to_pylist():
                if row.get("language") != language:
                    continue
                tag = row.get("category_tag") or {}
                domain = domain_from_arena_flags(
                    is_code=bool(row.get("is_code")),
                    is_math=bool((tag.get("math_v0.1") or {}).get("math")),
                    is_creative_writing=bool(
                        (tag.get("creative_writing_v0.1") or {}).get("creative_writing")),
                )
                if domain is None:
                    continue
                prompt = _first_user_text(row.get("conversation_a"))
                if len(prompt) < 15:
                    continue
                out.append(Example(prompt=prompt[:max_chars], source="arena_flag",
                                   subset=shard.rsplit("/", 1)[-1], capability=domain.value,
                                   meta={"arena_id": row.get("id")}))
    log.info("arena_flag: %d rows", len(out))
    return out


def load_mmlu_pro(*, max_rows: int | None = None) -> list[Example]:
    """MMLU-Pro: exam questions with 14 clean, intuitive categories."""
    path = hf_hub_download("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet",
                           repo_type="dataset")
    frame = pq.read_table(path, columns=["question", "options", "category"]).to_pandas()
    out: list[Example] = []
    for row in frame.itertuples():
        domain = domain_from_mmlu_pro(row.category)
        if domain is None:
            continue
        # row.options is a numpy array; `or []` would evaluate it as a bool.
        opts = list(row.options) if row.options is not None else []
        options = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        out.append(Example(prompt=f"{row.question}\n\n{options}".strip(), source="mmlu_pro",
                           subset=row.category, capability=domain.value))
    if max_rows:
        out = out[:max_rows]
    log.info("mmlu_pro: %d rows", len(out))
    return out


def load_routerarena_bycategory() -> list[Example]:
    """RouterArena, labelled from its ``Category`` (Dewey *sub*class).

    v1 used the parent ``Domain`` column and failed: "6 Technology" lumps
    medicine with engineering. "61 Medicine and health" does not.
    """
    path = hf_hub_download("RouteWorks/RouterArena", "data/full-00000-of-00001.parquet",
                           repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    out: list[Example] = []
    for row in rows:
        domain = domain_from_routerarena_category(row.get("Category") or "")
        if domain is None:
            continue
        question = (row.get("Question") or "").strip()
        if not question:
            continue
        context = (row.get("Context") or "").strip()
        options = row.get("Options")
        rendered = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options or []))
        prompt = "\n\n".join(p for p in (context, question, rendered) if p)
        out.append(Example(prompt=prompt, source="routerarena",
                           subset=(row.get("Dataset name") or "").strip() or None,
                           capability=domain.value,
                           difficulty=(row.get("Difficulty") or "").strip().lower() or None))
    log.info("routerarena: %d rows", len(out))
    return out


def load_bigbench(*, max_per_task: int = 60) -> list[Example]:
    """BIG-bench tasks whose name names an unambiguous subject.

    Capped per task, because the tasks vary in size by orders of magnitude and
    a handful would otherwise dominate. Its prompts are synthetic, so this is
    included to test whether task diversity helps -- not assumed to.
    """
    files = list_repo_files("tasksource/bigbench", repo_type="dataset")
    parquet = [f for f in files if f.endswith(".parquet") and "/" in f]
    by_task: dict[str, list[str]] = {}
    for f in parquet:
        by_task.setdefault(f.split("/")[0], []).append(f)

    out: list[Example] = []
    for task, task_files in sorted(by_task.items()):
        domain = domain_from_bigbench_task(task)
        if domain is None:
            continue
        try:
            path = hf_hub_download("tasksource/bigbench", sorted(task_files)[0],
                                   repo_type="dataset")
            frame = pq.read_table(path).to_pandas()
        except Exception as exc:  # noqa: BLE001 - one bad task must not stop the load
            log.warning("bigbench[%s] skipped: %s", task, exc)
            continue
        col = next((c for c in ("inputs", "question", "text") if c in frame.columns), None)
        if col is None:
            continue
        for text in frame[col].dropna().astype(str).head(max_per_task):
            text = text.strip()
            if 15 <= len(text) <= 3000:
                out.append(Example(prompt=text, source="bigbench", subset=task,
                                   capability=domain.value))
    log.info("bigbench: %d rows from mapped tasks", len(out))
    return out


SYNTHETIC_DIR = "data/synthetic"


def load_synthetic(domains: list[str] | None = None) -> list[Example]:
    """Hand-written synthetic prompts for the four weakest domains.

    Motivation: `humanities`, `language`, `law_politics` and `medicine_health`
    each had ~100-165 real examples and the model was confident on <13% of
    them. Real labelled data is the scarce resource; these are written to
    supplement it.

    **Written deliberately in real-user register** -- lowercase openings,
    typos, missing punctuation, bare fragments, context-free follow-ups. Clean,
    well-formed synthetic prompts would be trivially separable from real
    traffic, and the model would learn to detect *synthetic style* rather than
    domain. That is the same mechanism that made v1 score 91% on benchmarks and
    47% in the wild, so it is the specific thing to avoid here.

    Whether this helps is an empirical question, settled only by the frozen
    real-prompt eval. It is tagged ``source="synthetic"`` so its contribution
    is always measurable separately, and can be dropped in one line.
    """
    import importlib.util
    import sys

    modules = {
        "law_politics": "LAW_POLITICS",
        "medicine_health": "MEDICINE_HEALTH",
        "humanities": "HUMANITIES",
        "language": "LANGUAGE",
    }
    # Measured: adding synthetic data to a class that already has enough real
    # examples shifts the prior and costs accuracy elsewhere. Restricting it to
    # genuinely starved classes is the difference between +0.20 F1 and a net
    # regression, so the caller chooses which domains get supplemented.
    if domains is not None:
        modules = {k: v for k, v in modules.items() if k in domains}
    out: list[Example] = []
    for domain, symbol in modules.items():
        path = f"{SYNTHETIC_DIR}/{domain}.py"
        spec = importlib.util.spec_from_file_location(f"_syn_{domain}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for prompt in getattr(module, symbol):
            out.append(Example(prompt=prompt, source="synthetic", subset=domain,
                               capability=domain))
    log.info("synthetic: %d rows across %d domains", len(out), len(modules))
    return out


def load_real_tasks(path: str = REAL_TASKS_PATH) -> list[Example]:
    """1,000 real prompts hand-labelled with task type.

    This is the task-type training set. It is small, and it is the right
    distribution, which on measurement matters far more: a head trained on
    these 1,000 rows scores 0.828 on held-out real prompts where one trained on
    14,776 Dolly rows scores 0.700 -- below the 0.729 of always predicting
    `answer`. See :mod:`router.tasktype` for the full comparison.
    """
    frame = pd.read_parquet(path)
    out = [Example(prompt=row.prompt, source="handlabelled", subset="real_tasks",
                   capability=row.task) for row in frame.itertuples()]
    log.info("real tasks: %d rows", len(out))
    return out


def load_synthetic_tasks(tasks: list[str] | None = None) -> list[Example]:
    """Hand-written prompts for the three starved task types.

    `summarize` has 31 real examples (F1 0.444), `classify` 55 (0.476) and
    `extract` 6 (0.000). Targeted mining across the whole unlabelled pool found
    one further `extract`, so more real data is not available at any reasonable
    cost; writing it is the remaining option.

    Written in real-user register -- lowercase, typos, bare URLs, exam
    formatting pasted verbatim, fragments -- and shaped from the real examples
    rather than from an idea of what the task looks like. Clean instruction
    prose would be trivially separable from real traffic, and the model would
    learn to spot synthetic style instead of the task. That is exactly the
    mechanism that made the Dolly-trained head score 0.822 on Dolly and 0.700
    on real prompts.

    Tagged ``source="synthetic"`` so its contribution stays measurable and can
    be dropped in one line. Whether it helps is settled on real held-out
    prompts, never on itself.
    """
    import importlib.util
    import sys

    names = list(tasks) if tasks else ["summarize", "classify", "extract"]
    out: list[Example] = []
    for task in names:
        path = f"{SYNTHETIC_DIR}/task_{task}.py"
        spec = importlib.util.spec_from_file_location(f"synthetic_task_{task}", path)
        if spec is None or spec.loader is None:
            log.warning("no synthetic task file at %s", path)
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        prompts = getattr(module, task.upper())
        out.extend(Example(prompt=p, source="synthetic", subset=f"task_{task}",
                           capability=task) for p in prompts)
    log.info("synthetic tasks: %d rows over %s", len(out), names)
    return out


def load_generated_tasks(path: str = GENERATED_TASKS_PATH,
                         split: str = "train") -> list[Example]:
    """Agent-generated prompts for the four weakest task types.

    Eight agents wrote 100 prompts each, two per class, each pair given
    different subject angles and told to avoid the other's. **The split is by
    generator, not at random**: one agent's output is training, the other's is
    a held-out generated test set. Shuffling a single pool would put the same
    house style on both sides and the test score would mostly measure
    memorisation of it.

    Known limitation, measured rather than assumed: this text is **trivially
    separable from real prompts** -- a tf-idf classifier tells generated from
    real at AUC 0.936, and the hand-written set at 0.974. So a model trained on
    it can learn "synthetic style" alongside the task, which is the mechanism
    that made the Dolly-trained head score 0.822 on Dolly and 0.700 on real
    traffic. The same effect is visible here: on the held-out *generated* set
    the model reaches 0.849 top-1 with per-class F1 of 0.87-0.94, against 0.79
    and 0.36-0.89 on real prompts. Read the real numbers only.

    It helps anyway, because for `extract` there was nothing to contaminate:
    F1 0.000 -> 0.364. Restricting synthetic data to the starved classes was
    tried on the theory that it would keep the gains and drop the losses; it
    did not beat using all four, so all four are used.
    """
    frame = pd.read_parquet(path)
    frame = frame[frame["split"] == split]
    out = [Example(prompt=row.prompt, source="generated", subset=f"gen_{row.task}",
                   capability=row.task) for row in frame.itertuples()]
    log.info("generated tasks (%s): %d rows", split, len(out))
    return out


def load_mined_tasks(path: str = MINED_TASKS_PATH) -> list[Example]:
    """240 prompts found by targeted search for the rare task types.

    Random sampling cannot reach these: `extract` is 0.3% of traffic, so another
    thousand random draws would yield about three more. These were found by
    scoring the 1,441 unlabelled real prompts with the current head and
    hand-labelling its most confident candidates for each rare class.

    Mining precision, labelled class against the class searched for:

        ideate     47 of 60      summarize  16 of 60
        classify   31 of 60      extract     1 of 60

    That last row is the finding, not a failure. Hunting the whole unlabelled
    pool for `extract` turned up one example. It is not that the classifier
    cannot see `extract`; this traffic does not contain it.

    These rows are a biased sample and belong in training only -- see
    :func:`router.dataset.build_task_dataset`.
    """
    frame = pd.read_parquet(path)
    out = [Example(prompt=row.prompt, source="handlabelled", subset="mined_tasks",
                   capability=row.task) for row in frame.itertuples()]
    log.info("mined tasks: %d rows", len(out))
    return out


def load_dolly_tasks() -> list[Example]:
    """Dolly-15k as task-type supervision. **Superseded by :func:`load_real_tasks`.**

    Human-written instructions with task labels -- the only sizeable source of
    them. Only the instruction text is used, never the context passage: at
    serving time the router should decide from what the user asked, and
    training on the passage would teach it to read a document it may not have.

    Retained for reproducing the comparison, not for training. Mixing these rows
    into the real ones hurts at every ratio tried, including down-weighting them
    fifteen to one; the measurements are in :mod:`router.tasktype`.
    """
    from router.tasktype import task_from_dolly

    path = hf_hub_download(
        "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset"
    )
    frame = pd.read_json(path, lines=True)
    out: list[Example] = []
    for row in frame.itertuples():
        task = task_from_dolly(row.category)
        instruction = (row.instruction or "").strip()
        if task is None or len(instruction) < 10:
            continue
        out.append(Example(prompt=instruction, source="dolly", subset=row.category,
                           capability=task.value))
    log.info("dolly tasks: %d rows", len(out))
    return out

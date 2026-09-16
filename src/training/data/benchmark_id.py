"""Cross-source question identity for the standard-benchmark subset.

RouterBench and lm-evaluation-harness ask the *same underlying question* in
different prompt formats. To compare a RouterBench model and an lm-eval model in
the same routing matrix (pool-expansion phase E3) we join on question identity,
not prompt text:

    benchmark:{task}:{index}

where ``task`` is a canonical lm-eval task token and ``index`` is the 0-based
position in that task's evaluation split. This is valid only for the subset of
RouterBench ``eval_name``s that are verbatim standard benchmarks enumerated in
dataset order -- MMLU (57 subjects), HellaSwag, WinoGrande, ARC-Challenge. The
custom RouterBench tasks (Chinese riddles, abstract2title, consensus_summary,
mtbench, ...) have no lm-eval counterpart and return ``None`` (they stay on
their native ``routerbench:...`` id and simply do not join).

GSM8K / MBPP are deliberately excluded from the default map: RouterBench's
``grade-school-math`` (n=7450) and ``mbpp`` (n=427) counts do not match the
lm-eval test-split sizes (1319 / 500), so the index alignment is unverified.
"""

from __future__ import annotations

from typing import Optional

# RouterBench eval_name -> lm-eval task token (exact, non-MMLU).
_DIRECT: dict[str, str] = {
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc-challenge": "arc_challenge",
}

_MMLU_PREFIX = "mmlu-"


def rb_eval_to_task(eval_name: str) -> Optional[str]:
    """Canonical lm-eval task token for a RouterBench ``eval_name``, or ``None``
    if it is not in the alignable standard-benchmark subset."""
    e = str(eval_name)
    if e in _DIRECT:
        return _DIRECT[e]
    if e.startswith(_MMLU_PREFIX):
        return "mmlu_" + e[len(_MMLU_PREFIX):].replace("-", "_")
    return None


def lm_eval_task_to_token(task: str) -> str:
    """Canonical task token for an lm-eval task name (identity for the tasks we
    use; kept as a seam for future normalisation)."""
    return str(task)


def benchmark_uid(task_token: str, index) -> str:
    return f"benchmark:{task_token}:{int(index)}"


def rb_sample_to_uid(eval_name: str, sample_id: str) -> Optional[str]:
    """``benchmark:{task}:{index}`` for a RouterBench (eval_name, sample_id), or
    ``None`` if the eval is not alignable.

    ``sample_id`` is ``<eval>.<split-token>.<index>`` -- the trailing integer is
    the dataset-order index.
    """
    task = rb_eval_to_task(eval_name)
    if task is None:
        return None
    tail = str(sample_id).rsplit(".", 1)[-1]
    if not tail.isdigit():
        return None
    return benchmark_uid(task, tail)


def lm_eval_sample_to_uid(task: str, doc_id) -> Optional[str]:
    """``benchmark:{task}:{doc_id}`` for an lm-eval per-sample record."""
    try:
        return benchmark_uid(lm_eval_task_to_token(task), doc_id)
    except (TypeError, ValueError):
        return None


def alignable_eval_names(eval_names) -> dict[str, str]:
    """``{routerbench_eval_name: task_token}`` for the alignable ones in the input."""
    return {e: t for e in eval_names if (t := rb_eval_to_task(e)) is not None}

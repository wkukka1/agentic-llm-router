"""Task classifier: what kind of work the prompt asks for.

Self-contained -- the label space in `taxonomy`, the serving head in `head`.

0.844 top-1 / 0.967 top-2 over six classes, cross-validated on 1,000 real
prompts. Trained only on real traffic: a head trained on 14,776 Dolly rows
scores 0.700 on real prompts, below the 0.729 of always predicting `answer`.
"""

from prompt_decomposition.task_classifier.head import TaskHead, TaskPrediction
from prompt_decomposition.task_classifier.taxonomy import (
    TASK_DESCRIPTIONS,
    TASK_LABELS,
    TaskType,
    task_from_dolly,
)

__all__ = [
    "TASK_DESCRIPTIONS",
    "TASK_LABELS",
    "TaskHead",
    "TaskPrediction",
    "TaskType",
    "task_from_dolly",
]

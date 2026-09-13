"""Prompt decomposition: turning a raw prompt into the signals a router needs.

This is the first stage of the pipeline. It reads a prompt and returns what it
is about, what it asks to be done, and a set of cheap surface features -- as
calibrated distributions the routing stage can threshold on rather than as bare
labels.

    domain_classifier   what the prompt is about    0.923 top-1 / 0.980 top-2
    task_classifier     what it asks to be done     0.844 top-1 / 0.967 top-2
    length_estimator    how much work it is         rho 0.573, ceiling ~0.742
    signals             free regex/arithmetic features, no model, no labels
    composite           both classifiers over one prompt, plus the handoff vector
    core                shared machinery: training harness, metrics, settings

Each head is a self-contained package holding its own label space next to its
serving head, so adding a fourth -- difficulty, tool-need, decomposability --
means adding a directory rather than editing an existing one.

No LLM at inference and nothing fine-tuned: frozen encoders with linear heads,
so the output is deterministic and costs about 130 ms.
"""

from prompt_decomposition.composite import RouterHead, RouterPrediction
from prompt_decomposition.core.head import CalibratedHead
from prompt_decomposition.domain_classifier import DomainHead, DomainPrediction
from prompt_decomposition.length_estimator.head import LengthHead, LengthPrediction
from prompt_decomposition.task_classifier import TaskHead, TaskPrediction

__all__ = [
    "CalibratedHead",
    "DomainHead",
    "DomainPrediction",
    "LengthHead",
    "LengthPrediction",
    "RouterHead",
    "RouterPrediction",
    "TaskHead",
    "TaskPrediction",
]

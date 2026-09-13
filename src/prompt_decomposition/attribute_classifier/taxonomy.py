"""What the attribute head predicts, and how much each one is worth.

Twelve binary attributes of a prompt, all supervised by labels the arena dump
gives away. They split into two kinds, and the split matters more than the
accuracy numbers:

**Gates** are preconditions. "Does this need code execution" has a right answer
independent of anyone's preference, and getting it wrong is a failed request
rather than a slightly worse one. A gate earns its place by being accurate.

**Criteria** are the arena's seven LLM-judged difficulty dimensions. They are
learnable (0.77-0.83 AUC) and their *routing* value is unproven: the hardness
score composed from them predicts whether the stronger model was actually
needed at AUC 0.487, which is chance. They are served as inputs to a downstream
model that may combine them with behavioural signals -- never as a decision on
their own, and never as a difficulty prediction, which three separate
experiments in this project have now failed to make from prompt text.

Every label here is **LLM-judged**. A head trained on them reproduces an
automatic judge, which is close enough to the truth for "does this contain
code" and further from it for "is this complex".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Attribute:
    name: str
    kind: str          # "gate" or "criterion"
    means: str
    #: Probe AUC with a 384-d encoder, for calibrating expectations before a
    #: run. Measured on 15,968 held-out prompts.
    probe_auc: float


ATTRIBUTES: tuple[Attribute, ...] = (
    Attribute("code", "gate",
              "the prompt contains or asks for code", 0.879),
    Attribute("math", "gate",
              "the prompt requires mathematical work", 0.912),
    Attribute("creative_writing", "gate",
              "an open-ended creative piece, where 'correct' does not apply", 0.897),
    Attribute("constrained_instruction", "gate",
              "carries explicit constraints the answer must satisfy", 0.825),
    Attribute("complexity", "criterion", "judged to require complex reasoning", 0.773),
    Attribute("creativity", "criterion", "judged to require creativity", 0.791),
    Attribute("domain_knowledge", "criterion", "judged to require specialised knowledge", 0.816),
    Attribute("problem_solving", "criterion", "judged to require problem solving", 0.824),
    Attribute("real_world", "criterion", "judged to concern a real-world situation", 0.832),
    Attribute("specificity", "criterion", "judged to be specific rather than open", 0.826),
    Attribute("technical_accuracy", "criterion",
              "judged to demand technical accuracy", 0.804),
)

ATTRIBUTE_NAMES: tuple[str, ...] = tuple(a.name for a in ATTRIBUTES)
GATES: tuple[str, ...] = tuple(a.name for a in ATTRIBUTES if a.kind == "gate")
CRITERIA: tuple[str, ...] = tuple(a.name for a in ATTRIBUTES if a.kind == "criterion")

#: Deliberately absent. `non_english` probes at 0.990, and that is not a
#: reason to train one: the label comes from a language detector, so shipping a
#: model to imitate it is strictly worse than calling one. Detect the language;
#: do not predict it.
NOT_MODELLED: tuple[str, ...] = ("non_english",)


def describe(name: str) -> Attribute:
    for attribute in ATTRIBUTES:
        if attribute.name == name:
            return attribute
    raise KeyError(f"unknown attribute {name!r}; have {list(ATTRIBUTE_NAMES)}")

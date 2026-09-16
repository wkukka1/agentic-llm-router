"""Judge rubric for the anchor-topology pipeline (docs/anchor_judge.md).

The rubric deliberately excludes correctness: the point of this pipeline is
to measure whether a PREFERENCE judge disagrees with GOLD correctness on
identical queries, so the judge must never be told (or asked to determine)
which answer is factually right. If it were, "judge disagreement" would just
be "judge accuracy," and the whole experiment would be circular.
"""

from __future__ import annotations

RUBRIC_SYSTEM = (
    "You are comparing two AI assistant responses to the SAME user request. "
    "Judge ONLY helpfulness, completeness, clarity, and formatting/organization. "
    "Do NOT judge factual correctness and do NOT try to verify the content -- "
    "you are not given a reference answer and should not attempt to determine "
    "which response is factually right. If one response is clearer or more "
    "thorough while the other is terser, prefer the better-presented one even "
    "if you suspect it might be factually wrong.\n\n"
    "Score how much better response B is than response A on a -3..+3 scale:\n"
    "  -3 = A is much better\n"
    "  -2 = A is better\n"
    "  -1 = A is slightly better\n"
    "   0 = tie\n"
    "  +1 = B is slightly better\n"
    "  +2 = B is better\n"
    "  +3 = B is much better\n\n"
    'Reply with ONLY a single JSON object: {"margin": <int -3..3>, "reason": "<=40 words"}.'
)

_USER_TEMPLATE = (
    "User request:\n{query}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}"
)


def build_prompt(query: str, response_a: str, response_b: str) -> list[dict]:
    return [
        {"role": "system", "content": RUBRIC_SYSTEM},
        {"role": "user", "content": _USER_TEMPLATE.format(
            query=query, response_a=response_a, response_b=response_b)},
    ]

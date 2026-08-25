import numpy as np
import pandas as pd
import pytest

from router.data.build import assert_no_leakage, dedupe, split_frame
from router.data.schema import Example, normalize_prompt, to_frame
from router.data.sources.routerarena import render_prompt


def _frame(n=180, seed=0):
    rng = np.random.default_rng(seed)
    domains = ["science", "history", "language"]
    difficulties = ["easy", "medium", "hard"]
    return pd.DataFrame(
        {
            "prompt": [f"prompt number {i} about things" for i in range(n)],
            "domain": rng.choice(domains, n),
            "difficulty": rng.choice(difficulties, n),
        }
    )


def test_normalize_prompt_collapses_whitespace_and_case():
    assert normalize_prompt("  Hello\n\tWorld  ") == "hello world"


def test_uid_is_stable_and_source_scoped():
    a = Example(prompt="What is 2+2?", source="routerarena")
    b = Example(prompt="what is   2+2?", source="routerarena")
    c = Example(prompt="What is 2+2?", source="lmarena_140k")
    assert a.uid == b.uid, "uid must ignore whitespace/case so dedup is effective"
    assert a.uid != c.uid, "identical prompts from different sources are distinct rows"


def test_to_frame_serialises_meta_and_computes_n_chars():
    frame = to_frame([Example(prompt="abcde", source="s", meta={"k": 1})])
    assert frame.loc[0, "n_chars"] == 5
    assert frame.loc[0, "meta"] == '{"k": 1}'


def test_to_frame_empty_keeps_schema():
    assert list(to_frame([]).columns)[:3] == ["uid", "prompt", "source"]


def test_dedupe_drops_case_and_whitespace_variants():
    frame = pd.DataFrame({"prompt": ["Same thing", "same   thing", "different"]})
    assert len(dedupe(frame)) == 2


def test_split_is_disjoint_and_leakage_free():
    splits = split_frame(_frame(), stratify_on=["domain", "difficulty"])
    assert sum(len(v) for v in splits.values()) == 180
    assert_no_leakage(splits)


def test_split_preserves_label_proportions():
    frame = _frame(600)
    splits = split_frame(frame, stratify_on=["domain", "difficulty"])
    overall = frame["domain"].value_counts(normalize=True)
    for part in splits.values():
        share = part["domain"].value_counts(normalize=True)
        assert np.allclose(share[overall.index], overall, atol=0.05)


def test_assert_no_leakage_detects_a_shared_prompt():
    splits = {
        "train": pd.DataFrame({"prompt": ["shared prompt"]}),
        "test": pd.DataFrame({"prompt": ["Shared   Prompt"]}),
    }
    with pytest.raises(AssertionError, match="leaks between"):
        assert_no_leakage(splits)


def test_render_prompt_toggles_context_and_options():
    full = render_prompt("Q?", "Some context", ["a", "b"])
    assert full == "Some context\n\nQ?\n\nA. a\nB. b"
    assert render_prompt("Q?", "Some context", ["a", "b"], include_options=False) == "Some context\n\nQ?"
    assert render_prompt("Q?", "Some context", ["a", "b"], include_context=False) == "Q?\n\nA. a\nB. b"
    assert render_prompt("Q?", "", None) == "Q?"

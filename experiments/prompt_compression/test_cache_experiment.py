import numpy as np

from cache_experiment import (build_messages, cache_accounting, compress_prefix, input_cost,
                              parse_choice, select_sentences, shared_prefix_len)


def test_shared_prefix_len():
    assert shared_prefix_len([1, 2, 3, 4], [1, 2, 9]) == 2
    assert shared_prefix_len([], [1]) == 0
    assert shared_prefix_len([5, 6], [5, 6]) == 2


def test_cache_accounting_reads_writes_and_minimum():
    a = list(range(2000)) + [7]
    b = list(range(2000)) + [8]               # shares 2000 tokens with a
    c = [99] * 50                              # shares nothing
    acc = cache_accounting([a, b, c], min_cache=1024)
    assert [r["cached"] for r in acc] == [0, 2000, 0]
    assert [r["uncached"] for r in acc] == [2001, 1, 50]
    assert [r["written"] for r in acc] == [2000, 0, 0]   # a's prefix is written because b reuses it
    short = cache_accounting([[1, 2, 3, 4], [1, 2, 3, 5]], min_cache=1024)
    assert [r["cached"] for r in short] == [0, 0]       # below the provider minimum: no caching


def test_input_cost_prices_reads_and_writes():
    acc = [{"uncached": 1000, "cached": 0, "written": 800}, {"uncached": 10, "cached": 800, "written": 0}]
    # 1010 base + 800 * 0.1 read + 800 * 0.25 write premium
    assert input_cost(acc, read_mult=0.1, write_mult=1.25) == 1010 + 80 + 200


def test_build_messages_keeps_prefix_stable_across_questions():
    m1 = build_messages("STORY", "Who?", ["a", "b", "c", "d"])
    m2 = build_messages("STORY", "Where?", ["e", "f", "g", "h"])
    assert m1[0] == m2[0] and m1[0]["role"] == "system" and "STORY" in m1[0]["content"]
    assert "Who?" in m1[1]["content"] and "D) d" in m1[1]["content"]


def test_parse_choice():
    assert parse_choice("B") == 1
    assert parse_choice(" (c) the third") == 2
    assert parse_choice("Answer: D.") == 3
    assert parse_choice("no idea") == -1


def test_select_sentences_keeps_relevant_sentences_in_order_within_budget():
    article = "The sky is blue. Cats purr loudly. The ocean is deep. Dogs bark at night."
    vocab = ["cat", "dog", "sky", "ocean"]

    def embed(texts):  # bag-of-keywords stand-in for the sentence encoder
        return np.array([[float(w in t.lower()) for w in vocab] for t in texts]) + 1e-3

    out = select_sentences(article, "What do cats and dogs do?", rate=0.5, embed=embed)
    assert out == "Cats purr loudly. Dogs bark at night."


def test_compress_prefix_is_deterministic():
    text = ("The committee met on Tuesday to review the annual budget. After a long discussion, "
            "members agreed to increase funding for the library and reduce spending on events. ") * 3
    assert compress_prefix(text, rate=0.5) == compress_prefix(text, rate=0.5)
    assert len(compress_prefix(text, rate=0.5)) < len(text)

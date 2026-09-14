"""decompose.embedders -- both wrap router.embeddings.encoder.TextEncoder,
which is the real, already-tested encoder (see tests/router/embeddings/).
These just check the wrapper contract itself: dimension exposed, embed()
returns a plain float list of that length."""

from __future__ import annotations

import pytest

from decompose.embedders.base import Embedder
from decompose.embedders.bert import PromptEmbedder
from decompose.embedders.sentence_transformer import SentenceEmbedder


def test_prompt_embedder_is_an_embedder():
    assert issubclass(PromptEmbedder, Embedder)


def test_sentence_embedder_is_an_embedder():
    assert issubclass(SentenceEmbedder, Embedder)


def test_prompt_embedder_embeds_to_its_own_dimension():
    pytest.importorskip("transformers")
    e = PromptEmbedder()
    vec = e.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) == e.dimension == 768


def test_sentence_embedder_embeds_to_its_own_dimension():
    pytest.importorskip("sentence_transformers")
    e = SentenceEmbedder()
    vec = e.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) == e.dimension

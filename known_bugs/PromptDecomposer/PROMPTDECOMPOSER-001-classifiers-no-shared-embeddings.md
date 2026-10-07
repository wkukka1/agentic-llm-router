# PROMPTDECOMPOSER-001: Each head classifier recomputes its own embedding pass

- **Status:** confirmed (traced; code_review finding DC-02)
- **Severity:** low-medium (performance)
- **Class:** `PromptDecomposer` / `decompose.classifiers.prompt_heads` classifiers (`src/decompose/decomposer.py`, `src/decompose/classifiers/prompt_heads.py`)
- **Found:** 2026-09-22

## What

`PromptDecomposer.extract_signals` runs every registered classifier over the
same prompt. `DomainClassifier`, `TaskClassifier`, `LengthClassifier`,
`AttributeClassifier` (all `_HeadClassifier` subclasses in `prompt_heads.py`)
each call `self.head.predict()` independently — there is no analogue of
`decompose/classifiers/prompt_decomposition/composite.py`'s
`_shared_embeddings` ([composite.py:201](../../src/decompose/classifiers/prompt_decomposition/composite.py#L201)),
which memoizes one embedding pass and reuses it across sub-classifiers in
that sibling module.

## Failure scenario

Not a correctness bug — every classifier still gets a correct embedding, just
a redundantly recomputed one. With four head classifiers active
(`default_prompt_decomposer` wires up to four), a single `decompose()` call
does up to 4x the embedding-model forward passes it needs to, on the request
path.

## Suggested fix

Factor a shared embedding cache into `PromptDecomposer.extract_signals` (or a
common base the `_HeadClassifier` subclasses share), mirroring
`CompositeClassifier`'s `_shared_embeddings` pattern.

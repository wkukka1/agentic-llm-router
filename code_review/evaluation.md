# Code review: evaluation (top level, baselines, data)

**Files reviewed:** 6 (330 lines): `__init__.py`, `baselines/__init__.py`, `baselines/classical_irt.py`, `data/__init__.py`, `data/judge_responses.py`, `data/stratify.py` · **Context-only:** `src/training/data/loaders.py`, `src/training/data/normalize.py`, `src/training/data/response_matrix.py`
**Summary:** 3 findings: 3 correctness, 0 design, 0 performance · 3 confirmed, 0 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| EV-01 | `src/evaluation/baselines/classical_irt.py:96-108` | correctness | medium | confirmed |
| EV-02 | `src/evaluation/data/judge_responses.py:62-75` | correctness | low | confirmed |
| EV-03 | `src/evaluation/data/judge_responses.py:67` | correctness | low | confirmed |

---

## src/evaluation/baselines/classical_irt.py

### EV-01: `protocol="main_effects"` trains on the cells it scores unless `split="test"` `:96-108`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
97:        fit_sp = obs[obs["split"].isin(["train", "validation"])]
98:        eval_sp = obs[obs["split"] == split]
107:        held[len(fit_df):] = obs_mask[len(fit_df):]        # score only eval queries
```

**What breaks:** The fit matrix always includes train and validation queries. With `split="validation"` (or `"train"`), every eval query appears twice in `q_ids`: once in the fit block, where its outcomes are trained on, and once in the eval block, where it's scored. `b_m` is fit on the very outcomes it's evaluated against, so the "genuine lower bound" becomes optimistic with no error. Only the default `split="test"` is clean.
**Evidence:** Read `:96-109`. `held` masks only the second copy, and nothing rejects `split in {"train", "validation"}`.
**Direction:** Fit on train and validation rows *excluding* `split`, or raise for non-test splits.

---

## src/evaluation/data/judge_responses.py

### EV-02: Duplicate RouterBench prompts pair a judged response with a different, averaged score `:62-75`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
68:        frames.append(pd.DataFrame({
69:            "query_id": query_id,
70:            "model_id": canonical_model_id(m),
```

**What breaks:** RouterBench contains some prompts verbatim more than once, which is why `collapse_duplicate_observations` averages their scores (`training/data/response_matrix.py:31-64`). This module keeps every raw row, so one `(query_id, model_id)` maps to several response texts. `response_text_lookup` builds a dict, and the last row wins. The judge rates one sample's text, while the gold correctness it's compared against is the average over all copies. When the copies disagree (0 and 1 average to 0.5), judge-versus-gold disagreement is measured against a label that doesn't belong to the text that was judged.
**Evidence:** Read `:78-87` and the collapse rule. This module does no de-duplication.
**Direction:** Drop `(query_id, model_id)` pairs with more than one row, or keep one consistently, and report the count.

### EV-03: `render_prompt` rewrites model responses that look like list literals `:67`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
67:        text = wide[resp_col].map(render_prompt)
```

**What breaks:** `render_prompt` exists to unpack a `str(list[turns])` prompt. On a model answer that is a bracketed literal (a JSON array, `"[1, 2, 3]"`, printed Python list output), it runs `ast.literal_eval` and joins the elements with blank lines. The judge then sees a different answer from the one the model gave.
**Evidence:** Probe: `render_prompt('[1, 2, 3]')` returns `'1\n\n2\n\n3'`. The same behaviour on prompts is TD-06.
**Direction:** Apply only `sanitize_text` to response text.

---

## src/evaluation/data/stratify.py, `__init__` files

No findings.

---

## Checked and ruled out
- **Cell protocol with a fully held-out query row:** θ_q stays near its init, which is the intended transductive behaviour.
- **`a_m` receiving no gradient under `main_effects`:** θ is pinned at 0, so `a_m` is unused, and Adam's weight decay shrinking it does no harm.
- **`stratify` drawing `natural` before `discriminative`:** documented disjointness policy.
- **Model-column detection copied from `_melt_routerbench`:** filed as cross-cutting duplication XD-04.

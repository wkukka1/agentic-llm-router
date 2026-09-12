# Generating more task-type training data

Two prompts. The **first** generates new examples; the **second** labels prompts
you already have. Use the second wherever you can get real prompts — a real
labelled prompt is worth several written ones, and the whole reason this
classifier had to be rebuilt is that generated data drifts from real traffic.

## What we need most, in order

| class | real examples | F1 | need |
|---|---|---|---|
| `extract` | 6 | 0.25 | **most** — mining 1,441 unlabelled prompts found one |
| `summarize` | 31 | 0.50 | high |
| `classify` | 55 | 0.55 | high |
| `ideate` | 69 | 0.47 | medium — confuses with `answer` |
| `media` | 44 | 0.73 | low |
| `create` | 122 | 0.65 | low |
| `answer` | 729 | 0.89 | none, it is 73% of traffic already |

**We also need more real labelled prompts of the rare classes for the *eval*
set.** Right now 1,000 random real prompts contain 3 `extract` and 11
`summarize`, which is why we cannot prove the model has improved on them even
when it has. Generated data cannot fix that — only labelled real prompts can.

---

## Prompt 1 — generate new examples

> You are helping build a training set for a classifier that reads a user's
> prompt to an AI assistant and predicts what kind of work is being asked for.
>
> Write **60 prompts** that a real person would send to an AI assistant, all of
> which are examples of the task type **`<TASK>`**, defined as:
>
> **`<TASK>`: `<DEFINITION>`**
>
> Critical requirement — these must read like **real chat traffic, not like
> textbook instructions**. Real prompts in our data look like this:
>
> - lowercase, often no punctuation: `summarise the te`
> - typos left in: `can you please summawrize a word doc`
> - a bare pasted URL with three words: `https://youtu.be/abc123 explain this video`
> - exam or quiz text pasted verbatim, formatting and all:
>   `Question # 10 of 10 ( Start time: 01:03:52 AM ) Total Marks: 1 ...`
> - fragments with no context: `do u agree with that ?`
> - other languages, unannounced: `fasse diesen artikel kurz zusammen`
> - occasionally long and rambling with the actual request buried at the end
>
> Do **not** write clean, uniform, well-formed instructions. A classifier
> trained on tidy prose learns to detect tidy prose, and then fails on real
> users. We have measured this: a model trained on 14,776 clean instructions
> scored **below a constant predictor** on our real traffic.
>
> Vary across: subject matter (medicine, law, code, sport, cooking, finance,
> school, relationships), length (3 words to 80), register (rushed, polite,
> demanding, confused), and language (~10% non-English).
>
> Output one prompt per line, no numbering, no quotes, no commentary.

Substitute one of:

- **`extract`** — pull specific spans or fields *out of* supplied or named
  material, returning parts of the source selected by a stated criterion.
  *Distinguish from `summarize`: extraction returns pieces of the source;
  summarising returns a condensed account of it.*
- **`summarize`** — condense supplied or named material into a shorter account.
- **`classify`** — sort something into categories, choose between given
  options, judge true/false, rate, rank, or check correctness.
- **`ideate`** — brainstorm options, suggest possibilities, recommend, generate
  ideas. *Distinguish from `answer`: ideating asks for several options to
  choose from; answering asks for the one right response.*

---

## Prompt 2 — label prompts you already have (preferred)

> Below are real prompts sent to an AI assistant. Label each with exactly one
> task type — what kind of work is being asked for, regardless of subject.
>
> - `answer` — answer a question, explain, look something up, give advice
> - `ideate` — brainstorm, suggest options, recommend, generate ideas
> - `summarize` — condense supplied or named material
> - `extract` — pull specific spans or fields out of supplied material
> - `classify` — choose between options, judge true/false, rate, rank, check
> - `create` — write, rewrite or translate text as the deliverable
> - `media` — produce an image, video or other non-text artifact
>
> Rules that resolve most of the hard cases:
>
> 1. Label what is **asked for**, not the subject. "Summarise this contract"
>    is `summarize`, not law.
> 2. If the deliverable is text the user will use, it is `create` — including
>    rewriting, proofreading and translating.
> 3. `ideate` needs **several options**. One recommendation is `answer`.
> 4. `extract` returns **pieces of a source**; `summarize` returns a condensed
>    account of it.
> 5. A question with the options supplied ("A/B/C/D", "true or false", "which
>    of these") is `classify`, not `answer`.
> 6. When genuinely torn, prefer `answer` — it is 73% of traffic and the
>    fallback that costs least.
>
> Output `<index><tab><label>`, one per line, nothing else.

---

## Format to send back

Plain text, one prompt per line, or two columns for labelled data:

```
prompt<TAB>task
summarise the te	summarize
Question 4 of 20 ... Select the best answer	classify
```

Land it at `data/synthetic/task_<class>.py` (generated) or
`data/handlabelled/real_tasks.parquet` (real, labelled). Generated rows go into
**training only** — the evaluation set stays real, or the numbers stop meaning
anything.

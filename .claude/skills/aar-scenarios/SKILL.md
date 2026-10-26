---
name: aar-scenarios
description: Generate synthetic test prompts and scenarios for the AAR router and gateway, grade them, and report routing quality and gateway conformance separately. Use when asked to make test prompts or scenarios, stress the router on a kind of task (a language, a difficulty, multi-turn), check the gateway still handles threads, fallback, resume, corrections or session isolation, or compare routers on fresh prompts.
---

# AAR scenarios

One module, `evaluation.gateway.scenarios` (behaviour suite in `scenarios_behave`),
four steps. Output lives in `~/.cache/aar/routing-pool/synthetic/`.

| step | cost | what it does |
|---|---|---|
| `generate --n N` | cents (Haiku, batched) | grid → slots → filled prompts → dedup → `turns.jsonl` |
| `behave [--wave 1\|2]` | $0 | scripted request sequences through an isolated gateway + stub |
| `report` | $0 | coverage, conformance, and (once graded) routing quality |
| `grade --budget B` | **plan usage** | replays every tier + judge; **needs the user's yes** |

Run everything with the repo's venv: `.venv/Scripts/python.exe -m evaluation.gateway.scenarios ...`
(`.venv/bin/python` off Windows).

## Rules

1. **Ask before `grade`.** Run `grade --budget B` without `--yes` first. It prints the
   pending count, the turn cap and an estimated range; show those and wait for a yes,
   then rerun with `--yes`. The budget is an estimate, not a cap: say so. Keep to about
   $25 per 5-hour plan window.
2. `generate` is cheap but still plan usage. For more than 100 prompts, say the size first.
3. Never send scenario traffic to the live gateway (:8787). `behave` starts its own on
   :8790 with the stub on :8899 and checks the live log afterwards. If those ports are
   busy (a router bake-off is running), wait. Don't kill anything.
4. Never upload pools, grades or reports anywhere.
5. Synthetic prompts are an extra stratum. Don't present their numbers as the held-out
   result; the real pools (`pool.py`, `multiturn.py`) back the final claims.
6. `suggested_tier` is a hint for the writer, never a label. The oracle is measured.

## Workflow

1. **Shape the grid** for what the user wants to test. Edit `reference/grid.yaml`:
   narrow an axis, add archetypes. Any edit starts a new generation version, and earlier
   fills stay cached under their own version.
2. `generate --n 20` (or more), then **hand-review at least 12 rows** of `turns.jsonl`
   before grading. Look for template-sounding phrasing, follow-ups that stand alone,
   and missing code or logs. If they're bad, fix the grid or the writer prompt and
   regenerate.
3. `behave` (both waves). Each scenario gets a fresh gateway, so expect ~10–30 s each.
4. `report`. It is a partial report before grading.
5. Ask, then `grade --budget B --yes`, then `report` again.
6. Optional, for a bake-off on the new prompts:
   `python -m evaluation.gateway.router_compare run --graded ~/.cache/aar/routing-pool/synthetic/graded.jsonl`.

`run --n N` does steps 2–4 in one go and stops before grading.

## Adding things

- **Axis value / exclusion:** edit `axes` or `exclude` in `grid.yaml`. Every pair named
  in `pairs_min` is covered before any is repeated.
- **Archetype:** add `{id, axes (all four), brief, rubric}` to `archetypes`. Archetypes
  are always sampled first.
- **Behaviour scenario:** add a function `bNN_what_it_checks(gw, checks)` to
  `scenarios_behave.py` and a row to `SCENARIOS`. Use `pre(...)` for preconditions
  (unmet → skip, never pass), `_check(...)` for expectations and `forbid(...)` for
  what must not happen. Set up model state with `mode(gw, "pin", HAIKU)` rather than
  relying on router predictions. Use `rc.failing({model: status})` for upstream errors.
- **Oracle tolerance:** `oracle: {tau, q_min}` in the grid. Set it before seeing
  results. If it changes afterwards, say so in the report.

## Report format

Two verdict lines first, never merged:

```
Routing quality: PASS | PARTIAL | NOT RUN — why
Gateway conformance: PASS | PARTIAL | FAIL | NOT RUN — why
```

Then the numbers:

- savings and ΔQ vs **no router** (every prompt on the model asked for), with the
  bootstrap interval;
- regret vs the adequate oracle;
- generator calibration (agree / over- / under-escalate);
- per-stratum n;
- behaviour failures with their failing checks;
- the isolation line.

Quote the measured grading usage next to the estimate.

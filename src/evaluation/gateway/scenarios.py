"""Synthetic test prompts from a scenario grid, graded like the real pools,
plus a report that keeps routing quality and gateway conformance apart.

    python -m evaluation.gateway.scenarios generate --n 60 --seed 0   # cents: Haiku fills slots in batches
    python -m evaluation.gateway.scenarios behave                     # $0: evaluation.gateway.scenarios_behave
    python -m evaluation.gateway.scenarios report
    python -m evaluation.gateway.scenarios grade --budget 3 [--yes]   # plan usage: asks first
    python -m evaluation.gateway.scenarios run --n 60                 # generate, behave, report; stops before grade

The grid (``.claude/skills/aar-scenarios/reference/grid.yaml``) names the
axes; :func:`expand` picks a seeded sequence of slots that covers every pair
of the listed axes before repeating one, archetypes first. Each slot's id
hashes its axes, the seed, its occurrence and the generation version (the
grid plus the writer's prompts), so a larger ``--n`` fills only new slots and
an edited grid starts a namespace of its own. A slot's ``suggested_tier`` is
what the writer aims at, never a label: the measured oracle is.

Rows land in ``~/.cache/aar/routing-pool/synthetic/turns.jsonl`` in the
grader's shape. They are an extra stratum, never a replacement for the
held-out real pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import yaml
from aar_gateway.config import append_jsonl, read_jsonl
from aar_gateway.grading.rubric import CRITERIA, RUBRICS, criteria_prompt, parse_criteria
from aar_gateway.grading.rubric import VERSION as RUBRIC_VERSION

from evaluation.gateway.multiturn import with_context
from evaluation.gateway.pool import POOL_DIR

GRID = Path(__file__).resolve().parents[3] / ".claude" / "skills" / "aar-scenarios" / "reference" / "grid.yaml"
OUT_DIR = POOL_DIR / "synthetic"
WRITER_MODEL = "claude-haiku-5-5"
JUDGE = "claude-sonnet-5-5"
READ_ONLY = ("--permission-mode", "plan", "--strict-mcp-config", "--safe-mode")
BATCH = 20
CANDIDATES = 64  # combinations scored per expansion step
NEAR_DUP = 0.92  # cosine similarity of normalised embeddings at or above which a prompt is a near-duplicate
REFILL_ROUNDS = 2
#: per-turn plan usage when no graded file says otherwise (three replays + judge, see subscription limits)
FALLBACK_TURN_USD = (0.35, 0.15)
#: one judge call's plan usage until graded rows record it (``judge_usage_usd``)
FALLBACK_JUDGE_USD = 0.06
OTHER_POOLS = (POOL_DIR / "turns.jsonl", POOL_DIR / "multiturn" / "turns.jsonl")

#: ``complete(prompt) -> (reply text, plan usage in USD or None)``
Complete = Callable[[str], tuple[str, float | None]]

WRITER_PROMPT = """You write realistic messages a developer would send to an AI coding assistant, for testing
a model router. For each slot below write ONE conversation:

- "context": exactly 2*depth earlier entries alternating {"role": "user"} and {"role": "assistant"},
  starting with "user" (an empty list when depth is 0). Assistant entries are short, plausible replies.
- "message": the developer's next message. It must need the earlier context when depth > 0, and the
  earlier entries must not already answer it.

Make each one sound like a real person at work: specific code, file names, error text, constraints.
Include the code or log the message talks about. Never mention difficulty, tiers, slots or tests.
Aim the task's difficulty at the slot's "difficulty"; follow "brief" when one is given.

Reply with ONLY a JSON array, one object per slot, nothing else:
[{"slot_id": "...", "context": [{"role": "user", "text": "..."}, ...], "message": "..."}, ...]

Slots:
"""

CHECK_PROMPT = """Below are conversations between a developer and an AI coding assistant, each ending in the
developer's latest message. For each, answer two questions:
- "depends": does the latest message need the earlier conversation to be understood?
- "answered": do the earlier assistant replies already fully answer the latest message?

Reply with ONLY a JSON object: {"<id>": {"depends": true|false, "answered": true|false}, ...}

"""

REWRITE_HINT = ("\nA previous attempt failed a check: the follow-up must depend on the earlier exchanges, "
                "and the earlier exchanges must not already answer it.\n")


# -- grid and slots -------------------------------------------------------------------------------------


def load_grid(path: Path = GRID) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    grid = yaml.safe_load(text)
    grid["version"] = hashlib.sha1((text + WRITER_PROMPT + CHECK_PROMPT).encode()).hexdigest()[:12]
    grid["pair_axes"] = [tuple(a.strip() for a in key.split(" x ")) for key in grid.get("pairs_min") or {}]
    for a, b in grid["pair_axes"]:
        if a not in grid["axes"] or b not in grid["axes"]:
            raise ValueError(f"pairs_min names an unknown axis: {a} x {b}")
    for arch in grid.get("archetypes") or []:
        if set(arch["axes"]) != set(grid["axes"]):
            raise ValueError(f"archetype {arch['id']}: axes must be exactly {sorted(grid['axes'])}")
    return grid


def excluded(axes: dict, grid: dict) -> bool:
    return any(all(axes.get(k) == v for k, v in rule.items()) for rule in grid.get("exclude") or [])


def slot_id(axes: dict, archetype: str | None, seed: int, occurrence: int, version: str) -> str:
    canon = json.dumps({"axes": axes, "archetype": archetype}, sort_keys=True)
    return hashlib.sha1(f"{canon}|{seed}|{occurrence}|{version}".encode()).hexdigest()[:12]


def make_slot(grid: dict, axes: dict, *, seed: int, occurrence: int, archetype: dict | None = None) -> dict:
    arch = archetype["id"] if archetype else None
    return {"slot_id": slot_id(axes, arch, seed, occurrence, grid["version"]), "axes": dict(axes),
            "archetype": arch, "brief": (archetype or {}).get("brief"), "rubric": (archetype or {}).get("rubric"),
            "occurrence": occurrence, "seed": seed, "version": grid["version"],
            "suggested_tier": grid["suggested_tier"][axes["difficulty"]]}


def _combo_key(axes: dict, archetype: str | None) -> str:
    return json.dumps({"axes": axes, "archetype": archetype}, sort_keys=True)


def expand(grid: dict, n: int, seed: int = 0) -> list[dict]:
    """``n`` slots: the archetypes, then greedy pair coverage -- each step takes, of
    :data:`CANDIDATES` seeded random combinations, the one whose pairs are least
    covered so far. Each step depends only on the steps before it, so
    ``expand(g, n)`` is always ``expand(g, m)[:n]`` for ``m > n``."""
    rng = random.Random(seed)
    names = list(grid["axes"])
    pairs = grid["pair_axes"]
    counts: Counter = Counter()
    seen: Counter = Counter()
    slots: list[dict] = []

    def take(axes: dict, archetype: dict | None) -> None:
        key = _combo_key(axes, archetype["id"] if archetype else None)
        slots.append(make_slot(grid, axes, seed=seed, occurrence=seen[key], archetype=archetype))
        seen[key] += 1
        counts.update((a, b, axes[a], axes[b]) for a, b in pairs)

    for arch in (grid.get("archetypes") or [])[:n]:
        take(arch["axes"], arch)
    while len(slots) < n:
        best, best_score = None, -1.0
        for _ in range(CANDIDATES):
            axes = {k: rng.choice(grid["axes"][k]) for k in names}
            if excluded(axes, grid):
                continue
            score = sum(1.0 / (1 + counts[(a, b, axes[a], axes[b])]) for a, b in pairs)
            if score > best_score:
                best, best_score = axes, score
        if best is None:
            raise ValueError("the grid's exclusions leave nothing to sample")
        take(best, None)
    return slots


def coverage(axes_list: Iterable[dict], grid: dict) -> dict:
    """Per required pair of axes: the count of every allowed cell, the cells below
    ``pairs_min``, and the cells the exclusions rule out."""
    axes_list = list(axes_list)
    out = {}
    for (a, b), need in zip(grid["pair_axes"], (grid.get("pairs_min") or {}).values(), strict=True):
        got = Counter((ax[a], ax[b]) for ax in axes_list)
        cells, short, ruled_out = {}, [], []
        for va in grid["axes"][a]:
            for vb in grid["axes"][b]:
                if any(set(rule) <= {a, b} and rule.get(a, va) == va and rule.get(b, vb) == vb
                       for rule in grid.get("exclude") or []):
                    ruled_out.append(f"{va}/{vb}")
                    continue
                cells[f"{va}/{vb}"] = got[(va, vb)]
                if got[(va, vb)] < need:
                    short.append(f"{va}/{vb}")
        out[f"{a} x {b}"] = {"min": need, "cells": cells, "short": short, "excluded": ruled_out}
    return out


# -- filling slots --------------------------------------------------------------------------------------


def subscription_complete(model: str = WRITER_MODEL, timeout: float = 600) -> Complete:
    from aar_gateway.claude_cli import ClaudeSubscriptionAdapter

    llm = ClaudeSubscriptionAdapter(model, timeout=timeout)

    def complete(prompt: str) -> tuple[str, float | None]:
        out = llm.complete(prompt, extra_args=READ_ONLY)
        return out.content, out.meta.get("subscription_cost_usd_equivalent")

    return complete


_LIMIT = re.compile(r"usage limit|session limit|weekly limit|rate limit|resets|\b429\b", re.I)


def _json_in(text: str, opener: str):
    """The first JSON array (``[``) or object (``{``) in ``text``, else ``None``."""
    closer = "]" if opener == "[" else "}"
    start, end = (text or "").find(opener), (text or "").rfind(closer)
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def validate_fill(slot: dict, item) -> str | None:
    """Why ``item`` isn't a usable fill of ``slot``, or ``None``."""
    if not isinstance(item, dict):
        return "not an object"
    message, context = item.get("message"), item.get("context")
    if not isinstance(message, str) or not message.strip():
        return "empty message"
    if not isinstance(context, list):
        return "context is not a list"
    depth = int(slot["axes"]["depth"])
    if len(context) != 2 * depth:
        return f"context has {len(context)} entries, wants {2 * depth}"
    for i, entry in enumerate(context):
        want = "user" if i % 2 == 0 else "assistant"
        if not isinstance(entry, dict) or entry.get("role") != want:
            return f"context entry {i} is not {want}"
        if not isinstance(entry.get("text"), str) or not entry["text"].strip():
            return f"context entry {i} is empty"
    return None


def _writer_prompt(slots: Sequence[dict], hint: str = "") -> str:
    listing = [{"slot_id": s["slot_id"], **s["axes"], **({"brief": s["brief"]} if s.get("brief") else {})}
               for s in slots]
    return WRITER_PROMPT + hint + json.dumps(listing, indent=1)


def fill_batch(complete: Complete, slots: Sequence[dict], *, hint: str = "", attempt: int = 1,
               spent: list | None = None) -> dict[str, dict]:
    """``{slot_id: {"item": ...} | {"error": ...}}`` for every slot. A reply that
    can't be parsed is retried once, then split in half; a slot whose item fails
    :func:`validate_fill` gets one more try. A usage limit is raised, not recorded."""
    spent = [] if spent is None else spent
    try:
        reply, cost = complete(_writer_prompt(slots, hint))
        spent.append(cost or 0.0)
    except RuntimeError as exc:
        if _LIMIT.search(str(exc)):
            raise
        reply = None
    items = _json_in(reply, "[") if reply is not None else None
    if not isinstance(items, list):
        if attempt == 1:
            return fill_batch(complete, slots, hint=hint, attempt=2, spent=spent)
        if len(slots) > 1:
            mid = len(slots) // 2
            return {**fill_batch(complete, slots[:mid], hint=hint, attempt=2, spent=spent),
                    **fill_batch(complete, slots[mid:], hint=hint, attempt=2, spent=spent)}
        return {slots[0]["slot_id"]: {"error": "no parseable reply"}}
    by_id: dict[str, list] = {}
    for it in items:
        if isinstance(it, dict):
            by_id.setdefault(str(it.get("slot_id")), []).append(it)
    out, retry = {}, []
    for s in slots:
        got = by_id.get(s["slot_id"], [])
        why = "missing from reply" if not got else "duplicated in reply" if len(got) > 1 else validate_fill(s, got[0])
        if why is None:
            out[s["slot_id"]] = {"item": got[0]}
        elif attempt == 1:
            retry.append(s)
        else:
            out[s["slot_id"]] = {"error": why}
    if retry:
        out.update(fill_batch(complete, retry, hint=hint, attempt=2, spent=spent))
    return out


def check_dependency(complete: Complete, rows: Sequence[dict], batch: int = 40,
                     spent: list | None = None) -> set[str]:
    """Slot ids of multi-turn rows whose follow-up doesn't need its context or is
    already answered there. A batch whose reply can't be parsed fails nobody."""
    spent = [] if spent is None else spent
    bad: set[str] = set()
    multi = [r for r in rows if int(r["axes"]["depth"]) > 0]
    for start in range(0, len(multi), batch):
        chunk = multi[start:start + batch]
        listing = "\n\n".join(f"=== {r['slot_id']} ===\n{r['prompt'][:3500]}" for r in chunk)
        reply, cost = complete(CHECK_PROMPT + listing)
        spent.append(cost or 0.0)
        verdicts = _json_in(reply, "{") or {}
        for r in chunk:
            v = verdicts.get(r["slot_id"]) if isinstance(verdicts, dict) else None
            if isinstance(v, dict) and (v.get("depends") is False or v.get("answered") is True):
                bad.add(r["slot_id"])
    return bad


def normalised_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()[:16]


def to_row(slot: dict, item: dict) -> dict:
    conv = [(e["role"], e["text"]) for e in item["context"]]
    message = item["message"].strip()
    prompt = with_context(conv, message)
    return {"prompt_hash": normalised_hash(prompt), "prompt": prompt, "route_text": message,
            "source": "synthetic", "group": f"syn-{slot['axes']['task']}", "cwd": None,
            "slot_id": slot["slot_id"], "axes": slot["axes"], "archetype": slot["archetype"],
            "suggested_tier": slot["suggested_tier"], "rubric": slot["rubric"],
            "generation_version": slot["version"]}


# -- dedup ----------------------------------------------------------------------------------------------


def dedup(rows: Sequence[dict], existing: Sequence[str], encode: Callable | None,
          threshold: float = NEAR_DUP) -> tuple[list[dict], list[tuple[dict, str]]]:
    """``(kept, [(dropped row, why)])``, in order: an exact (normalised) duplicate of
    an existing prompt or an earlier row, or a near-duplicate (cosine similarity at
    or above ``threshold``) of either."""
    seen = {normalised_hash(t) for t in existing}
    kept, dropped = [], []
    vecs = None
    if encode is not None and rows:
        texts = list(existing) + [r["prompt"] for r in rows]
        v = np.asarray(encode(texts), dtype=float)
        vecs = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    pool_idx = list(range(len(existing)))
    for i, r in enumerate(rows):
        if r["prompt_hash"] in seen:
            dropped.append((r, "exact duplicate"))
            continue
        if vecs is not None and pool_idx:
            sims = vecs[pool_idx] @ vecs[len(existing) + i]
            if float(sims.max()) >= threshold:
                dropped.append((r, f"near duplicate (cosine {float(sims.max()):.3f})"))
                continue
        seen.add(r["prompt_hash"])
        kept.append(r)
        pool_idx.append(len(existing) + i)
    return kept, dropped


def default_encoder() -> Callable:
    from router.embeddings.encoder import EncoderConfig, TextEncoder, light_backend

    return TextEncoder(EncoderConfig(name="scenarios", backend=light_backend())).encode


# -- generate -------------------------------------------------------------------------------------------


def _fills(out_dir: Path, version: str) -> dict[str, dict]:
    """The latest cached fill of each slot of this generation version."""
    out = {}
    for rec in read_jsonl(out_dir / "fills.jsonl"):
        if rec.get("version") == version:
            out[rec["slot_id"]] = rec
    return out


def _fill_missing(complete: Complete, slots: Sequence[dict], cache: dict, out_dir: Path, *,
                  batch: int, spent: list, hint: str = "") -> None:
    todo = [s for s in slots if (cache.get(s["slot_id"]) or {}).get("status") != "ok"]
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        result = fill_batch(complete, chunk, hint=hint, spent=spent)
        recs = []
        for s in chunk:
            got = result[s["slot_id"]]
            rec = {"version": s["version"], "slot_id": s["slot_id"], "slot": s,
                   **({"status": "ok", "item": got["item"]} if "item" in got else {"status": "error",
                                                                                    "error": got["error"]})}
            cache[s["slot_id"]] = rec
            recs.append(rec)
        append_jsonl(out_dir / "fills.jsonl", *recs)


def generate(n: int, seed: int = 0, *, grid_path: Path = GRID, out_dir: Path = OUT_DIR,
             complete: Complete | None = None, encode: Callable | None = None, batch: int = BATCH,
             other_pools: Sequence[Path] = OTHER_POOLS) -> dict:
    """Fill the first ``n`` slots (cached fills are reused), check multi-turn
    dependency, drop duplicates and refill them in their stratum, and add the
    accepted rows to ``turns.jsonl``. Returns the summary ``report`` reads."""
    grid = load_grid(grid_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = complete or subscription_complete()
    slots = expand(grid, n, seed)
    cache = _fills(out_dir, grid["version"])
    spent: list = []
    _fill_missing(complete, slots, cache, out_dir, batch=batch, spent=spent)

    # multi-turn rows whose follow-up stands alone (or is already answered) get one rewrite
    rows = {s["slot_id"]: to_row(s, cache[s["slot_id"]]["item"]) for s in slots
            if cache[s["slot_id"]]["status"] == "ok"}
    weak = check_dependency(complete, [r for r in rows.values() if not cache[r["slot_id"]].get("checked")],
                            spent=spent)
    for sid in [s for s in rows if s not in weak]:
        if not cache[sid].get("checked"):
            cache[sid] = {**cache[sid], "checked": True}
            append_jsonl(out_dir / "fills.jsonl", cache[sid])
    if weak:
        redo = [s for s in slots if s["slot_id"] in weak]
        for s in redo:
            cache.pop(s["slot_id"])
        _fill_missing(complete, redo, cache, out_dir, batch=batch, spent=spent, hint=REWRITE_HINT)
        again = [to_row(s, cache[s["slot_id"]]["item"]) for s in redo if cache[s["slot_id"]]["status"] == "ok"]
        still = check_dependency(complete, again, spent=spent)
        for s in redo:
            rec = cache[s["slot_id"]]
            if rec["status"] == "ok" and s["slot_id"] in still:
                rec = {**rec, "status": "error", "error": "follow-up does not depend on its context",
                       "dependency_failed": True}
            elif rec["status"] == "ok":
                rec = {**rec, "checked": True}
            cache[s["slot_id"]] = rec
            append_jsonl(out_dir / "fills.jsonl", rec)
            rows.pop(s["slot_id"], None)
            if rec["status"] == "ok":
                rows[s["slot_id"]] = to_row(s, rec["item"])

    # dedup against the other pools and earlier synthetic rows; refill drops in their stratum
    ours = {s["slot_id"] for s in slots}
    synthetic = [r for r in read_jsonl(out_dir / "turns.jsonl") if r.get("slot_id") not in ours]
    existing = [r["prompt"] for p in other_pools for r in read_jsonl(p) if r.get("prompt")]
    existing += [r["prompt"] for r in synthetic]
    if encode is None and (existing or len(rows) > 1):
        encode = default_encoder()
    order = [rows[s["slot_id"]] for s in slots if s["slot_id"] in rows]
    kept, dropped = dedup(order, existing, encode)
    all_dropped = list(dropped)
    occurrences = Counter(_combo_key(s["axes"], s["archetype"]) for s in slots)
    refills: list[dict] = []
    for _ in range(REFILL_ROUNDS):
        if not dropped:
            break
        fresh = []
        for row, _why in dropped:
            key = _combo_key(row["axes"], row["archetype"])
            arch = next((a for a in grid.get("archetypes") or [] if a["id"] == row["archetype"]), None)
            fresh.append(make_slot(grid, row["axes"], seed=seed, occurrence=occurrences[key], archetype=arch))
            occurrences[key] += 1
        refills += fresh
        _fill_missing(complete, fresh, cache, out_dir, batch=batch, spent=spent)
        new = [to_row(s, cache[s["slot_id"]]["item"]) for s in fresh if cache[s["slot_id"]]["status"] == "ok"]
        more, dropped = dedup(new, existing + [r["prompt"] for r in kept], encode)
        kept += more
        all_dropped += dropped

    from aar_gateway.grading.history import merge_into

    added = merge_into(out_dir / "turns.jsonl", kept)
    unrecoverable = [{"slot_id": s["slot_id"], "axes": s["axes"], "error": cache[s["slot_id"]].get("error")}
                     for s in [*slots, *refills] if cache.get(s["slot_id"], {}).get("status") != "ok"]
    summary = {"n": n, "seed": seed, "version": grid["version"], "accepted": len(kept), "added": added,
               "slots": len(slots), "refill_slots": len(refills), "duplicates": len(all_dropped),
               "duplicate_reasons": Counter(why.split(" (")[0] for _, why in all_dropped),
               "unrecoverable": unrecoverable, "fill_usd": round(sum(spent), 4), "llm_calls": len(spent),
               "accepted_hashes": [r["prompt_hash"] for r in kept],
               "coverage": coverage([r["axes"] for r in kept], grid)}
    (out_dir / "generate.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# -- grade ----------------------------------------------------------------------------------------------


def per_turn_cost(graded_files: Iterable[Path]) -> tuple[float, float, int]:
    """Mean and sd of one graded prompt's plan usage (every tier's replay, summed;
    the judge is not recorded per row, so this is a floor), and how many it rests on."""
    totals: dict[str, float] = {}
    for path in graded_files:
        for r in read_jsonl(path):
            if r.get("usage_usd") is not None:
                totals[f"{path}|{r['prompt_hash']}"] = totals.get(f"{path}|{r['prompt_hash']}", 0.0) + r["usage_usd"]
    vals = list(totals.values())
    if len(vals) < 5:
        return (*FALLBACK_TURN_USD, 0)
    return statistics.fmean(vals), statistics.stdev(vals), len(vals)


def judge_call_cost(graded_files: Iterable[Path]) -> tuple[float, int]:
    """Mean plan usage of one judge call, from rows that record the judge's usage (once
    per prompt, divided by its number of judgments), and how many prompts it rests on."""
    per: dict[str, float] = {}
    for path in graded_files:
        for r in read_jsonl(path):
            if r.get("judge_usage_usd") is not None:
                per[f"{path}|{r['prompt_hash']}"] = r["judge_usage_usd"] / max(1, len(r.get("judge_scores") or [0]))
    return (statistics.fmean(per.values()), len(per)) if len(per) >= 5 else (FALLBACK_JUDGE_USD, 0)


def grade_plan(out_dir: Path, budget: float, *, repeat_frac: float = 0.1,
               history: Sequence[Path] | None = None, judgments: int = 2) -> dict:
    """What a grading run would do: pending and graded counts, the turn cap the
    budget allows, and the estimated plan usage with a range. An estimate only."""
    from aar_gateway.grading.grade import pending_turns

    turns = read_jsonl(out_dir / "turns.jsonl")
    graded = read_jsonl(out_dir / "graded.jsonl")
    pending = pending_turns(turns, graded)
    history = history if history is not None else [POOL_DIR / "graded.jsonl", POOL_DIR / "multiturn" / "graded.jsonl",
                                                   out_dir / "graded.jsonl"]
    replay, sd, based_on = per_turn_cost(history)
    judge_call, judge_based_on = judge_call_cost(history)
    mean = replay + judgments * judge_call
    per = mean * (1 + repeat_frac)  # the repeat sample is graded again in full
    cap = min(len(pending), max(0, math.floor(budget / per))) if per > 0 else len(pending)
    repeat = math.ceil(cap * repeat_frac) if cap else 0
    return {"pending": len(pending), "graded": len({g["prompt_hash"] for g in graded if g.get("score") is not None}),
            "turn_cap": cap, "repeat": repeat, "budget_usd": budget,
            "estimate_usd": round(cap * mean + repeat * mean, 2),
            "estimate_range_usd": [round(max(0.0, (cap + repeat) * (mean - sd)), 2),
                                   round((cap + repeat) * (mean + sd), 2)],
            "per_turn_usd": [round(mean, 3), round(sd, 3)], "estimate_based_on": based_on,
            "judgments": judgments, "judge_call_usd": round(judge_call, 3), "judge_estimate_based_on": judge_based_on,
            "note": "an estimate: earlier prompts' replay usage plus the judge calls"
                    + ("" if judge_based_on else f" (judge call assumed ${FALLBACK_JUDGE_USD})")}


def _grader(turns: Path, out: Path, n: int, judge: str, workers: int = 2, rubric: str = "standard",
            judgments: int = 1) -> int:
    cmd = [sys.executable, "-m", "aar_gateway.grading.grade", "--turns", str(turns), "--out", str(out),
           "--judge", judge, "--n", str(n), "--workers", str(workers), "--rubric", rubric, "--judgments", str(judgments)]
    log = out.parent / "grade.log"  # the grader's progress and any traceback, kept past a crash
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n== {' '.join(cmd[2:])}\n")
        f.flush()
        code = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    if code:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        print(f"grader exited {code}; last lines of {log}:\n" + "\n".join(tail), file=sys.stderr)
    return code


def grade(out_dir: Path, budget: float, *, judge: str = JUDGE, repeat_frac: float = 0.1, seed: int = 0,
          workers: int = 2, rubric: str = "criteria", judgments: int = 2) -> dict:
    """Grade up to the plan's turn cap, then re-grade a seeded ``repeat_frac``
    sample into ``graded-repeat.jsonl`` for judge/replay noise. Returns the plan
    with the measured usage next to the estimate."""
    from aar_gateway.grading.grade import pending_turns

    plan = grade_plan(out_dir, budget, repeat_frac=repeat_frac, judgments=judgments)
    graded_before = read_jsonl(out_dir / "graded.jsonl")
    before = {(g["prompt_hash"], g["model"]) for g in graded_before}
    # the grader shuffles what it is given: hand it only the next prompts in slot order,
    # so a partly graded pool is the pair-covering prefix, not a random subset
    nxt = pending_turns(read_jsonl(out_dir / "turns.jsonl"), graded_before)[:plan["turn_cap"]]
    batch = out_dir / "grade-turns.jsonl"
    batch.write_text("".join(json.dumps(t) + "\n" for t in nxt), encoding="utf-8")
    code = _grader(batch, out_dir / "graded.jsonl", len(nxt), judge, workers, rubric, judgments) if nxt else 0
    graded = read_jsonl(out_dir / "graded.jsonl")
    new = [g for g in graded if (g["prompt_hash"], g["model"]) not in before]
    measured = sum(g.get("usage_usd") or 0.0 for g in new)
    measured += sum({g["prompt_hash"]: g.get("judge_usage_usd") or 0.0 for g in new}.values())
    done = sorted({g["prompt_hash"] for g in graded if g.get("score") is not None})
    repeat_file = out_dir / "graded-repeat.jsonl"
    already = {g["prompt_hash"] for g in read_jsonl(repeat_file) if g.get("score") is not None}
    sample = [h for h in random.Random(seed).sample(done, min(plan["repeat"], len(done))) if h not in already]
    if sample and code == 0:
        by_hash = {t["prompt_hash"]: t for t in read_jsonl(out_dir / "turns.jsonl")}
        sub = out_dir / "repeat-turns.jsonl"
        sub.write_text("".join(json.dumps(by_hash[h]) + "\n" for h in sample if h in by_hash), encoding="utf-8")
        before_r = {(g["prompt_hash"], g["model"]) for g in read_jsonl(repeat_file)}
        _grader(sub, repeat_file, len(sample), judge, workers, rubric, judgments)
        measured += sum(g.get("usage_usd") or 0.0 for g in read_jsonl(repeat_file)
                        if (g["prompt_hash"], g["model"]) not in before_r)
        measured += sum({g["prompt_hash"]: g.get("judge_usage_usd") or 0.0 for g in read_jsonl(repeat_file)
                         if (g["prompt_hash"], g["model"]) not in before_r}.values())
    return {**plan, "grader_exit": code, "measured_usd": round(measured, 2)}


# -- re-judging: does the judge's own tier move the result? ---------------------------------------------


STORED_ANSWER_CHARS = 4000  # graded.jsonl keeps this much of each answer

def rejudge_sample(graded: Sequence[dict], n: int, seed: int = 0) -> list[dict]:
    """``n`` seeded prompts whose three answers were all scored and are stored whole
    (shorter than the stored cap), so a new judge sees exactly what the first one saw.
    Each: ``{prompt_hash, prompt, answers: {model: text}, scores: {model: score}}``."""
    from evaluation.gateway.pool_eval import TIERS

    by: dict[str, dict] = {}
    for r in graded:
        if r.get("score") is not None and r.get("model") in TIERS:
            by.setdefault(r["prompt_hash"], {})[r["model"]] = r
    ok = [h for h, rows in sorted(by.items())
          if len(rows) == len(TIERS) and all(0 < len(rows[t].get("answer") or "") < STORED_ANSWER_CHARS for t in TIERS)]
    picked = random.Random(seed).sample(ok, min(n, len(ok)))
    return [{"prompt_hash": h, "prompt": by[h][TIERS[0]]["prompt"],
             "answers": {t: by[h][t]["answer"] for t in TIERS}, "scores": {t: by[h][t]["score"] for t in TIERS}}
            for h in picked]


def judge_key(model: str, rubric: str = "standard", run: int = 1) -> str:
    """How a re-judging pass is named in ``rejudge.jsonl``: the bare model for the grader's
    own rubric (first run), else ``model|rubric|rN``; the criteria rubric carries its text
    version, so a reworded rubric is a new pass."""
    if rubric == "criteria":
        rubric = f"criteria-{RUBRIC_VERSION}"
    return model if rubric == "standard" and run == 1 else f"{model}|{rubric}|r{run}"


def rejudge_one(item: dict, complete: Complete, judge: str, seed: int) -> list[dict]:
    """One prompt's stored answers, shuffled afresh (per pass) and judged under the pass's
    rubric; the letter each tier got is kept, so position bias can be measured."""
    from aar_gateway.grading.grade import judge_prompt, parse_verdicts

    rubric = judge.split("|")[1] if "|" in judge else "standard"
    order = list(item["answers"])
    random.Random(f"{seed}|{judge}|{item['prompt_hash']}").shuffle(order)
    labels = {chr(ord("A") + i): m for i, m in enumerate(order)}
    shown = {lab: item["answers"][m] for lab, m in labels.items()}
    error, verdicts = None, {}
    try:
        if rubric.startswith("criteria"):  # synthetic rows were answered in an empty folder
            reply, _cost = complete(criteria_prompt(item["prompt"], shown, empty_folder=True))
            verdicts = parse_criteria(reply, labels)
        else:
            reply, _cost = complete(judge_prompt(item["prompt"], shown))
            verdicts = {lab: {"score": s, "matches_best": mb, "reason": why}
                        for lab, (s, why, mb) in parse_verdicts(reply, labels).items()}
    except RuntimeError as exc:
        error = str(exc)[:300]
    return [{"prompt_hash": item["prompt_hash"], "model": m, "label": lab, "judge_model": judge,
             "first_score": item["scores"][m], "error": error,
             **{"score": None, "matches_best": None, **(verdicts.get(lab) or {})}} for lab, m in labels.items()]


def rejudge(out_dir: Path, n: int, judges: Sequence[str], *, seed: int = 0, workers: int = 4,
            complete_for: Callable[[str], Complete] | None = None) -> Path:
    """Re-judge ``n`` graded prompts' stored answers with each of ``judges`` into
    ``rejudge.jsonl`` (prompts already re-judged by a judge are skipped). A usage limit
    stops the run; what was judged is kept."""
    from concurrent.futures import ThreadPoolExecutor

    complete_for = complete_for or (lambda j: subscription_complete(j.split("|")[0]))
    path = out_dir / "rejudge.jsonl"
    done = {(r["judge_model"], r["prompt_hash"]) for r in read_jsonl(path) if r.get("score") is not None}
    items = rejudge_sample(read_jsonl(out_dir / "graded.jsonl"), n, seed)
    for judge in judges:
        todo = [it for it in items if (judge, it["prompt_hash"]) not in done]
        if not todo:
            continue
        complete = complete_for(judge)
        with ThreadPoolExecutor(max(1, workers)) as pool:
            for rows in pool.map(lambda it, c=complete, j=judge: rejudge_one(it, c, j, seed), todo):
                append_jsonl(path, *rows)
                if rows[0]["error"] and _LIMIT.search(rows[0]["error"]):
                    print(f"stopped: usage limit ({rows[0]['error'][:80]})", file=sys.stderr)
                    pool.shutdown(cancel_futures=True)
                    return path
    return path


def judge_agreement(rows: Iterable[dict], tau: float, q_min: float, keys: Sequence[str] | None = None) -> dict:
    """Per judging pass (``keys``: all when None), on the prompts every chosen pass scored:
    each tier's mean score (first judge and this one), the Sonnet-minus-Opus gap, agreement
    of the adequate-oracle tier with the first judge, score correlation, mean score by answer
    letter, per-criterion tier means (criteria rubric), and the same agreement between every
    pair of passes -- two runs of one rubric say how repeatable it is."""
    from evaluation.gateway.pool_eval import TIERS

    by: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("score") is not None and (keys is None or r["judge_model"] in keys):
            by.setdefault(r["judge_model"], {}).setdefault(r["prompt_hash"], {})[r["model"]] = r
    full = {j: {h for h, m in d.items() if len(m) == len(TIERS)} for j, d in by.items()}
    common = set.intersection(*full.values()) if full else set()
    out = {"n": len(common), "judges": {}}
    hs = sorted(common)
    for judge, d in by.items():
        if not hs:
            break
        new = np.array([[d[h][t]["score"] for t in TIERS] for h in hs])
        first = np.array([[d[h][t]["first_score"] for t in TIERS] for h in hs])
        letters: dict[str, list] = {}
        for h in hs:
            for t in TIERS:
                letters.setdefault(d[h][t]["label"], []).append(d[h][t]["score"])
        out["judges"][judge] = {
            "mean": {n_: float(new[:, i].mean()) for i, n_ in enumerate(TIER_NAMES)},
            "first_mean": {n_: float(first[:, i].mean()) for i, n_ in enumerate(TIER_NAMES)},
            "sonnet_minus_opus": float((new[:, 1] - new[:, 2]).mean()),
            "first_sonnet_minus_opus": float((first[:, 1] - first[:, 2]).mean()),
            "oracle_agreement": float((adequate_oracle(new, tau, q_min) == adequate_oracle(first, tau, q_min)).mean()),
            "score_corr": float(np.corrcoef(new.ravel(), first.ravel())[0, 1]),
            "mean_abs_diff": float(np.abs(new - first).mean()),
            "by_letter": {k: float(np.mean(v)) for k, v in sorted(letters.items())}}
        crit = {}
        for k, _ in CRITERIA:
            vals = {t: [d[h][t]["criteria"][k] for h in hs if (d[h][t].get("criteria") or {}).get(k) is not None]
                    for t in TIERS}
            if any(vals.values()):
                crit[k] = {n_: (float(np.mean(vals[t])) if vals[t] else None) for n_, t in zip(TIER_NAMES, TIERS,
                                                                                               strict=True)}
        if crit:
            out["judges"][judge]["criteria"] = crit
            out["judges"][judge]["hallucination_rate"] = {
                n_: float(np.mean([bool(d[h][t].get("hallucinations")) for h in hs]))
                for n_, t in zip(TIER_NAMES, TIERS, strict=True)}
    mats = {j: np.array([[d[h][t]["score"] for t in TIERS] for h in hs]) for j, d in by.items()} if hs else {}
    names = sorted(mats)
    out["pairs"] = [{"a": a, "b": b,
                     "oracle_agreement": float((adequate_oracle(mats[a], tau, q_min)
                                                == adequate_oracle(mats[b], tau, q_min)).mean()),
                     "score_corr": float(np.corrcoef(mats[a].ravel(), mats[b].ravel())[0, 1]),
                     "mean_abs_diff": float(np.abs(mats[a] - mats[b]).mean())}
                    for i, a in enumerate(names) for b in names[i + 1:]]
    return out


# -- report ---------------------------------------------------------------------------------------------


TIER_NAMES = ("haiku", "sonnet", "opus")  # pool_eval.TIERS order


def adequate_oracle(S: np.ndarray, tau: float, q_min: float) -> np.ndarray:
    """Per prompt, the cheapest tier scoring within ``tau`` of the best and at least
    ``q_min``; the best-scoring tier when none qualifies. Ties go to the cheaper tier."""
    ok = (S >= S.max(axis=1, keepdims=True) - tau) & (S >= q_min)
    return np.where(ok.any(axis=1), ok.argmax(axis=1), S.argmax(axis=1))


def quality_section(out_dir: Path, grid: dict, accepted: set[str], encode: Callable | None) -> dict:
    from evaluation.gateway import pool_eval as pe
    from evaluation.gateway.router_compare import paired_bootstrap

    turns = {t["prompt_hash"]: t for t in read_jsonl(out_dir / "turns.jsonl")}
    prompts = [p for p in pe.load_prompts(out_dir / "graded.jsonl") if p["prompt_hash"] in accepted]
    graded_any = {g["prompt_hash"] for g in read_jsonl(out_dir / "graded.jsonl")} & accepted
    excluded_rows = sorted(graded_any - {p["prompt_hash"] for p in prompts})
    rubrics = Counter((g.get("rubric") or "standard").split("-")[0] for g in read_jsonl(out_dir / "graded.jsonl")
                      if g["prompt_hash"] in accepted and g.get("score") is not None)
    section = {"eligible": len(prompts), "accepted": len(accepted), "excluded": excluded_rows,
               "rubrics": {k: v // 3 for k, v in rubrics.items()}}
    if not prompts:
        return section
    S = np.stack([p["score"] for p in prompts])
    tau, q_min = grid["oracle"]["tau"], grid["oracle"]["q_min"]
    picks = pe.policies(prompts, margin=0.05, k=10, encode=encode if len(prompts) > 10 else None)
    picks["oracle (adequate)"] = adequate_oracle(S, tau, q_min)
    results = pe.evaluate(prompts, picks=picks)
    best = S[np.arange(len(S)), picks["oracle (adequate)"]]
    for r in results:
        pick = picks[r["policy"]]
        r["regret"] = float((best - S[np.arange(len(S)), pick]).mean())
        r["dq_vs_opus"] = float(S[np.arange(len(S)), pick].mean() - S[:, -1].mean())
        if r["policy"].startswith("pool kNN"):
            r["vs_opus_ci"] = paired_bootstrap(prompts, pick, np.full(len(S), len(TIER_NAMES) - 1))
    oracle = picks["oracle (adequate)"]
    order = {t: i for i, t in enumerate(TIER_NAMES)}
    cal = Counter()
    strata: dict[str, dict] = {}
    for p, o in zip(prompts, oracle, strict=True):
        t = turns.get(p["prompt_hash"]) or {}
        hint = order.get(t.get("suggested_tier"))
        if hint is not None:
            cal["agree" if hint == o else "over-escalate" if hint > o else "under-escalate"] += 1
        for axis in ("difficulty", "task", "depth"):
            key = f"{axis}={(t.get('axes') or {}).get(axis)}"
            st = strata.setdefault(key, {"n": 0, "oracle_mix": Counter()})
            st["n"] += 1
            st["oracle_mix"][TIER_NAMES[o]] += 1
    section.update({"policies": results, "calibration": dict(cal), "strata": strata,
                    "oracle": {"tau": tau, "q_min": q_min}, "noise": repeat_noise(out_dir)})
    return section


def repeat_noise(out_dir: Path) -> dict | None:
    """Mean |score difference| and oracle-tier agreement between the first grading
    and the repeat, on the prompts graded twice."""
    from evaluation.gateway import pool_eval as pe

    again = {p["prompt_hash"]: p for p in pe.load_prompts(out_dir / "graded-repeat.jsonl")}
    if not again:
        return None
    first = [p for p in pe.load_prompts(out_dir / "graded.jsonl") if p["prompt_hash"] in again]
    if not first:
        return None
    diff = np.concatenate([np.abs(p["score"] - again[p["prompt_hash"]]["score"]) for p in first])
    a = np.stack([p["score"] for p in first])
    b = np.stack([again[p["prompt_hash"]]["score"] for p in first])
    return {"n": len(first), "mean_abs_score_diff": float(diff.mean()),
            "best_tier_agreement": float((a.argmax(axis=1) == b.argmax(axis=1)).mean())}


def verdicts(gen: dict | None, quality: dict | None, behave: dict | None) -> dict:
    """Two verdicts, never merged. Quality is PASS only with every accepted prompt
    graded and no stratum short; conformance only with every scenario run and passing."""
    if not gen:
        q = ("NOT RUN", "nothing generated yet")
    elif not quality or not quality.get("eligible"):
        q = ("NOT RUN", "no prompt graded on every tier yet")
    else:
        short = [k for k, v in gen["coverage"].items() if v["short"]]
        why = []
        if quality["eligible"] < quality["accepted"]:
            why.append(f"{quality['eligible']}/{quality['accepted']} accepted prompts fully graded")
        if short:
            why.append(f"short strata in {', '.join(short)}")
        if gen.get("unrecoverable"):
            why.append(f"{len(gen['unrecoverable'])} unrecoverable slot(s)")
        q = ("PARTIAL", "; ".join(why)) if why else ("PASS", "every accepted prompt graded, every stratum met")
    if not behave:
        c = ("NOT RUN", "run `behave`")
    else:
        res = behave["results"]
        failed = [r["id"] for r in res if r["status"] == "fail"]
        skipped = [r["id"] for r in res if r["status"] == "skip"]
        if failed:
            c = ("FAIL", f"failed: {', '.join(failed)}")
        elif skipped:
            c = ("PARTIAL", f"skipped (precondition unmet): {', '.join(skipped)}")
        else:
            c = ("PASS", f"{sum(r['status'] == 'pass' for r in res)} scenario(s) passed"
                         + (f", {sum(r['status'] == 'info' for r in res)} recorded" if any(
                             r["status"] == "info" for r in res) else ""))
        if not behave.get("isolation", {}).get("ok", True):
            c = ("FAIL", f"scenario traffic reached the live gateway; {c[1]}")
    return {"quality": {"verdict": q[0], "why": q[1]}, "conformance": {"verdict": c[0], "why": c[1]}}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render(rep: dict) -> str:
    v = rep["verdicts"]
    lines = ["# Scenario report", "",
             f"- **Routing quality: {v['quality']['verdict']}** — {v['quality']['why']}",
             f"- **Gateway conformance: {v['conformance']['verdict']}** — {v['conformance']['why']}", ""]
    gen = rep.get("generate")
    if gen:
        lines += ["## Generation", "",
                  f"{gen['accepted']} accepted of {gen['slots']} slots (+{gen['refill_slots']} refills), "
                  f"{gen['duplicates']} duplicate(s) dropped {dict(gen['duplicate_reasons'])}, "
                  f"{len(gen['unrecoverable'])} unrecoverable; fill usage ${gen['fill_usd']:.3f} "
                  f"over {gen['llm_calls']} call(s); version `{gen['version']}`.", ""]
        for pair, cov in gen["coverage"].items():
            filled = sum(1 for c in cov["cells"].values() if c >= cov["min"])
            lines.append(f"- {pair}: {filled}/{len(cov['cells'])} cells at >= {cov['min']}"
                         + (f"; short: {', '.join(cov['short'][:12])}{' ...' if len(cov['short']) > 12 else ''}"
                            if cov["short"] else "")
                         + (f"; excluded: {', '.join(cov['excluded'])}" if cov["excluded"] else ""))
        for u in gen["unrecoverable"][:10]:
            lines.append(f"- unrecoverable `{u['slot_id']}` {u['axes']}: {u['error']}")
        lines.append("")
    q = rep.get("quality") or {}
    if q.get("policies"):
        lines += ["## Routing quality", "",
                  f"{q['eligible']} prompts graded on every tier (of {q['accepted']} accepted). Oracle: cheapest tier "
                  f"with score >= best - {q['oracle']['tau']} and >= {q['oracle']['q_min']}. Baseline: no router "
                  f"(every prompt on Opus, the model asked for). Judged by rubric: "
                  f"{', '.join(f'{k} {v}' for k, v in sorted((q.get('rubrics') or {}).items()))}.", "",
                  "| policy | quality | ΔQ vs no router | saved vs no router | vs mix | regret | H/S/O |",
                  "|---|---|---|---|---|---|---|"]
        for r in q["policies"]:
            m = r["mix"]
            lines.append(f"| {r['policy']} | {r['quality']:.3f} | {r['dq_vs_opus']:+.3f} | {r['vs_opus']:.0%} | "
                         f"{r['vs_mix']:+.3f} | {r['regret']:.3f} | {m.get('haiku', 0)}/{m.get('sonnet', 0)}/{m.get('opus', 0)} |")
            if r.get("vs_opus_ci"):
                ci = r["vs_opus_ci"]["quality"]
                lines.append(f"| ↳ 95% CI ΔQ | [{ci['lo']:+.3f}, {ci['hi']:+.3f}] {ci['verdict']} | | | | | |")
        cal = q.get("calibration") or {}
        total = sum(cal.values()) or 1
        lines += ["", "Generator calibration (suggested tier vs measured oracle): "
                      + ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(cal.items())), ""]
        lines += ["| stratum | n | oracle H/S/O |", "|---|---|---|"]
        for key, st in sorted(q["strata"].items()):
            om = st["oracle_mix"]
            lines.append(f"| {key} | {st['n']} | {om.get('haiku', 0)}/{om.get('sonnet', 0)}/{om.get('opus', 0)} |")
        if q.get("noise"):
            nz = q["noise"]
            lines += ["", f"Repeat grading on {nz['n']} prompts: mean |Δscore| {nz['mean_abs_score_diff']:.3f}, "
                          f"best-tier agreement {nz['best_tier_agreement']:.0%}."]
        if q.get("excluded"):
            lines.append(f"\nExcluded (not graded on every tier): {len(q['excluded'])}.")
        lines.append("")
    b = rep.get("behave")
    if b:
        lines += ["## Gateway conformance", "", "| id | kind | status | detail |", "|---|---|---|---|"]
        for r in b["results"]:
            failed = [c for c in r["checks"] if not c["ok"]]
            detail = "; ".join(f"{c['name']}: {c['detail']}" for c in failed) or r.get("note") or ""
            lines.append(f"| {r['id']} | {r['kind']} | {r['status']} | {detail[:300]} |")
        iso = b.get("isolation") or {}
        lines += ["", f"Isolation: {'ok' if iso.get('ok') else 'LEAKED'} — {iso.get('detail', '')}", ""]
    return "\n".join(lines)


def report(out_dir: Path = OUT_DIR, *, grid_path: Path = GRID, encode: Callable | None = None) -> dict:
    grid = load_grid(grid_path)
    gen = _read_json(out_dir / "generate.json")
    behave = _read_json(out_dir / "behave.json")
    quality = None
    if gen and (out_dir / "graded.jsonl").exists():
        quality = quality_section(out_dir, grid, set(gen["accepted_hashes"]), encode or default_encoder())
    rep = {"generate": gen, "quality": quality, "behave": behave, "verdicts": verdicts(gen, quality, behave)}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(render(rep), encoding="utf-8")
    return rep


# -- CLI ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.gateway.scenarios", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    parsers = {name: sub.add_parser(name) for name in ("generate", "behave", "report", "grade", "run", "rejudge")}
    rj = parsers["rejudge"]
    rj.add_argument("--n", type=int, default=30)
    rj.add_argument("--judges", default="claude-opus-5-5,claude-sonnet-5-5",
                    help="comma list; the original judge again measures plain judge noise")
    rj.add_argument("--rubric", choices=RUBRICS, default="standard",
                    help="criteria: relevance, accuracy/hallucinations, completeness, conciseness, "
                         "consistency, balance, uncertainty, then overall")
    rj.add_argument("--runs", type=int, default=1, help="passes per judge; 2 measures repeatability")
    rj.add_argument("--workers", type=int, default=4)
    rj.add_argument("--yes", action="store_true", help="the user approved this run (plan usage)")
    for name in ("generate", "run"):
        parsers[name].add_argument("--n", type=int, required=True)
        parsers[name].add_argument("--seed", type=int, default=0)
        parsers[name].add_argument("--batch", type=int, default=BATCH)
    for name in ("behave", "run"):
        parsers[name].add_argument("--wave", type=int, choices=(1, 2), action="append",
                                   help="scenario wave(s) to run; default both")
    g = parsers["grade"]
    g.add_argument("--budget", type=float, required=True, help="estimated plan usage in USD (an estimate, not a cap)")
    g.add_argument("--judge", default=JUDGE)
    g.add_argument("--repeat-frac", type=float, default=0.1)
    g.add_argument("--rubric", choices=RUBRICS, default="criteria",
                   help="judge rubric for newly graded prompts (criteria: per-criterion scores + overall)")
    g.add_argument("--judgments", type=int, default=2, help="judge each prompt this many times and average")
    g.add_argument("--workers", type=int, default=2, help="prompts graded at once (wall clock only; same plan usage)")
    g.add_argument("--yes", action="store_true", help="the user approved this run")
    for p in parsers.values():
        p.add_argument("--out-dir", default=str(OUT_DIR))
        p.add_argument("--grid", default=str(GRID))
    args = ap.parse_args(argv)
    out_dir, grid_path = Path(args.out_dir).expanduser(), Path(args.grid).expanduser()
    if hasattr(sys.stdout, "reconfigure"):  # the report's dashes and Δ on a cp1252 Windows console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.cmd in ("generate", "run"):
        summary = generate(args.n, args.seed, grid_path=grid_path, out_dir=out_dir, batch=args.batch)
        print(f"accepted {summary['accepted']} (added {summary['added']}), {summary['duplicates']} duplicate(s), "
              f"{len(summary['unrecoverable'])} unrecoverable, fill ${summary['fill_usd']:.3f}", file=sys.stderr)
    if args.cmd in ("behave", "run"):
        from evaluation.gateway.scenarios_behave import run_suite

        result = run_suite(out_dir, waves=tuple(args.wave or (1, 2)))
        bad = [r["id"] for r in result["results"] if r["status"] == "fail"]
        print(f"behaviour: {len(result['results']) - len(bad)} ok, failed {bad or 'none'}; "
              f"isolation {'ok' if result['isolation']['ok'] else 'LEAKED'}", file=sys.stderr)
    if args.cmd == "rejudge":
        judges = [judge_key(j.strip(), args.rubric, run) for j in args.judges.split(",") if j.strip()
                  for run in range(1, args.runs + 1)]
        if not args.yes:
            print(f"would re-judge {args.n} prompts with {', '.join(judges)}: {args.n * len(judges)} judge calls "
                  f"(judge only, no replays). Rerun with --yes once approved.", file=sys.stderr)
            return 3
        rejudge(out_dir, args.n, judges, workers=args.workers)
        grid = load_grid(grid_path)
        print(json.dumps(judge_agreement(read_jsonl(out_dir / "rejudge.jsonl"), grid["oracle"]["tau"],
                                         grid["oracle"]["q_min"], keys=judges), indent=1))
        return 0
    if args.cmd == "grade":
        if not args.yes:
            print(json.dumps(grade_plan(out_dir, args.budget, repeat_frac=args.repeat_frac,
                                        judgments=args.judgments), indent=1))
            print("not graded: rerun with --yes once the user has approved this estimate", file=sys.stderr)
            return 3
        print(json.dumps(grade(out_dir, args.budget, judge=args.judge, repeat_frac=args.repeat_frac,
                               workers=args.workers, rubric=args.rubric,
                               judgments=args.judgments), indent=1))
    if args.cmd in ("report", "run", "grade"):
        rep = report(out_dir, grid_path=grid_path)
        print(render(rep))
        if args.cmd == "run":
            plan = grade_plan(out_dir, 3.0)
            print(f"\nstopped before grading: {plan['pending']} pending. Next, with the user's approval:\n"
                  f"  python -m evaluation.gateway.scenarios grade --budget <usd>", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

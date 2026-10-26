import json
import re

import numpy as np
import pytest

from evaluation.gateway import router_compare as rc
from evaluation.gateway import scenarios as sc
from evaluation.gateway import scenarios_behave as sb
from evaluation.gateway.pool_eval import TIERS

TINY = """
axes:
  task: [debug]
  difficulty: [trivial]
  language: [python]
  depth: [0]
suggested_tier: {trivial: haiku}
pairs_min:
  task x difficulty: 1
oracle: {tau: 0.1, q_min: 0.6}
"""


@pytest.fixture
def grid():
    return sc.load_grid(sc.GRID)


def _listing(prompt):
    rest = prompt.split("Slots:\n", 1)[1]
    return json.loads(rest[rest.index("["):])


def writer(text_for=lambda s: f"please fix bug {s['slot_id']} in parser.py", *, check=None, calls=None):
    """A fake Haiku: fills every slot validly; dependency checks pass unless ``check`` says."""
    calls = [] if calls is None else calls

    def complete(prompt):
        calls.append(prompt)
        if prompt.startswith(sc.CHECK_PROMPT):
            return json.dumps(check(prompt) if check else {}), 0.001
        slots = _listing(prompt)
        items = [{"slot_id": s["slot_id"], "message": text_for(s),
                  "context": [{"role": "user" if i % 2 == 0 else "assistant", "text": f"ctx {i} {s['slot_id']}"}
                              for i in range(2 * int(s["depth"]))]} for s in slots]
        return json.dumps(items), 0.01

    return complete, calls


def onehot_encoder(texts):
    """Distinct texts are orthogonal; identical texts coincide."""
    vocab = {t: i for i, t in enumerate(dict.fromkeys(texts))}
    out = np.zeros((len(texts), len(vocab)))
    for row, t in enumerate(texts):
        out[row, vocab[t]] = 1.0
    return out


# -- slots ----------------------------------------------------------------------------------------------


def test_expand_is_seeded_prefix_stable_and_archetypes_first(grid):
    a, b = sc.expand(grid, 40, seed=3), sc.expand(grid, 60, seed=3)
    assert a == b[:40]
    assert [s["archetype"] for s in a[:len(grid["archetypes"])]] == [x["id"] for x in grid["archetypes"]]
    assert len({s["slot_id"] for s in b}) == 60
    assert not any(sc.excluded(s["axes"], grid) for s in b)
    assert sc.expand(grid, 40, seed=4) != a


def test_repeated_combination_gets_distinct_ids_and_a_new_grid_a_new_namespace(tmp_path, grid):
    path = tmp_path / "grid.yaml"
    path.write_text(TINY, encoding="utf-8")
    tiny = sc.load_grid(path)
    slots = sc.expand(tiny, 5, seed=0)
    assert [s["occurrence"] for s in slots] == [0, 1, 2, 3, 4]
    assert len({s["slot_id"] for s in slots}) == 5
    path.write_text(TINY + "\n# edited\n", encoding="utf-8")
    edited = sc.load_grid(path)
    assert edited["version"] != tiny["version"]
    assert {s["slot_id"] for s in sc.expand(edited, 5)}.isdisjoint(s["slot_id"] for s in slots)


def test_sixty_slots_cover_every_required_pair(grid):
    cov = sc.coverage([s["axes"] for s in sc.expand(grid, 60, seed=0)], grid)
    assert {k: v["short"] for k, v in cov.items()} == {k: [] for k in cov}
    assert "shell-git/sql" not in cov["task x difficulty"]["cells"]  # exclusions only bite their own pairs


def test_coverage_reports_short_and_excluded_cells(grid):
    cov = sc.coverage([grid["archetypes"][0]["axes"]], grid)
    td = cov["task x difficulty"]
    assert td["cells"]["debug/trivial"] == 1 and "explain/hard" in td["short"]
    assert td["excluded"] == []


# -- filling --------------------------------------------------------------------------------------------


def test_validate_fill_wants_exact_alternating_context(grid):
    slot = sc.make_slot(grid, {"task": "debug", "difficulty": "hard", "language": "go", "depth": 1}, seed=0,
                        occurrence=0)
    ok = {"message": "and now?", "context": [{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"}]}
    assert sc.validate_fill(slot, ok) is None
    assert "entries" in sc.validate_fill(slot, {**ok, "context": ok["context"][:1]})
    assert "not user" in sc.validate_fill(slot, {**ok, "context": ok["context"][::-1]})
    assert sc.validate_fill(slot, {**ok, "message": "  "}) == "empty message"


def test_fill_batch_retries_bad_slots_then_reports_them(grid):
    slots = sc.expand(grid, 3, seed=0)
    good, _ = writer()
    tries = {"n": 0}

    def flaky(prompt):
        tries["n"] += 1
        reply, cost = good(prompt)
        items = json.loads(reply)
        if slots[0]["slot_id"] in prompt:  # this slot always comes back empty
            for it in items:
                if it["slot_id"] == slots[0]["slot_id"]:
                    it["message"] = ""
        if tries["n"] == 1:
            items = [it for it in items if it["slot_id"] != slots[1]["slot_id"]]  # missing on the first reply
        return json.dumps(items), cost

    out = sc.fill_batch(flaky, slots)
    assert out[slots[0]["slot_id"]] == {"error": "empty message"}
    assert "item" in out[slots[1]["slot_id"]] and "item" in out[slots[2]["slot_id"]]
    assert tries["n"] == 2


def test_unparseable_replies_retry_then_split_down_to_single_slots(grid):
    slots = sc.expand(grid, 4, seed=0)
    good, _ = writer()
    seen = []

    def garbled_when_batched(prompt):
        n = len(_listing(prompt))
        seen.append(n)
        return ("not json", None) if n > 1 else good(prompt)

    out = sc.fill_batch(garbled_when_batched, slots)
    assert all("item" in v for v in out.values())
    assert seen == [4, 4, 2, 1, 1, 2, 1, 1]


def test_a_usage_limit_is_raised_not_recorded(grid):
    def limited(_prompt):
        raise RuntimeError("claude exited 1: You've hit your usage limit · resets 5pm")

    with pytest.raises(RuntimeError, match="usage limit"):
        sc.fill_batch(limited, sc.expand(grid, 2))


# -- dedup ----------------------------------------------------------------------------------------------


def _r(prompt, i=0):
    return {"prompt": prompt, "prompt_hash": sc.normalised_hash(prompt), "slot_id": str(i)}


def test_dedup_drops_exact_and_near_duplicates_by_similarity_not_distance():
    def enc(texts):
        table = {"fix the parser": [1, 0, 0], "Fix  the PARSER": [1, 0, 0], "repair the parser": [0.95, 0.31, 0],
                 "write docs for the api": [0, 0, 1], "old pool prompt": [0, 1, 0]}
        return np.array([table[t] for t in texts], dtype=float)

    rows = [_r("fix the parser", 1), _r("Fix  the PARSER", 2), _r("repair the parser", 3),
            _r("write docs for the api", 4)]
    kept, dropped = sc.dedup(rows, ["old pool prompt"], enc)
    assert [r["slot_id"] for r in kept] == ["1", "4"]
    assert [why.split(" (")[0] for _, why in dropped] == ["exact duplicate", "near duplicate"]
    kept, _ = sc.dedup([_r("repair the parser")], ["fix the parser"], enc, threshold=0.99)
    assert len(kept) == 1  # 0.95 similarity is below a 0.99 threshold


# -- generate -------------------------------------------------------------------------------------------


def test_generate_caches_fills_and_extends_with_n(tmp_path, grid):
    complete, calls = writer()
    first = sc.generate(14, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    assert first["accepted"] == 14 and first["unrecoverable"] == []
    writes = [c for c in calls if not c.startswith(sc.CHECK_PROMPT)]
    calls.clear()
    again = sc.generate(14, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    assert again["added"] == 0 and calls == []  # everything cached, dependency already checked
    sc.generate(20, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    new = [c for c in calls if not c.startswith(sc.CHECK_PROMPT)]
    assert len(new) == 1 and len(_listing(new[0])) == 6 and len(writes) == 1
    rows = sc.read_jsonl(tmp_path / "turns.jsonl")
    assert len(rows) == 20 and all(r["source"] == "synthetic" and r["suggested_tier"] for r in rows)
    multi = next(r for r in rows if r["axes"]["depth"])
    assert multi["prompt"].startswith("<earlier conversation>") and multi["prompt"].endswith(multi["route_text"])


def test_duplicates_are_refilled_in_their_stratum(tmp_path, grid):
    texts = iter(["same text"] * 2 + [f"distinct {i}" for i in range(100)])
    complete, _ = writer(lambda s: next(texts))
    out = sc.generate(4, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    assert out["duplicates"] == 1 and out["refill_slots"] == 1 and out["accepted"] == 4
    rows = sc.read_jsonl(tmp_path / "turns.jsonl")
    slots = sc.expand(grid, 4)
    dropped_axes = slots[1]["axes"]  # the second "same text" lost
    assert sum(r["axes"] == dropped_axes for r in rows) == 1


def test_weak_follow_ups_get_one_rewrite_then_are_unrecoverable(tmp_path, grid):
    slots = sc.expand(grid, 3)  # archetype 3 is multi-turn
    target = slots[2]["slot_id"]

    def check(prompt):
        return {target: {"depends": False, "answered": False}} if target in prompt else {}

    complete, calls = writer(check=check)
    out = sc.generate(3, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    assert [u["slot_id"] for u in out["unrecoverable"]] == [target]
    assert out["accepted"] == 2
    assert any(sc.REWRITE_HINT in c for c in calls)


# -- grade and report -----------------------------------------------------------------------------------


def test_grade_without_yes_only_prints_the_estimate(tmp_path, monkeypatch, capsys):
    complete, _ = writer()
    sc.generate(4, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    monkeypatch.setattr(sc, "_grader", lambda *a, **k: pytest.fail("graded without approval"))
    monkeypatch.setattr(sc, "POOL_DIR", tmp_path / "no-history")  # no earlier graded files: the fallback cost
    assert sc.main(["grade", "--budget", "1", "--out-dir", str(tmp_path)]) == 3
    plan = json.loads(capsys.readouterr().out)
    assert plan["pending"] == 4 and plan["turn_cap"] == 1  # $1 / (($0.35 + 2 judge calls x $0.06) * 1.1) per turn
    assert plan["estimate_range_usd"][0] < plan["estimate_usd"] < plan["estimate_range_usd"][1]


def test_adequate_oracle_takes_the_cheapest_tier_within_tau_and_above_q_min():
    S = np.array([[0.85, 0.9, 0.95],   # haiku within 0.1 of best
                  [0.5, 0.55, 0.58],   # none reaches q_min: the best
                  [0.7, 0.9, 0.9]])    # sonnet ties opus: the cheaper
    assert sc.adequate_oracle(S, 0.1, 0.6).tolist() == [0, 2, 1]


def test_verdicts_never_pass_partial_work():
    cov_ok = {"task x difficulty": {"short": []}}
    gen = {"coverage": cov_ok, "unrecoverable": []}
    behave_ok = {"results": [{"id": "B01", "status": "pass"}, {"id": "B09", "status": "info"}],
                 "isolation": {"ok": True}}
    v = sc.verdicts(gen, {"eligible": 10, "accepted": 10}, behave_ok)
    assert v["quality"]["verdict"] == "PASS" and v["conformance"]["verdict"] == "PASS"
    assert sc.verdicts(gen, {"eligible": 9, "accepted": 10}, None)["quality"]["verdict"] == "PARTIAL"
    assert sc.verdicts({**gen, "coverage": {"x": {"short": ["a/b"]}}}, {"eligible": 1, "accepted": 1},
                       None)["quality"]["verdict"] == "PARTIAL"
    assert sc.verdicts(gen, None, None)["quality"]["verdict"] == "NOT RUN"
    skipped = {**behave_ok, "results": [{"id": "B01", "status": "skip"}]}
    assert sc.verdicts(gen, None, skipped)["conformance"]["verdict"] == "PARTIAL"
    leaked = {**behave_ok, "isolation": {"ok": False}}
    assert sc.verdicts(gen, None, leaked)["conformance"]["verdict"] == "FAIL"


def test_report_before_grading_is_partial_and_renders(tmp_path):
    complete, _ = writer()
    sc.generate(4, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    rep = sc.report(tmp_path, encode=onehot_encoder)
    assert rep["verdicts"]["quality"]["verdict"] == "NOT RUN"
    assert "## Generation" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_report_scores_graded_prompts(tmp_path):
    complete, _ = writer()
    sc.generate(12, out_dir=tmp_path, complete=complete, encode=onehot_encoder, other_pools=())
    rows = []
    for i, t in enumerate(sc.read_jsonl(tmp_path / "turns.jsonl")):
        for j, model in enumerate(TIERS):
            rows.append({"prompt_hash": t["prompt_hash"], "prompt": t["prompt"], "model": model,
                         "score": [0.8, 0.9, 0.9][j] if i % 2 else [0.3, 0.6, 0.9][j], "usage_usd": [0.01, 0.03, 0.1][j],
                         "matches_best": j >= 1})
    (tmp_path / "graded.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = sc.report(tmp_path, encode=onehot_encoder)
    q = rep["quality"]
    assert q["eligible"] == 12
    oracle = next(r for r in q["policies"] if r["policy"] == "oracle (adequate)")
    assert oracle["regret"] == 0.0 and oracle["mix"] == {"haiku": 6, "sonnet": 0, "opus": 6}
    assert sum(q["calibration"].values()) == 12
    assert "| oracle (adequate) |" in (tmp_path / "report.md").read_text(encoding="utf-8")


# -- behaviour plumbing ---------------------------------------------------------------------------------


def test_stub_failing_is_scoped_and_ids_are_unique():
    with rc.failing({"claude-haiku-5-5": 529}):
        status, _, body = rc.record_call("/v1/messages", b'{"model": "claude-haiku-5-5"}', {})
        ok, _, _ = rc.record_call("/v1/messages", b'{"model": "claude-opus-5-5"}', {})
    assert (status, ok) == (529, 200) and json.loads(body)["type"] == "error"
    after, _, a = rc.record_call("/v1/messages", b'{"model": "claude-haiku-5-5"}', {})
    _, _, b = rc.record_call("/v1/messages", b'{"model": "claude-haiku-5-5"}', {})
    assert after == 200 and json.loads(a)["id"] != json.loads(b)["id"]


def test_scenario_status_and_exceptions():
    def boom(gw, checks):
        sb.pre(checks, "ready", True)
        raise ValueError("nope")

    res = sb.run_scenario({}, "BX", "mechanics", boom)
    assert res["status"] == "fail" and "ValueError" in res["checks"][-1]["detail"]
    assert sb.status_of([{"kind": "pre", "ok": False}, {"kind": "expect", "ok": False}], "mechanics") == "skip"
    assert sb.status_of([{"kind": "forbid", "ok": False}], "mechanics") == "fail"
    assert sb.status_of([{"kind": "expect", "ok": True}], "observe") == "info"


def test_isolation_reads_only_lines_appended_since_the_start(tmp_path):
    log = tmp_path / "requests.jsonl"
    log.write_text(json.dumps({"claude_session": "scn123456789"}) + "\n", encoding="utf-8")
    offset = log.stat().st_size
    assert sb.isolation(log, offset, {"scn123456789"})["ok"]
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"claude_session": "other"}) + "\n" + json.dumps({"claude_session": "scn123456789"}) + "\n")
    assert not sb.isolation(log, offset, {"scn123456789"})["ok"]


def _graded_rows(n, long_first=False):
    rows = []
    for i in range(n):
        for j, model in enumerate(TIERS):
            text = "x" * (5000 if long_first and i == 0 else 100 + j)
            rows.append({"prompt_hash": f"h{i}", "prompt": f"p{i}", "model": model, "score": [0.5, 0.8, 0.9][j],
                         "answer": text[:sc.STORED_ANSWER_CHARS], "usage_usd": 0.1, "matches_best": j > 0})
    return rows


def test_rejudge_samples_only_prompts_stored_whole():
    picked = sc.rejudge_sample(_graded_rows(5, long_first=True), 10)
    assert sorted(p["prompt_hash"] for p in picked) == ["h1", "h2", "h3", "h4"]
    assert picked[0]["scores"][TIERS[0]] == 0.5


def test_rejudge_records_letters_and_compares_judges(tmp_path):
    (tmp_path / "graded.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _graded_rows(6)), encoding="utf-8")

    def judge_for(name):
        def complete(prompt):
            # this fake judge scores by answer length: the longest (Opus's) answer best
            answers = re.findall(r"=== Answer (\w) ===\n(x+)", prompt)
            return json.dumps({lab: {"score": len(txt) - 95, "matches_best": True} for lab, txt in answers}), 0.0
        return complete

    sc.rejudge(tmp_path, 6, ["opus-judge", "sonnet-judge"], workers=2, complete_for=judge_for)
    rows = sc.read_jsonl(tmp_path / "rejudge.jsonl")
    assert len(rows) == 6 * 3 * 2 and {r["label"] for r in rows} == {"A", "B", "C"}
    agree = sc.judge_agreement(rows, 0.1, 0.6)
    assert agree["n"] == 6
    j = agree["judges"]["opus-judge"]
    assert j["first_sonnet_minus_opus"] == pytest.approx(-0.1) and j["sonnet_minus_opus"] == pytest.approx(-0.1)
    assert set(j["by_letter"]) == {"A", "B", "C"}
    sc.rejudge(tmp_path, 6, ["opus-judge"], complete_for=lambda n: pytest.fail("re-judged a done prompt"))


def test_criteria_rubric_is_parsed_and_summarised(tmp_path):
    (tmp_path / "graded.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _graded_rows(4)), encoding="utf-8")
    keys = [sc.judge_key("j", "criteria", r) for r in (1, 2)]
    prompts = []

    def judge_for(_name):
        def complete(prompt):
            prompts.append(prompt)
            labs = re.findall(r"=== Answer (\w) ===", prompt)
            return "Here you go: " + json.dumps({lab: {"relevance": 9, "accuracy": 8, "completeness": 7,
                                                       "conciseness": 6, "consistency": 9, "balance": None,
                                                       "uncertainty": "n/a", "hallucinations": ["fake --flag"],
                                                       "overall": 7, "matches_best": lab == "A", "reason": "ok"}
                                                for lab in labs}), 0.0
        return complete

    sc.rejudge(tmp_path, 4, keys, workers=1, complete_for=judge_for)
    assert all("hallucinations" in p and "conciseness" in p for p in prompts)
    rows = sc.read_jsonl(tmp_path / "rejudge.jsonl")
    assert rows[0]["score"] == 0.7 and rows[0]["criteria"]["balance"] is None
    assert rows[0]["criteria"]["uncertainty"] is None and rows[0]["hallucinations"] == ["fake --flag"]
    agree = sc.judge_agreement(rows, 0.1, 0.6, keys=keys)
    j = agree["judges"][keys[0]]
    assert j["criteria"]["conciseness"]["haiku"] == pytest.approx(0.6) and "balance" not in j["criteria"]
    assert j["hallucination_rate"]["opus"] == 1.0
    (pair,) = agree["pairs"]
    assert pair["oracle_agreement"] == 1.0
    assert sc.judge_key("m") == "m" and sc.judge_key("m", "criteria", 2) == f"m|criteria-{sc.RUBRIC_VERSION}|r2"

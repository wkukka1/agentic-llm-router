"""The router-specific half of :mod:`evaluation.gateway.router_compare`: how each installed
router is started against the recorder stub, its activation tests, and the full run.

Installs live in ``~/.aar-router-bench/`` (``scripts/router_bench/install.ps1``), config templates
in ``scripts/router_bench/``. Each router runs from a fresh per-run folder with its own HOME or
data dir, so none touches the user's real configs, and its upstream is the stub with a dummy key.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from evaluation.gateway import router_compare as rc
from evaluation.gateway.pool_eval import TIERS, evaluate, load_prompts, policies

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "scripts" / "router_bench"
HOME = rc.BENCH_HOME
HAIKU, SONNET, OPUS = TIERS
EXE = ".exe" if os.name == "nt" else ""
BIN = "Scripts" if os.name == "nt" else "bin"

#: name -> (label in the report, track). Order is the report's order.
ROUTERS: dict[str, tuple[str, str]] = {
    "aar": ("aar-gateway MLP router (live default)", "native"),
    "aar-nirt": ("aar-gateway NIRT router", "native"),
    "aar-manifold": ("aar-gateway manifold NIRT (judged, BERT)", "native"),
    "ccr": ("claude-code-router 1.0.73", "native"),
    "ccp": ("claude-code-proxy (name mapper)", "native"),
    "litellm-complexity": ("LiteLLM complexity router", "native"),
    "litellm-semantic": ("LiteLLM auto-router, custom tier utterances", "native"),
    "litellm-passthrough": ("LiteLLM alias passthrough (control)", "native"),
    "routellm-haiku": ("RouteLLM bert, Opus/Haiku", "native"),
    "routellm-sonnet": ("RouteLLM bert, Opus/Sonnet", "native"),
}

PORTS = {"aar": 8790, "ccr": 8791, "ccp": 8793, "litellm": 8794, "aar-nirt": 8795, "aar-manifold": 8796}
#: the bundled router files the gateway runs serve (``aar-gateway router list``)
GATEWAY_ROUTERS = REPO / "aar-gateway" / "src" / "aar_gateway" / "routers"
AAR_ROUTER_FILES = {"aar": None,  # None: whatever the live config serves
                    "aar-nirt": GATEWAY_ROUTERS / "llmrouterbench-nirt.npz",
                    "aar-manifold": GATEWAY_ROUTERS / "manifold-nirt-claude-judged.npz"}


def _fill(text: str, **values: str) -> str:
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


# -- launch specs: (cmd, env, cwd, base url, health url) ------------------------------------------------


def aar_spec(run_dir: Path, upstream: str, server: str = "aar") -> dict:
    """A second, isolated gateway: its own log folder (so its own pid, history and preferences),
    the live config with prompt capture off, history off, no share consent, and the stub upstream.
    ``server`` picks the router file it serves (:data:`AAR_ROUTER_FILES`): only ``router.artifact``
    differs between the runs. The live gateway's ``~/.aar-gateway`` is only read."""
    home = run_dir / f"{server}-home"
    home.mkdir(parents=True, exist_ok=True)
    live = Path.home() / ".aar-gateway" / "config.yaml"
    spec = yaml.safe_load(live.read_text(encoding="utf-8")) if live.exists() else {}
    spec.setdefault("interceptor", {})["capture_prompts"] = False
    if AAR_ROUTER_FILES.get(server):
        spec["interceptor"].setdefault("router", {})["artifact"] = str(AAR_ROUTER_FILES[server])
    (home / "config.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    (home / "preferences.json").write_text(json.dumps({"keep_history": False}), encoding="utf-8")
    models = HOME / "aar-models"  # a copy of the encoder cache, so the live folder is never written
    if not models.exists():
        shutil.copytree(Path.home() / ".aar-gateway" / "models", models)
    # ONNX text encoders a router file names (e.g. the BERT one): the gateway reads them from its own home
    encoders = Path.home() / ".aar-gateway" / "encoders"
    if encoders.exists() and not (home / "encoders").exists():
        shutil.copytree(encoders, home / "encoders")
    port = PORTS[server]
    return {"cmd": [sys.executable, "-m", "aar_gateway.cli", "serve"], "cwd": run_dir,
            "env": {"AAR_GATEWAY_LOG": str(home / "requests.jsonl"), "AAR_GATEWAY_CONFIG": str(home / "config.yaml"),
                    "AAR_GATEWAY_HOST": "127.0.0.1", "AAR_GATEWAY_PORT": str(port),
                    "AAR_GATEWAY_UPSTREAM": upstream, "FASTEMBED_CACHE_PATH": str(models)},
            "base": f"http://127.0.0.1:{port}", "health": f"http://127.0.0.1:{port}/health"}


def ccr_spec(run_dir: Path, upstream: str) -> dict:
    home = run_dir / "ccr-home"
    (home / ".claude-code-router").mkdir(parents=True, exist_ok=True)
    (home / ".claude-code-router" / "config.json").write_text(
        _fill((CONFIGS / "ccr.config.json").read_text(encoding="utf-8"), UPSTREAM=upstream), encoding="utf-8")
    cli = HOME / "ccr-v1" / "node_modules" / "@musistudio" / "claude-code-router" / "dist" / "cli.js"
    port = PORTS["ccr"]
    return {"cmd": ["node", str(cli), "start"], "cwd": home,
            "env": {"HOME": str(home), "USERPROFILE": str(home)},
            "base": f"http://127.0.0.1:{port}", "health": f"http://127.0.0.1:{port}/health"}


def ccp_spec(run_dir: Path, upstream: str) -> dict:
    """claude-code-proxy maps haiku -> SMALL and sonnet -> BIG and passes anything else through.
    Both of its upstreams (OpenAI-shaped for mapped names, LiteLLM's Anthropic provider for the
    rest) point at the stub, so no request can reach a real provider."""
    port = PORTS["ccp"]
    py = HOME / "ccp" / ".venv" / BIN / f"python{EXE}"
    return {"cmd": [str(py), "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
                    "--log-level", "warning"], "cwd": HOME / "ccp",
            "env": {"PREFERRED_PROVIDER": "openai", "OPENAI_BASE_URL": upstream + "/v1",
                    "OPENAI_API_KEY": "aar-bench-dummy", "ANTHROPIC_API_KEY": "aar-bench-dummy",
                    "ANTHROPIC_API_BASE": upstream, "ANTHROPIC_BASE_URL": upstream, "GEMINI_API_KEY": "",
                    "BIG_MODEL": SONNET, "SMALL_MODEL": HAIKU, "LITELLM_LOG": "ERROR",
                    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            "base": f"http://127.0.0.1:{port}", "health": f"http://127.0.0.1:{port}/"}


def litellm_spec(run_dir: Path, upstream: str) -> dict:
    cfg = yaml.safe_load(_fill((CONFIGS / "litellm.config.yaml").read_text(encoding="utf-8"), UPSTREAM=upstream))
    routes = json.loads((CONFIGS / "litellm.auto_router.json").read_text(encoding="utf-8"))
    for m in cfg["model_list"]:
        params = m["litellm_params"]
        if params.pop("auto_router_config_path", None) is not None:
            params["auto_router_config"] = json.dumps({"routes": routes["routes"]})
    out = run_dir / "litellm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    port = PORTS["litellm"]
    exe = HOME / "litellm" / ".venv" / BIN / f"litellm{EXE}"
    key = "sk-bench-" + os.urandom(16).hex()  # per run, never saved: the proxy refuses to start without one
    return {"cmd": [str(exe), "--config", str(out / "config.yaml"), "--host", "127.0.0.1", "--port", str(port)],
            "cwd": out, "env": {"LITELLM_TELEMETRY": "False", "LITELLM_LOG": "ERROR", "ANTHROPIC_API_KEY": "aar-bench-dummy",
                                "LITELLM_MASTER_KEY": key, "PYTHONIOENCODING": "utf-8"},
            "headers": {"authorization": f"Bearer {key}", "x-api-key": key},
            "base": f"http://127.0.0.1:{port}", "health": f"http://127.0.0.1:{port}/health/liveliness"}


SPECS = {"aar": aar_spec, "aar-nirt": lambda run_dir, upstream: aar_spec(run_dir, upstream, "aar-nirt"),
         "aar-manifold": lambda run_dir, upstream: aar_spec(run_dir, upstream, "aar-manifold"),
         "ccr": ccr_spec, "ccp": ccp_spec, "litellm": litellm_spec}
#: router name -> (server, model it is asked for or None for the requested tier, request format)
VIA = {"aar": ("aar", None, "anthropic"), "aar-nirt": ("aar-nirt", None, "anthropic"),
       "aar-manifold": ("aar-manifold", None, "anthropic"), "ccr": ("ccr", None, "anthropic"), "ccp": ("ccp", None, "anthropic"),
       "litellm-complexity": ("litellm", "complexity-router", "anthropic"),
       "litellm-semantic": ("litellm", "semantic-router", "anthropic"),
       "litellm-passthrough": ("litellm", None, "anthropic")}


# -- template capture -----------------------------------------------------------------------------------


def capture_template(out: Path) -> int:
    """One ``claude -p "hi"`` against the stub: Claude Code's real request shape. ``--settings``
    overrides a user-level ``ANTHROPIC_BASE_URL`` (the live gateway), so the request can only reach
    the stub. Only the body and the version/beta/user-agent headers are kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cwd = HOME / "capture-cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    claude = shutil.which("claude")
    if not claude:
        print("`claude` not on PATH", file=sys.stderr)
        return 1
    with rc.recorder(capture=out) as url:
        settings = json.dumps({"env": {"ANTHROPIC_BASE_URL": url}})
        subprocess.run([claude, "-p", rc.TEMPLATE_PROMPT, "--model", OPUS, "--settings", settings],
                       cwd=cwd, env={**os.environ, "ANTHROPIC_BASE_URL": url}, capture_output=True, timeout=180,
                       check=False)
    if not out.exists():
        print("no tool-carrying request reached the stub", file=sys.stderr)
        return 1
    text = out.read_text(encoding="utf-8")
    if "sk-ant" in text or "Bearer " in text:
        out.unlink()
        print("template held a credential; deleted", file=sys.stderr)
        return 1
    t = rc.load_template(out)
    print(f"template: {out} (model {t['body'].get('model')}, {len(t['body']['tools'])} tools, "
          f"thinking={'thinking' in t['body']})")
    return 0


# -- activation tests: routing must actually change -------------------------------------------------------

ACTIVATION_PROMPTS = {
    "lookup": "what is the default port",
    "design": ("Design the architecture for a new distributed job scheduler across these packages: think through "
               "step by step the consistency model, failure recovery, the database schema, and the API, then "
               "compare three alternatives and justify the trade-offs."),
}


def _send(base: str, template: Mapping, prompt: str, *, model: str, tag: str, thinking: bool = True,
          pad: int = 0, send_model: str | None = None) -> dict:
    import httpx

    bid = rc.bench_id("activation", tag, model, hashlib.sha256(prompt.encode()).hexdigest()[:16])
    body = rc.native_request(template, prompt + (" filler" * pad), model=send_model or model, bid=bid, thinking=thinking)
    headers = {**rc.request_headers(template, bid), **_server_headers}
    with httpx.Client() as client:
        return rc.send_one(client, base + "/v1/messages", body, headers, kind="anthropic", bid=bid, timeout=180)


#: auth the running server needs (LiteLLM's per-run master key); set while it runs
_server_headers: dict[str, str] = {}


def activation(name: str, base: str, template: Mapping) -> dict:
    """``{"checks": {...: bool}, "observed": {...}}`` for one HTTP router."""
    p = ACTIVATION_PROMPTS
    obs: dict[str, str | None] = {}
    if name == "ccr":
        cases = {"background (haiku asked)": ({"model": HAIKU}, HAIKU),
                 "think (thinking on)": ({"model": OPUS}, OPUS),
                 "longContext (>60k tokens)": ({"model": OPUS, "thinking": False, "pad": 70_000}, OPUS),
                 "default (no thinking)": ({"model": OPUS, "thinking": False}, SONNET)}
        checks = {}
        for label, (kw, want) in cases.items():
            got = _send(base, template, p["lookup"], tag=label, **kw)
            obs[label] = got["selected_model"]
            checks[label] = got["selected_model"] == want and got["error"] is None
        return {"checks": checks, "observed": obs}
    if name == "ccp":
        got = {m: _send(base, template, p["lookup"], model=m, tag="map")["selected_model"] for m in (HAIKU, OPUS)}
        return {"checks": {"haiku -> SMALL": got[HAIKU] == HAIKU, "opus passes through": got[OPUS] == OPUS},
                "observed": got}
    if name == "litellm-passthrough":
        got = {m: _send(base, template, p["lookup"], model=m, tag="pass")["selected_model"] for m in TIERS}
        return {"checks": {f"{m} honoured": got[m] == m for m in TIERS}, "observed": got}
    if name == "litellm-semantic":
        utter = json.loads((CONFIGS / "litellm.auto_router.json").read_text(encoding="utf-8"))["routes"]
        checks = {}
        for route in utter:
            got = _send(base, template, route["utterances"][0], model=OPUS, tag="sem", send_model="semantic-router")
            obs[route["name"]] = got["selected_model"]
            checks[f"utterance -> {route['name']}"] = got["selected_model"] == route["name"]
        return {"checks": checks, "observed": obs}
    send_model = VIA[name][1]
    for label, prompt in p.items():
        got = _send(base, template, prompt, model=OPUS, tag=label, send_model=send_model)
        obs[label] = got["selected_model"]
    # a router whose decision never changes is reported, not forced
    return {"checks": {"lookup and design get different tiers": obs["lookup"] != obs["design"]}, "observed": obs}


# -- RouteLLM (in-process in its own venv) ------------------------------------------------------------------


def routellm_scores(prompts: Sequence[Mapping], run_dir: Path) -> dict:
    """Win rates and Arena calibration from RouteLLM's venv. The prompt file is deleted after."""
    src, out = run_dir / "routellm-prompts.jsonl", run_dir / "routellm-scores.json"
    src.write_text("".join(json.dumps({"prompt_hash": p["prompt_hash"], "prompt": p["prompt"]}) + "\n"
                           for p in prompts), encoding="utf-8")
    py = HOME / "routellm" / ".venv" / BIN / f"python{EXE}"
    try:
        r = subprocess.run([str(py), "-I", str(CONFIGS / "routellm_scores.py"), str(src), str(out)],
                           capture_output=True, text=True, timeout=1800, check=False,
                           # RouteLLM builds an OpenAI client at import (used by mf/sw_ranking, not bert):
                           # a dummy key and a dead local base, so nothing can leave the machine
                           env={**os.environ, "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false",
                                "OPENAI_API_KEY": "aar-bench-dummy", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"})
    finally:
        src.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"routellm_scores failed: {r.stderr[-800:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def routellm_rows(prompts: Sequence[Mapping], scores: Mapping, *, weak: str, strong_share: float,
                  name: str) -> tuple[list[dict], dict]:
    """Picks at the Arena-calibrated threshold for ``strong_share`` (no graded prompt tunes it)."""
    key = f"{round(strong_share, 2):.2f}"
    threshold = scores["arena_thresholds"][key]
    rows = []
    for p in prompts:
        rate = scores["win_rate"][p["prompt_hash"]]
        tier = OPUS if rate >= threshold else weak
        rows.append({"router": name, "track": "native", "condition": "primary", "prompt_hash": p["prompt_hash"],
                     "requested_model": OPUS, "selected_model": tier, "tier": tier, "error": None,
                     "mapping_reason": "Controller win rate >= threshold", "upstream_calls": 0, "win_rate": rate})
    got = float(np.mean([r["tier"] == OPUS for r in rows]))
    return rows, {"target_strong_share": strong_share, "threshold": threshold, "achieved_share": got,
                  "share_gap": got - strong_share}


def routellm_sweep(prompts: Sequence[Mapping], scores: Mapping, weak: str) -> list[dict]:
    """Every Arena-calibrated operating point: the cost/quality curve (exploratory)."""
    out = []
    for key in sorted(scores["arena_thresholds"], key=float)[::5]:
        rows, op = routellm_rows(prompts, scores, weak=weak, strong_share=float(key), name="sweep")
        r = rc.score_rows(prompts, {"sweep": rows})[0]
        out.append({**op, "quality": r["quality"], "est_cost_usd": r["est_cost_usd"], "vs_mix": r["vs_mix"]})
    return out


def routellm_activation(scores: Mapping) -> dict:
    rates = np.array(list(scores["win_rate"].values()))
    th = scores["arena_thresholds"]
    all_weak, all_strong = (rates >= th["0.00"]).mean(), (rates >= th["1.00"]).mean()
    mid = (rates >= th["0.50"]).mean()
    return {"checks": {"0% strong -> (almost) all weak": bool(all_weak <= 0.05),
                       "100% strong -> (almost) all strong": bool(all_strong >= 0.95),
                       "50% strong -> a mix": bool(0 < mid < 1), "Controller.route agrees": bool(scores["route_agrees"])},
            "observed": {"strong@0%": float(all_weak), "strong@100%": float(all_strong), "strong@50%": float(mid)}}


# -- the run ----------------------------------------------------------------------------------------------------


def claude_smoke(spec: Mapping) -> dict:
    """One real ``claude -p "say ok"`` through the running router. Passes when Claude Code exits 0
    with the stub's canned "ok" and at least one upstream call reached the stub. The auth token is
    a dummy (LiteLLM's run key for LiteLLM): the user's login never goes to a router."""
    claude = shutil.which("claude")
    if not claude:
        return {"pass": False, "why": "claude not on PATH"}
    key = (spec.get("headers") or {}).get("x-api-key", "aar-bench-dummy")
    env = {**os.environ, "ANTHROPIC_BASE_URL": spec["base"], "ANTHROPIC_AUTH_TOKEN": key}
    settings = json.dumps({"env": {"ANTHROPIC_BASE_URL": spec["base"], "ANTHROPIC_AUTH_TOKEN": key}})
    start = rc.call_count()
    try:
        r = subprocess.run([claude, "-p", "say ok", "--model", OPUS, "--settings", settings], cwd=HOME / "capture-cwd",
                           env=env, capture_output=True, timeout=180, check=False)
        code, out = r.returncode, r.stdout.decode("utf-8", "ignore").strip()
    except subprocess.TimeoutExpired:
        code, out = None, ""
    calls = [c for c in rc.calls_since(start) if c["kind"] != "count_tokens"]
    return {"pass": code == 0 and out.lower().startswith("ok") and bool(calls), "exit": code,
            "answered_ok": out.lower().startswith("ok"), "upstream_calls": len(calls),
            "upstream_models": sorted({c["model"] for c in calls})}


MARGIN_HEADER = "x-aar-quality-margin"  # aar_gateway.interceptor.MARGIN_HEADER


def margin_sweep(prompts: Sequence[Mapping], pool_rows: Sequence[Mapping], margins: Sequence[float],
                 routers: Sequence[str], lines: list[str], pool: str, *, splits: int = 200) -> dict:
    """The gateway's cost aversion (``router.margin``) swept through the running gateways, asked for
    Opus. Per margin: savings vs no router with a paired CI. Then an honest pick: choose the margin
    with the best ``vs mix`` on a random half of the prompts and score it on the other half."""
    out: dict = {}
    lines.append(f"\n-- {pool}: cost aversion sweep (router.margin via x-aar-quality-margin), asked Opus")
    lines.append(f"{'router':40s} {'margin':>6s} {'quality':>7s} {'vs mix':>7s} {'saved $':>8s} {'saved':>6s} "
                 f"{'saved $ 95% CI':>17s} {'dQ':>7s} {'dQ 95% CI':>17s} {'H/S/O':>9s} {'err':>4s}")
    opus = np.full(len(prompts), TIERS.index(OPUS))
    S = np.stack([p["score"] for p in prompts])
    C = np.stack([p["cost"] for p in prompts])
    for n in routers:
        picks = {}
        for m in margins:
            mine = [r for r in pool_rows if r["router"] == n and r["track"] == "margin" and r["condition"] == f"margin {m:g}"]
            if not mine:
                continue
            row = rc.score_rows(prompts, {ROUTERS[n][0]: mine})[0]
            picks[m], _ = rc.picks_of(mine, prompts)
            b = rc.paired_bootstrap(prompts, picks[m], opus)
            q, c = b["quality"], b["est_cost_usd"]
            out.setdefault(n, {})[f"{m:g}"] = {**row, "bootstrap": b}
            mix = row["mix"]
            lines.append(f"{ROUTERS[n][0][:40]:40s} {m:6.3f} {row['quality']:7.3f} {row['vs_mix']:+7.3f} "
                         f"{row['saved_usd']:+8.2f} {row['saved_pct']:6.0%} [{-c['hi']:+7.2f},{-c['lo']:+7.2f}] "
                         f"{q['delta']:+7.3f} [{q['lo']:+.3f},{q['hi']:+.3f}] "
                         f"{mix.get('haiku', 0)}/{mix.get('sonnet', 0)}/{mix.get('opus', 0):<3d} {row['n_errors']:4d}")
        if len(picks) < 2 or len(prompts) < 10:
            continue
        rng = np.random.default_rng(0)
        held, chosen = [], []
        idx_all = np.arange(len(prompts))
        for _ in range(splits):
            a = rng.permutation(len(prompts))[: len(prompts) // 2]
            b_ = np.setdiff1d(idx_all, a)
            def vs_mix(ix, pk):
                fixed = sorted((C[ix, j].sum(), S[ix, j].mean()) for j in range(S.shape[1]))
                xs, ys = zip(*fixed, strict=True)
                return S[ix, pk[ix]].mean() - float(np.interp(C[ix, pk[ix]].sum(), xs, ys))
            best = max(picks, key=lambda m: vs_mix(a, picks[m]))
            pk = picks[best]
            chosen.append(best)
            held.append((vs_mix(b_, pk), 1 - C[b_, pk[b_]].sum() / C[b_, -1].sum(), S[b_, pk[b_]].mean() - S[b_, -1].mean()))
        h = np.array(held)
        vals, counts = np.unique(chosen, return_counts=True)
        out.setdefault(n, {})["split_half"] = {"held_out_vs_mix": float(h[:, 0].mean()), "held_out_saved": float(h[:, 1].mean()),
                                               "held_out_dq": float(h[:, 2].mean()),
                                               "chosen": {f"{v:g}": int(c) for v, c in zip(vals, counts, strict=True)}}
        lines.append(f"{ROUTERS[n][0][:40]:40s} split-half pick (best vs mix on one half, scored on the other, "
                     f"{splits}x): held-out vs mix {h[:, 0].mean():+.3f}, saved {h[:, 1].mean():.0%}, "
                     f"dQ vs Opus {h[:, 2].mean():+.3f}; margins chosen {dict(zip([f'{v:g}' for v in vals], counts.tolist(), strict=True))}")
    return out


def aar_diagnostics(base: str, run_dir: Path, server: str = "aar") -> dict:
    """Was the isolated gateway's router loaded, and why did it decide what it did: ``/health``
    and the decision reasons from its own request log (which holds no prompt text)."""
    from collections import Counter

    import httpx

    try:
        health = httpx.get(base + "/health", timeout=5).json()
    except (httpx.HTTPError, ValueError) as exc:
        health = {"error": type(exc).__name__}
    keep = {k: v for k, v in health.items() if not isinstance(v, (dict, list)) or k in ("router", "calibration")}
    log = run_dir / f"{server}-home" / "requests.jsonl"
    reasons = Counter()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError):
                reasons[str(json.loads(line).get("reason") or "")[:70]] += 1
    return {"health": keep, "decision_reasons": dict(reasons.most_common(8))}


_reference: dict = {}


def reference_inputs() -> dict:
    """The benchmark router (the live gateway's router file, each tier's stand-in) and the encoder
    for the pool kNN, loaded once: ``pool_eval``'s reference policies."""
    if not _reference:
        from aar_gateway.config import DEFAULT_MODELS
        from aar_gateway.routing import load_router_model

        from router.embeddings.encoder import EncoderConfig, TextEncoder, light_backend

        router_file = Path.home() / ".aar-gateway" / "router.npz"
        _reference["proxies"] = {t: DEFAULT_MODELS[t]["proxy"] for t in TIERS}
        if router_file.exists():
            _reference["router"] = load_router_model({"artifact": str(router_file)})
        if AAR_ROUTER_FILES["aar-nirt"].exists():
            _reference["nirt"] = load_router_model({"artifact": str(AAR_ROUTER_FILES["aar-nirt"])})
        _reference["encode"] = TextEncoder(EncoderConfig(name="pool-eval", backend=light_backend())).encode
    return _reference


def _versions() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
        except OSError:
            return None

    def pyver(venv, pkgs):
        py = HOME / venv / ".venv" / BIN / f"python{EXE}"
        code = "import importlib.metadata as m,sys;print({p:m.version(p) for p in sys.argv[1:]})"
        return run([str(py), "-c", code, *pkgs])

    return {"python": sys.version.split()[0], "node": run(["node", "--version"]), "platform": platform.platform(),
            "repo_commit": run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "repo_dirty": bool(run(["git", "-C", str(REPO), "status", "--porcelain", "aar-gateway", "src/evaluation"])),
            "ccr": "1.0.73", "ccp_commit": run(["git", "-C", str(HOME / "ccp"), "rev-parse", "HEAD"]),
            "litellm": pyver("litellm", ["litellm", "semantic-router"]),
            "routellm": pyver("routellm", ["routellm", "torch", "transformers"])}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def privacy_sweep(run_dir: Path, prompts: Sequence[Mapping], extra: Sequence[Path] = ()) -> list[str]:
    """Files under the run folder that hold prompt text or a credential. Must be empty."""
    needles = [p["prompt"].strip()[:40] for p in prompts if len(p["prompt"].strip()) >= 40][:25]
    hits = []
    for f in [*run_dir.rglob("*"), *extra]:
        if not f.is_file() or f.suffix in (".npz", ".onnx", ".bin", ".sqlite"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(n in text for n in needles):
            hits.append(f"{f}: prompt text")
        if "sk-ant-" in text:
            hits.append(f"{f}: credential")
    return hits


LIVE_FILES = ("requests.jsonl", "turns.jsonl", "history/history.jsonl")


def _live_sizes() -> dict[str, int]:
    home = Path.home() / ".aar-gateway"
    return {f: (home / f).stat().st_size if (home / f).exists() else 0 for f in LIVE_FILES}


def _bench_traffic_in_live(before: Mapping[str, int], rows: Sequence[Mapping]) -> list[str]:
    """Live gateway files whose lines added during the run name a bench session. The live gateway
    keeps serving the user's own sessions meanwhile, so a changed file alone says nothing."""
    home = Path.home() / ".aar-gateway"
    marks = {rc._session_uuid(r["bench_id"])[:12] for r in rows if r.get("bench_id")} | {"aar-bench"}
    hits = []
    for name, size in before.items():
        f = home / name
        if not f.exists() or f.stat().st_size <= size:
            continue
        with open(f, "rb") as fh:
            fh.seek(size)
            added = fh.read().decode("utf-8", "ignore")
        if any(m in added for m in marks):
            hits.append(str(f))
    return hits


def run_all(args) -> int:
    template = rc.load_template(Path(args.template))
    pools = {Path(g).expanduser().parent.name: load_prompts(Path(g).expanduser()) for g in args.graded}
    pools = {k: v for k, v in pools.items() if v}
    seen, everything = set(), []
    for v in pools.values():
        for p in v:
            if p["prompt_hash"] not in seen:
                seen.add(p["prompt_hash"])
                everything.append(p)
    if len(pools) > 1:
        pools["all"] = everything
    wanted = [r for r in args.routers.split(",") if r]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / stamp
    run_dir = HOME / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    conditions = [("primary", OPUS, True)]
    margins = [float(m) for m in (getattr(args, "margins", None) or "").split(",") if m.strip()]
    if args.sensitivity:
        conditions += [("asked sonnet", SONNET, True), ("asked haiku", HAIKU, True), ("no thinking", OPUS, False)]

    rows: list[dict] = []
    compat: dict[str, dict] = {}
    activations: dict[str, dict] = {}
    routellm_ops: dict[str, dict] = {}
    sweeps: dict[str, list] = {}
    diagnostics: dict[str, dict] = {}
    smoke: dict[str, dict] = {}
    live_before = _live_sizes()
    try:
        with rc.recorder() as upstream:
            servers = sorted({VIA[r][0] for r in wanted if r in VIA})
            for server in servers:
                spec = SPECS[server](run_dir, upstream)
                print(f"starting {server} ...", flush=True)
                try:
                    with rc.running(server, spec["cmd"], env=spec["env"], cwd=spec["cwd"], health=spec["health"],
                                    log_dir=run_dir / "logs"):
                        _server_headers.clear()
                        _server_headers.update(spec.get("headers") or {})
                        for name in [r for r in wanted if VIA.get(r, ("",))[0] == server]:
                            _, send_model, fmt = VIA[name]
                            activations[name] = activation(name, spec["base"], template)
                            print(f"  {name}: activation {activations[name]['checks']}", flush=True)
                            # native: the plain Messages API turn, every condition; compat: the captured
                            # Claude Code request, primary condition only
                            runs = [("native", *c) for c in conditions] + [("compat", *conditions[0])]
                            for track, cond, model, thinking in runs:
                                rows += rc.replay(name, spec["base"], "/v1/messages", everything, template,
                                                  track=track, condition=cond, model=model, thinking=thinking,
                                                  fmt=fmt, send_model=send_model, extra_headers=_server_headers)
                            if server.startswith("aar"):
                                # the cost-aversion sweep: the gateway's own per-request margin override
                                for m in margins:
                                    rows += rc.replay(name, spec["base"], "/v1/messages", everything, template,
                                                      track="margin", condition=f"margin {m:g}", model=OPUS,
                                                      extra_headers={**_server_headers, MARGIN_HEADER: f"{m:g}"})
                            cc = [r for r in rows if r["router"] == name and r["track"] == "compat"]
                            compat[name] = {"accepted_claude_code_request": all(r.get("compat_error") is None for r in cc),
                                            "compat_errors": sorted({r["compat_error"] for r in cc if r.get("compat_error")})}
                            print(f"  {name}: Claude Code request {'ok' if compat[name]['accepted_claude_code_request'] else compat[name]['compat_errors']}",
                                  flush=True)
                        if getattr(args, "smoke", False):
                            smoke[server] = claude_smoke(spec)
                            print(f"  {server}: claude -p smoke {smoke[server]}", flush=True)
                        if server.startswith("aar"):
                            diagnostics[server] = aar_diagnostics(spec["base"], run_dir, server)
                        _server_headers.clear()
                except RuntimeError as exc:
                    print(f"  {server} failed: {exc}", flush=True)
                    for name in [r for r in wanted if VIA.get(r, ("",))[0] == server]:
                        compat[name] = {"accepted_claude_code_request": False, "compat_errors": [str(exc)[:300]]}
                        rows += [{"router": name, "track": t, "condition": c[0], "prompt_hash": p["prompt_hash"],
                                  "requested_model": c[1], "tier": None, "error": "router_not_started"}
                                 for t, cs in (("native", conditions), ("compat", conditions[:1]))
                                 for c in cs for p in everything]
        if any(r.startswith("routellm") for r in wanted):
            scores = routellm_scores(everything, run_dir)
            activations["routellm"] = routellm_activation(scores)
            aar_prim = [r for r in rows if r["router"] == "aar" and r["track"] == "native"
                        and r["condition"] == "primary" and r.get("tier")]
            share = float(np.mean([r["tier"] == OPUS for r in aar_prim])) if aar_prim else 0.5
            for name, weak in (("routellm-haiku", HAIKU), ("routellm-sonnet", SONNET)):
                if name in wanted:
                    got, op = routellm_rows(everything, scores, weak=weak, strong_share=share, name=name)
                    rows += got
                    routellm_ops[name] = op
                    sweeps[name] = routellm_sweep(everything, scores, weak)
            compat["routellm"] = {"accepted_claude_code_request": False,
                                  "compat_errors": ["its server speaks OpenAI chat completions only; not run in the Claude Code track"]}
    finally:
        live_touched = _bench_traffic_in_live(live_before, rows)

    # -- report
    report: dict = {"pools": {}, "activation": activations, "compatibility": compat, "routellm": routellm_ops,
                    "routellm_sweep": sweeps, "diagnostics": diagnostics, "claude_smoke": smoke}
    lines = [f"Zero-cost, stub-backed routing-policy bake-off ({stamp}). Actual spend: $0. "
             "est $ = plan usage measured when the prompts were graded, at API prices."]
    for pool, prompts in pools.items():
        hashes = {p["prompt_hash"] for p in prompts}
        pool_rows = [r for r in rows if r["prompt_hash"] in hashes]
        block = {}
        groups = [("native", c[0], "installed routers, plain Messages API request" if c[0] == "primary"
                   else f"sensitivity: {c[0]}") for c in conditions]
        groups.append(("compat", "primary", "Claude Code compatibility track: the captured request as sent today"))
        for track, cond, title in groups:
            res = {ROUTERS[n][0]: [r for r in pool_rows if r["router"] == n and r["track"] == track
                                   and r["condition"] == cond] for n in wanted if n in ROUTERS}
            res = {k: v for k, v in res.items() if v}
            if not res:
                continue
            asked = TIERS.index(next(m for c, m, _ in conditions if c == cond))
            scored = rc.score_rows(prompts, res, baseline=asked)
            block[f"{track}/{cond}"] = scored
            lines.append("\n" + rc.table(scored, f"== {pool}: {len(prompts)} prompts -- {title}"))
        refs = reference_inputs()
        picks = policies(prompts, margin=0.05, k=10, router=refs.get("router"), proxies=refs.get("proxies"),
                         encode=refs["encode"])
        picks = {("benchmark router (MLP, offline)" if k == "benchmark router" else k): v for k, v in picks.items()}
        if refs.get("nirt"):  # the NIRT file through the same offline replay: cross-checks the NIRT gateway
            nirt = policies(prompts, margin=0.05, k=10, router=refs["nirt"], proxies=refs["proxies"])
            picks["benchmark router (NIRT, offline)"] = nirt["benchmark router"]
        ref = evaluate(prompts, picks=picks)
        for r in ref:
            r.update(est_cost_usd=r.pop("usage_usd"), n_prompts=len(prompts), n_errors=0, unknown_models=0,
                     anomalies=0, actual_opus_share=r["mix"].get("opus", 0) / len(prompts))
            rc.add_savings(r, prompts, TIERS.index(OPUS))
        block["reference"] = ref
        lines.append("\n" + rc.table(ref, f"-- {pool}: reference policies (not installed routers; oracle is a ceiling)"))
        aar_rows = [r for r in pool_rows if r["router"] == "aar" and r["track"] == "native" and r["condition"] == "primary"]
        if aar_rows:
            aar_pick, aar_fail = rc.picks_of(aar_rows, prompts)
            boots = {}
            for n in wanted:
                if n == "aar" or n not in ROUTERS:
                    continue
                mine = [r for r in pool_rows if r["router"] == n and r["track"] == "native" and r["condition"] == "primary"]
                pick, fail = rc.picks_of(mine, prompts)
                ok = ~(fail | aar_fail)
                if ok.sum() < 5:
                    continue
                sub = [p for p, k in zip(prompts, ok, strict=True) if k]
                boots[ROUTERS[n][0]] = {**rc.paired_bootstrap(sub, pick[ok], aar_pick[ok]), "n_paired": int(ok.sum())}
            block["vs_aar_bootstrap"] = boots
            lines.append(f"\n-- {pool}: router minus aar-gateway, paired bootstrap 95% CI (prompts both answered)")
            for k, b in boots.items():
                q, c = b["quality"], b["est_cost_usd"]
                lines.append(f"{k:44s} dQ {q['delta']:+.3f} [{q['lo']:+.3f},{q['hi']:+.3f}] {q['verdict']:24s} "
                             f"d$ {c['delta']:+7.2f} [{c['lo']:+.2f},{c['hi']:+.2f}]  n={b['n_paired']}")
        # savings against no router (always the model asked for), per condition, with paired CIs
        savings = {}
        lines.append(f"\n-- {pool}: savings vs no router (every prompt on the model asked for), paired bootstrap 95% CI")
        for cond, model, _ in conditions:
            asked = TIERS.index(model)
            for n in wanted:
                mine = [r for r in pool_rows if r["router"] == n and r["track"] == "native" and r["condition"] == cond]
                if n not in ROUTERS or not mine:
                    continue
                pick, _ = rc.picks_of(mine, prompts)  # errors stay in, charged as Opus
                b = rc.paired_bootstrap(prompts, pick, np.full(len(prompts), asked))
                q, c = b["quality"], b["est_cost_usd"]
                savings.setdefault(cond, {})[ROUTERS[n][0]] = b
                lines.append(f"[{cond:12s}] {ROUTERS[n][0][:40]:40s} saved ${-c['delta']:+7.2f} "
                             f"[{-c['hi']:+.2f},{-c['lo']:+.2f}]  dQ {q['delta']:+.3f} [{q['lo']:+.3f},{q['hi']:+.3f}] "
                             f"{q['verdict']}")
        block["savings_vs_no_router"] = savings
        if margins:
            block["margin_sweep"] = margin_sweep(prompts, pool_rows, margins, [n for n in wanted if n.startswith("aar")],
                                                 lines, pool)
        report["pools"][pool] = block
    lines.append("\n-- activation tests")
    for k, a in activations.items():
        lines.append(f"{k:22s} {'PASS' if all(a['checks'].values()) else 'FAIL'}  {a['checks']}  observed={a['observed']}")
    for k, s in smoke.items():
        lines.append(f"claude -p through {k:9s} {'PASS' if s['pass'] else 'FAIL'}  {s}")
    for k, d in diagnostics.items():
        lines.append(f"{k} diagnostics: {json.dumps(d)[:900]}")
    lines.append("\n-- Claude Code compatibility")
    for k, c in compat.items():
        lines.append(f"{k:22s} {'ok' if c['accepted_claude_code_request'] else 'NOT OK'}  {c['compat_errors']}")
    for k, op in routellm_ops.items():
        lines.append(f"{k}: Arena-calibrated threshold {op['threshold']:.4f} for Opus share {op['target_strong_share']:.0%} "
                     f"(aar's); achieved {op['achieved_share']:.0%} (gap {op['share_gap']:+.0%})")

    (out_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    leaks = privacy_sweep(run_dir, everything, extra=[out_dir / "rows.jsonl"])
    report["privacy"] = {"hits": leaks, "bench_traffic_in_live_gateway": live_touched}
    lines.append(f"\n-- privacy: {len(leaks)} file(s) with prompt text or credentials; bench requests seen by "
                 f"the live ~/.aar-gateway: {live_touched or 'none'}")
    manifest = {"command": sys.argv, "versions": _versions(), "conditions": conditions,
                "datasets": {g: {"sha256": _sha(Path(g).expanduser()), "prompts": len(load_prompts(Path(g).expanduser()))}
                             for g in args.graded},
                "configs": {f.name: f.read_text(encoding="utf-8") for f in CONFIGS.glob("*") if f.suffix in (".json", ".yaml")},
                "routellm_threshold_procedure": "RouteLLM calibrate_threshold: quantile(1 - share) of bert win rates on "
                                                "routellm/lmsys-arena-human-preference-55k-thresholds; share = aar's Opus "
                                                "share on the primary condition. No graded prompt tunes it.",
                "aar": "isolated gateway subprocess: own AAR_GATEWAY_LOG folder, live config with capture_prompts=false, "
                       "keep_history=false, no share consent, AAR_GATEWAY_UPSTREAM=stub",
                "attribution": "sequential replay; one upstream call per request = a pick; bench id in a trailing system block"}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nresults: {out_dir}")
    if not leaks and all(c["accepted_claude_code_request"] or k == "routellm" for k, c in compat.items()):
        shutil.rmtree(run_dir, ignore_errors=True)  # router homes, logs, configs: nothing worth keeping
    return 1 if leaks or live_touched else 0

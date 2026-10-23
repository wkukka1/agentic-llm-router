"""Does a router file work behind the Aptic plugin, and how long does a turn wait for it?

Aptic (``mods/aptic``) asks the gateway ``POST /api/route`` at the start of every
typed turn and waits for the answer before the turn starts, so the router's
latency is added to every turn. This starts an ISOLATED gateway (its own home
folder and port, no upstream; the live ``~/.aar-gateway`` is only read), serving
``--router``, and calls ``/api/route`` with the plugin's exact body:

* **contract** -- every answer is ``{"model": <a Claude tier or null>, "requested", "reason"}``;
* **lambda** -- the picks move with the ``cost_weight`` preference (set over ``/api/preferences``);
* **latency** -- time to a healthy gateway (router + encoder load), the first
  ``/api/route`` (cold), then p50 / p95 / max over warm calls, short and long prompts.

    python scripts/routing/plugin_route_check.py \\
        --router aar-gateway/src/aar_gateway/routers/manifold-nirt-claude-judged.npz \\
        --router ~/.aar-gateway/router.npz
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

PORT = 8797
TIERS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")


def _post(base: str, path: str, body: dict, timeout: float = 120.0) -> tuple[dict, float]:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def _get(base: str, path: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def _home(root: Path, router: Path) -> Path:
    """An isolated gateway home: the live config with this router file, prompt capture
    and history off, and the live encoders folder linked (read only)."""
    home = root / router.stem
    home.mkdir(parents=True, exist_ok=True)
    live = Path.home() / ".aar-gateway"
    spec = yaml.safe_load((live / "config.yaml").read_text(encoding="utf-8")) if (live / "config.yaml").exists() else {}
    icfg = spec.setdefault("interceptor", {})
    icfg["capture_prompts"] = False
    icfg.setdefault("router", {}).update(enabled=True, artifact=str(router.resolve()))
    (home / "config.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    (home / "preferences.json").write_text(json.dumps({"keep_history": False}), encoding="utf-8")
    enc = home / "encoders"
    if (live / "encoders").exists() and not enc.exists():
        if os.name == "nt":   # a junction needs no admin rights on Windows
            subprocess.run(["cmd", "/c", "mklink", "/J", str(enc), str(live / "encoders")], check=True,
                           capture_output=True)
        else:
            enc.symlink_to(live / "encoders", target_is_directory=True)
    return home


def check(router: Path, root: Path, prompts: list[str], warm: int) -> dict:
    home = _home(root, router)
    base = f"http://127.0.0.1:{PORT}"
    env = {**os.environ, "AAR_GATEWAY_LOG": str(home / "requests.jsonl"), "AAR_GATEWAY_CONFIG": str(home / "config.yaml"),
           "AAR_GATEWAY_HOST": "127.0.0.1", "AAR_GATEWAY_PORT": str(PORT),
           "AAR_GATEWAY_UPSTREAM": "http://127.0.0.1:9", "FASTEMBED_CACHE_PATH": str(root / "fastembed")}
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "aar_gateway.cli", "serve"], env=env,
                            stdout=open(home / "gateway.out", "w"), stderr=subprocess.STDOUT)
    try:
        health = None
        while time.perf_counter() - t0 < 300:
            try:
                health = _get(base, "/health")
                break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError(f"gateway exited: {(home / 'gateway.out').read_text()[-2000:]}") from None
                time.sleep(0.25)
        startup = time.perf_counter() - t0
        prefs = _get(base, "/api/preferences")
        if not prefs.get("router_enabled"):
            raise RuntimeError(f"router did not load: {prefs.get('router_error')}")
        body = lambda text, i: {"text": text, "model": "claude-opus-5-5", "session_id": f"check-{i}",  # noqa: E731
                                "blocked": [], "resumed": False}
        first, cold_ms = _post(base, "/api/route", body(prompts[0], "cold"))
        lat, picks, bad = {"short": [], "long": []}, {}, []
        for lam in (None, 0.02, 0.3):
            _post(base, "/api/preferences", {"cost_weight": lam})
            for i, text in enumerate(prompts[:warm]):
                ans, ms = _post(base, "/api/route", body(text, f"{lam}-{i}"))
                lat["long" if len(text) > 2000 else "short"].append(ms)
                if ans.get("model") not in (*TIERS, None) or "reason" not in ans:
                    bad.append(ans)
                picks.setdefault(str(lam), []).append(ans.get("model") or "keep")
        allms = lat["short"] + lat["long"]
        pct = lambda xs, q: round(float(np.percentile(xs, q)), 1) if xs else None  # noqa: E731
        return {"router": router.name, "health_router": health.get("router") if health else None,
                "startup_s": round(startup, 1), "cold_route_ms": round(cold_ms, 1), "first_answer": first,
                "warm_calls": len(allms), "p50_ms": pct(allms, 50), "p95_ms": pct(allms, 95),
                "max_ms": round(max(allms), 1),
                "short_p50_ms": pct(lat["short"], 50), "long_p50_ms": pct(lat["long"], 50),
                "n_long": len(lat["long"]), "contract_violations": bad,
                "picks": {lam: {m: p.count(m) for m in sorted(set(p))} for lam, p in picks.items()}}
    finally:
        proc.terminate()      # our own child, by handle -- never by name
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", action="append", required=True, help="router file(s) to check")
    ap.add_argument("--graded", default=str(Path.home() / ".cache/aar/routing-pool/multiturn/graded.jsonl"),
                    help="prompts to route (their route text); any graded jsonl")
    ap.add_argument("--warm", type=int, default=40, help="warm calls per lambda setting")
    ap.add_argument("--out", default="investigations/profile_manifold/plugin_route_check.json")
    args = ap.parse_args(argv)

    from evaluation.gateway.pool_eval import load_prompts

    prompts = [p["route_text"] for p in load_prompts(Path(args.graded).expanduser())]
    print(f"{len(prompts)} prompts; {sum(len(p) > 2000 for p in prompts)} over 2,000 characters")
    root = Path(tempfile.mkdtemp(prefix="aar-plugin-check-"))
    live_models = Path.home() / ".aar-gateway" / "models"   # fastembed's cache (MiniLM): a copy, never written
    if live_models.exists():
        shutil.copytree(live_models, root / "fastembed")
    results = []
    try:
        for r in args.router:
            res = check(Path(r).expanduser(), root, prompts, args.warm)
            results.append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "first_answer"}, indent=1))
    finally:
        # unlink every encoders link FIRST so the delete below can never reach the live folder
        for link in root.glob("*/encoders"):
            if os.path.islink(link) or (hasattr(os.path, "isjunction") and os.path.isjunction(link)):
                os.unlink(link) if os.path.islink(link) else os.rmdir(link)   # removes the link, not the target
        if not any(os.path.isjunction(p) or os.path.islink(p) for p in root.rglob("*")):
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"left {root} in place: it still holds a link")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

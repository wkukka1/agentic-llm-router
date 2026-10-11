"""The gateway's own routing data: the user's Claude Code turns, graded on the
real Claude tiers by ``aar-gateway-grade`` (``evaluation.gateway.grade``).

For a new prompt, a tier's predicted quality is the similarity-weighted mean
of its grades over the most similar graded prompts. When too few graded
prompts are similar enough, there is no prediction and the router falls back
to the benchmark router with its stand-ins (``proxy``). The file is re-read
when it changes, so grades added while the gateway runs are picked up.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

Encoder = Callable[[Sequence[str]], np.ndarray]  # unit-normalized rows
GradedPredictor = Callable[[str], Optional[dict]]


def load_graded(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """``(prompts, models, scores)``: one row per graded prompt, ``nan`` where
    a model has no grade for it (failed replay, unparseable verdict)."""
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("score") is not None]
    prompts: dict[str, str] = {}
    for r in rows:
        prompts.setdefault(r["prompt_hash"], r["prompt"])
    hashes, models = list(prompts), sorted({r["model"] for r in rows})
    scores = np.full((len(hashes), len(models)), np.nan)
    at, col = {h: i for i, h in enumerate(hashes)}, {m: j for j, m in enumerate(models)}
    for r in rows:
        scores[at[r["prompt_hash"]], col[r["model"]]] = float(r["score"])
    return [prompts[h] for h in hashes], models, scores


def graded_predictor(path: Path, encode: Encoder, *, k: int = 10, min_similarity: float = 0.5,
                     min_neighbors: int = 3, min_prompts: int = 20) -> GradedPredictor:
    state: dict = {"mtime": None, "prompts": [], "models": [], "scores": None, "vectors": None}
    lock = threading.Lock()

    def refresh() -> None:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if mtime == state["mtime"]:
            return
        prompts, models, scores = load_graded(path)
        vectors = encode(prompts) if prompts else None
        state.update(mtime=mtime, prompts=prompts, models=models, scores=scores, vectors=vectors)

    def predict(text: str) -> Optional[dict]:
        with lock:
            refresh()
            if len(state["prompts"]) < min_prompts:
                return None
            sims = state["vectors"] @ encode([text])[0]
            top = np.argsort(-sims)[:k]
            top = top[sims[top] >= min_similarity]
            if len(top) < min_neighbors:
                return None
            weights = sims[top]
            quality = {}
            for j, model in enumerate(state["models"]):
                col = state["scores"][top, j]
                ok = ~np.isnan(col)
                if ok.sum() >= min_neighbors:
                    quality[model] = float(np.average(col[ok], weights=weights[ok]))
            return {"quality": quality, "neighbors": int(len(top)),
                    "max_similarity": round(float(sims[top].max()), 3),
                    "graded_prompts": len(state["prompts"])}

    return predict

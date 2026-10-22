"""What the ZOIB router would do in the gateway, before it can be served there.

The ZOIB run (``lrb-zoib-projected``, a ``BaselineNIRT`` with a ZOIB response head,
an interaction term and taxonomy relevance) doesn't fold into the gateway's ``mlp``
router file (``training.trainers.nirt_router_export`` refuses it), so it can't run
behind a live gateway yet. This replays it offline the way the gateway would use it:
raw prompt text -> MiniLM (``retrieval``) embedding -> taxonomy relevance from the
cluster centroids -> the model's expected score (``pi1 + pic * mu``) per pool model
-> each Claude tier's stand-in (``proxy``) -> the gateway's margin rule.

Two checks make the replay worth trusting:

* **text-path parity** -- the same path, from raw text, reproduces the run's own
  predictions on held-out dataset queries (stored embeddings + stored relevance);
* **gateway parity** -- the MLP and NIRT router files replayed here give the same
  picks the live gateways gave in :mod:`evaluation.gateway.router_compare`.

    python -m evaluation.gateway.zoib_replay --graded ~/.aar-gateway/eval-rules/graded.jsonl \\
        --graded ~/.aar-gateway/eval40/graded.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluation.gateway import router_bench as rb
from evaluation.gateway import router_compare as rc
from evaluation.gateway.pool_eval import TIERS, load_prompts, margin_pick

REPO = rb.REPO
RUN = REPO / "data" / "processed" / "nirt_runs" / "lrb-zoib-projected"
CONFIG = REPO / "configs" / "llmrouterbench.yaml"
MARGINS = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3)


def load_zoib(run: Path = RUN, config: Path = CONFIG):
    """``(model, blob, settings, data, centroids)`` for the ZOIB run and its dataset."""
    from router.config import load_config
    from training.data.facade import load_training_data
    from training.nirt.baseline.checkpoint import load_baseline_run
    from training.taxonomy.clustering import load_centroids

    model, blob, settings = load_baseline_run(run)
    cfg = load_config(str(config))
    data = load_training_data(cfg)
    centroids = load_centroids(cfg)
    if model.use_relevance and centroids.shape[0] != model.relevance_dim:
        raise ValueError(f"taxonomy has {centroids.shape[0]} clusters, the checkpoint was trained on "
                         f"{model.relevance_dim}: retrain it against the current taxonomy")
    return model, blob, settings, data, centroids


def _model_refs(data, blob, pathway: str) -> tuple[list[str], np.ndarray]:
    """Pool order and each model's profile embedding, as the run saw them."""
    from training.nirt.baseline.data import build_arrays

    order = sorted(blob["model_index"], key=lambda m: blob["model_index"][m])
    ev = build_arrays(data.cfg, split="test", data=data, pathway=pathway, use_relevance=False, use_warmup=False,
                      model_index=blob["model_index"])
    refs = {}
    for m, e in zip(ev.model_ids, ev.e_m, strict=True):
        refs.setdefault(str(m), e)
    missing = [m for m in order if m not in refs]
    if missing:
        raise ValueError(f"no profile embedding for {missing}")
    return [str(m) for m in order], np.stack([refs[str(m)] for m in order]).astype(np.float32)


def score_embeddings(model, blob, e_q: np.ndarray, r_q: np.ndarray | None, refs: np.ndarray,
                     order: list[str]) -> np.ndarray:
    """``[Q, M]`` expected scores for queries given as embeddings (+ relevance)."""
    from training.nirt.baseline.data import BaselineArrays, batched_forward

    q, m = len(e_q), len(order)
    rows = np.repeat(np.arange(q), m)
    cols = np.tile(np.arange(m), q)
    arrays = BaselineArrays(
        e_q=np.ascontiguousarray(e_q[rows], dtype=np.float32), midx=np.array([blob["model_index"][order[c]] for c in cols]),
        y=np.zeros(q * m, np.float32), y_soft=np.zeros(q * m, np.float32),
        query_ids=rows.astype(str), model_ids=np.array([order[c] for c in cols]),
        e_m=np.ascontiguousarray(refs[cols], dtype=np.float32),
        r_q=None if r_q is None else np.ascontiguousarray(r_q[rows], dtype=np.float32))
    return batched_forward(model, arrays, fields=("mean",))["mean"].reshape(q, m)


def score_texts(texts, model, blob, settings, data, centroids, refs, order, *, tau: float) -> np.ndarray:
    """``[Q, M]`` expected scores for raw prompt text: the path the gateway would serve."""
    from router.routing.routers import _encode_texts
    from training.taxonomy.relevance import relevance_from_embeddings

    e_q = _encode_texts(data, list(texts), settings["pathway"]).astype(np.float32)
    r_q = relevance_from_embeddings(e_q, centroids, tau=tau) if model.use_relevance else None
    return score_embeddings(model, blob, e_q, r_q, refs, order)


def text_path_parity(model, blob, settings, data, centroids, refs, order, *, n: int = 300) -> dict:
    """Raw text vs the run's stored features on held-out queries, for each relevance temperature
    the store could have used. The temperature that reproduces the store is the one to serve."""
    from training.nirt.baseline.data import build_arrays
    from training.taxonomy.relevance import DEFAULT_TAU

    ev = build_arrays(data.cfg, split="test", data=data, pathway=settings["pathway"], use_relevance=model.use_relevance,
                      use_warmup=False, model_index=blob["model_index"],
                      relevance_dim=model.relevance_dim if model.use_relevance else None)
    qids = list(dict.fromkeys(ev.query_ids))[:n]
    first = {q: i for i, q in reversed(list(enumerate(ev.query_ids)))}
    stored_e = np.stack([ev.e_q[first[q]] for q in qids])
    stored_r = None if ev.r_q is None else np.stack([ev.r_q[first[q]] for q in qids])
    want = score_embeddings(model, blob, stored_e, stored_r, refs, order)
    lookup = dict(zip(data.queries["query_id"].astype(str), data.queries["query"].astype(str), strict=True))
    have = [q for q in qids if str(q) in lookup]
    idx = [qids.index(q) for q in have]
    out = {"n": len(have)}
    for tau in sorted({DEFAULT_TAU, 0.05, 0.1, 0.2, 0.5, 1.0}):
        got = score_texts([lookup[str(q)] for q in have], model, blob, settings, data, centroids, refs, order, tau=tau)
        out[f"tau={tau:g}"] = float(np.abs(got - want[idx]).max())
    from router.routing.routers import _encode_texts

    emb = _encode_texts(data, [lookup[str(q)] for q in have], settings["pathway"])
    out["embedding_max_abs_diff"] = float(np.abs(emb - stored_e[idx]).max())
    return out


def proxy_scores(frame_or_matrix, order_or_none=None) -> np.ndarray:
    from aar_gateway.config import DEFAULT_MODELS

    prox = [DEFAULT_MODELS[t]["proxy"] for t in TIERS]
    if order_or_none is None:  # a gateway router's score frame
        return np.stack([frame_or_matrix[p].to_numpy(float) for p in prox], axis=1)
    return frame_or_matrix[:, [order_or_none.index(p) for p in prox]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.gateway.zoib_replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("--graded", action="append", required=True)
    ap.add_argument("--out", default=str(rc.BENCH_HOME / "results" / "zoib_replay.json"))
    args = ap.parse_args(argv)
    seen, prompts = set(), []
    for g in args.graded:
        for p in load_prompts(Path(g).expanduser()):
            if p["prompt_hash"] not in seen:
                seen.add(p["prompt_hash"])
                prompts.append(p)

    model, blob, settings, data, centroids = load_zoib()
    order, refs = _model_refs(data, blob, settings["pathway"])
    parity = text_path_parity(model, blob, settings, data, centroids, refs, order)
    taus = {k: v for k, v in parity.items() if k.startswith("tau=")}
    best = min(taus, key=taus.get)
    tau = float(best.split("=")[1])
    print(f"text-path parity on {parity['n']} held-out queries: embedding max |d| {parity['embedding_max_abs_diff']:.2g}; "
          + ", ".join(f"{k} max |dE[y]| {v:.2g}" for k, v in taus.items()) + f"  -> serving {best}")

    texts = [p["route_text"] for p in prompts]
    preds = {"ZOIB (lrb-zoib-projected)": proxy_scores(score_texts(texts, model, blob, settings, data, centroids,
                                                                   refs, order, tau=tau), order)}
    from aar_gateway.routing import load_router_model

    for name, f in (("MLP (live default file)", Path.home() / ".aar-gateway" / "router.npz"),
                    ("NIRT (llmrouterbench-nirt)", rb.AAR_ROUTER_FILES["aar-nirt"])):
        preds[name] = proxy_scores(load_router_model({"artifact": str(f)}).predict_scores_text(texts))

    rows, labels = [], {}
    for name, pred in preds.items():
        key = "offline:" + name.split(" ")[0].lower()
        labels[key] = name
        for m in MARGINS:
            pick = margin_pick(pred, m)
            rows += [{"router": key, "track": "margin", "condition": f"margin {m:g}", "prompt_hash": p["prompt_hash"],
                      "tier": TIERS[k], "error": None} for p, k in zip(prompts, pick, strict=True)]
    rb.ROUTERS.update({k: (v, "offline") for k, v in labels.items()})
    lines: list[str] = []
    sweep = rb.margin_sweep(prompts, rows, MARGINS, list(labels), lines, f"all ({len(prompts)} prompts)")
    gaps = {name: {"opus_minus_sonnet_median": float(np.median(p[:, 2] - p[:, 1])),
                   "opus_minus_haiku_median": float(np.median(p[:, 2] - p[:, 0])),
                   "opus_minus_haiku_min": float((p[:, 2] - p[:, 0]).min()),
                   "corr_with_real_opus_minus_haiku": float(np.corrcoef(
                       p[:, 2] - p[:, 0], np.stack([q["score"] for q in prompts])[:, 2]
                       - np.stack([q["score"] for q in prompts])[:, 0])[0, 1])}
            for name, p in preds.items()}
    print("\n".join(lines))
    print("\npredicted tier gaps (stand-ins):")
    for name, g in gaps.items():
        print(f"  {name:28s} Opus-Sonnet median {g['opus_minus_sonnet_median']:+.3f}  Opus-Haiku median "
              f"{g['opus_minus_haiku_median']:+.3f} (min {g['opus_minus_haiku_min']:+.3f})  "
              f"corr with real Opus-Haiku gap {g['corr_with_real_opus_minus_haiku']:+.2f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"parity": parity, "tau": tau, "gaps": gaps, "sweep": sweep}, indent=1, default=float),
                   encoding="utf-8")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

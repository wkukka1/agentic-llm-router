"""Prompt-compression experiment for the NIRT router.

Target A: QueryBottleneck squeezes a 768-dim query embedding to a 64-dim latent,
with a linear head that scores each candidate model.
Target B: compress_prompt_text prunes prompt tokens with LLMLingua-2.

Running this file retrains the bottleneck (and an uncompressed 768-dim linear head
as the "original") on the RouterBench train split and benchmarks both against the
ZOIB checkpoint on the test split, with the same lambda sweep as
scripts/nirt/baseline/routing_benchmark.py:

    python experiments/prompt_compression/experiment.py

With --compress-rates, the test prompts are instead compressed with LLMLingua-2 and
re-embedded (BERT e_q and the MiniLM-based relevance r_q), and every router scores
the compressed view. Labels and realized costs stay those of the ORIGINAL prompt, so
this measures routing on a compressed prompt while the full prompt goes to the model.
Rate 1.0 re-embeds the uncompressed text as a pipeline control:

    python experiments/prompt_compression/experiment.py --compress-rates 1.0,0.7,0.5,0.3
"""
import argparse
import dataclasses
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
ZOIB_CKPT = "artifacts/phase2/zoib_cost_length_v2"
OUT = "experiments/prompt_compression/results_test.json"
COMPRESS_OUT = "experiments/prompt_compression/results_compressed_test.json"
TEXT_CACHE = Path(__file__).with_name("compressed_texts")


class QueryBottleneck(nn.Module):
    def __init__(self, n_models: int, in_dim: int = 768, hidden: int = 256, latent: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.head = nn.Linear(latent, n_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self(x))


def full_linear_head(n_models: int, in_dim: int = 768) -> nn.Module:
    """The uncompressed comparison: one linear layer straight off the 768-dim embedding."""
    return nn.Linear(in_dim, n_models)


def fit_quality_head(net: nn.Module, X_tr, Y_tr, X_va, Y_va, epochs: int = 60, lr: float = 1e-3,
                     batch_size: int = 512, patience: int = 5, seed: int = 42):
    """Train ``net`` to predict each model's graded score (sigmoid + BCE on soft targets),
    early-stopping on validation BCE. Returns ``predict(X) -> [n, n_models]`` probabilities."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = getattr(net, "classify", net)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)  # noqa: E731
    Xt, Yt, Xv, Yv = t((X_tr - mu) / sd), t(Y_tr), t((X_va - mu) / sd), t(Y_va)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(Xt), device=device).split(batch_size):
            opt.zero_grad()
            bce(logits(Xt[idx]), Yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = bce(logits(Xv), Yv).item()
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif (bad := bad + 1) >= patience:
            break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(logits(t((X - mu) / sd))).cpu().numpy().astype(np.float64)

    return predict


@lru_cache(maxsize=1)
def _compressor():
    from llmlingua import PromptCompressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PromptCompressor(model_name=LLMLINGUA2_MODEL, use_llmlingua2=True, device_map=device)


def compress_prompt_text(prompt: str, rate: float = 0.5) -> str:
    return _compressor().compress_prompt(prompt, rate=rate, force_tokens=["?"])["compressed_prompt"]


def compress_texts(texts, rate: float) -> list[str]:
    """``compress_prompt_text`` over a list; ``rate >= 1`` keeps the text as is."""
    if rate >= 1.0:
        return list(texts)
    return [compress_prompt_text(t, rate=rate) if t.strip() else t for t in texts]


def replace_rows(ids, base: np.ndarray, new_by_id: dict) -> np.ndarray:
    """Copy of the cell-level ``base`` with every row whose id is in ``new_by_id`` replaced."""
    out = base.copy()
    for i, q in enumerate(ids):
        if q in new_by_id:
            out[i] = new_by_id[q]
    return out


def _predictive(P: np.ndarray, Y: np.ndarray) -> dict:
    """Cell-level fit and quality-only routing regret of a [Q, M] quality prediction."""
    Pc = np.clip(P, 1e-6, 1 - 1e-6)
    picked = Y[np.arange(len(Y)), P.argmax(1)]
    return {"bce": float(-(Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc)).mean()),
            "mae": float(np.abs(P - Y).mean()),
            "quality@argmax": float(picked.mean()),
            "regret@argmax": float((Y.max(1) - picked).mean())}


def _setup(ckpt_arg: str):
    """Data, ZOIB test predictions and the dense RouterBench train/val/test matrices."""
    from evaluation.nirt.routing import eval_matrices
    from evaluation.nirt.routing_benchmark import REFERENCE_MODELS, baseline_cell_predictions
    from router.config import load_config
    from training.cli import resolve
    from training.data.facade import load_training_data

    cfg = load_config()
    d = load_training_data(cfg)
    ckpt = resolve(ckpt_arg, cfg.root)
    long, _ = baseline_cell_predictions(ckpt, cfg, d, "test")
    src = "routerbench"
    Y_df, C_df = eval_matrices(d, "test", models=sorted(long["model_id"].unique()), source=src)
    models = list(Y_df.columns)
    tr, _ = eval_matrices(d, "train", models=models, source=src)
    va, _ = eval_matrices(d, "validation", models=models, source=src)
    return cfg, d, ckpt, long, Y_df, C_df, tr, va, models, REFERENCE_MODELS[src]


def _train_heads(store, tr, va, models, epochs: int, seed: int) -> dict:
    """``{name: predict}`` for the bottleneck and the uncompressed linear head."""
    emb = lambda df: store.gather(list(df.index)).astype(np.float32)  # noqa: E731
    X_tr, X_va = emb(tr), emb(va)
    Y_tr, Y_va = tr[models].to_numpy(np.float32), va[models].to_numpy(np.float32)
    print(f"{len(models)} models | train {len(X_tr)} / val {len(X_va)} queries")
    heads = {"Bottleneck64": QueryBottleneck(len(models)), "Full768": full_linear_head(len(models))}
    out = {}
    for name, net in heads.items():
        print(f"training {name} ({sum(p.numel() for p in net.parameters()):,} params) ...")
        out[name] = fit_quality_head(net, X_tr, Y_tr, X_va, Y_va, epochs=epochs, seed=seed)
    return out


def _zoib_long(ckpt, cfg, d, e_q_by_id: dict, r_q_by_id: dict):
    """``baseline_cell_predictions`` on test with each query's e_q / r_q swapped in."""
    import pandas as pd
    from training.data.tokens import cell_token_table
    from training.nirt.baseline.checkpoint import load_baseline_run
    from training.nirt.baseline.data import attach_cost_offset, batched_forward, build_arrays
    from training.nirt.baseline.length import attach_length_inputs, attach_tokens

    model, blob, s = load_baseline_run(ckpt)
    ev = build_arrays(cfg, split="test", data=d, pathway=s["pathway"],
                      binary_threshold=s["binary_threshold"], score_kind=s["score_kind"],
                      use_relevance=model.use_relevance, use_warmup=model.use_warmup,
                      model_index=blob["model_index"],
                      relevance_dim=model.relevance_dim if model.use_relevance else None)
    ev = dataclasses.replace(ev, e_q=replace_rows(ev.query_ids, ev.e_q, e_q_by_id))
    if ev.r_q is not None:
        ev = dataclasses.replace(ev, r_q=replace_rows(ev.query_ids, ev.r_q, r_q_by_id))
    ev = attach_cost_offset(ev, blob["model_cfg"])
    if blob["model_cfg"].get("cost_mode") == "length":
        ev = attach_length_inputs(attach_tokens(ev, cell_token_table(d, cfg)), blob["model_cfg"])
    f = batched_forward(model, ev, fields=("mean", "cost"))
    return pd.DataFrame({"query_id": ev.query_ids, "model_id": ev.model_ids,
                         "quality_pred": f["mean"], "cost_pred": f["cost"]})


def _benchmark(quality: dict, zc, Cm, Y, C, models, ref, lams) -> tuple[dict, dict]:
    from evaluation.nirt.routing_benchmark import source_picks, summarize

    routers = {}
    for k, q in quality.items():
        routers[f"{k} + C_hat"] = (q, zc)  # ZOIB's per-query cost head, shared by all
        routers[f"{k} + C(m)"] = (q, Cm)  # static per-model train-mean cost
    return ({k: _predictive(v, Y) for k, v in quality.items()},
            summarize(source_picks(Y, C, models, ref, routers, lams)))


def _print_predictive(predictive: dict) -> None:
    print("\n| quality model | test BCE | MAE | quality @ argmax | regret @ argmax |\n|---|---|---|---|---|")
    for k, m in predictive.items():
        print(f"| {k} | {m['bce']:.4f} | {m['mae']:.4f} | {m['quality@argmax']:.3f} | {m['regret@argmax']:.3f} |")


def _cached_compression(ids, texts, rate: float) -> list[str]:
    """Compressed test prompts for ``rate``, cached as parquet beside this file."""
    import pandas as pd

    path = TEXT_CACHE / f"rate_{rate:g}.parquet"
    if path.exists():
        cached = pd.read_parquet(path).set_index("query_id")["text"]
        if set(ids) <= set(cached.index):
            return [cached[q] for q in ids]
    t0 = time.time()
    out = compress_texts(texts, rate=rate)
    print(f"  compressed {len(ids)} prompts at rate {rate:g} in {time.time() - t0:.0f}s")
    TEXT_CACHE.mkdir(exist_ok=True)
    pd.DataFrame({"query_id": list(ids), "text": out}).to_parquet(path)
    return out


def run_compression(args) -> None:
    from evaluation.nirt.routing import align
    from evaluation.nirt.routing_benchmark import DEFAULT_LAMS, per_model_mean_cost
    from router.embeddings import load_encoder
    from router.nirt.frames import pivot_qm
    from training.cli import resolve, write_json
    from training.taxonomy.clustering import load_centroids
    from training.taxonomy.relevance import relevance_from_embeddings

    cfg, d, ckpt, _, Y_df, C_df, tr, va, models, ref = _setup(args.zoib_ckpt)
    store = d.query_embeddings("irt")
    heads = _train_heads(store, tr, va, models, args.epochs, args.seed)
    ids = [str(q) for q in Y_df.index]
    all_texts = d.query_texts("test")
    texts = [all_texts[q] for q in ids]
    stored = store.gather(ids).astype(np.float32)
    irt_enc, ret_enc = load_encoder(cfg, "irt"), load_encoder(cfg, "retrieval")
    centroids = load_centroids(cfg)
    Y, C = Y_df.to_numpy(np.float64), C_df.to_numpy(np.float64)
    Cm = per_model_mean_cost(d.nirt_observations()).reindex(models).to_numpy(np.float64)
    lams = list(DEFAULT_LAMS)
    base_words = sum(len(t.split()) for t in texts)

    results, base_picks = {}, None
    for rate in [float(r) for r in args.compress_rates.split(",")]:
        print(f"\n=== rate {rate:g} ===")
        ctexts = _cached_compression(ids, texts, rate)
        e_q = irt_enc.encode(ctexts).astype(np.float32)
        r_q = relevance_from_embeddings(ret_enc.encode(ctexts), centroids, tau=args.tau)
        cos = (e_q * stored).sum(1) / (np.linalg.norm(e_q, axis=1) * np.linalg.norm(stored, axis=1) + 1e-12)
        zl = _zoib_long(ckpt, cfg, d, dict(zip(ids, e_q)), dict(zip(ids, r_q)))
        zq, zc = align(pivot_qm(zl, "quality_pred"), Y_df), align(pivot_qm(zl, "cost_pred"), Y_df)
        quality = {"ZOIB": zq, **{k: f(e_q) for k, f in heads.items()}}
        predictive, policies = _benchmark(quality, zc, Cm, Y, C, models, ref, lams)
        picks = {k: v.argmax(1) for k, v in quality.items()}
        base_picks = base_picks or picks
        for k in predictive:
            predictive[k]["same_pick_as_first_rate"] = float((picks[k] == base_picks[k]).mean())
        word_ratio = sum(len(t.split()) for t in ctexts) / base_words
        results[f"{rate:g}"] = {"word_ratio": word_ratio,
                                "cos_to_stored_e_q": {"mean": float(cos.mean()), "min": float(cos.min())},
                                "predictive": predictive, "policies": policies}

        print(f"words kept {word_ratio:.0%} | cos(e_q, stored) mean {cos.mean():.4f} min {cos.min():.4f}")
        _print_predictive(predictive)
        print("same pick as first rate: " + ", ".join(
            f"{k} {m['same_pick_as_first_rate']:.0%}" for k, m in predictive.items()))
        for k in quality:
            cells = (policies[f"{k} + C_hat@{x:g}"] for x in (0, 20, 50, 200))
            print(f"  {k} + C_hat, lam 0/20/50/200: "
                  + "  ".join(f"{p['quality']:.3f} / ${p['cost_per_1k']:.2f}" for p in cells))

    write_json(resolve(args.out or COMPRESS_OUT, cfg.root),
               {"zoib_ckpt": str(ckpt), "split": "test", "source": "routerbench", "models": models,
                "n_test": len(ids), "seed": args.seed, "lams": lams, "rates": results})


def main() -> None:
    from evaluation.nirt.routing import align
    from evaluation.nirt.routing_benchmark import DEFAULT_LAMS, per_model_mean_cost
    from router.nirt.frames import pivot_qm
    from training.cli import resolve, write_json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zoib-ckpt", default=ZOIB_CKPT)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compress-rates", default=None, help="comma list, e.g. 1.0,0.7,0.5,0.3")
    ap.add_argument("--tau", type=float, default=0.1, help="relevance softmax temperature (the store's tau)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.compress_rates:
        return run_compression(args)

    cfg, d, ckpt, long, Y_df, C_df, tr, va, models, ref = _setup(args.zoib_ckpt)
    store = d.query_embeddings("irt")
    heads = _train_heads(store, tr, va, models, args.epochs, args.seed)
    X_te = store.gather(list(Y_df.index)).astype(np.float32)
    preds = {k: f(X_te) for k, f in heads.items()}

    zq = align(pivot_qm(long, "quality_pred"), Y_df)
    zc = align(pivot_qm(long, "cost_pred"), Y_df)
    Y, C = Y_df.to_numpy(np.float64), C_df.to_numpy(np.float64)
    keep = np.isfinite(zq).all(1) & np.isfinite(zc).all(1)
    Y, C, zq, zc = Y[keep], C[keep], zq[keep], zc[keep]
    preds = {k: v[keep] for k, v in preds.items()}
    Cm = per_model_mean_cost(d.nirt_observations()).reindex(models).to_numpy(np.float64)
    lams = list(DEFAULT_LAMS)
    quality = {"ZOIB": zq, **preds}
    predictive, policies = _benchmark(quality, zc, Cm, Y, C, models, ref, lams)

    print(f"\n{int(keep.sum())} RouterBench test queries ({int((~keep).sum())} dropped), ref = {ref}")
    _print_predictive(predictive)
    r, o = policies["reference"], policies["oracle_best_quality"]
    print(f"\nAlways {ref}: q {r['quality']:.3f} @ ${r['cost_per_1k']:.2f}/1k;  "
          f"oracle: q {o['quality']:.3f} @ ${o['cost_per_1k']:.2f}/1k")
    print("\n| router | " + " | ".join(f"lam {x:g}" for x in lams) + " |")
    print("|---|" + "---|" * len(lams))
    for k in quality:
        for name in (f"{k} + C_hat", f"{k} + C(m)"):
            cells = (policies[f"{name}@{x:g}"] for x in lams)
            print(f"| {name} | " + " | ".join(f"{p['quality']:.3f} / ${p['cost_per_1k']:.2f}" for p in cells) + " |")

    write_json(resolve(args.out or OUT, cfg.root),
               {"zoib_ckpt": str(ckpt), "split": "test", "source": "routerbench", "models": models,
                "n_train": len(tr), "n_val": len(va), "n_test": int(keep.sum()),
                "predictive": predictive, "lams": lams, "policies": policies})


if __name__ == "__main__":
    main()

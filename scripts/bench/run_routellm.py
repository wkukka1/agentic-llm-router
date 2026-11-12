"""RouteLLM 0.2.0 routers for the 2-model sub-study (gpt_4o vs gpt_4o_mini).

Runs in ``.venv-routellm`` only; never imported by the repo. Reads the bake-off
bundle (``scripts/bench/export_bakeoff.py``) plus a query-text file and writes
one strong-model win rate per evaluation query and router:

    .venv-routellm/bin/python -I scripts/bench/run_routellm.py \\
        --bundle data/processed/router_bakeoff/primary --texts <texts.parquet>

Arms
* ``bert (pretrained, Arena)``: RouteLLM's released ``routellm/bert_gpt4_augmented``
  checkpoint (``BERTRouter``), off the shelf, with no training on our data. Win rate
  = 1 - P(tie or weak wins), the library's ``calculate_strong_win_rate``, batched.
* ``mf (retrained, 20-model pairs)``: the library's ``MFModel_Train`` and
  ``train_loops`` with its published defaults (dim 128, lr 3e-4, wd 1e-5,
  alpha 0.1, batch 64), trained on pairwise wins built from our train split
  across all 20 models, as RouteLLM trains on all Arena model pairs. Prompt
  embeddings are our 768-d BERT vectors, L2-normalized (RouteLLM's OpenAI
  embeddings are unit-norm, which its noise scale alpha assumes).
* ``mf (retrained, 4o-vs-mini pairs)``: same, only gpt_4o vs gpt_4o_mini pairs.

Not run: the pretrained ``mf`` and ``sw_ranking`` embed every prompt with
OpenAI's API (no key here). ``causal_llm`` is an 8B Llama classifier (no GPU).

Output: ``<bundle>/routellm/<arm>.parquet`` (query_id, strong_win_rate).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

# routellm builds an OpenAI client at import time; nothing here calls it (we pass our
# own embeddings), so a placeholder satisfies the constructor
os.environ.setdefault("OPENAI_API_KEY", "unused-no-api-calls")

STRONG, WEAK = "gpt_4o", "gpt_4o_mini"
MF_HP = {"dim": 128, "batch_size": 64, "alpha": 0.1, "lr": 3e-4, "weight_decay": 1e-5}


def run_bert(texts: pd.Series, batch: int, max_len: int) -> np.ndarray:
    from routellm.routers.routers import BERTRouter

    r = BERTRouter(checkpoint_path="routellm/bert_gpt4_augmented")
    r.model.eval()
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        enc = r.tokenizer(list(texts.iloc[i:i + batch]), return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len)
        with torch.no_grad():
            p = torch.softmax(r.model(**enc).logits, dim=-1).numpy()
        out.append(1.0 - p[:, -2:].sum(axis=1))      # = calculate_strong_win_rate, batched
        if (i // batch) % 20 == 0:
            print(f"[bert] {i + batch}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out)


def pairs_from_scores(Y: pd.DataFrame, models: list[str], per_query: int | None, rng) -> list[dict]:
    """RouteLLM-format battles: one per (query, model pair) with different graded scores."""
    out = []
    A = Y[models].to_numpy()
    for qi in range(len(Y)):
        cand = [(a, b) for a in range(len(models)) for b in range(a + 1, len(models))
                if np.isfinite(A[qi, a]) and np.isfinite(A[qi, b]) and A[qi, a] != A[qi, b]]
        if per_query is not None and len(cand) > per_query:
            cand = [cand[j] for j in rng.choice(len(cand), per_query, replace=False)]
        for a, b in cand:
            out.append({"idx": qi, "model_a": models[a], "model_b": models[b],
                        "winner": "model_a" if A[qi, a] > A[qi, b] else "model_b"})
    return out


def run_mf(Y: pd.DataFrame, E: np.ndarray, models: list[str], eval_E: np.ndarray,
           per_query: int | None, epochs: int, seed: int) -> np.ndarray:
    import routellm.routers.matrix_factorization.model as mfm
    import routellm.routers.matrix_factorization.train_matrix_factorization as tmf

    # the library's PairwiseDataset maps names through this module-level dict
    mfm.MODEL_IDS.clear(); mfm.MODEL_IDS.update({m: i for i, m in enumerate(models)})
    tmf.MODEL_IDS = mfm.MODEL_IDS
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed); random.seed(seed)
    data = pairs_from_scores(Y, models, per_query, rng)
    random.shuffle(data)
    print(f"[mf] {len(models)} models, {len(data)} battles", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        npy = os.path.join(tmp, "emb.npy")
        np.save(npy, E.astype(np.float32))
        net = tmf.MFModel_Train(dim=MF_HP["dim"], num_models=len(models), num_prompts=len(E),
                                text_dim=E.shape[1], use_proj=True, npy_path=npy)
    cut = int(len(data) * 0.95)                      # the library's own 95/5 battle holdout
    loader = tmf.PairwiseDataset(data[:cut]).get_dataloaders(MF_HP["batch_size"], shuffle=True)
    hold = tmf.PairwiseDataset(data[cut:]).get_dataloaders(1024, shuffle=False)
    tmf.train_loops(net, loader, hold, lr=MF_HP["lr"], weight_decay=MF_HP["weight_decay"],
                    alpha=MF_HP["alpha"], num_epochs=epochs, device="cpu")
    net.eval()
    with torch.no_grad():
        Pn = torch.nn.functional.normalize(net.P.weight, p=2, dim=1)
        diff = Pn[models.index(STRONG)] - Pn[models.index(WEAK)]
        z = net.classifier(diff * net.text_proj(torch.from_numpy(eval_E))).squeeze(-1)
    return torch.sigmoid(z).numpy()                  # = MFModel.pred_win_rate(strong, weak)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--texts", required=True, help="parquet: query_id, text (written by routellm_substudy.py)")
    ap.add_argument("--arms", default="mf20,mfpair,bert")
    ap.add_argument("--splits", default="test,ood")
    ap.add_argument("--mf-epochs", type=int, default=100)
    ap.add_argument("--mf-pairs-per-query", type=int, default=5)
    ap.add_argument("--bert-batch", type=int, default=16)
    ap.add_argument("--bert-max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    b = args.bundle
    q = pd.read_parquet(os.path.join(b, "queries.parquet"))
    emb = np.load(os.path.join(b, "embeddings.npy"))
    obs = pd.read_parquet(os.path.join(b, "observations.parquet"))
    models = sorted(obs["model_id"].unique())
    ev = q[q["split"].isin(args.splits.split(","))].reset_index(drop=True)
    tr = q[q["split"] == "train"].reset_index(drop=True)
    out_dir = os.path.join(b, "routellm")
    os.makedirs(out_dir, exist_ok=True)

    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    Y = (obs[obs["query_id"].isin(set(tr["query_id"]))]
         .pivot(index="query_id", columns="model_id", values="target").reindex(tr["query_id"]))
    E_tr = unit[tr["emb_id"].to_numpy()]
    E_ev = unit[ev["emb_id"].to_numpy()].astype(np.float32)

    for arm in args.arms.split(","):
        name = {"bert": "bert (pretrained, Arena)", "mf20": "mf (retrained, 20-model pairs)",
                "mfpair": "mf (retrained, 4o-vs-mini pairs)"}[arm]
        path = os.path.join(out_dir, f"{arm}.parquet")
        if os.path.exists(path):
            print(f"[skip] {path}")
            continue
        t0 = time.time()
        if arm == "bert":
            texts = pd.read_parquet(args.texts).set_index("query_id")["text"].reindex(ev["query_id"])
            if texts.isna().any():
                raise SystemExit(f"{int(texts.isna().sum())} eval queries have no text")
            w = run_bert(texts.reset_index(drop=True), args.bert_batch, args.bert_max_len)
        elif arm == "mf20":
            w = run_mf(Y, E_tr, models, E_ev, args.mf_pairs_per_query, args.mf_epochs, args.seed)
        else:
            w = run_mf(Y, E_tr, [STRONG, WEAK] if STRONG < WEAK else [WEAK, STRONG],
                       E_ev, None, args.mf_epochs, args.seed)
        pd.DataFrame({"query_id": ev["query_id"], "strong_win_rate": w, "router": name,
                      "seconds": time.time() - t0}).to_parquet(path, index=False)
        print(f"[{arm}] done in {time.time() - t0:.0f}s -> {path}", flush=True)
    json.dump({"strong": STRONG, "weak": WEAK, "mf_hparams": MF_HP, "args": vars(args)},
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

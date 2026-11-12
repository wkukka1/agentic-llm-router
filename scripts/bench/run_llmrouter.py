"""Train and score llmrouter-lib's routers on the bake-off bundle.

Runs ONLY in the isolated venv (``.venv-llmrouter``, ``llmrouter-lib`` 0.4.0 +
``torch-geometric``); it never imports this repo. Router bake-off step 1,
docs/router_benchmark_experiment.md.

    .venv-llmrouter/bin/python scripts/bench/run_llmrouter.py \
        --bundle data/processed/router_bakeoff/primary \
        --routers knn,svm,mlp,mf,elo,graphrouter,routerdc --seed 0

What it does, per router and cost weight lambda:

* Training rows are the bundle's ``train`` split. Each (query, model) row gets
  ``performance = rescale(target - lambda * C_m)``, where ``C_m`` is the model's
  train-mean cost and ``rescale`` is one global affine map onto [0, 1]. That map
  keeps every per-query argmax and ordering, and keeps RouterDC's
  ``> 0`` / ``> 0.5`` thresholds meaningful. At lambda=0 it is the raw target.
* Within a query, rows are ordered by the model's train-mean score (strongest
  first), so the library's ``idxmax`` / ``argmax`` "best model" label breaks
  exact ties toward the stronger model. (Cheapest-first would make the free
  ``glm_4_flash`` the label on most queries even at lambda=0, so the classifier
  routers could never reach the high-quality end of the frontier.) For
  lambda > 0 the cost term already separates tied models.
* The router is built and trained with the library's own classes and trainers
  (constructor + ``*Trainer.train()``). Scoring calls the fitted model directly
  on our precomputed query embeddings, because the library's ``route_batch``
  re-embeds text with Longformer and then calls a live LLM API.

Output: ``<bundle>/llmrouter/<router>/seed<k>/lam<lambda>.npz`` with
``query_ids``, ``model_ids``, ``scores`` (n_eval x n_models; higher = route
there, ``-inf`` = never), for the validation/test/ood (or heldout) splits.
Resumable: an existing npz is skipped.
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
import yaml

LAMBDAS = (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)

HPARAMS = {
    # library / paper defaults; validation tuning is step 2 of the design
    "knn": {"n_neighbors": 5, "metric": "cosine", "algorithm": "brute"},
    "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "decision_function_shape": "ovr"},
    "mlp": {"hidden_layer_sizes": [128, 64], "activation": "relu", "lr": 1e-3,
            "epochs": 50, "batch_size": 32, "alpha": 1e-4},
    "mf": {"latent_dim": 128, "text_dim": 768, "lr": 1e-3, "epochs": 3, "batch_size": 64},
    "elo": {},
    "graphrouter": {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-4,
                    "train_epoch": 100, "batch_size": 4, "train_mask_rate": 0.3,
                    "val_split_ratio": 0.2},
    "routerdc": {"learning_rate": 1e-3, "top_k": 3, "last_k": 3, "temperature": 1.0,
                 "cluster_loss_weight": 1.0, "H": 3, "n_clusters": 3, "batch_size": 64,
                 "training_steps": 3000, "hidden_state_dim": 768},
}


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class Bundle:
    def __init__(self, path: str):
        self.path = path
        self.queries = pd.read_parquet(os.path.join(path, "queries.parquet"))
        self.emb = np.load(os.path.join(path, "embeddings.npy")).astype(np.float32)
        self.obs = pd.read_parquet(os.path.join(path, "observations.parquet"))
        self.models = pd.read_parquet(os.path.join(path, "models.parquet"))
        self.model_ids = self.models["model_id"].tolist()
        self.cost = dict(zip(self.models["model_id"], self.models["train_mean_cost"]))
        self.emb_id = dict(zip(self.queries["query_id"], self.queries["emb_id"]))
        eval_splits = [s for s in ("validation", "test", "ood", "heldout")
                       if (self.queries["split"] == s).any()]
        self.eval_q = self.queries[self.queries["split"].isin(eval_splits)]["query_id"].tolist()

    def train_rows(self, lam: float) -> pd.DataFrame:
        tq = set(self.queries.loc[self.queries["split"] == "train", "query_id"])
        r = self.obs[self.obs["query_id"].isin(tq)].copy()
        r["c"] = r["model_id"].map(self.cost)
        u = r["target"] - lam * r["c"]
        lo, hi = -lam * max(self.cost.values()), 1.0
        r["performance"] = (u - lo) / (hi - lo)
        r["strength"] = r["model_id"].map(r.groupby("model_id")["target"].mean())
        r = r.sort_values(["query_id", "strength", "model_id"], ascending=[True, False, True],
                          kind="stable")
        r["embedding_id"] = r["query_id"].map(self.emb_id)
        return r.rename(columns={"query_id": "query", "model_id": "model_name"})


def _write_inputs(b: Bundle, rows: pd.DataFrame, work: str) -> dict:
    """The files the library's DataLoader reads (absolute paths)."""
    torch.save(torch.from_numpy(b.emb), os.path.join(work, "query_emb.pt"))
    cols = ["query", "model_name", "performance", "embedding_id"]
    with open(os.path.join(work, "routing_train.jsonl"), "w", encoding="utf-8") as f:
        for rec in rows[cols].to_dict("records"):
            rec["embedding_id"] = int(rec["embedding_id"])
            f.write(json.dumps(rec) + "\n")
    return {
        "query_embedding_data": os.path.join(work, "query_emb.pt"),
        "routing_data_train": os.path.join(work, "routing_train.jsonl"),
    }


def _yaml(work: str, data_path: dict, hparam: dict, extra: dict | None = None) -> str:
    cfg = {"data_path": data_path, "hparam": hparam,
           "model_path": {"ini_model_path": os.path.join(work, "none.pkl"),
                          "save_model_path": os.path.join(work, "model.pkl"),
                          "load_model_path": os.path.join(work, "model.pkl")}}
    cfg.update(extra or {})
    p = os.path.join(work, "cfg.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return p


def _full(b: Bundle, classes, cls_scores: np.ndarray) -> np.ndarray:
    """Spread per-class scores over the full model list (-inf for never-best models)."""
    out = np.full((cls_scores.shape[0], len(b.model_ids)), -np.inf, dtype=np.float64)
    for j, m in enumerate(classes):
        out[:, b.model_ids.index(str(m))] = cls_scores[:, j]
    return out


def run_sklearn(kind, b, rows, work, hp):
    from llmrouter.models.knnrouter.router import KNNRouter
    from llmrouter.models.knnrouter.trainer import KNNRouterTrainer
    from llmrouter.models.svmrouter.router import SVMRouter
    from llmrouter.models.svmrouter.trainer import SVMRouterTrainer

    R, T = {"knn": (KNNRouter, KNNRouterTrainer), "svm": (SVMRouter, SVMRouterTrainer)}[kind]
    router = R(_yaml(work, _write_inputs(b, rows, work), hp))
    trainer = T(router)
    trainer.train()
    clf = trainer.model
    X = b.emb[[b.emb_id[q] for q in b.eval_q]]
    s = clf.predict_proba(X) if kind == "knn" else clf.decision_function(X)
    if s.ndim == 1:   # binary case
        s = np.stack([-s, s], axis=1)
    return _full(b, clf.classes_, s)


def run_mlp(b, rows, work, hp):
    from llmrouter.models.mlprouter.router import MLPRouter
    from llmrouter.models.mlprouter.trainer import MLPTrainer

    router = MLPRouter(_yaml(work, _write_inputs(b, rows, work), hp))
    trainer = MLPTrainer(router)
    trainer.train()
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(torch.from_numpy(b.emb[[b.emb_id[q] for q in b.eval_q]])).numpy()
    classes = [router.idx_to_model[i] for i in range(router.num_classes)]
    return _full(b, classes, logits)


def run_mf(b, rows, work, hp):
    from llmrouter.models.mfrouter.router import MFRouter
    from llmrouter.models.mfrouter.trainer import MFRouterTrainer

    router = MFRouter(_yaml(work, _write_inputs(b, rows, work), hp))
    trainer = MFRouterTrainer(router)
    trainer.train()
    model = trainer.model.eval()
    X = torch.from_numpy(b.emb[[b.emb_id[q] for q in b.eval_q]])
    with torch.no_grad():
        s = np.stack([model.score_all(model.project_text(x)).numpy() for x in X])
    classes = [router.idx_to_model[i] for i in range(len(router.idx_to_model))]
    return _full(b, classes, s)


def run_elo(b, rows, work, hp):
    from llmrouter.models.elorouter.router import EloRouter
    from llmrouter.models.elorouter.trainer import EloRouterTrainer
    from llmrouter.utils import load_model

    router = EloRouter(_yaml(work, _write_inputs(b, rows, work), hp))
    EloRouterTrainer(router).train()
    elo = dict(load_model(os.path.join(work, "model.pkl")))
    row = np.array([elo.get(m, -np.inf) for m in b.model_ids], dtype=np.float64)
    return np.tile(row, (len(b.eval_q), 1))


def run_graphrouter(b, rows, work, hp):
    from sklearn.preprocessing import MinMaxScaler
    from llmrouter.models.graphrouter.router import GraphRouter
    from llmrouter.models.graphrouter.trainer import GraphTrainer

    llm_data = {m: {"embedding": [float(v) for v in p]}
                for m, p in zip(b.models["model_id"], b.models["profile"])}
    cfg = _yaml(work, _write_inputs(b, rows, work), hp, {"llm_data": llm_data})
    with open(cfg, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["model_path"] = {k: v.replace("model.pkl", "model.pt") for k, v in c["model_path"].items()}
    with open(cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f)
    router = GraphRouter(cfg)
    GraphTrainer(router, device="cpu").train()
    router.gnn_predictor.model.load_state_dict(torch.load(os.path.join(work, "model.pt"), map_location="cpu"))

    # the library's route_batch graph (train queries visible, eval edges masked,
    # eval performance zeroed), but returning the edge scores instead of argmax
    test_emb = b.emb[[b.emb_id[q] for q in b.eval_q]]
    sc = MinMaxScaler().fit(np.vstack([router.query_embedding_list, test_emb]))
    allq = np.vstack([router.query_embedding_list, sc.transform(test_emb)])
    n_tr, n_te, L = router.num_queries_train, len(test_emb), router.num_llms
    nq = n_tr + n_te
    perf = np.concatenate([router.performance_list, np.zeros(n_te * L)])
    m_train = torch.zeros(nq * L); m_train[: n_tr * L] = 1
    m_test = torch.zeros(nq * L); m_test[n_tr * L:] = 1
    label = np.eye(L)[np.argmax(perf.reshape(nq, L), axis=1)].reshape(-1, 1)
    data = router.form_data.formulation(
        query_feature=allq, llm_feature=router.llm_embedding,
        org_node=[q for q in range(nq) for _ in range(L)], des_node=list(range(L)) * nq,
        edge_feature=perf, label=label, edge_mask=m_test, train_mask=m_train,
        valide_mask=torch.zeros(nq * L), test_mask=m_test)
    gp = router.gnn_predictor
    gp.model.eval()
    with torch.no_grad():
        e = gp.model(query_features=data.query_features, llm_features=data.llm_features,
                     edge_index=data.edge_index, edge_mask=data.edge_mask.clone().bool(),
                     edge_can_see=data.train_mask.clone().bool(), edge_weight=data.edge_attr)
    s = e.reshape(-1, L).numpy()[-n_te:]
    return _full(b, router.model_names, s)


def run_routerdc(b, rows, work, hp):
    """RouterDC with the text encoder frozen: the library's RouterModule and its
    dual-contrastive losses, over our fixed BERT embeddings plus a trainable
    768->768 projection (standing in for the fine-tuned DeBERTa, which needs a GPU).
    Clusters are k-means on the train embeddings, as in the library's preprocess."""
    from sklearn.cluster import KMeans
    from llmrouter.models.routerdc.dcmodel import RouterModule

    class FrozenBackbone(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.Tanh(), torch.nn.Linear(d, d))

        def forward(self, input_ids=None, attention_mask=None):
            return {"last_hidden_state": self.proj(input_ids).unsqueeze(1)}

    Y = rows.pivot(index="query", columns="model_name", values="performance")[b.model_ids]
    Y = Y.dropna()
    X = torch.from_numpy(b.emb[[b.emb_id[q] for q in Y.index]])
    T = torch.tensor(Y.to_numpy(), dtype=torch.float32)
    clusters = torch.tensor(KMeans(n_clusters=hp["n_clusters"], n_init=10, random_state=0)
                            .fit_predict(X.numpy()), dtype=torch.long)
    model = RouterModule(FrozenBackbone(X.shape[1]), hidden_state_dim=hp["hidden_state_dim"],
                         node_size=len(b.model_ids), similarity_function="cos")
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"])
    model.train()
    for step in range(hp["training_steps"]):
        idx = torch.randint(0, len(X), (hp["batch_size"],))
        scores, hidden = model(t=hp["temperature"], input_ids=X[idx])
        loss = model.compute_sample_llm_loss(scores, T[idx], hp["top_k"], hp["last_k"])
        if hp["cluster_loss_weight"] > 0:
            loss = loss + hp["cluster_loss_weight"] * model.compute_cluster_loss(
                hidden, clusters[idx], hp["temperature"], hp["H"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[routerdc] step {step} loss {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        s, _ = model(t=hp["temperature"], input_ids=torch.from_numpy(b.emb[[b.emb_id[q] for q in b.eval_q]]))
    return s.numpy().astype(np.float64)


RUNNERS = {
    "knn": lambda b, r, w, h: run_sklearn("knn", b, r, w, h),
    "svm": lambda b, r, w, h: run_sklearn("svm", b, r, w, h),
    "mlp": run_mlp, "mf": run_mf, "elo": run_elo,
    "graphrouter": run_graphrouter, "routerdc": run_routerdc,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--routers", default=",".join(RUNNERS))
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDAS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b = Bundle(args.bundle)
    for kind in args.routers.split(","):
        for lam in (float(x) for x in args.lambdas.split(",")):
            out = os.path.join(args.bundle, "llmrouter", kind, f"seed{args.seed}", f"lam{lam:g}.npz")
            if os.path.exists(out):
                continue
            _seed(args.seed)
            t0 = time.time()
            with tempfile.TemporaryDirectory() as work:
                scores = RUNNERS[kind](b, b.train_rows(lam), work, dict(HPARAMS[kind]))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(out, query_ids=np.array(b.eval_q), model_ids=np.array(b.model_ids),
                                scores=scores.astype(np.float32), seconds=time.time() - t0,
                                hparams=json.dumps(HPARAMS[kind]))
            print(f"[{kind}] lam={lam:g} seed={args.seed} done in {time.time() - t0:.0f}s -> {out}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Export a trained NIRT run as a gateway ``mlp`` router file.

With the candidate pool fixed, a ``query_latent`` / ``projected`` NIRT with a
``Linear -> ReLU -> Linear`` query head and scalar difficulty is exactly a
one-hidden-layer MLP (row-vector convention, ``y = x @ W.T + b``)::

    h     = relu(e @ W1.T + b1)
    theta = h @ W2.T + b2                 # [N, K]
    logit = theta @ A.T - B               # A [M, K], B [M]: the pool's (a_m, b_m)
          = h @ (A @ W2).T + (A @ b2 - B)

so it is saved in the existing ``mlp`` format (``w2' = A @ W2``,
``b2' = A @ b2 - B``) and the gateway serves it with numpy, no PyTorch. Any
architecture this does not hold for is refused, and every export is checked
against ``NIRTRouter``'s own scores.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PARITY_TOL = 1e-5


def check_foldable(model, run_cfg: dict | None = None) -> None:
    """Raise ``ValueError`` naming the first reason ``model`` is not exactly an MLP
    over a fixed pool. Checks the loaded module, not just the run's config."""
    from torch import nn

    from router.nirt.model import NIRTModel

    def refuse(why: str):
        raise ValueError(f"this NIRT can't be exported as a gateway router file: {why}")

    if not isinstance(model, NIRTModel):
        refuse(f"{type(model).__name__} (orientation={getattr(model, 'orientation', None)!r}); "
               "only query_latent NIRTModel folds")
    if model.model_params != "projected":
        refuse(f"model_params={model.model_params!r}; only 'projected' is supported")
    if model.interaction:
        refuse("interaction=True (the interaction residual is not bilinear)")
    if model.query_difficulty_on:
        refuse("query_difficulty=True")
    if model.difficulty != "scalar":
        refuse(f"difficulty={model.difficulty!r}; only 'scalar' is supported")
    if model.query_head_batchnorm:
        refuse("query_head_batchnorm=True")
    head = model.query_head
    if not (isinstance(head, nn.Sequential) and len(head) == 3
            and isinstance(head[0], nn.Linear) and type(head[1]) is nn.ReLU and isinstance(head[2], nn.Linear)):
        refuse(f"query head is {head!r}; only Linear -> ReLU -> Linear folds")
    if head[0].in_features != model.query_dim or head[2].out_features != model.dim \
            or head[0].out_features != head[2].in_features:
        refuse("query head shapes don't match query_dim / dim")
    dcfg = (run_cfg or {}).get("data", {}) or {}
    if dcfg.get("query_features"):
        refuse(f"query_features={dcfg['query_features']!r} (live text has no feature row)")
    if dcfg.get("query_pathway") not in (None, "", dcfg.get("pathway", "irt")):
        refuse(f"query_pathway={dcfg['query_pathway']!r} differs from pathway={dcfg.get('pathway')!r}")


def fold_nirt(model, refs: np.ndarray, pool_ref: np.ndarray | None = None,
              offsets: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """``{w1, b1, w2, b2}`` (float32) of the MLP equal to ``model`` scored against
    ``refs`` (the ``[M, p]`` profile matrix, one row per output column).
    ``offsets`` = ``(da [M, K], db [M])`` added to each pool member's ``(a_m, b_m)``
    -- the per-tier corrections learned by :mod:`training.nirt.judge_finetune`."""
    import torch

    from router.nirt.predict import pool_centering

    check_foldable(model)
    head = model.query_head
    model.eval()
    with torch.no_grad(), pool_centering(model, pool_ref):
        a, b = model.model_parameters(torch.from_numpy(np.ascontiguousarray(refs, dtype=np.float32)))
    A = a.double().numpy()
    B = b.double().numpy()
    if offsets is not None:
        A = A + np.asarray(offsets[0], np.float64)
        B = B + np.asarray(offsets[1], np.float64)
    M, K = len(refs), model.dim
    if A.shape != (M, K) or B.shape != (M,):
        raise ValueError(f"pool parameters have shape A{A.shape} B{B.shape}, expected A({M}, {K}) B({M},)")
    W2 = head[2].weight.detach().double().numpy()
    b2 = head[2].bias.detach().double().numpy()
    state = {"w1": head[0].weight.detach().numpy(), "b1": head[0].bias.detach().numpy(),
             "w2": A @ W2, "b2": A @ b2 - B}
    state = {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}
    hidden = head[0].out_features
    want = {"w1": (hidden, model.query_dim), "b1": (hidden,), "w2": (M, hidden), "b2": (M,)}
    for k, shape in want.items():
        if state[k].shape != shape:
            raise ValueError(f"folded {k} has shape {state[k].shape}, expected {shape}")
        if not np.isfinite(state[k]).all():
            raise ValueError(f"folded {k} has non-finite values")
    return state


def ensemble_fold(states: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """One MLP whose logit is the MEAN of ``states``' logits, exactly: the hidden
    layers side by side (``w1``/``b1`` stacked), each output layer scaled by
    ``1/n`` (``w2`` concatenated along the hidden axis), the output biases averaged."""
    if not states:
        raise ValueError("ensemble_fold needs at least one folded state")
    n = len(states)
    return {"w1": np.concatenate([s["w1"] for s in states], axis=0),
            "b1": np.concatenate([s["b1"] for s in states]),
            "w2": np.concatenate([s["w2"] for s in states], axis=1) / n,
            "b2": np.mean([s["b2"] for s in states], axis=0)}


def mlp_scores(state: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """The gateway's ``mlp`` forward: ``sigmoid(relu(x @ w1.T + b1) @ w2.T + b2)``."""
    h = np.maximum(x @ state["w1"].T + state["b1"], 0.0)
    return 1.0 / (1.0 + np.exp(-(h @ state["w2"].T + state["b2"])))


def _check(got: np.ndarray, want: np.ndarray, tol: float, where: str) -> float:
    diff = float(np.max(np.abs(np.asarray(got, np.float64) - want)))
    if not diff <= tol:
        raise ValueError(f"{where} disagrees with the NIRT run: max |dP| = {diff:.3g} > {tol:g}")
    return diff


def nirt_scores(router, e_q: np.ndarray) -> np.ndarray:
    """``router``'s own ``[N, M]`` P(correct) on ``e_q``, columns in ``router.model_ids`` order."""
    return router._score_embeddings(e_q)[router.model_ids].to_numpy(np.float64)


def parity_max_diff(state: dict[str, np.ndarray], want: np.ndarray, e_q: np.ndarray,
                    tol: float = PARITY_TOL) -> float:
    """Max |P(correct)| difference between the folded MLP and the NIRT's scores
    ``want`` on ``e_q``; raises if it exceeds ``tol``."""
    return _check(mlp_scores(state, np.asarray(e_q, dtype=np.float32)), want, tol, "the folded router")


def parity_max_diff_file(path, model_ids, e_q: np.ndarray, want: np.ndarray, tol: float = PARITY_TOL) -> float:
    """The same check on a saved file, loaded and scored by the gateway's own code."""
    from aar_gateway._core.scorers import load_router

    router = load_router(path)
    if router.model_ids != [str(m) for m in model_ids]:
        raise ValueError(f"{path}: model order {router.model_ids} != {list(model_ids)}")
    return _check(router._forward(np.asarray(e_q, dtype=np.float32)), want, tol, f"{path}")


def parity_embeddings(data, pathway: str, n: int = 500, seed: int = 0) -> np.ndarray:
    """Up to ``n`` test-split query embeddings (the train split if there is no test split)."""
    store = data.query_embeddings(pathway)
    if store is None:
        raise FileNotFoundError(f"query embeddings for pathway {pathway!r} not built")
    ids = data.split_query_ids("test") or data.split_query_ids("train") or set(store.ids)
    ids = sorted(i for i in ids if i in store._index)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    return np.stack([np.asarray(store.get(ids[i]), np.float32) for i in sorted(pick)])


#: encoder backends the gateway can load (``aar_gateway._core.encoder._BACKENDS``)
GATEWAY_BACKENDS = ("sentence_transformer", "fastembed")


def gateway_encoder_spec(enc_cfg, dim: int) -> dict:
    """The router file's ``encoder`` for a training encoder config. A raw ``bert``
    encoder has no fastembed model, so it is named as a LOCAL ONNX folder under the
    gateway's ``encoders/`` (``scripts/routing/export_bert_encoder.py`` writes it)."""
    spec = {"backend": enc_cfg.backend, "model_name": enc_cfg.model_name, "normalize": bool(enc_cfg.normalize),
            "max_seq_length": int(enc_cfg.max_seq_length or 0)}
    if enc_cfg.backend == "bert":
        folder = f"{enc_cfg.model_name.rstrip('/').split('/')[-1]}-{enc_cfg.pooling}"
        spec.update(backend="fastembed", model_name=f"aar/{folder}", model_path=folder,
                    pooling=enc_cfg.pooling, dim=int(dim))
    return spec


def check_servable_encoder(encoder: dict, encoders_dir=None) -> None:
    """Raise unless the gateway can load ``encoder``. Parity runs on precomputed
    embeddings, so it never exercises the gateway's encoder -- a file whose encoder
    the gateway can't load would pass every check and then route nothing.
    A local ONNX encoder (``model_path``) must already be exported under ``encoders_dir``."""
    backend, name = encoder.get("backend"), encoder.get("model_name")
    if encoder.get("model_path"):
        if encoders_dir is None:
            from aar_gateway.config import encoders_dir as gateway_encoders_dir

            encoders_dir = gateway_encoders_dir()
        path = Path(encoders_dir) / encoder["model_path"]
        if not (path / "model.onnx").is_file():
            raise ValueError(f"the gateway would look for encoder {name!r} in {path}, which has no model.onnx: "
                             "run scripts/routing/export_bert_encoder.py first")
        return
    if backend not in GATEWAY_BACKENDS:
        raise ValueError(f"the gateway can't load encoder backend {backend!r} (it serves {GATEWAY_BACKENDS}); "
                         "train the run on a pathway it can serve, e.g. data.pathway=retrieval")
    try:
        from fastembed import TextEmbedding
    except ImportError:          # the gateway then serves sentence_transformer as is
        return
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if name not in supported:    # with fastembed installed the gateway always uses it
        raise ValueError(f"fastembed has no model {name!r}, so the gateway can't encode prompts for this file")


def export_nirt_router(data, run_name: str, *, dataset: str = "", runs_dir=None, n_check: int = 500,
                       pool_ids=None):
    """Fold ``run_name`` into the gateway's ``mlp`` format and check it. Returns
    ``(payload, encoder, model_costs, header, (e_q, want))``: the first four for
    :meth:`~router.models.artifacts.MLPArtifactPayload.save`, then the parity
    check's embeddings and NIRT scores, for :func:`parity_max_diff_file`.

    ``pool_ids`` folds over that pool instead of the run's trained one: each
    model's profile from the store, else its curated entry in
    ``configs/model_profiles.yaml`` embedded with the run's own encoder
    (:meth:`~router.routing.routers.NIRTRouter.resolve_pool_profile_matrix`), so
    unseen models (the Claude tiers) are scored from their own profiles, and the
    file's columns are named by ``pool_ids``. The pool is centred on itself, as
    ``NIRTRouter.predict_pool_scores_text`` does."""
    from router.config import DEFAULT_NIRT_RUNS_DIR, REPO_ROOT
    from router.embeddings.encoder import load_encoder
    from router.models.artifacts import MLPArtifactPayload
    from router.nirt.checkpoint import load_run
    from router.routing.routers import NIRTRouter, _mean_train_costs

    model, cfg, model_index = load_run(run_name, runs_dir=runs_dir)
    check_foldable(model, cfg)
    pathway = (cfg.get("data", {}) or {}).get("pathway", "irt")
    encoder = gateway_encoder_spec(load_encoder(data.cfg, pathway).cfg, dim=int(model.query_dim))
    check_servable_encoder(encoder)
    router = NIRTRouter(model, model_index, data=data, pathway=pathway, name=f"nirt:{run_name}")
    e_q = parity_embeddings(data, pathway, n=n_check)
    if pool_ids:
        model_ids = [str(m) for m in pool_ids]
        refs = router.resolve_pool_profile_matrix(model_ids)
        state = fold_nirt(model, refs, refs)
        want = router._forward_pool(e_q, refs, refs)
    else:
        model_ids = router.model_ids
        refs, pool_ref = router._pool_refs()
        if len(refs) != len(model_ids):
            raise ValueError(f"{len(refs)} profile rows for {len(model_ids)} models")
        state = fold_nirt(model, refs, pool_ref)
        want = nirt_scores(router, e_q)
    diff = parity_max_diff(state, want, e_q)

    payload = MLPArtifactPayload(
        state_dict={"0.weight": state["w1"], "0.bias": state["b1"], "3.weight": state["w2"], "3.bias": state["b2"]},
        model_ids=[str(m) for m in model_ids], in_dim=int(state["w1"].shape[1]), hidden=int(state["w1"].shape[0]),
        dropout=0.0, pathway=pathway)
    # an ad hoc pool has no training observations to average; the gateway prices tiers itself
    costs = None if pool_ids else _mean_train_costs(data, model_ids)
    run_json = Path(REPO_ROOT / DEFAULT_NIRT_RUNS_DIR if runs_dir is None else runs_dir) / run_name / "run.json"
    git_sha = json.loads(run_json.read_text()).get("git_sha") if run_json.exists() else None
    header = {"artifact_id": f"nirt-{run_name}", "dataset": dataset, "source_run": run_name,
              "source_git_sha": git_sha, "parity_max_abs_diff": diff, "parity_n": int(len(e_q)),
              "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    return payload, encoder, ([float(c) for c in costs] if costs is not None else None), header, (e_q, want)

"""Train the Phase 1 plain Bernoulli / BCE NIRT baseline.

    from router.nirt.baseline.train import fit
    res = fit(yaml.safe_load(open("configs/phase1.yaml")))

Phase 1 facade -> ``build_arrays`` (materialised numpy, CPU RAM, never a block on
GPU) -> ``DataLoader`` -> Adam + early stop on validation BCE -> checkpoint.
Correctness signal only (Arena excluded); binary targets.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from router.config import Config, load_config
from router.determinism import seed_everything
from training.nirt.metrics import log_loss, prediction_metrics
from router.provenance import file_digest

from .checkpoint import load_checkpoint, save_checkpoint
from .data import (attach_cost_offset, batched_forward, build_arrays, cost_offset_table, log_cost,
                   to_tensors)
from .losses import RegConfig, bce_loss, class_balance_pos_weight, regularization
from .model import build_baseline_model

_DEFAULT_DIR = "artifacts/phase1/baseline"


@dataclass
class BaselineRun:
    config: dict
    history: list = field(default_factory=list)
    best_epoch: int = -1
    val_metrics: dict = field(default_factory=dict)
    train_imbalance: dict = field(default_factory=dict)
    model: object = None
    model_index: dict = field(default_factory=dict)
    dims: dict = field(default_factory=dict)
    path: Optional[Path] = None


@dataclass
class TrainCfg:
    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4096
    epochs: int = 60
    patience: int = 8
    device: str = "cpu"
    target: str = "binary"          # {binary, soft}
    class_weighting: bool = False    # BCEWithLogits pos_weight = #neg/#pos
    grad_clip: float = 0.0           # max grad-norm (0 = off); applied only for continuous heads

    @classmethod
    def from_dict(cls, d: dict) -> "TrainCfg":
        d = d or {}

        def g(k, alt=None):
            return d.get(k, d.get(alt, getattr(cls, k)))

        return cls(
            seed=int(d.get("seed", 42)), lr=float(g("lr", "learning_rate")),
            weight_decay=float(g("weight_decay")), batch_size=int(g("batch_size")),
            epochs=int(g("epochs")), patience=int(g("patience")), device=str(g("device")),
            target=str(g("target")), class_weighting=bool(g("class_weighting")),
            grad_clip=float(g("grad_clip")),
        )


def fit(
    phase1_cfg: dict, *, phase0_cfg: Optional[Config] = None, arrays: Optional[tuple] = None,
    save: bool = True, out_dir: Optional[str] = None, resume_from: Optional[str] = None,
    verbose: bool = True,
) -> BaselineRun:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = json.loads(json.dumps(phase1_cfg))
    tcfg = TrainCfg.from_dict({**cfg.get("train", {}), "seed": cfg.get("seed", 42)})
    mcfg = cfg.get("model", {}) or {}
    abl = cfg.get("ablation", {}) or {}
    rcfg = RegConfig.from_dict(cfg.get("regularization"))
    resp = cfg.get("response", {}) or {}
    use_relevance = bool(abl.get("use_relevance", mcfg.get("use_relevance", True)))
    use_warmup = bool(abl.get("use_warmup", mcfg.get("use_warmup", False)))
    use_interaction = bool(abl.get("use_interaction", mcfg.get("use_interaction", True)))
    model_params = mcfg.get("model_params", "free")
    seed_everything(tcfg.seed)

    p0 = phase0_cfg or load_config()
    d = None
    if arrays is not None:
        tr_arr, va_arr = arrays
    else:
        from training.data.facade import load_training_data

        d = load_training_data(p0)
        common = dict(data=d, pathway=cfg.get("data", {}).get("pathway", "irt"),
                      binary_threshold=float(resp.get("binary_threshold", 0.5)),
                      score_kind=resp.get("score_kind", "effective"),
                      use_relevance=use_relevance, use_warmup=use_warmup)
        tr_arr = build_arrays(p0, split="train", **common)
        va_arr = build_arrays(p0, split="validation", model_index=tr_arr.meta["model_index"], **common)

    # Beta: interior observations only -- filtered BEFORE building the model so
    # the cost probe's init statistic is taken over exactly the rows it trains on
    from .response_head import _registry

    response_model = (resp.get("model", mcfg.get("response_model", "bernoulli")) or "bernoulli").lower()
    if getattr(_registry().get(response_model), "interior_only", False):
        tr_arr, va_arr = _interior(tr_arr), _interior(va_arr)

    ccfg = cfg.get("cost", {}) or {}
    use_cost = bool(ccfg.get("enabled", False))
    cost_weight = float(ccfg.get("weight", 1.0))
    if use_cost and bool(ccfg.get("model_price_features", False)):
        tr_arr, va_arr = _with_price_feats(tr_arr, p0), _with_price_feats(va_arr, p0)
    cost_mode = str(ccfg.get("mode", "log_cost") or "log_cost")
    if use_cost and cost_mode == "length" and tr_arr.out_tokens is None:
        if d is None:
            raise ValueError("cost.mode=length with explicit arrays= needs in_tokens / "
                             "out_tokens on them (training.nirt.baseline.length.attach_tokens)")
        from training.data.tokens import cell_token_table

        from .length import attach_tokens

        tokens = cell_token_table(d, p0)
        tr_arr, va_arr = attach_tokens(tr_arr, tokens), attach_tokens(va_arr, tokens)

    model_index = tr_arr.meta["model_index"]
    dims = {"query_dim": tr_arr.query_dim, "profile_dim": tr_arr.profile_dim,
            "relevance_dim": tr_arr.relevance_dim if use_relevance else 0,
            "n_models": len(model_index)}
    # like relevance (off when relevance_dim == 0): warm-up requested but no
    # warm-up store at training time -> bake it OFF, or a later eval would blend
    # neighbour means the model was never trained with
    if use_warmup and tr_arr.nbr is None:
        if verbose:
            print("[phase1] use_warmup requested but no warm-up store -- training without it")
        use_warmup = False

    # bake the resolved ablation + response model into the model cfg so a
    # checkpoint rebuilds the exact same architecture.
    binary_threshold = float(resp.get("binary_threshold", tr_arr.meta.get("binary_threshold", 0.5)))
    mcfg = {**mcfg, "ablation": {"use_relevance": use_relevance, "use_interaction": use_interaction,
                                 "use_warmup": use_warmup},
            "response_model": resp.get("model", mcfg.get("response_model", "bernoulli")),
            "response_cfg": {**(resp.get("cfg", mcfg.get("response_cfg", {})) or {}),
                             "binary_threshold": binary_threshold}}
    if use_cost:
        mcfg = {**mcfg, "use_cost_head": True, "cost_hidden": ccfg.get("hidden", 64),
                "cost_price_dim": 2 if tr_arr.price_feats is not None else 0}
        if cost_mode == "length":
            from .length import attach_length_inputs, fit_length_tables

            # bucket values, per-model priors and token prices: TRAIN split only
            mcfg = {**mcfg, "cost_mode": "length", **fit_length_tables(tr_arr)}
            tr_arr, va_arr = attach_length_inputs(tr_arr, mcfg), attach_length_inputs(va_arr, mcfg)
        elif bool(ccfg.get("residual", False)):
            import numpy as np

            # baseline log C(m) from the TRAIN split only, frozen into the checkpoint
            mcfg = {**mcfg, "cost_residual": True,
                    "cost_offsets": cost_offset_table(tr_arr.model_ids, tr_arr.cost)}
            tr_arr, va_arr = attach_cost_offset(tr_arr, mcfg), attach_cost_offset(va_arr, mcfg)
            res = log_cost(tr_arr.cost, len(tr_arr.y)) - tr_arr.cost_offset
            mcfg["cost_init_mu"] = float(np.nanmean(res.astype(np.float64)))   # ~0 by construction
        else:
            mcfg["cost_init_mu"] = train_log_cost_mean(tr_arr)
    cfg["model"] = mcfg
    model = build_baseline_model(
        mcfg, n_models=dims["n_models"], query_dim=dims["query_dim"],
        profile_dim=dims["profile_dim"] or dims["query_dim"], relevance_dim=dims["relevance_dim"],
    ).to(tcfg.device)
    if resume_from:
        model.load_state_dict(load_checkpoint(resume_from, map_location=tcfg.device)[0].state_dict())
        if verbose:
            print(f"[phase1] resumed weights from {resume_from}")

    head = model.response_head
    bernoulli = model.response_model == "bernoulli"

    tt = dict(model_params=model_params, use_relevance=use_relevance, use_warmup=use_warmup)
    tr, va = to_tensors(tr_arr, **tt), to_tensors(va_arr, **tt)
    zeros = torch.zeros(len(tr["y"]), 1)
    cols = [tr["e_q"], tr["ref"], tr["y"], tr["y_soft"],
            tr["r_q"] if tr["r_q"] is not None else zeros,
            tr["nbr"] if tr["nbr"] is not None else zeros,
            tr["log_cost"],
            tr["price_feats"] if tr["price_feats"] is not None else zeros,
            tr["cost_offset"] if tr["cost_offset"] is not None else zeros,
            tr["length_label"] if tr["length_label"] is not None else zeros,
            tr["length_prior"] if tr["length_prior"] is not None else zeros]
    # extra columns don't change the shuffle: the sampler permutes indices only
    loader = DataLoader(TensorDataset(*cols), batch_size=tcfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(tcfg.seed))
    has_r, has_n = tr["r_q"] is not None, tr["nbr"] is not None
    has_p = tr["price_feats"] is not None
    has_o = tr["cost_offset"] is not None
    has_lp = use_cost and cost_mode == "length"

    pos_weight = (class_balance_pos_weight(tr["y"]).to(tcfg.device)
                  if tcfg.class_weighting else None)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    # identifiability regularisers touch only the ability rows we actually fit
    # (a model absent from training gets no gradient -> no cold-start leakage).
    # free: one theta row per training model. projected: one PRE-sigmoid
    # W_theta e_m row per distinct training profile -- per model (not weighted
    # by observation count), and unbounded, so the zero-mean convention is reachable.
    train_rows = torch.unique(tr["ref"]) if model_params == "free" else None
    train_profiles = (torch.unique(tr["ref"], dim=0).float().to(tcfg.device)
                      if model_params == "projected" else None)

    history: list = []
    best_val, best_epoch, best_state, since = float("inf"), -1, None, 0
    t0 = time.time()

    # the head declares its target column; train.target: soft is a Bernoulli-only override
    soft_target = head.target == "soft" or (bernoulli and tcfg.target == "soft")
    for epoch in range(tcfg.epochs):
        model.train()
        running = seen = 0.0
        for e_q, ref, y, y_soft, r_q, nbr, log_c, pf, co, lab, lp in loader:
            e_q, ref = e_q.to(tcfg.device), ref.to(tcfg.device)
            target = (y_soft if soft_target else y).to(tcfg.device)
            out = model(e_q, ref, r_q.to(tcfg.device) if has_r else None,
                        nbr.to(tcfg.device) if has_n else None,
                        pf.to(tcfg.device) if has_p else None,
                        co.to(tcfg.device) if has_o else None,
                        lp.to(tcfg.device) if has_lp else None)
            loss = (bce_loss(out.logit, target, pos_weight=pos_weight) if bernoulli
                    else head.loss(target, out.response))
            theta_all = (model.theta.weight[train_rows] if model_params == "free"
                         else model.theta_proj(train_profiles))
            reg = regularization(rcfg, theta_all=theta_all, a_q=out.a_q, b_q=out.b_q).to(tcfg.device)
            total = loss + reg
            if use_cost and cost_mode == "length":
                total = total + cost_weight * _masked_length_loss(
                    model.cost_head, lab.to(tcfg.device), out.length_logits)
            elif use_cost:
                total = total + cost_weight * _masked_cost_loss(
                    model.cost_head, log_c.to(tcfg.device), out.cost_mu, out.cost_log_sigma,
                    beta=float(ccfg.get("beta", 0.5)))
            opt.zero_grad()
            total.backward()
            if not bernoulli and tcfg.grad_clip > 0:
                _clip_grads(model, tcfg.grad_clip)
            opt.step()
            running += float(loss.item()) * len(y)
            seen += len(y)
        train_loss = running / max(seen, 1)

        val_prob = batched_forward(model, va, device=tcfg.device)["proba"]
        vm = prediction_metrics(va_arr.y, val_prob)
        if not bernoulli:
            val_nll = _mean_nll(model, va, va_arr, device=tcfg.device)
        elif soft_target:
            # select on the objective actually optimised: soft BCE against y_soft
            val_nll = log_loss(va_arr.y_soft, val_prob)
        else:
            val_nll = vm["bce"]
        stop = val_nll
        row = {"epoch": epoch, "train_loss": train_loss, "val_nll": val_nll,
               **{f"val_{k}": v for k, v in vm.items() if isinstance(v, (int, float))}}
        if use_cost:
            row.update({f"val_{k}": v for k, v in
                        _cost_metrics(model, va, va_arr, device=tcfg.device).items()})
        history.append(row)
        if verbose:
            print(f"[phase1] epoch {epoch:3d}  train_loss {train_loss:.4f}  val_nll {val_nll:.4f}  "
                  f"val_acc {vm['accuracy']:.4f}  val_brier {vm['brier']:.4f}")

        if stop < best_val - 1e-5:
            best_val, best_epoch, since = stop, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= tcfg.patience:
                if verbose:
                    print(f"[phase1] early stop at epoch {epoch} (best {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if use_cost:
        # the joint phase's cost head is rolled back with the quality-best state;
        # finish fitting it as a probe of THAT frozen representation
        probe_hist = _fit_cost_probe(model, tr, va, tr_arr, va_arr, ccfg, tcfg)
        if history:
            history[-1]["cost_probe"] = probe_hist
    val_metrics = prediction_metrics(va_arr.y, batched_forward(model, va, device=tcfg.device)["proba"])
    if not bernoulli:
        val_metrics["nll"] = _mean_nll(model, va, va_arr, device=tcfg.device)
    if use_cost:
        val_metrics.update(_cost_metrics(model, va, va_arr, device=tcfg.device))

    run = BaselineRun(config=cfg, history=history, best_epoch=best_epoch, val_metrics=val_metrics,
                      train_imbalance=tr_arr.imbalance_report(), model=model,
                      model_index=model_index, dims=dims)
    if save:
        out = Path(out_dir or cfg.get("out_dir", _DEFAULT_DIR))
        out = out if out.is_absolute() else p0.root / out
        save_checkpoint(
            out, model=model, config=cfg, model_index=model_index, query_dim=dims["query_dim"],
            profile_dim=dims["profile_dim"] or dims["query_dim"], relevance_dim=dims["relevance_dim"],
            metrics={"validation": val_metrics, "best_epoch": best_epoch,
                     "train_imbalance": run.train_imbalance},
            history=history, provenance=_provenance(p0, cfg, tr_arr, va_arr, tcfg.seed))
        run.path = out
        if verbose:
            print(f"[phase1] wrote {out} ({round(time.time() - t0, 1)}s)")
    return run


def _interior(arr):
    """Rows with ``0 < y_soft < 1`` (Beta support)."""
    from dataclasses import replace

    m = (arr.y_soft > 0.0) & (arr.y_soft < 1.0)
    sub = {f: getattr(arr, f)[m] for f in
           ("e_q", "midx", "y", "y_soft", "query_ids", "model_ids")}
    for f in ("e_m", "r_q", "nbr", "cost", "price_feats", "cost_offset", "in_tokens", "out_tokens",
              "length_prior", "token_price"):
        v = getattr(arr, f)
        sub[f] = v[m] if v is not None else None
    return replace(arr, **sub)


def train_log_cost_mean(arr) -> float:
    """Mean finite log-cost over ``arr`` -- the cost probe's init location. Callers
    pass the TRAIN split only."""
    import numpy as np

    lc = log_cost(arr.cost, len(arr.y))
    lc = lc[np.isfinite(lc)]
    if not len(lc):
        raise ValueError("cost.enabled but the train split has no finite positive cost")
    return float(lc.astype(np.float64).mean())


def _fit_cost_probe(model, tr: dict, va: dict, tr_arr, va_arr, ccfg: dict, tcfg) -> list:
    """Train ONLY the cost head on the frozen quality representation, early-stopped
    on validation cost NLL (``cost.probe_epochs`` / ``cost.probe_patience``).
    Quality parameters are not in the optimizer, so they cannot move. Features
    are recomputed per batch (no_grad) rather than materialised: a full copy is
    ``N x (query_dim + K)`` floats, several GB on the real train split."""
    import torch

    dev = tcfg.device
    head = model.cost_head

    length = getattr(model, "cost_mode", "log_cost") == "length"

    def feats(t, idx):
        return model.cost_features(
            t["e_q"][idx].to(dev), t["ref"][idx].to(dev),
            t["nbr"][idx].to(dev) if t["nbr"] is not None else None,
            t["price_feats"][idx].to(dev) if t["price_feats"] is not None else None)

    if length:
        tr_rows = torch.nonzero(tr["length_label"] >= 0).squeeze(1)
        va_rows = torch.nonzero(va["length_label"] >= 0).squeeze(1)
        y_tr, y_va = tr["length_label"].to(dev), va["length_label"].to(dev)
    else:
        tr_rows = torch.nonzero(torch.isfinite(tr["log_cost"])).squeeze(1)
        va_rows = torch.nonzero(torch.isfinite(va["log_cost"])).squeeze(1)
        y_tr, y_va = tr["log_cost"].to(dev), va["log_cost"].to(dev)
        if tr.get("cost_offset") is not None:      # residual mode: the head fits log c - log C(m)
            y_tr = y_tr - tr["cost_offset"].to(dev)
            y_va = y_va - va["cost_offset"].to(dev)
    if not len(tr_rows) or not len(va_rows):
        return []

    def row_loss(t, y, idx, *, train: bool):
        x = feats(t, idx)
        yi = y[idx.to(dev)]
        if length:
            return head.nll(yi, head(x, t["length_prior"][idx].to(dev)))
        mu, ls = head(x)
        return head.train_loss(yi, mu, ls, beta=beta) if train else head.nll(yi, mu, ls)

    # streaming mean / std of the train features (float64 accumulators)
    s1 = s2 = None
    for idx in tr_rows.split(65536):
        x = feats(tr, idx).double()
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
    n = len(tr_rows)
    mean = s1 / n
    std = ((s2 / n - mean * mean).clamp_min(0) * n / max(n - 1, 1)).sqrt()
    head.set_standardization(mean.float(), std.float())

    opt = torch.optim.Adam(head.parameters(), lr=float(ccfg.get("probe_lr", tcfg.lr)))
    gen = torch.Generator().manual_seed(tcfg.seed)
    bs = int(ccfg.get("probe_batch_size", tcfg.batch_size))
    patience = int(ccfg.get("probe_patience", 5))
    beta = float(ccfg.get("beta", 0.5))

    def val_nll() -> float:
        tot = 0.0
        with torch.no_grad():
            for idx in va_rows.split(65536):
                tot += float(row_loss(va, y_va, idx, train=False).sum())
        return tot / len(va_rows)

    best, best_state, since, hist = val_nll(), None, 0, []
    for epoch in range(int(ccfg.get("probe_epochs", 50))):
        head.train()
        for b in torch.randperm(n, generator=gen).split(bs):
            loss = row_loss(tr, y_tr, tr_rows[b], train=True).mean()
            opt.zero_grad()
            loss.backward()
            if tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), tcfg.grad_clip)
            opt.step()
        head.eval()
        v = val_nll()
        hist.append({"epoch": epoch, "val_cost_nll": v})
        if v < best - 1e-5:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return hist


def _masked_cost_loss(cost_head, log_c, mu, log_sigma, beta: float = 0.5):
    """Mean cost NLL over rows with a finite label; exactly 0 (graph-connected,
    NaN-free grads) when the batch has none."""
    import torch

    ok = torch.isfinite(log_c)
    if not bool(ok.any()):
        return mu.sum() * 0.0
    return cost_head.train_loss(log_c[ok], mu[ok], log_sigma[ok], beta=beta).mean()


def _clip_grads(model, max_norm: float) -> None:
    """Clip the quality params and the cost probe SEPARATELY: one global norm
    would let cost gradients rescale the quality step."""
    import torch

    torch.nn.utils.clip_grad_norm_(model.quality_parameters(), max_norm)
    cost = model.cost_parameters()
    if cost:
        torch.nn.utils.clip_grad_norm_(cost, max_norm)


def _masked_length_loss(head, labels, logits):
    """Mean bucket cross-entropy over rows with an output-token label; exactly 0
    (graph-connected) when the batch has none."""
    ok = labels.long() >= 0
    if not bool(ok.any()):
        return logits.sum() * 0.0
    return head.nll(labels[ok], logits[ok]).mean()


def _length_metrics(model, tensors: dict, arr, *, device: str = "cpu") -> dict:
    """Length classifier quality (``length_ce``, ``length_acc``, ``length_mae_log``:
    |log1p E[L] - log1p L|) and the priced expected cost against REALIZED cost
    (``cost_mae_log`` / ``cost_spearman`` over rows with a price and a cost)."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    fields = ("length_logits", "expected_length") + (
        ("cost",) if tensors.get("token_price") is not None else ())
    f = batched_forward(model, tensors, fields=fields, device=device)
    lab = tensors["length_label"].numpy()
    ok = lab >= 0
    out = {"length_ce": float("nan"), "length_acc": float("nan"), "length_mae_log": float("nan"),
           "cost_mae_log": float("nan"), "cost_spearman": float("nan")}
    if ok.any():
        logits = torch.from_numpy(f["length_logits"][ok])
        out["length_ce"] = float(model.cost_head.nll(torch.from_numpy(lab[ok]), logits).mean())
        out["length_acc"] = float((f["length_logits"][ok].argmax(1) == lab[ok]).mean())
        out["length_mae_log"] = float(np.abs(np.log1p(f["expected_length"][ok])
                                             - np.log1p(np.asarray(arr.out_tokens)[ok])).mean())
    if "cost" in f:
        c = np.asarray(arr.cost, np.float64)
        p = f["cost"]
        okc = np.isfinite(p) & np.isfinite(c) & (p > 0) & (c > 0)
        if okc.sum() > 1:
            out["cost_mae_log"] = float(np.abs(np.log(p[okc]) - np.log(c[okc])).mean())
            out["cost_spearman"] = float(spearmanr(p[okc], c[okc]).correlation)
    return out


def _cost_metrics(model, tensors: dict, arr, *, device: str = "cpu") -> dict:
    """``cost_nll`` / ``cost_mae_log`` / ``cost_spearman`` of the probe against the
    REALIZED per-cell cost (rows with a finite label)."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    if getattr(model, "cost_mode", "log_cost") == "length":
        return _length_metrics(model, tensors, arr, device=device)

    f = batched_forward(model, tensors, fields=("cost_mu", "cost_log_sigma"), device=device)
    lc = log_cost(arr.cost, len(arr.y))
    ok = np.isfinite(lc)
    if ok.sum() < 2:
        return {"cost_nll": float("nan"), "cost_mae_log": float("nan"), "cost_spearman": float("nan")}
    mu, ls = f["cost_mu"][ok], f["cost_log_sigma"][ok]
    nll = model.cost_head.nll(torch.from_numpy(lc[ok]), torch.from_numpy(mu), torch.from_numpy(ls))
    return {"cost_nll": float(nll.mean()), "cost_mae_log": float(np.abs(mu - lc[ok]).mean()),
            "cost_spearman": float(spearmanr(mu, lc[ok]).correlation)}


def _with_price_feats(arr, cfg: Config):
    """Attach ``[log input $/1M, log output $/1M]`` per row (cost-probe ablation).
    Every model in ``arr`` must be priced in ``configs/model_prices.yaml``."""
    from dataclasses import replace

    import numpy as np

    from training.data.pricing import load_prices

    prices = load_prices(cfg)
    missing = sorted({str(m) for m in np.unique(arr.model_ids) if str(m) not in prices})
    if missing:
        raise ValueError(f"cost.model_price_features: no price for {missing} in "
                         "configs/model_prices.yaml (RouterBench models ship a precomputed "
                         "cost and are intentionally unpriced there)")
    pf = np.array([[np.log(float(prices[str(m)]["input_per_1m"]) + 1e-6),
                    np.log(float(prices[str(m)]["output_per_1m"]) + 1e-6)] for m in arr.model_ids],
                  dtype=np.float32).reshape(-1, 2)
    return replace(arr, price_feats=pf)


def _mean_nll(model, tensors: dict, arr, *, device: str = "cpu") -> float:
    return float(batched_forward(model, tensors, fields=("nll",), y=arr.y_soft,
                                 device=device)["nll"].mean())


def _provenance(cfg: Config, phase1_cfg: dict, tr_arr, va_arr, seed: int) -> dict:
    def _hash(p: Path) -> Optional[str]:
        return file_digest(p, algo="sha1")

    proc, tax = cfg.path("processed"), cfg.path("taxonomy")
    return {
        "seed": seed,
        "phase1_config": phase1_cfg,
        "phase0_artifacts": {
            k: _hash(base / f)
            for k, (base, f) in {
                "nirt_observations": (proc, "nirt_observations.parquet"),
                "queries": (proc, "queries.parquet"),
                "model_profiles": (proc, "model_profiles.parquet"),
                "clusters": (tax, "clusters.parquet"),
                "centroids": (tax, "centroids.npy"),
            }.items()
        },
        "taxonomy_version": int(cfg.get("taxonomy.version", 1)),
        "dataset_stats": {
            "train": tr_arr.imbalance_report(),
            "validation_n": int(len(va_arr)),
            "binary_threshold": tr_arr.meta["binary_threshold"],
            "score_kind": tr_arr.meta["score_kind"],
            "relevance_available": tr_arr.meta["relevance_available"],
            "warmup_available": tr_arr.meta["warmup_available"],
        },
    }

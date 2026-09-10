"""Phase 2 continuous-response evaluation + boundary statistics + comparison.

Reuses the Phase 1 representation diagnostics (``theta_spectrum``,
``parameter_summary``, cold-start) and adds proper-likelihood / mean-error /
calibration / interval-coverage / boundary-calibration metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from router.config import Config, load_config

from .checkpoint import load_run
from .data import batched_forward, build_arrays
from .eval import _per_group, family_labels, theta_matrix
from .diagnostics import _write_json, parameter_summary, plot_theta_spectrum, pyplot, savefig, \
    save_theta_spectrum, theta_spectrum
from router.nirt.metrics import prediction_metrics, reliability_curve


# --------------------------------------------------------------------------- #
# boundary statistics                                                         #
# --------------------------------------------------------------------------- #
def boundary_statistics(cfg: Optional[Config] = None, *, splits=("train", "validation", "test"),
                        write: bool = True) -> dict:
    cfg = cfg or load_config()
    from router.data.phase1 import load_phase1

    d = load_phase1(cfg)
    out: dict = {"by_split": {}, "by_model": {}}
    for sp in splits:
        long = d.correctness(split=sp, models="warm", score_kind="effective")
        y = np.clip(long["score"].to_numpy(np.float64), 0, 1)
        out["by_split"][sp] = _frac(y)
        if sp == "test":
            for m, g in long.groupby("model_id"):
                out["by_model"][str(m)] = _frac(np.clip(g["score"].to_numpy(np.float64), 0, 1))
    out["note"] = "fractions of the graded RouterBench correctness score at exactly 0, exactly 1, " \
                  "and strictly interior (0,1). Determines whether ZOIB's boundary masses matter."
    if write:
        (cfg.root / "artifacts/phase2").mkdir(parents=True, exist_ok=True)
        (cfg.root / "artifacts/phase2/response_boundary_statistics.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
    return out


def _frac(y: np.ndarray) -> dict:
    n = max(len(y), 1)
    return {"n": int(len(y)), "y==0": float((y <= 0).mean()), "y==1": float((y >= 1).mean()),
            "0<y<1": float(((y > 0) & (y < 1)).mean()), "mean": float(y.mean())}


# --------------------------------------------------------------------------- #
# continuous metrics                                                          #
# --------------------------------------------------------------------------- #
def continuous_metrics(y, mean, proba, nll, lower=None, upper=None, lower50=None, upper50=None,
                       interior_mask=None) -> dict:
    y = np.asarray(y, np.float64)
    mean = np.asarray(mean, np.float64)
    err = mean - y
    m = np.ones(len(y), bool) if interior_mask is None else np.asarray(interior_mask)
    out = {
        "nll_mean": float(np.mean(nll[m])),
        "nll_median": float(np.median(nll[m])),
        "nll_finite": bool(np.isfinite(nll[m]).all()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_pred": float(mean.mean()),
        "observed_mean": float(y.mean()),
        # secondary BCE-after-threshold diagnostic (NOT the training objective)
        **{f"bce_diag_{k}": v for k, v in prediction_metrics((y >= 0.5).astype(float), proba).items()
           if k in ("bce", "accuracy", "brier", "auc", "ece")},
    }
    rc = reliability_curve(y, np.clip(mean, 0, 1), n_bins=15)
    out["mean_calibration_ece"] = rc["ece"]
    out["mean_calibration_mce"] = rc["mce"]
    out["_reliability"] = rc["bins"]
    if lower is not None:
        out["coverage_90"] = float(np.mean((y >= lower) & (y <= upper)))
        out["interval_width_90"] = float(np.mean(np.asarray(upper) - np.asarray(lower)))
    if lower50 is not None:
        out["coverage_50"] = float(np.mean((y >= lower50) & (y <= upper50)))
    return out


def evaluate_continuous(directory: str | Path, *, phase0_cfg: Optional[Config] = None,
                        split: str = "test", level: float = 0.9, write: bool = True) -> dict:
    d = Path(directory)
    cfg = phase0_cfg or load_config()
    model, blob, s = load_run(d)
    mi = blob["model_index"]
    head = model.response_head
    key = model.response_model

    ev = build_arrays(cfg, split=split, pathway=s["pathway"], binary_threshold=s["binary_threshold"],
                      score_kind=s["score_kind"], use_relevance=model.use_relevance,
                      use_warmup=model.use_warmup, model_index=mi)

    fields = ["mean", "proba", "nll", "lower", "upper", "std"]
    fw = batched_forward(model, ev, fields=tuple(fields), y=ev.y_soft, level=level)
    fw50 = batched_forward(model, ev, fields=("lower", "upper"), level=0.5)
    interior = (ev.y_soft > 0) & (ev.y_soft < 1)
    imask = interior if getattr(head, "interior_only", False) else None

    metrics = continuous_metrics(ev.y_soft, fw["mean"], fw["proba"], fw["nll"],
                                 fw["lower"], fw["upper"], fw50["lower"], fw50["upper"], imask)
    metrics["mean_uncertainty"] = float(fw["std"].mean())

    # boundary calibration for ZOIB
    boundary = None
    if key == "zoib":
        bf = batched_forward(model, ev, fields=("pi0", "pi1"))
        boundary = {
            "pred_P(y=0)_mean": float(bf["pi0"].mean()),
            "actual_freq(y=0)": float((ev.y_soft <= 0).mean()),
            "pred_P(y=1)_mean": float(bf["pi1"].mean()),
            "actual_freq(y=1)": float((ev.y_soft >= 1).mean()),
            "P(y=0)_calibration": _binned(bf["pi0"], (ev.y_soft <= 0).astype(float)),
            "P(y=1)_calibration": _binned(bf["pi1"], (ev.y_soft >= 1).astype(float)),
        }

    theta = theta_matrix(model, cfg, mi, s["pathway"])
    spec = theta_spectrum(theta)
    a_all = batched_forward(model, ev, fields=("a_q", "b_q"))
    psum = parameter_summary(theta=theta, a_q=a_all["a_q"], b_q=a_all["b_q"],
                             discrimination_constrained=bool(model.disc_head.constrain))
    psum["response_params"], psum["response_flags"] = _response_param_summary(model, ev)

    cold = None
    if model.model_params == "projected":
        from .eval import cold_start_baseline

        cold = cold_start_baseline(model, cfg, blob, split=split, **s)

    result = {
        "split": split, "response_model": key, "confidence_level": level,
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
        "per_model": _per_group(ev.y_soft, fw["mean"], ev.model_ids),
        "per_family": _per_group(ev.y_soft, fw["mean"], family_labels(ev.query_ids, cfg)),
        "boundary_calibration": boundary,
        "theta_spectrum": spec,
        "parameter_summary": psum,
        "cold_start": cold,
    }
    if write:
        prev = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        (d / "metrics.json").write_text(json.dumps({**prev, split: result}, indent=2, default=float),
                                        encoding="utf-8")
        save_theta_spectrum(spec, d)
        _write_json({"bins": metrics["_reliability"], "ece": metrics["mean_calibration_ece"]},
                    d, "mean_calibration.json")
        plots = d / "plots"
        plot_theta_spectrum(spec, plots / "theta_spectrum.png")
        _plot_mean_calibration(metrics["_reliability"], plots / "mean_calibration.png", key)
        _plot_uncertainty_calibration(model, ev, fw, plots / "uncertainty_calibration.png", level)
        if boundary:
            _plot_boundary_calibration(boundary, plots / "boundary_calibration.png")
    return result


def _response_param_summary(model, ev) -> tuple:
    key = model.response_model
    want = {"normal": ["sigma"], "beta": ["kappa", "mu"], "zoib": ["kappa", "mu", "pi0", "pi1", "pic"]}.get(key, [])
    if not want:
        return {}, []
    fw = batched_forward(model, ev, fields=tuple(want))
    from .diagnostics import _dist

    summ = {k: _dist(fw[k]) for k in want}
    flags = []
    if "kappa" in fw:
        if fw["kappa"].max() > 0.9 * float(getattr(model.response_head, "max_conc", 1e4)):
            flags.append("kappa saturating at max_concentration")
        if fw["kappa"].min() < 1e-3:
            flags.append("kappa collapsing toward 0")
    if "sigma" in fw:
        if fw["sigma"].min() < 1e-4:
            flags.append("sigma -> 0")
        if fw["sigma"].max() > 100:
            flags.append("sigma enormous")
    for p in ("pi0", "pi1"):
        if p in fw and fw[p].mean() > 0.95:
            flags.append(f"{p} ~ 1 everywhere")
    return summ, flags


def _binned(pred, actual, n_bins: int = 10) -> list:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "pred": float(pred[m].mean()),
                         "actual": float(actual[m].mean())})
    return rows


# --------------------------------------------------------------------------- #
# comparison                                                                  #
# --------------------------------------------------------------------------- #
def compare_response_models(dirs: dict, *, cfg: Optional[Config] = None, split: str = "test",
                            write: bool = True) -> dict:
    cfg = cfg or load_config()
    rows = {}
    for name, path in dirs.items():
        p = Path(path)
        if not (p / "model.pt").exists():
            continue
        model, blob, _ = load_run(p)
        if model.response_model == "bernoulli":
            from .eval import evaluate_checkpoint

            r = evaluate_checkpoint(p, phase0_cfg=cfg, split=split, write=False)
            pred = r["prediction"]
            rows[name] = {
                "response_model": "bernoulli",
                "nll": pred["bce"], "mae": pred["mae"], "rmse": pred["mse"] ** 0.5,
                "mean_calibration_ece": r["calibration"]["ece"],
                "bce_diag": pred["bce"], "accuracy": pred["accuracy"],
                "cold_start": _cs(r.get("cold_start")),
                "theta_effective_rank": r["theta_spectrum"]["effective_rank"],
            }
        else:
            r = evaluate_continuous(p, phase0_cfg=cfg, split=split, write=False)
            m = r["metrics"]
            rows[name] = {
                "response_model": r["response_model"],
                "nll": m["nll_mean"], "mae": m["mae"], "rmse": m["rmse"],
                "mean_calibration_ece": m["mean_calibration_ece"],
                "coverage_90": m.get("coverage_90"), "coverage_50": m.get("coverage_50"),
                "bce_diag": m["bce_diag_bce"], "accuracy": m["bce_diag_accuracy"],
                "boundary_calibration": r["boundary_calibration"],
                "cold_start": _cs(r.get("cold_start")),
                "theta_effective_rank": r["theta_spectrum"]["effective_rank"],
                "response_flags": r["parameter_summary"]["response_flags"],
            }
    out = {"split": split, "models": rows,
           "note": "NLL is each model's own proper score (Bernoulli BCE vs Normal/Beta/ZOIB NLL) "
                   "-- not directly comparable across rows; use it within a family and read "
                   "MAE/RMSE/calibration/coverage across rows."}
    if write:
        (cfg.root / "artifacts/phase2").mkdir(parents=True, exist_ok=True)
        (cfg.root / "artifacts/phase2/comparison.json").write_text(json.dumps(out, indent=2, default=float),
                                                                   encoding="utf-8")
    return out


def _cs(cold) -> Optional[dict]:
    if not cold or "pooled" not in cold:
        return None if not cold else {"skipped": cold.get("skipped")}
    return cold["pooled"]


# --------------------------------------------------------------------------- #
# plots                                                                       #
# --------------------------------------------------------------------------- #
def _plot_mean_calibration(bins, path, key):
    plt = pyplot()
    if plt is None:
        return
    b = [x for x in bins if x["count"]]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], "o-", color="#1f77b4")
    ax.set(xlabel="predicted E[Y] (bin mean)", ylabel="observed mean score",
           xlim=(0, 1), ylim=(0, 1), title=f"{key}: mean calibration")
    savefig(fig, path, plt)


def _plot_uncertainty_calibration(model, ev, fw, path, level):
    plt = pyplot()
    if plt is None:
        return
    levels = np.linspace(0.1, 0.95, 10)
    cov = []
    for lv in levels:
        q = batched_forward(model, ev, fields=("lower", "upper"), level=float(lv))
        cov.append(float(np.mean((ev.y_soft >= q["lower"]) & (ev.y_soft <= q["upper"]))))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(levels, cov, "o-", color="#ff7f0e")
    ax.set(xlabel="nominal central interval", ylabel="empirical coverage",
           xlim=(0, 1), ylim=(0, 1), title=f"{model.response_model}: uncertainty calibration")
    savefig(fig, path, plt)


def _plot_boundary_calibration(boundary, path):
    plt = pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, keyname in zip(axes, ("P(y=0)_calibration", "P(y=1)_calibration")):
        rows = boundary[keyname]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot([r["pred"] for r in rows], [r["actual"] for r in rows], "o-", color="#2ca02c")
        ax.set(xlabel="predicted", ylabel="actual frequency", title=keyname)
    savefig(fig, path, plt)

"""NIRT response model (Phase 2).

``router.data.nirt`` holds the training *representation* (observation table +
``NIRTDataset``); this package holds the *model*:

* :class:`~router.nirt.model.NIRTModel` -- ``theta_q = f_theta(e_q)``,
  ``y_hat = sigmoid(a_m . theta_q - b_m)``.
* :func:`~router.nirt.train.fit` -- train + early-stop + checkpoint.
* :func:`~router.nirt.evaluate.evaluate_split` -- prediction-quality metrics.

v1 is a deliberately simple predictor. Ranking / routing / cold-start evaluation,
the classical-IRT baseline, and the 2D/4D ladder are the next stage.
"""

from .baselines import (
    ClassicalIRTResult,
    fit_classical_irt,
    fit_mlp_router,
    knn_router_matrix,
    mlp_router_matrix,
)
from .ood import ood_datasets, ood_matrices, split_observations as ood_split_observations
from .evaluate import (
    cold_start_eval,
    evaluate_split,
    nearest_profile_model,
    predict_dataset,
    predict_matrix,
    ranking_metrics,
)
from .metrics import (
    brier_score,
    log_loss,
    marginal_baselines,
    prediction_metrics,
    reliability_curve,
)
from .model import IRTRouterModel, NIRTModel, build_model
from .baseline import BaselineNIRT, build_baseline_model
from .response_head import (
    RESPONSE_MODELS,
    BernoulliResponseHead,
    ResponseHead,
    ResponseOutput,
    build_response_head,
)
from .continuous_normal import NormalResponseHead
from .continuous_beta import BetaResponseHead
from .continuous_zoib import ZOIBResponseHead
from .losses import RegConfig, bce_loss, regularization
from .baseline_data import BaselineArrays, batched_forward, build_arrays
from .baseline_train import fit as fit_baseline
from .baseline_eval import evaluate_checkpoint
from .continuous_eval import (
    boundary_statistics,
    compare_response_models,
    evaluate_continuous,
)
from .checkpoint import load_checkpoint, save_checkpoint
from . import calibration, continuous_synthetic, diagnostics, synthetic
from .routing import (
    add_reward_columns,
    aiq,
    dense_matrices,
    eval_matrices,
    linear_cost,
    pareto,
    reward,
    routing_report,
)
from .train import RunResult, fit, load_run, seed_everything

__all__ = [
    "NIRTModel",
    "IRTRouterModel",
    "build_model",
    "BaselineNIRT",
    "build_baseline_model",
    "ResponseHead",
    "ResponseOutput",
    "BernoulliResponseHead",
    "NormalResponseHead",
    "BetaResponseHead",
    "ZOIBResponseHead",
    "build_response_head",
    "RESPONSE_MODELS",
    "RegConfig",
    "bce_loss",
    "regularization",
    "BaselineArrays",
    "build_arrays",
    "batched_forward",
    "fit_baseline",
    "evaluate_checkpoint",
    "evaluate_continuous",
    "compare_response_models",
    "boundary_statistics",
    "load_checkpoint",
    "save_checkpoint",
    "brier_score",
    "log_loss",
    "reliability_curve",
    "calibration",
    "diagnostics",
    "synthetic",
    "continuous_synthetic",
    "fit",
    "load_run",
    "RunResult",
    "seed_everything",
    "evaluate_split",
    "cold_start_eval",
    "nearest_profile_model",
    "predict_dataset",
    "predict_matrix",
    "ranking_metrics",
    "prediction_metrics",
    "marginal_baselines",
    "fit_classical_irt",
    "ClassicalIRTResult",
    "knn_router_matrix",
    "fit_mlp_router",
    "mlp_router_matrix",
    "ood_datasets",
    "ood_matrices",
    "ood_split_observations",
    "eval_matrices",
    "dense_matrices",
    "routing_report",
    "pareto",
    "reward",
    "linear_cost",
    "add_reward_columns",
    "aiq",
]

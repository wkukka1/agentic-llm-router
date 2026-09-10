"""NIRT response model (Phase 2).

``router.data.nirt`` holds the training *representation* (observation table +
``NIRTDataset``); this package holds the *model*. Import the submodule you need
directly -- ``from router.nirt.train import fit``, ``from router.nirt.routing
import routing_report``, etc. -- rather than this namespace, so touching one
corner of the package does not eagerly load torch + every response head.

The frozen Bernoulli / continuous-response control arm lives isolated under
``router.nirt.baseline`` (e.g. ``from router.nirt.baseline.model import
build_baseline_model``, ``from router.nirt.baseline.response_head import
RESPONSE_MODELS``) -- it is not re-exported here.

A few common entry points are re-exported here for convenience:

* :func:`router.nirt.model.build_model` -- build the model for a config.
* :func:`router.nirt.train.fit` / :func:`router.nirt.train.load_run`.
* :func:`router.nirt.evaluate.evaluate_split`.
"""

from .model import build_model
from .train import RunResult, fit, load_run

__all__ = [
    "build_model",
    "fit",
    "load_run",
    "RunResult",
]

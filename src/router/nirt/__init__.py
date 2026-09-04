"""NIRT response model (Phase 2).

``router.data.nirt`` holds the training *representation* (observation table +
``NIRTDataset``); this package holds the *model*. Import the submodule you need
directly -- ``from router.nirt.train import fit``, ``from router.nirt.routing
import routing_report``, etc. -- rather than this namespace, so touching one
corner of the package does not eagerly load torch + every response head.

A few common entry points are re-exported here for convenience:

* :func:`router.nirt.model.build_model` -- build the model for a config.
* :func:`router.nirt.train.fit` / :func:`router.nirt.train.load_run`.
* :func:`router.nirt.evaluate.evaluate_split`.
* :data:`router.nirt.response_head.RESPONSE_MODELS` +
  :func:`~router.nirt.response_head.build_response_head`.
"""

from .model import build_model
from .response_head import RESPONSE_MODELS, build_response_head
from .train import RunResult, fit, load_run

__all__ = [
    "build_model",
    "build_response_head",
    "RESPONSE_MODELS",
    "fit",
    "load_run",
    "RunResult",
]

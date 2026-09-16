"""NIRT response model -- serving side (Phase 2).

``router.data.nirt`` holds the batch representation (observation table +
``NIRTDataset``); this package holds the *model* plus everything needed to
score a fitted one: :mod:`router.nirt.checkpoint` (load a saved run),
:mod:`router.nirt.predict` (label-free inference), :mod:`router.nirt.frames`
(the ``[query x model]`` pivot), :mod:`router.nirt.baselines_infer` (kNN / MLP
router scoring) and :mod:`router.nirt.routing_decision` (the argmax policy).

Fitting (``training.nirt.train.fit``, ``training.trainers.mlp_router``) and
label-needing evaluation (``evaluation.nirt.evaluate``,
``evaluation.routing.oracle``) live in their own top-level packages -- this
package must never import them. Import the submodule you need directly
rather than this namespace, so touching one corner of the package does not
eagerly load torch + every response head.

The frozen Bernoulli / continuous-response control arm (training-only) lives
under ``training.nirt.baseline``.

A few common entry points are re-exported here for convenience, resolved
lazily (PEP 562) so ``import router.nirt.routing_decision`` -- which
``router.routing`` needs -- does not import torch:

* :func:`router.nirt.model.build_model` -- build the model for a config.
* :func:`router.nirt.checkpoint.load_run` -- load a saved run for inference.
"""

from __future__ import annotations

__all__ = [
    "build_model",
    "load_run",
]


def __getattr__(name: str):
    if name == "build_model":
        from .model import build_model

        return build_model
    if name == "load_run":
        from .checkpoint import load_run

        return load_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

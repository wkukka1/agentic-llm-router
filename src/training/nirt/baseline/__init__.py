"""Frozen baseline / control-arm pipeline (plain Bernoulli + continuous response heads).

This is the control arm compared against in ``scripts/route_compare.py``, kept isolated from
the active capacity-workstream pipeline (``router.nirt.model`` / ``training.nirt.train`` / ...).
Import the submodule you need directly, e.g. ``from training.nirt.baseline.model import
build_baseline_model``, ``from training.nirt.baseline.train import fit`` -- not this namespace.
"""

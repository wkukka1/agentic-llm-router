"""Frozen baseline / control-arm pipeline (plain Bernoulli + continuous response heads).

This is the control arm compared against in ``scripts/route_compare.py``, kept isolated from
the active capacity-workstream pipeline (``router.nirt.model`` / ``router.nirt.train`` / ...).
Import the submodule you need directly, e.g. ``from router.nirt.baseline.model import
build_baseline_model``, ``from router.nirt.baseline.train import fit`` -- not this namespace.
"""

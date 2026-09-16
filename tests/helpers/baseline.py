"""Training-config builder for the ``training.nirt.baseline`` (Phase 1/2) tests."""

from __future__ import annotations


def baseline_cfg(*, response: str = "bernoulli", k: int = 3, query_hidden: int = 64,
                 epochs: int = 40, patience: int = 12, lr: float = 3e-3,
                 batch_size: int | None = None, seed: int = 0, **train_over):
    """A ``fit`` config for ``training.nirt.baseline.train``.

    ``response="bernoulli"`` gives the binary target; any continuous head
    (``normal`` / ``beta`` / ``zoib``) switches to the soft target and adds the
    ``response`` block. Extra keyword args are merged into ``cfg["train"]``.
    """
    binary = response == "bernoulli"
    cfg = {
        "seed": seed,
        "model": {"theta_dim": k, "model_params": "free", "query_hidden": query_hidden,
                  "use_length_head": False},
        "ablation": {"use_relevance": False, "use_warmup": False, "use_interaction": False},
        "regularization": {"theta_l2": 1e-4, "theta_center_l2": 1e-3, "difficulty_l2": 1e-4},
        "train": {"target": "binary" if binary else "soft", "lr": lr,
                  "batch_size": batch_size or (1024 if binary else 2048),
                  "epochs": epochs, "patience": patience, "device": "cpu"},
    }
    if not binary:
        cfg["response"] = {"model": response, "cfg": {}}
    cfg["train"].update(train_over)
    return cfg

"""Experiment configuration.

An experiment is a YAML file; nothing about a run lives in code. That is what
makes the sweep honest -- every variant is described by a file that is written
back into the results directory alongside its metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    """Which built dataset variant to train on."""

    variant: str = "full_prompt"
    text_column: str = "prompt"
    label_column: str = "domain"
    #: Optional cap for smoke runs; ``None`` uses everything.
    max_train_rows: int | None = None


@dataclass(slots=True)
class ModelConfig:
    """Registry name plus its constructor kwargs."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 20260824
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        if "name" not in raw or "model" not in raw:
            raise ValueError("experiment config requires 'name' and 'model'")
        model_raw = raw["model"]
        model = ModelConfig(name=model_raw["name"], params=dict(model_raw.get("params") or {}))
        data = DataConfig(**(raw.get("data") or {}))
        return cls(
            name=raw["name"],
            model=model,
            data=data,
            seed=int(raw.get("seed", 20260824)),
            notes=raw.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "notes": self.notes,
            "data": {
                "variant": self.data.variant,
                "text_column": self.data.text_column,
                "label_column": self.data.label_column,
                "max_train_rows": self.data.max_train_rows,
            },
            "model": {"name": self.model.name, "params": self.model.params},
        }


def load_experiments(paths: list[str | Path]) -> list[ExperimentConfig]:
    """Load configs from files and/or directories of ``*.yaml``."""
    configs: list[ExperimentConfig] = []
    for entry in paths:
        path = Path(entry)
        files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
        configs.extend(ExperimentConfig.from_yaml(f) for f in files)
    return configs

"""``RouterModelFactory``: build a live :class:`~router.routing.base.RouterModel`
from a fitted :class:`~router.models.artifacts.RouterModelArtifact` -- the
artifact-driven construction path ``docs/architecture.md`` describes.

Coexists with, does not replace, :mod:`router.routing.registry`'s
``REGISTRY``/``register``/``build_router``: that registry serves a real,
different need -- constructing a router from raw kwargs with no artifact at
all (``RandomRouter(model_ids)``, ``MatrixRouter(scores_df)``). This factory
is specifically for the artifact+cost-model path, keyed by the same
``kind``/``router_model_name`` vocabulary (``NIRTRouter.kind == "nirt"``)
rather than inventing a third naming scheme.

Python has no signature-based overloading, so the diagram's two ``create()``
overloads (by artifact, and by artifact id + store) are one method here,
dispatched on whether the first argument is a bare id string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..llm.cost import CostModel
    from ..routing.base import RouterModel
    from .artifacts import RouterModelArtifact
    from .store import ArtifactStore


class RouterModelFactory:
    def __init__(self):
        self._registry: dict[str, type] = {}

    def register(self, router_model_name: str, model_class: type) -> None:
        self._registry[router_model_name] = model_class

    def create(
        self,
        artifact_or_id: "RouterModelArtifact | str",
        store_or_cost_model: "ArtifactStore | CostModel | None" = None,
        cost_model: Optional["CostModel"] = None,
    ) -> "RouterModel":
        """``create(artifact, cost_model)`` or ``create(artifact_id, store, cost_model)``."""
        if isinstance(artifact_or_id, str):
            store = store_or_cost_model
            if store is None:
                raise TypeError("create(artifact_id, store, cost_model) needs a store")
            artifact = store.load(artifact_or_id)
        else:
            artifact = artifact_or_id
            cost_model = store_or_cost_model

        model_class = self._registry.get(artifact.router_model_name)
        if model_class is None:
            raise ValueError(
                f"no RouterModel registered for router_model_name="
                f"{artifact.router_model_name!r}; registered: {sorted(self._registry)}"
            )
        from_artifact = getattr(model_class, "from_artifact", None)
        if from_artifact is None:
            raise TypeError(
                f"{model_class.__name__} has no from_artifact(artifact, cost_model) classmethod"
            )
        return from_artifact(artifact, cost_model)

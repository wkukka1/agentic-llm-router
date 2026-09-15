"""``LLMRegistry``: the pool of :class:`~router.llm.profile.LLMProfile`\\ s a
:class:`~router.router.RoutingPipeline` builds a request's candidate list
from.

Named ``registry.py`` like :mod:`router.routing.registry` on purpose --
repo-structure.md calls this out explicitly: different registries (model
profiles vs. router-strategy classes), both genuinely registries. Import both
qualified (``from router.llm import registry as llm_registry``) rather than
by bare name.
"""

from __future__ import annotations

from .profile import LLMProfile


class LLMRegistry:
    def __init__(self):
        self._profiles: dict[str, LLMProfile] = {}

    def register(self, profile: LLMProfile, *, replace: bool = False) -> None:
        """Add ``profile``. A second profile with the same ``model_id`` (e.g.
        the same model served by two providers at different prices) raises
        unless ``replace=True`` -- silently keeping the last one would route
        with the wrong price."""
        if profile.model_id in self._profiles and not replace:
            raise ValueError(
                f"model_id {profile.model_id!r} is already registered "
                f"(provider={self._profiles[profile.model_id].provider!r}); "
                "pass replace=True to overwrite"
            )
        self._profiles[profile.model_id] = profile

    def get(self, model_id: str) -> LLMProfile:
        return self._profiles[model_id]

    def list(self) -> list[LLMProfile]:
        return list(self._profiles.values())

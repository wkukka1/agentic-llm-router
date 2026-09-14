"""The framework-agnostic :class:`Tool` interface."""

from __future__ import annotations

import abc
from typing import Any


class Tool(abc.ABC):
    name: str = "tool"
    version: str = "0"

    @abc.abstractmethod
    def execute(self, input: Any) -> Any:
        raise NotImplementedError

"""Project settings, read once from the environment and a `.env` file.

Every experiment in this repository has to agree on a few things or its results
are not comparable with the others': the random seed above all, but also where
data and artifacts live. Those were defaults scattered across function
signatures, which meant a sweep run from a different entry point could silently
use a different seed and produce a number that looked like the others and was
not. Seed variance here is ±3.4 points, wide enough to invent an effect.

Precedence is the usual one: an explicit argument beats the environment, the
environment beats `.env`, and `.env` beats the defaults below. `.env` is
gitignored; `.env.example` is committed and lists every variable.

Read the values through :func:`settings`, not by reaching for ``os.environ``
directly, so that there is one place where a default lives and one place to
look when two runs disagree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: The seed every experiment shares unless one deliberately overrides it.
DEFAULT_SEED = 20260824

_TRUE = frozenset({"1", "true", "yes", "on"})


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal `.env` reader: ``KEY=value`` lines, ``#`` comments, no expansion.

    Deliberately not python-dotenv. This needs to run during import of a config
    module, and a hard dependency on a package that may not be installed --
    `pip install -e .` does not pull it in -- would turn a missing optional into
    an ImportError at startup.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True, slots=True)
class Settings:
    """Values shared across every experiment and every entry point."""

    seed: int = DEFAULT_SEED
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    embedding_cache_dir: Path = Path("data/processed/embeddings")
    #: Where `.env` was read from, or None. Printed by ``router describe-data``
    #: so a surprising seed can be traced to the file that set it.
    source: Path | None = None

    @property
    def handlabelled_dir(self) -> Path:
        return self.data_dir / "handlabelled"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache(maxsize=1)
def settings(env_file: str | Path = ".env") -> Settings:
    """The project settings, cached.

    Cached because these are read from module-level defaults in several places
    and must not change between two calls inside one process -- a seed that
    differs between the split and the fit is worse than a wrong seed.
    Call ``settings.cache_clear()`` in a test that needs a different value.
    """
    path = Path(env_file)
    from_file = _load_dotenv(path)

    def get(key: str, default: str) -> str:
        return os.environ.get(key, from_file.get(key, default))

    return Settings(
        seed=int(get("ROUTER_SEED", str(DEFAULT_SEED))),
        data_dir=Path(get("ROUTER_DATA_DIR", "data")),
        artifacts_dir=Path(get("ROUTER_ARTIFACTS_DIR", "artifacts")),
        embedding_cache_dir=Path(
            get("ROUTER_EMBEDDING_CACHE", "data/processed/embeddings")
        ),
        source=path if from_file else None,
    )


def truthy(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment."""
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE

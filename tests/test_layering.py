"""The dependency ladder: router <- training <- evaluation, one direction only.

`router` is what a serving deployment installs. The moment it imports `training`
it drags the training stack with it, and the moment an offline convenience
lands in `router` it becomes a serving dependency nobody chose. Neither failure
announces itself -- both work fine on a developer machine with everything
installed -- so the rule is enforced here rather than trusted.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

#: What each tier is allowed to import from. Order is the ladder.
ALLOWED = {
    "router": {"router"},
    "training": {"router", "training"},
    "evaluation": {"router", "training", "evaluation"},
}


def _first_party_imports(path: pathlib.Path) -> set[str]:
    """Top-level package of every in-repo import, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found & set(ALLOWED)


def _modules(tier: str) -> list[pathlib.Path]:
    return sorted((SRC / tier).rglob("*.py"))


@pytest.mark.parametrize("tier", sorted(ALLOWED))
def test_the_tier_exists_and_has_modules(tier):
    assert _modules(tier), f"{tier}/ has no modules; the ladder has a missing rung"


@pytest.mark.parametrize("tier", sorted(ALLOWED))
def test_no_module_imports_from_a_tier_above_it(tier):
    """Lazy imports inside functions count -- they are the usual way this rule
    gets broken, because they do not show up at the top of the file."""
    offences = []
    for path in _modules(tier):
        illegal = _first_party_imports(path) - ALLOWED[tier]
        if illegal:
            offences.append(f"{path.relative_to(SRC)} imports {sorted(illegal)}")
    assert not offences, (
        f"{tier}/ may only import {sorted(ALLOWED[tier])}:\n  " + "\n  ".join(offences))


def test_router_alone_carries_no_offline_dependency():
    """The point of the rule, stated as the thing it protects: a serving
    install is `router` and its imports, nothing else."""
    for path in _modules("router"):
        assert _first_party_imports(path) <= {"router"}, path

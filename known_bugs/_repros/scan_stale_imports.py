"""Resolve every `import` / `from ... import ...` in src/ (incl. function-local ones).

Reports imports of first-party modules (router, training, evaluation, decompose) that
cannot be resolved, and `from pkg import name` where `name` is neither an attribute
nor a submodule. Third-party import errors are ignored (optional deps).
"""
import ast, importlib, importlib.util, sys
from pathlib import Path

SRC = Path("src")
FIRST = {"router", "training", "evaluation", "decompose"}
sys.path.insert(0, str(SRC))
bad = []

def modname(path):
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts), path.name == "__init__.py"

def resolve(mod, name=None):
    try:
        m = importlib.import_module(mod)
    except Exception as e:  # noqa
        return f"cannot import module {mod!r}: {type(e).__name__}: {str(e)[:90]}"
    if name and name != "*" and not hasattr(m, name):
        try:
            importlib.import_module(f"{mod}.{name}")
        except Exception:
            return f"{mod!r} has no attribute/submodule {name!r}"
    return None

for path in sorted(SRC.rglob("*.py")):
    me, is_pkg = modname(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = me.split(".") if is_pkg else me.split(".")[:-1]
                base = base[: len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module
            targets = [(mod, a.name) for a in node.names]
        else:
            continue
        for mod, name in targets:
            if mod and mod.split(".")[0] in FIRST:
                err = resolve(mod, name)
                if err:
                    bad.append((str(path), node.lineno, err))

for p, ln, e in sorted(set(bad)):
    print(f"{p}:{ln}: {e}")
print(f"\n{len(set(bad))} unresolved first-party imports")

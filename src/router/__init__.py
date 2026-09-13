"""Serving. Everything here runs at inference, and imports nothing above it.

The dependency rule is one-way and load-bearing: `router` may not import
`training` or `evaluation`. It keeps a serving deployment from dragging in the
training stack, and stops an offline convenience from quietly becoming a
serving dependency. `tests/test_layering.py` enforces it.

Subsystems sit side by side under this tier -- `prompt_decomposition` is one --
with shared infrastructure (`embeddings`, `settings`) at the root.
"""

"""Tests for the prompt-decomposition heads.

A package rather than a bare directory so these modules get qualified names:
several basenames here (`test_data`, `test_metrics`, `test_taxonomy`,
`test_review_fixes`) also exist under `tests/router/` and `tests/training/`,
and without this pytest imports them as top-level modules and the second one
collides with the first.
"""

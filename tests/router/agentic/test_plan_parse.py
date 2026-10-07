"""Planner output with dependencies: parse_plan_lines, LLMDecomposer.plan and
AgenticRouter.decompose_plan (router.agentic.decompose / router)."""

from __future__ import annotations

import pandas as pd
import pytest

from router.agentic import AgenticRouter, ClientRegistry, LLMDecomposer
from router.agentic.decompose import SubTask, parse_plan_lines
from router.routing import RouterModel


def test_after_marker_becomes_a_dependency():
    text = ("1. Implement quicksort in Rust.\n"
            "2. [after 1] Analyse the time complexity of the Rust quicksort from step 1.")
    assert parse_plan_lines(text) == [
        SubTask("Implement quicksort in Rust."),
        SubTask("Analyse the time complexity of the Rust quicksort from step 1.", (1,)),
    ]


def test_unmarked_lines_are_independent():
    assert [t.depends_on for t in parse_plan_lines("Explain A in detail.\nExplain B in detail.")] == [(), ()]


def test_forward_self_and_unknown_references_are_dropped():
    text = ("1. [after 2] Do the first thing.\n"
            "2. [after 2, 9] Do the second thing.\n"
            "3. [after 1 and 2] Do the third thing.")
    assert [t.depends_on for t in parse_plan_lines(text)] == [(), (), (1, 2)]


def test_reference_to_a_dropped_repeat_points_at_the_original():
    text = ("1. Implement quicksort in Rust.\n"
            "2. implement quicksort in rust\n"
            "3. [after 2] Test the Rust quicksort.")
    tasks = parse_plan_lines(text)
    assert [t.text for t in tasks] == ["Implement quicksort in Rust.", "Test the Rust quicksort."]
    assert tasks[1].depends_on == (1,)


def test_none_and_preamble_lines_are_not_tasks():
    assert parse_plan_lines("Here is the plan:\nnone\n1. Do the thing.") == [SubTask("Do the thing.")]


def test_overflow_folds_into_the_last_kept_task_with_its_dependencies():
    text = "1. Task one here.\n2. Task two here.\n3. [after 1] Task three here."
    assert parse_plan_lines(text, max_parts=2) == [
        SubTask("Task one here."),
        SubTask("Task two here. Task three here.", (1,)),
    ]


def test_external_references_are_checked_against_ref_limit_only():
    text = "[after 2] Fix the Rust quicksort from step 2.\n[after 7] Something else to do."
    assert [t.depends_on for t in parse_plan_lines(text, ref_limit=3)] == [(2,), ()]

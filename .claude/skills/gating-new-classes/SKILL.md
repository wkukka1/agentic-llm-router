---
name: gating-new-classes
description: Use when about to write a new Python class anywhere under src/ in this repo, before writing any implementation code — including when told to skip confirmation, move fast, or "just use your judgment."
---

# Gating New Classes

## Overview

This repo keeps its class hierarchy documented as a Mermaid `classDiagram` in
[docs/architecture.md](../../../docs/architecture.md). A new class is a change
to that shared, everyone-reads-it design — not an implementation detail — so
it needs the user's explicit go-ahead before it's written, and again before
the diagram is edited to match.

**Violating the letter of this gate (writing the class first, asking after)
is violating its spirit** — the point is the user deciding *before* a new
entity exists, not being informed afterward.

## When to Use

Applies to: any `class Foo:` you're about to add under `src/`.

Does not apply to: local closures, test fixtures, a `dataclass`-as-struct
used only inside one function, or editing/renaming an *existing* class.

## The Rule

1. **Before writing any code**, read the classDiagram in
   `docs/architecture.md` and locate where the new class would fit
   (namespace, base class, what it would depend on / be depended on by).
2. **Tell the user** that placement — namespace, base class, relationships —
   and ask them to confirm before you write the class. Use AskUserQuestion or
   a direct question in your reply.
3. **Wait for a yes.** This holds even if the user said "don't ask," "trust
   your judgment," "I'm in a rush," or gave precise behavioral spec ("try X,
   fall back to Y on error"). Those waive sign-off on *how the class works
   internally* — they do not waive sign-off on *whether a new named entity
   gets added to the architecture*. A rushed user who says "no need to check
   in" on the implementation has not pre-approved a new class existing.
4. **After they confirm and you've written the class**, draft the diagram
   edit — the new `class X { ... }` block plus its relationship edges — and
   ask again before applying it to `docs/architecture.md`. Do not treat this
   as an optional follow-up or something to add "if you remember": the task
   isn't done until the diagram edit has been proposed and either applied or
   explicitly declined.

## Rationalization Table

| Excuse | Reality |
|---|---|
| "The user said no need to ask" | That waives sign-off on implementation details, not on adding a new architectural entity. |
| "The instructions were explicit and unambiguous" | A precise behavioral spec (try X, fall back to Y) doesn't cover *whether this should be a new class*, where it lives, or what it's named — that's still open. |
| "This is an ordinary engineering judgment call" | Method bodies are ordinary judgment calls. Adding a new class to the shared diagram is not — it's the thing this gate exists for. |
| "I'll update the diagram after, if there's time" | Diagram updates get silently dropped under time pressure. Propose-or-decline happens before you report the task done. |
| "I already read the diagram, that's diligence enough" | Reading confirms fit; it isn't the user's yes. The gate is their answer, not your own judgment that it fits. |

## Red Flags — Stop and Ask

- About to type `class ` in a `src/` file and haven't shown the user where it fits yet.
- The reason you're skipping the question is "user said trust your judgment" or "user's in a rush."
- The class is written and you're about to report done without having proposed a diagram edit.

## Quick Reference

1. Need a new class → read `docs/architecture.md` diagram → tell user the proposed placement → ask → wait for yes → write it.
2. Class written → draft the diagram diff → ask → apply only after yes.

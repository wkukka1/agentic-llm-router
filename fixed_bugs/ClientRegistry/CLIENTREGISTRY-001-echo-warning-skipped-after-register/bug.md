# CLIENTREGISTRY-001: the "answering with an echo placeholder" warning depends on how clients were added

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low
- **Class:** `ClientRegistry` ([llm_clients.py:71](../../../src/router/agentic/llm_clients.py#L71), [92-107](../../../src/router/agentic/llm_clients.py#L92-L107))
- **Found:** 2026-09-19

## What

`get()` warns before synthesising an echo client only when `self._has_real_clients` is true.
That flag is computed once in `__init__` from the constructor's `clients` mapping.
`register()` adds to `_clients` but never updates it.

## Failure scenario

Repro (`known_bugs/_repros/repro_agentic.py`, section `CLIENTREGISTRY-001`):

```
clients passed to constructor -> warnings: 1
clients added via register()  -> warnings: 0
```

A registry built empty and populated with real clients through `register()` (the usual way
to configure one) silently answers any unregistered model id with an echo string. The warning
exists to catch exactly that.

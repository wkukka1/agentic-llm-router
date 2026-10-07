# LANGCHAINADAPTER-002: A partially-priced profile costs calls at a discount, not an error

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Class:** `LangChainAdapter` ([router/llm/adapters/langchain_adapter.py:122-126](../../../../src/router/llm/adapters/langchain_adapter.py#L122-L126))
- **Found:** 2026-09-22

## What

`_cost` prices a call from `input_cost_per_token` /
`output_cost_per_token`. When only one of the two is set on a profile, the
unset side is treated as `0.0` rather than triggering a fallback to a
known-good price or raising — the same silent-zero pattern as
[LLMCLIENT-001](../../../LLMClient/LLMCLIENT-001-unpriced-profile-prices-calls-at-zero/bug.md),
but here it's a *partial* miss (one side priced, one silently free) rather
than a fully unpriced profile.

## Failure scenario

A registry entry with `input_cost_per_token` set but `output_cost_per_token`
omitted (a plausible copy-paste/config-entry mistake, since output tokens are
usually the pricier side) has every real, billed completion under-costed —
output tokens are counted for free in the adapter's own accounting, with no
signal that the profile is only half-priced.

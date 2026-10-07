# LLMCLIENTFACTORY-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`client_for` now caches only the `ProviderAdapter` (the expensive SDK client) per `(provider, model, api_key, client_kwargs)` and always constructs a fresh `LLMClient(adapter, profile)` around the `profile` passed in, so a re-priced profile is never served through a client bound to the first profile seen for that provider/model. Covered by `tests/router/llm/test_client.py::test_client_factory_serves_a_repriced_profile_with_fresh_prices`.

## Fix suggested when the bug was filed

Include the profile's pricing (or `id(profile)` / the profile itself, if hashable) in the
key, or store only the adapter in the cache and construct a fresh `LLMClient(adapter, profile)`
per call.

## Regression tests

`tests/router/llm/test_client.py:32` `test_client_factory_serves_a_repriced_profile_with_fresh_prices`

```python
def test_client_factory_serves_a_repriced_profile_with_fresh_prices():
    """LLMCLIENTFACTORY-001: the adapter/SDK client may be cached, but a
    later ``client_for`` call with a re-priced profile for the same
    provider/model must not keep the first profile's prices."""
    factory = LLMClientFactory(adapters={"echo": EchoAdapter})
    old = _profile(output_cost_per_token=1e-6)
    new = _profile(output_cost_per_token=5e-6)

    factory.client_for(old)
    client = factory.client_for(new)

    assert client.price(0, 1000) == 5e-6 * 1000
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/llm/client.py`

```diff
--- a/src/router/llm/client.py
+++ b/src/router/llm/client.py
@@ -96,29 +107,33 @@ class LLMClientFactory:
 
     ``adapters`` extends / overrides the provider -> adapter table;
     ``fallback`` (e.g. an OpenAI-compatible endpoint adapter) serves any
-    provider not in it. Clients are cached per ``(provider, model, api_key,
-    client_kwargs)`` so SDK clients aren't rebuilt on every request.
+    provider not in it. **Adapters** (the SDK client construction) are cached
+    per ``(provider, model, api_key, client_kwargs)`` so they aren't rebuilt on
+    every request; the returned :class:`LLMClient` is always built fresh
+    around the ``profile`` passed in, so a re-priced profile (see
+    ``LLMRegistry.register(..., replace=True)``) is never served stale prices
+    from an earlier call for the same provider/model.
     """
 
     def __init__(self, adapters: Optional[Mapping[str, AdapterSpec]] = None, *,
                  fallback: Optional[AdapterSpec] = None):
         self._adapters: dict[str, AdapterSpec] = {**_ADAPTERS, **dict(adapters or {})}
         self._fallback = fallback
-        self._cache: dict[tuple, LLMClient] = {}
+        self._cache: dict[tuple, "ProviderAdapter"] = {}
 
     def client_for(self, profile: "LLMProfile", *, api_key: Optional[str] = None,
                    **client_kwargs) -> LLMClient:
         provider = (getattr(profile, "serving_provider", None) or profile.provider).lower()
         key = (provider, profile.model_id, api_key, repr(sorted(client_kwargs.items())))
-        if key in self._cache:
-            return self._cache[key]
-        spec = self._adapters.get(provider, self._fallback)
-        if spec is None:
-            raise ValueError(
-                f"no ProviderAdapter for provider={provider!r}; "
-                f"supported: {sorted(self._adapters)} (pass adapters= or fallback=)"
-            )
-        Adapter = _adapter_class(spec)
-        client = LLMClient(Adapter(profile.model_id, api_key=api_key, **client_kwargs), profile)
-        self._cache[key] = client
-        return client
+        adapter = self._cache.get(key)
+        if adapter is None:
+            spec = self._adapters.get(provider, self._fallback)
+            if spec is None:
+                raise ValueError(
+                    f"no ProviderAdapter for provider={provider!r}; "
+                    f"supported: {sorted(self._adapters)} (pass adapters= or fallback=)"
+                )
+            Adapter = _adapter_class(spec)
+            adapter = Adapter(profile.model_id, api_key=api_key, **client_kwargs)
+            self._cache[key] = adapter
+        return LLMClient(adapter, profile)
```

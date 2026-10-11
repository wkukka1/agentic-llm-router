"""Every gateway test gets its own log folder, so none of them reads or
writes the real ~/.aar-gateway (preferences, overrides, request log)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_gateway_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    # FastAPI's TestClient calls itself "testserver"; the real gateway only
    # answers to localhost names (server._local_callers_only)
    monkeypatch.setenv("AAR_GATEWAY_ALLOWED_HOSTS", "testserver")

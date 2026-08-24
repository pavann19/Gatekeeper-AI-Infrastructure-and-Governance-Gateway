"""
Tests for GET /api/v1/tools and GET /api/v1/gateway/providers (Phase 7,
Developer UI) -- read-only catalogue views over the REAL shared tool
registry and provider registry, not synthesized data.
"""
import json

import pytest
from fastapi.testclient import TestClient

import api.main as main_mod
from api.main import app
from core import auth as auth_mod
from core.auth import KeyStore, generate_key, hash_key
from core.demo_tools import register_demo_tools
from core.tools import get_tool_registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def _demo_tools_registered():
    """Same pattern tests/test_tools_endpoint.py uses -- these hit the
    real app/TestClient, which resolves tools via the module-level shared
    registry, not an injectable one."""
    reg = register_demo_tools()
    yield reg
    for name in ("demo.echo", "demo.calculator.add", "demo.database.query", "demo.database.delete"):
        reg._tools.pop(name, None)
        reg._handlers.pop(name, None)


@pytest.fixture
def key_store(tmp_path, monkeypatch):
    path = tmp_path / "api_keys.json"
    store = {}

    def issue(capability="GENERAL", tenant="default", key_id="test-key"):
        plaintext = generate_key()
        store[hash_key(plaintext)] = {"capability": capability, "tenant": tenant, "key_id": key_id}
        path.write_text(json.dumps(store), encoding="utf-8")
        monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
        return plaintext

    return issue


# --- /api/v1/tools ---------------------------------------------------------------

def test_tools_requires_internal_capability(key_store):
    key = key_store(capability="ELEVATED")
    response = client.get("/api/v1/tools", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_tools_lists_the_real_registered_tools(key_store):
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/tools", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    names = {t["name"] for t in response.json()["tools"]}
    assert names == {"demo.echo", "demo.calculator.add", "demo.database.query", "demo.database.delete"}


def test_tools_response_carries_real_spec_fields(key_store):
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/tools", headers={"Authorization": f"Bearer {key}"})
    by_name = {t["name"]: t for t in response.json()["tools"]}
    delete_tool = by_name["demo.database.delete"]
    assert delete_tool["risk_level"] == get_tool_registry().get("demo.database.delete").risk_level
    assert delete_tool["capability_required"] == get_tool_registry().get("demo.database.delete").capability_required
    assert "parameters" in delete_tool and "description" in delete_tool


# --- /api/v1/gateway/providers ---------------------------------------------------

def test_gateway_providers_requires_internal_capability(key_store):
    key = key_store(capability="GENERAL")
    response = client.get("/api/v1/gateway/providers", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_gateway_providers_lists_real_supported_providers(key_store):
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/gateway/providers", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    body = response.json()
    assert set(body["providers"]) == {"ollama", "openai_compatible", "anthropic_compatible"}
    assert body["default_provider"] == main_mod.settings.LLM_GATEWAY_DEFAULT_PROVIDER

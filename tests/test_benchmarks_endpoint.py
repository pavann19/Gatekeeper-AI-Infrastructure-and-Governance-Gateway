"""Tests for GET /api/v1/benchmarks (Phase 7, Developer UI)."""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core.auth import KeyStore, generate_key, hash_key

client = TestClient(app)


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


def test_requires_internal_capability(key_store):
    key = key_store(capability="ELEVATED")
    response = client.get("/api/v1/benchmarks", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_returns_benchmark_catalog_for_internal_even_when_public_checkout_has_no_raw_evidence(key_store):
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/benchmarks", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    body = response.json()
    assert sorted(body.keys()) == ["errors", "runs"]
    assert isinstance(body["runs"], list)
    assert isinstance(body["errors"], list)

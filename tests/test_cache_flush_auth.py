"""
Tests for POST /api/v1/cache/flush's auth requirement (issue #5).

Previously had NO check at all -- any unauthenticated caller could
repeatedly flush the semantic cache, a latency/cost amplification vector.
Now requires INTERNAL capability, same bar as the review endpoints
(GET /api/v1/review, POST /api/v1/review/{id}/resolve).
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core.auth import KeyStore, generate_key, hash_key

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("api.main.settings.RATE_LIMIT_ENABLED", False)


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

    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
    return issue


def test_anonymous_caller_rejected(monkeypatch):
    response = client.post("/api/v1/cache/flush")
    assert response.status_code == 403


def test_general_capability_rejected(key_store):
    key = key_store(capability="GENERAL")
    response = client.post("/api/v1/cache/flush", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_elevated_capability_rejected(key_store):
    key = key_store(capability="ELEVATED")
    response = client.post("/api/v1/cache/flush", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_internal_capability_succeeds(key_store, monkeypatch):
    flushed = {"called": False}
    monkeypatch.setattr("api.main.flush_cache", lambda: flushed.__setitem__("called", True))
    key = key_store(capability="INTERNAL")
    response = client.post("/api/v1/cache/flush", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert flushed["called"] is True


def test_rejected_caller_never_triggers_a_flush(key_store, monkeypatch):
    flushed = {"called": False}
    monkeypatch.setattr("api.main.flush_cache", lambda: flushed.__setitem__("called", True))
    key = key_store(capability="GENERAL")
    client.post("/api/v1/cache/flush", headers={"Authorization": f"Bearer {key}"})
    assert flushed["called"] is False

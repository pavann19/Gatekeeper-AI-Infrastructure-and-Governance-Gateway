"""
Tests for GET /api/v1/whoami (Phase 7) -- the identity-check endpoint the
client UI's login page uses to verify a pasted API key against the real
KeyStore before trusting it, rather than trusting whatever the browser
happened to store.
"""
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


def test_valid_key_returns_its_own_capability_tenant_and_key_id(key_store):
    key = key_store(capability="ELEVATED", tenant="acme-corp", key_id="acme-elevated-1")
    response = client.get("/api/v1/whoami", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    body = response.json()
    assert body == {"capability": "ELEVATED", "tenant": "acme-corp", "key_id": "acme-elevated-1"}


def test_missing_credential_is_401():
    response = client.get("/api/v1/whoami")
    assert response.status_code == 401


def test_unrecognised_key_is_401(key_store):
    key_store()  # a key store exists, but the caller presents something else
    response = client.get("/api/v1/whoami", headers={"Authorization": "Bearer gk_not_a_real_key"})
    assert response.status_code == 401


def test_malformed_authorization_header_is_401(key_store):
    key_store()
    response = client.get("/api/v1/whoami", headers={"Authorization": "not-bearer-shaped"})
    assert response.status_code == 401


def test_response_never_contains_the_credential_itself(key_store):
    key = key_store(capability="INTERNAL", tenant="default", key_id="ops-key")
    response = client.get("/api/v1/whoami", headers={"Authorization": f"Bearer {key}"})
    assert key not in response.text


def test_response_has_no_extra_fields(key_store):
    """WhoAmIResponse is extra='forbid' -- this must stay the minimal,
    audit-safe shape (Principal.to_audit() minus 'authenticated', which is
    implied by a 200 at all) and never grow a field by accident."""
    key = key_store()
    response = client.get("/api/v1/whoami", headers={"Authorization": f"Bearer {key}"})
    assert set(response.json().keys()) == {"capability", "tenant", "key_id"}

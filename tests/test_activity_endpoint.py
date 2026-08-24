"""
Tests for GET /api/v1/activity (Phase 7) -- the client UI's activity feed.
Writes real records into a real audit.jsonl file (via monkeypatching
core.activity.settings.AUDIT_LOG_PATH, the same setting the endpoint's own
get_recent_activity() call reads) and drives the real HTTP layer, so these
exercise the actual tenant-scoping decision, not a mocked stand-in for it.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core.auth import KeyStore, generate_key, hash_key

client = TestClient(app)


@pytest.fixture(autouse=True)
def _audit_log(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr("core.activity.settings.AUDIT_LOG_PATH", str(path))

    def write(records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    return write


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


def _event(tenant="acme", event_type="tool_call", request_id="r1"):
    return {"timestamp": "2026-08-24T00:00:00", "event_type": event_type,
            "tenant": tenant, "capability": "GENERAL", "decision": "ALLOW",
            "request_id": request_id}


def test_caller_sees_only_their_own_tenant_by_default(_audit_log, key_store):
    _audit_log([_event(tenant="acme", request_id="a1"), _event(tenant="beta", request_id="b1")])
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/activity", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"a1"}


def test_general_caller_cannot_see_another_tenant_via_query_param(_audit_log, key_store):
    """A GENERAL/ELEVATED caller passing ?tenant=beta must NOT be able to
    read another tenant's activity just by asking -- only INTERNAL may
    cross tenants."""
    _audit_log([_event(tenant="acme", request_id="a1"), _event(tenant="beta", request_id="b1")])
    key = key_store(capability="ELEVATED", tenant="acme")
    response = client.get("/api/v1/activity?tenant=beta", headers={"Authorization": f"Bearer {key}"})
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"a1"}


def test_internal_caller_can_request_a_specific_other_tenant(_audit_log, key_store):
    _audit_log([_event(tenant="acme", request_id="a1"), _event(tenant="beta", request_id="b1")])
    key = key_store(capability="INTERNAL", tenant="ops")
    response = client.get("/api/v1/activity?tenant=beta", headers={"Authorization": f"Bearer {key}"})
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"b1"}


def test_internal_caller_can_request_all_tenants(_audit_log, key_store):
    _audit_log([_event(tenant="acme", request_id="a1"), _event(tenant="beta", request_id="b1")])
    key = key_store(capability="INTERNAL", tenant="ops")
    response = client.get("/api/v1/activity?tenant=__all__", headers={"Authorization": f"Bearer {key}"})
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"a1", "b1"}


def test_internal_caller_without_tenant_param_still_only_sees_own_tenant(_audit_log, key_store):
    """No ?tenant= at all means the same default every capability gets:
    the caller's own tenant, not an implicit cross-tenant view."""
    _audit_log([_event(tenant="ops", request_id="o1"), _event(tenant="beta", request_id="b1")])
    key = key_store(capability="INTERNAL", tenant="ops")
    response = client.get("/api/v1/activity", headers={"Authorization": f"Bearer {key}"})
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"o1"}


def test_event_type_filter(_audit_log, key_store):
    _audit_log([
        _event(tenant="acme", event_type="tool_call", request_id="t1"),
        _event(tenant="acme", event_type="gateway_call", request_id="g1"),
    ])
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/activity?event_type=tool_call", headers={"Authorization": f"Bearer {key}"})
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"t1"}


def test_limit_is_honoured(_audit_log, key_store):
    _audit_log([_event(tenant="acme", request_id=str(i)) for i in range(10)])
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/activity?limit=3", headers={"Authorization": f"Bearer {key}"})
    assert len(response.json()["events"]) == 3


def test_no_audit_log_yet_is_an_empty_feed_not_an_error(key_store, monkeypatch, tmp_path):
    monkeypatch.setattr("core.activity.settings.AUDIT_LOG_PATH", str(tmp_path / "nope.jsonl"))
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/activity", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert response.json()["events"] == []

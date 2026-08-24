"""
Tests for GET /api/v1/activity/trace/{request_id} and GET /api/v1/logs
(Phase 7, Developer UI) -- both read the same audit.jsonl the activity
feed does, so these focus on what's DIFFERENT about them: trace's
request_id correlation and chronological ordering, and logs' INTERNAL-only
cross-tenant-by-default gate.
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


# --- trace ---------------------------------------------------------------------

def test_trace_returns_events_for_that_request_id_chronologically(_audit_log, key_store):
    _audit_log([
        _event(tenant="acme", event_type="input_assessment", request_id="req-1"),
        _event(tenant="acme", event_type="tool_call", request_id="req-1"),
        _event(tenant="acme", event_type="input_assessment", request_id="req-2"),
    ])
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/activity/trace/req-1", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    types = [e["event_type"] for e in response.json()["events"]]
    assert types == ["input_assessment", "tool_call"]


def test_trace_scoped_to_callers_own_tenant_for_non_internal(_audit_log, key_store):
    _audit_log([_event(tenant="beta", request_id="shared-id")])
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/activity/trace/shared-id", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert response.json()["events"] == []


def test_trace_internal_can_cross_tenants(_audit_log, key_store):
    _audit_log([_event(tenant="beta", request_id="shared-id")])
    key = key_store(capability="INTERNAL", tenant="ops")
    response = client.get(
        "/api/v1/activity/trace/shared-id?tenant=beta", headers={"Authorization": f"Bearer {key}"}
    )
    assert len(response.json()["events"]) == 1


def test_trace_unknown_request_id_is_empty_not_404(_audit_log, key_store):
    _audit_log([_event(tenant="acme", request_id="req-1")])
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/activity/trace/no-such-id", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert response.json()["events"] == []


# --- logs ------------------------------------------------------------------------

def test_logs_requires_internal_capability(_audit_log, key_store):
    _audit_log([_event(tenant="acme")])
    key = key_store(capability="ELEVATED", tenant="acme")
    response = client.get("/api/v1/logs", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_logs_defaults_to_cross_tenant_for_internal(_audit_log, key_store):
    _audit_log([_event(tenant="acme", request_id="a1"), _event(tenant="beta", request_id="b1")])
    key = key_store(capability="INTERNAL", tenant="ops")
    response = client.get("/api/v1/logs", headers={"Authorization": f"Bearer {key}"})
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"a1", "b1"}


def test_logs_can_be_scoped_to_one_tenant(_audit_log, key_store):
    _audit_log([_event(tenant="acme", request_id="a1"), _event(tenant="beta", request_id="b1")])
    key = key_store(capability="INTERNAL", tenant="ops")
    response = client.get("/api/v1/logs?tenant=acme", headers={"Authorization": f"Bearer {key}"})
    ids = {e["request_id"] for e in response.json()["events"]}
    assert ids == {"a1"}

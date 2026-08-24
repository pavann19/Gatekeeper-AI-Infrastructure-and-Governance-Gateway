"""
Tests for the Policy Editor endpoints (Phase 7, Developer UI):
GET /api/v1/policy, POST /api/v1/policy/validate, POST /api/v1/policy/deploy,
POST /api/v1/policy/rollback.

These are thin HTTP wrappers over core.policy.validate_policy_file and
core.policy_versioning's real deploy/rollback machinery -- tests isolate
the live policy file and versions dir to tmp_path (same pattern
tests/test_policy_versioning.py uses) and drive the REAL functions, not
mocks, so a bug in the wiring (e.g. skipping validation before deploy)
would actually be caught.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core import policy as policy_mod
from core.auth import KeyStore, generate_key, hash_key
from core.policy import PolicyStore

client = TestClient(app)

VALID_POLICY = {
    "default_action": "BLOCK",
    "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK", "LOW": "ALLOW"}}}},
}


@pytest.fixture(autouse=True)
def _isolated_policy_files(tmp_path, monkeypatch):
    live = tmp_path / "policy_rules.json"
    live.write_text(json.dumps(VALID_POLICY), encoding="utf-8")
    versions_dir = tmp_path / "policy_versions"
    # settings.POLICY_RULES_FILE drives deploy_policy/validate_policy_file
    # (read fresh on every call), but core.policy's module-level `_store`
    # singleton captures its path once at construction -- monkeypatching
    # settings alone would leave `_store` (and therefore policy_decision)
    # still reading the REAL project's live policy file. Same isolation
    # approach tests/test_policy.py's own `policy_store` fixture uses.
    monkeypatch.setattr("core.config.settings.POLICY_RULES_FILE", str(live))
    monkeypatch.setattr("core.config.settings.POLICY_VERSIONS_DIR", str(versions_dir))
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(live)))
    yield live


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


# --- GET /api/v1/policy -----------------------------------------------------------

def test_get_policy_requires_internal(key_store):
    key = key_store(capability="ELEVATED")
    response = client.get("/api/v1/policy", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_get_policy_returns_real_live_content(key_store):
    key = key_store(capability="INTERNAL")
    response = client.get("/api/v1/policy", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == VALID_POLICY
    assert body["versions"] == []


# --- POST /api/v1/policy/validate --------------------------------------------------

def test_validate_requires_internal(key_store):
    key = key_store(capability="GENERAL")
    response = client.post("/api/v1/policy/validate", json={"content": "{}"},
                           headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_validate_accepts_a_valid_candidate(key_store):
    key = key_store(capability="INTERNAL")
    response = client.post(
        "/api/v1/policy/validate",
        json={"content": json.dumps(VALID_POLICY)},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 200
    assert response.json() == {"valid": True, "errors": []}


def test_validate_rejects_missing_default_tenant(key_store):
    key = key_store(capability="INTERNAL")
    bad = {"default_action": "BLOCK", "tenants": {"acme": {"policies": {}}}}
    response = client.post(
        "/api/v1/policy/validate", json={"content": json.dumps(bad)},
        headers={"Authorization": f"Bearer {key}"},
    )
    body = response.json()
    assert body["valid"] is False
    assert len(body["errors"]) > 0


def test_validate_does_not_touch_the_live_file(key_store, _isolated_policy_files):
    key = key_store(capability="INTERNAL")
    bad = "{ not even valid json"
    client.post("/api/v1/policy/validate", json={"content": bad},
               headers={"Authorization": f"Bearer {key}"})
    assert json.loads(_isolated_policy_files.read_text(encoding="utf-8")) == VALID_POLICY


# --- POST /api/v1/policy/deploy ----------------------------------------------------

def test_deploy_requires_internal(key_store):
    key = key_store(capability="ELEVATED")
    response = client.post("/api/v1/policy/deploy", json={"content": json.dumps(VALID_POLICY)},
                           headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_deploy_refuses_an_invalid_policy_and_leaves_the_live_file_untouched(key_store, _isolated_policy_files):
    key = key_store(capability="INTERNAL")
    bad = {"default_action": "BLOCK", "tenants": {"acme": {"policies": {}}}}
    response = client.post("/api/v1/policy/deploy", json={"content": json.dumps(bad)},
                           headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 422
    assert json.loads(_isolated_policy_files.read_text(encoding="utf-8")) == VALID_POLICY


def test_deploy_a_valid_candidate_actually_takes_effect(key_store, _isolated_policy_files):
    key = key_store(capability="INTERNAL")
    new_policy = {
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "ALLOW", "LOW": "ALLOW"}}}},
    }
    response = client.post("/api/v1/policy/deploy", json={"content": json.dumps(new_policy)},
                           headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert response.json()["deployed"] is True
    assert json.loads(_isolated_policy_files.read_text(encoding="utf-8")) == new_policy

    from core.policy import policy_decision
    decision, _reason = policy_decision("GENERAL", "HIGH", "default")
    assert decision == "ALLOW"  # the reloaded live policy is what's now enforced


def test_deploy_snapshots_the_previous_version(key_store):
    key = key_store(capability="INTERNAL")
    new_policy = {"default_action": "BLOCK", "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}}}}
    response = client.post("/api/v1/policy/deploy", json={"content": json.dumps(new_policy)},
                           headers={"Authorization": f"Bearer {key}"})
    assert response.json()["previous_version"] is not None

    versions_response = client.get("/api/v1/policy", headers={"Authorization": f"Bearer {key}"})
    assert len(versions_response.json()["versions"]) == 1


# --- POST /api/v1/policy/rollback --------------------------------------------------

def test_rollback_requires_internal(key_store):
    key = key_store(capability="GENERAL")
    response = client.post("/api/v1/policy/rollback", json={"version": "whatever"},
                           headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403


def test_rollback_unknown_version_is_404(key_store):
    key = key_store(capability="INTERNAL")
    response = client.post("/api/v1/policy/rollback", json={"version": "no-such-version.json"},
                           headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 404


def test_rollback_restores_the_previous_live_policy(key_store, _isolated_policy_files):
    key = key_store(capability="INTERNAL")
    new_policy = {"default_action": "BLOCK", "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}}}}
    deploy_response = client.post("/api/v1/policy/deploy", json={"content": json.dumps(new_policy)},
                                  headers={"Authorization": f"Bearer {key}"})
    previous_version = deploy_response.json()["previous_version"]

    rollback_response = client.post("/api/v1/policy/rollback", json={"version": previous_version},
                                    headers={"Authorization": f"Bearer {key}"})
    assert rollback_response.status_code == 200
    assert json.loads(_isolated_policy_files.read_text(encoding="utf-8")) == VALID_POLICY


# --- Phase 8 hardening: policy changes are a distinct, security-relevant metric ---

def _counter_value(action, outcome):
    from core.metrics import policy_changes_total
    return policy_changes_total.labels(action=action, outcome=outcome)._value.get()


def test_successful_deploy_increments_the_success_counter(key_store):
    key = key_store(capability="INTERNAL")
    before = _counter_value("deploy", "success")
    new_policy = {"default_action": "BLOCK", "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}}}}
    client.post("/api/v1/policy/deploy", json={"content": json.dumps(new_policy)},
               headers={"Authorization": f"Bearer {key}"})
    assert _counter_value("deploy", "success") == before + 1


def test_rejected_deploy_increments_the_rejected_counter_not_success(key_store):
    key = key_store(capability="INTERNAL")
    before_rejected = _counter_value("deploy", "rejected")
    before_success = _counter_value("deploy", "success")
    bad = {"default_action": "BLOCK", "tenants": {"acme": {"policies": {}}}}
    client.post("/api/v1/policy/deploy", json={"content": json.dumps(bad)},
               headers={"Authorization": f"Bearer {key}"})
    assert _counter_value("deploy", "rejected") == before_rejected + 1
    assert _counter_value("deploy", "success") == before_success


def test_successful_rollback_increments_the_success_counter(key_store):
    key = key_store(capability="INTERNAL")
    new_policy = {"default_action": "BLOCK", "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}}}}
    deploy_response = client.post("/api/v1/policy/deploy", json={"content": json.dumps(new_policy)},
                                  headers={"Authorization": f"Bearer {key}"})
    previous_version = deploy_response.json()["previous_version"]

    before = _counter_value("rollback", "success")
    client.post("/api/v1/policy/rollback", json={"version": previous_version},
               headers={"Authorization": f"Bearer {key}"})
    assert _counter_value("rollback", "success") == before + 1


def test_failed_rollback_increments_the_rejected_counter(key_store):
    key = key_store(capability="INTERNAL")
    before = _counter_value("rollback", "rejected")
    client.post("/api/v1/policy/rollback", json={"version": "no-such-version.json"},
               headers={"Authorization": f"Bearer {key}"})
    assert _counter_value("rollback", "rejected") == before + 1

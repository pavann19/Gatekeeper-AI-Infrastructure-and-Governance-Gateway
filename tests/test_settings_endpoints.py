"""
Tests for GET /api/v1/settings/privacy and GET /api/v1/settings/protection
(Phase 7, Client UI remainder). Both read REAL running configuration --
core/privacy.py's actual regex/NER config, and core/tenancy.py +
core/policy.py's actual tenant SLA and policy mapping -- not a hand-written
description of them.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import auth as auth_mod
from core import policy as policy_mod
from core import tenancy as tenancy_mod
from core.auth import KeyStore, generate_key, hash_key
from core.policy import PolicyStore
from core.privacy import NER_LABELS, REGEX_PATTERNS
from core.tenancy import TenantStore

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


@pytest.fixture
def tenant_and_policy(tmp_path, monkeypatch):
    tenants_path = tmp_path / "tenants.json"
    tenants_path.write_text(json.dumps({
        "acme": {"display_name": "Acme Corp", "status": "active",
                 "rate_limit_rpm": 500, "token_quota_daily": 200000},
    }), encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(tenants_path)))

    policy_path = tmp_path / "policy_rules.json"
    policy_path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {
            "default": {"policies": {"GENERAL": {"HIGH": "BLOCK", "LOW": "ALLOW"}}},
            "acme": {"policies": {"GENERAL": {"HIGH": "RESTRICT", "LOW": "ALLOW"},
                                  "ELEVATED": {"HIGH": "ALLOW"}}},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(policy_path)))


# --- /api/v1/settings/privacy ------------------------------------------------------

def test_privacy_settings_reflects_real_regex_categories(key_store):
    key = key_store(capability="GENERAL")
    response = client.get("/api/v1/settings/privacy", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert set(response.json()["regex_categories"]) == set(REGEX_PATTERNS.keys())
    assert set(response.json()["ner_labels"]) == set(NER_LABELS)


def test_privacy_settings_same_for_every_capability(key_store):
    general_key = key_store(capability="GENERAL", key_id="g")
    internal_key = key_store(capability="INTERNAL", key_id="i")
    r1 = client.get("/api/v1/settings/privacy", headers={"Authorization": f"Bearer {general_key}"})
    r2 = client.get("/api/v1/settings/privacy", headers={"Authorization": f"Bearer {internal_key}"})
    assert r1.json() == r2.json()


# --- /api/v1/settings/protection ---------------------------------------------------

def test_protection_settings_returns_callers_own_tenant_and_policy(key_store, tenant_and_policy):
    key = key_store(capability="GENERAL", tenant="acme")
    response = client.get("/api/v1/settings/protection", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    body = response.json()
    assert body["tenant"]["tenant_id"] == "acme"
    assert body["tenant"]["rate_limit_rpm"] == 500
    assert body["tenant"]["token_quota_daily"] == 200000
    assert body["policy"]["capability"] == "GENERAL"
    assert body["policy"]["risk_to_action"] == {"HIGH": "RESTRICT", "LOW": "ALLOW"}
    assert body["policy"]["default_action"] == "BLOCK"


def test_protection_settings_shows_the_callers_own_capability_mapping_not_others(key_store, tenant_and_policy):
    key = key_store(capability="ELEVATED", tenant="acme")
    response = client.get("/api/v1/settings/protection", headers={"Authorization": f"Bearer {key}"})
    body = response.json()
    assert body["policy"]["capability"] == "ELEVATED"
    assert body["policy"]["risk_to_action"] == {"HIGH": "ALLOW"}


def test_non_internal_cannot_view_another_tenants_settings(key_store, tenant_and_policy):
    key = key_store(capability="GENERAL", tenant="default")
    response = client.get("/api/v1/settings/protection?tenant=acme",
                          headers={"Authorization": f"Bearer {key}"})
    assert response.json()["tenant"]["tenant_id"] == "default"


def test_internal_can_view_another_tenants_settings(key_store, tenant_and_policy):
    key = key_store(capability="INTERNAL", tenant="ops")
    response = client.get("/api/v1/settings/protection?tenant=acme",
                          headers={"Authorization": f"Bearer {key}"})
    assert response.json()["tenant"]["tenant_id"] == "acme"


def test_unconfigured_tenant_falls_back_to_default_tenant_config(key_store):
    key = key_store(capability="GENERAL", tenant="never-configured")
    response = client.get("/api/v1/settings/protection", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200
    assert response.json()["tenant"]["tenant_id"] == "default"

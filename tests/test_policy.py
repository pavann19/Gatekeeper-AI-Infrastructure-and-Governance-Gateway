"""
Tests for tenant-scoped policy (core/policy.py) — the "Policy Context" box
in the V2 reference architecture, immediately after Tenant Resolver.

Scope, per the module docstring: (capability, risk_level) -> action, per
tenant. NOT risk-threshold policy — that stays identical across tenants.
Mirrors tests/test_tenancy.py's structure, which mirrors tests/test_auth.py's
— all three stores (auth, tenancy, policy) are built to the same shape.
"""
import json

import pytest

from core import policy as policy_mod
from core.policy import (
    FAIL_SAFE,
    PolicyStore,
    policy_decision,
    reload_policies,
    resolve_policy_set,
)

DEFAULT_POLICIES = {
    "GENERAL": {"HIGH": "BLOCK", "MEDIUM": "RESTRICT", "LOW": "ALLOW"},
}


@pytest.fixture
def policy_store(tmp_path, monkeypatch):
    """Installs an isolated policy store and returns a helper that writes it."""
    path = tmp_path / "policy_rules.json"
    data = {"default_action": "BLOCK", "tenants": {}}

    def write(tenant_id, policies, default_action=None):
        data["tenants"][tenant_id] = {"policies": policies}
        if default_action is not None:
            data["default_action"] = default_action
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))

    write("default", DEFAULT_POLICIES)
    write.path = path
    write.data = data
    return write


# --- backward compatibility: unmodified call site, single global policy -----

def test_default_tenant_matches_the_pre_tenancy_behaviour(policy_store):
    """policy_decision(capability, risk) with no tenant_id must behave
    exactly as the pre-tenancy single-policy module did."""
    assert policy_decision("GENERAL", "HIGH") == (
        "BLOCK", "Policy applied for GENERAL (Risk: HIGH, Tenant: default)"
    )
    assert policy_decision("GENERAL", "MEDIUM")[0] == "RESTRICT"
    assert policy_decision("GENERAL", "LOW")[0] == "ALLOW"


def test_unconfigured_tenant_falls_back_to_default_policy(policy_store):
    action, reason = policy_decision("GENERAL", "HIGH", "never-configured")
    assert action == "BLOCK"
    assert "never-configured" in reason


def test_fallback_reason_states_the_fallback_explicitly(policy_store):
    """
    An audit reason of 'Tenant: broken' when tenant 'broken' actually got
    the DEFAULT tenant's policy would be misleading — an investigator would
    conclude 'broken' has its own distinct policy. The reason must say a
    fallback happened, not just name whichever tenant was requested.
    """
    _, reason = policy_decision("GENERAL", "HIGH", "never-configured")
    assert "never-configured -> default" in reason


# --- the actual point: per-tenant divergence ---------------------------------

def test_two_tenants_get_different_decisions_for_the_same_risk(policy_store):
    """THE FEATURE. Same capability, same risk_level, different tenants,
    different action — this is what Policy Context exists to enable."""
    policy_store("default", {"GENERAL": {"MEDIUM": "RESTRICT"}})
    policy_store("strict-acme", {"GENERAL": {"MEDIUM": "BLOCK"}})

    default_action, _ = policy_decision("GENERAL", "MEDIUM", "default")
    strict_action, _ = policy_decision("GENERAL", "MEDIUM", "strict-acme")

    assert default_action == "RESTRICT"
    assert strict_action == "BLOCK"


def test_one_tenants_policy_is_independent_of_anothers(policy_store):
    policy_store("a", {"GENERAL": {"HIGH": "BLOCK"}})
    policy_store("b", {"GENERAL": {"HIGH": "ALLOW"}})

    assert policy_decision("GENERAL", "HIGH", "a")[0] == "BLOCK"
    assert policy_decision("GENERAL", "HIGH", "b")[0] == "ALLOW"


# --- undefined capability / risk, per-tenant default_action -----------------

def test_undefined_capability_uses_the_global_default_action(policy_store):
    policy_store("default", DEFAULT_POLICIES, default_action="RESTRICT")
    action, reason = policy_decision("NOT_A_REAL_CAPABILITY", "HIGH", "default")
    assert action == "RESTRICT"
    assert "not defined" in reason


def test_undefined_risk_level_uses_the_default_action(policy_store):
    policy_store("default", {"GENERAL": {"HIGH": "BLOCK"}}, default_action="RESTRICT")
    action, _ = policy_decision("GENERAL", "SOME_UNMAPPED_RISK", "default")
    assert action == "RESTRICT"


def test_default_action_is_shared_across_tenants(policy_store):
    """
    default_action is a global fail-safe backstop, deliberately NOT
    per-tenant (see module docstring) — every tenant falls back to the same
    value when their own capability policy doesn't cover a case.
    """
    policy_store("default", DEFAULT_POLICIES, default_action="ALLOW")
    policy_store("acme", {"GENERAL": {"HIGH": "BLOCK"}})  # no MEDIUM/LOW entries

    action, _ = policy_decision("GENERAL", "LOW", "acme")
    assert action == "ALLOW", "acme should fall back to the SAME global default_action"


# --- malformed input, per-entry not global -----------------------------------

def test_malformed_tenant_entry_falls_back_to_default_others_unaffected(tmp_path, monkeypatch):
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {
            "default": {"policies": DEFAULT_POLICIES},
            "broken": {"policies": "not-an-object"},
            "good": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))

    # broken falls back to default's real policy, not FAIL_SAFE — and the
    # reason says so explicitly rather than claiming "Tenant: broken".
    assert policy_decision("GENERAL", "HIGH", "broken") == (
        "BLOCK", "Policy applied for GENERAL (Risk: HIGH, Tenant: broken -> default (fallback))"
    )
    # good is entirely unaffected by broken's malformed entry.
    assert policy_decision("GENERAL", "HIGH", "good")[0] == "ALLOW"


def test_invalid_action_value_is_dropped_capability_falls_back(tmp_path, monkeypatch):
    """An action outside {BLOCK, RESTRICT, ALLOW} (typo, or a truncated
    write) must not become a silent no-op ALLOW."""
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {
            "default": {
                "policies": {"GENERAL": {"HIGH": "not-a-real-action", "LOW": "ALLOW"}}
            },
        },
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))

    # HIGH's bad value is dropped -> falls through to default_action.
    assert policy_decision("GENERAL", "HIGH", "default")[0] == "BLOCK"
    # LOW is untouched.
    assert policy_decision("GENERAL", "LOW", "default")[0] == "ALLOW"


def test_empty_policies_object_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {}}},
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))

    # "default" itself failed validation -> no usable policy at all.
    assert policy_decision("GENERAL", "HIGH", "default") == (
        "BLOCK", "System Error: Policies not loaded"
    )


# --- fail-safe: two DIFFERENT "something is wrong" states -------------------

def test_missing_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(tmp_path / "absent.json")))
    assert policy_decision("GENERAL", "HIGH") == ("BLOCK", "System Error: Policies not loaded")


def test_corrupt_json_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "policy_rules.json"
    path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))
    assert policy_decision("GENERAL", "HIGH") == ("BLOCK", "System Error: Policies not loaded")


def test_missing_default_tenant_fails_closed_for_everyone(tmp_path, monkeypatch):
    """
    Unlike TenantStore (unconfigured tenant -> DEFAULT_TENANT, active), a
    policy store with no usable 'default' tenant must fail closed for EVERY
    tenant, not just the unconfigured ones — there is nothing safe to fall
    back to when the fallback itself doesn't exist.
    """
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"acme": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}}},
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))

    assert policy_decision("GENERAL", "HIGH", "acme") == (
        "BLOCK", "System Error: Policies not loaded"
    )
    assert resolve_policy_set("acme") is FAIL_SAFE


def test_non_dict_tenants_value_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({"default_action": "BLOCK", "tenants": "not-an-object"}),
                    encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))
    assert policy_decision("GENERAL", "HIGH") == ("BLOCK", "System Error: Policies not loaded")


@pytest.mark.parametrize("bad_default", ["not-a-real-action", 123, None])
def test_invalid_top_level_default_action_falls_back_to_block(tmp_path, monkeypatch, bad_default):
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": bad_default,
        "tenants": {"default": {"policies": {"GENERAL": {}}}},
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))

    action, _ = policy_decision("GENERAL", "ANYTHING", "default")
    assert action == "BLOCK"


# --- reload -------------------------------------------------------------------

def test_reload_picks_up_a_policy_change(policy_store):
    policy_store("acme", {"GENERAL": {"HIGH": "ALLOW"}})
    assert policy_decision("GENERAL", "HIGH", "acme")[0] == "ALLOW"

    policy_store("acme", {"GENERAL": {"HIGH": "BLOCK"}})
    reload_policies()
    assert policy_decision("GENERAL", "HIGH", "acme")[0] == "BLOCK"


def test_without_reload_the_store_is_cached(tmp_path, monkeypatch):
    """Load-once-and-cache, matching KeyStore/TenantStore — pinned so a
    future change to it is a conscious decision."""
    path = tmp_path / "policy_rules.json"
    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}}},
    }), encoding="utf-8")
    monkeypatch.setattr(policy_mod, "_store", PolicyStore(str(path)))
    policy_decision("GENERAL", "HIGH")  # triggers initial load

    path.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK"}}}},
    }), encoding="utf-8")
    assert policy_decision("GENERAL", "HIGH")[0] == "ALLOW"  # stale cache, on purpose

    reload_policies()
    assert policy_decision("GENERAL", "HIGH")[0] == "BLOCK"

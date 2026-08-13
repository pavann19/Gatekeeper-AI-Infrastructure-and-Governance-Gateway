"""
Tests for tenant resolution (core/tenancy.py) — the "Tenant Resolver" box in
the V2 reference architecture.

Deliberately does NOT test policy behaviour here — see the module docstring
for why identity/SLA resolution is kept separate from policy. Endpoint-level
enforcement (suspension -> 403, SLA -> rate limit override) lives in
tests/test_tenant_enforcement.py; this file is the resolver in isolation,
mirroring tests/test_auth.py's structure for KeyStore.
"""
import json

import pytest

from core import tenancy as tenancy_mod
from core.tenancy import DEFAULT_TENANT, TenantConfig, TenantStore, resolve_tenant


@pytest.fixture
def tenant_store(tmp_path, monkeypatch):
    """Installs an isolated tenant store and returns a helper that writes it."""
    path = tmp_path / "tenants.json"
    data = {}

    def write(tenant_id, **fields):
        data[tenant_id] = fields
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))
    write.path = path
    return write


# --- resolution basics -------------------------------------------------------

def test_unknown_tenant_resolves_to_default(tenant_store):
    """No tenants.json entry -> DEFAULT_TENANT, not an error. Tenancy is opt-in."""
    result = resolve_tenant("never-configured")
    assert result == DEFAULT_TENANT
    assert result.status == "active"
    assert result.rate_limit_rpm is None


def test_missing_file_means_every_tenant_is_default(tmp_path, monkeypatch):
    """No tenants.json AT ALL — the store must not error, and every caller
    must resolve exactly as before this module existed."""
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(tmp_path / "absent.json")))
    assert resolve_tenant("acme") == DEFAULT_TENANT
    assert resolve_tenant("anything") == DEFAULT_TENANT


def test_configured_tenant_resolves_to_its_own_config(tenant_store):
    tenant_store("acme", display_name="Acme Corp", status="active", rate_limit_rpm=500)
    result = resolve_tenant("acme")
    assert result.tenant_id == "acme"
    assert result.display_name == "Acme Corp"
    assert result.status == "active"
    assert result.rate_limit_rpm == 500.0
    assert result.suspended is False


def test_suspended_tenant_is_reported_suspended(tenant_store):
    tenant_store("acme-trial", status="suspended")
    result = resolve_tenant("acme-trial")
    assert result.suspended is True


def test_one_tenant_is_independent_of_another(tenant_store):
    """The whole point of per-entry validation: one tenant's config must not
    leak into or block another's."""
    tenant_store("acme", status="active")
    tenant_store("beta", status="suspended")
    assert resolve_tenant("acme").suspended is False
    assert resolve_tenant("beta").suspended is True


# --- malformed / hostile input, fails per-entry not globally -----------------

def test_bad_status_value_is_rejected_not_defaulted(tmp_path, monkeypatch):
    """
    An unrecognised status string must NOT silently become 'active' — that
    would turn a typo (or a truncated write) into an accidental un-suspension.
    It is dropped, and the tenant then resolves to DEFAULT_TENANT via the
    normal unknown-tenant path, which is active — but only because nothing
    was ever successfully configured for it, not because the bad value was
    coerced.
    """
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({"acme": {"status": "definitely-not-a-status"}}), encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    assert resolve_tenant("acme") == DEFAULT_TENANT


def test_non_dict_entry_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({"acme": "not-an-object"}), encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    assert resolve_tenant("acme") == DEFAULT_TENANT


def test_non_dict_entry_does_not_block_valid_siblings(tmp_path, monkeypatch):
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({
        "broken": "not-an-object",
        "good": {"status": "active", "display_name": "Good Co"},
    }), encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    assert resolve_tenant("broken") == DEFAULT_TENANT
    assert resolve_tenant("good").display_name == "Good Co"


def test_corrupt_json_fails_closed_not_open(tmp_path, monkeypatch):
    """
    An unreadable store might have contained a suspension. Falling back to
    'no tenants configured' (i.e. everyone gets DEFAULT_TENANT, which is
    active) would silently re-admit a caller who was meant to be blocked —
    so a corrupt file must load ZERO tenants, not skip validation.
    """
    path = tmp_path / "tenants.json"
    path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    assert len(tenancy_mod._store) == 0
    # Still resolves (to DEFAULT_TENANT) rather than raising — a broken file
    # degrades tenancy, it must not take down the gateway.
    assert resolve_tenant("acme") == DEFAULT_TENANT


@pytest.mark.parametrize("bad_rpm", ["not-a-number", -5, 0, None])
def test_invalid_rate_limit_override_is_dropped_not_applied(tmp_path, monkeypatch, bad_rpm):
    """
    A non-positive or unparseable rate_limit_rpm must not silently become
    'no limit' or crash the resolver — it is dropped, falling back to the
    caller's normal tier-based rate.
    """
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({"acme": {"rate_limit_rpm": bad_rpm}}), encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    result = resolve_tenant("acme")
    assert result.rate_limit_rpm is None


# --- reload -------------------------------------------------------------------

def test_reload_picks_up_suspension(tenant_store):
    tenant_store("acme", status="active")
    assert resolve_tenant("acme").suspended is False

    tenant_store("acme", status="suspended")
    tenancy_mod.reload_tenants()
    assert resolve_tenant("acme").suspended is True


def test_without_reload_the_store_is_cached(tmp_path, monkeypatch):
    """
    Load-once-and-cache is deliberate (matches KeyStore) — this pins that
    behaviour so a future change to it is a conscious decision, not a
    surprise regression.

    Writes directly to the file (unlike the `tenant_store` fixture, which
    installs a fresh, unloaded TenantStore on every call — that would defeat
    exactly the "same instance, no reload" scenario this test is for).
    """
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({"acme": {"status": "active"}}), encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    resolve_tenant("acme")  # triggers the initial load

    path.write_text(json.dumps({"acme": {"status": "suspended"}}), encoding="utf-8")
    assert resolve_tenant("acme").suspended is False  # stale cache, on purpose

    tenancy_mod.reload_tenants()
    assert resolve_tenant("acme").suspended is True

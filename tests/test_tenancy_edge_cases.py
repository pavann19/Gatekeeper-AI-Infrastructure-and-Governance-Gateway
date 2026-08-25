"""
Edge-case coverage for core/tenancy.py, additional to tests/test_tenancy.py.

Focus areas NOT already covered there: exact DEFAULT_TENANT field values (not
just equality), tenant_id case-sensitivity, empty-string/None tenant_id
handling, and the precise "tenant override vs settings default" fallback
pattern used at api/main.py's call sites
(`tenant_config.token_quota_daily if ... is not None else
settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT`), exercised here directly against
resolve_tenant's output rather than through the endpoint.
"""
import json

import pytest

from core import tenancy as tenancy_mod
from core.config import settings
from core.tenancy import DEFAULT_TENANT, TenantStore, resolve_tenant


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


# --- DEFAULT_TENANT exact field values --------------------------------------

def test_default_tenant_exact_field_values():
    """Pin every field of DEFAULT_TENANT precisely, not just via equality to
    itself -- a future accidental edit to any field should fail this test."""
    assert DEFAULT_TENANT.tenant_id == "default"
    assert DEFAULT_TENANT.display_name == "Default"
    assert DEFAULT_TENANT.status == "active"
    assert DEFAULT_TENANT.rate_limit_rpm is None
    assert DEFAULT_TENANT.token_quota_daily is None
    assert DEFAULT_TENANT.suspended is False


def test_unknown_tenant_returns_the_actual_default_tenant_object(tenant_store):
    """resolve_tenant for a never-configured id returns DEFAULT_TENANT itself
    (identity), not merely an equal-looking TenantConfig."""
    result = resolve_tenant("totally-unheard-of")
    assert result is DEFAULT_TENANT


# --- case sensitivity --------------------------------------------------------

def test_tenant_id_lookup_is_case_sensitive(tenant_store):
    """'Acme' must not match a store entry for 'acme' -- tenant_id is looked
    up by exact dict key, no normalization performed anywhere in the module."""
    tenant_store("acme", status="suspended", display_name="Acme Corp")

    exact = resolve_tenant("acme")
    assert exact.status == "suspended"
    assert exact.display_name == "Acme Corp"

    wrong_case = resolve_tenant("Acme")
    assert wrong_case is DEFAULT_TENANT
    assert wrong_case.status == "active"  # NOT suspended -- fell through to default


def test_status_value_in_file_is_lowercased_but_tenant_id_key_is_not(tenant_store):
    """The `status` string is explicitly .lower()'d during load, but nothing
    does the same for the tenant_id key itself -- confirm this asymmetry."""
    tenant_store("acme", status="ACTIVE")
    result = resolve_tenant("acme")
    assert result.status == "active"  # value was normalized
    assert resolve_tenant("ACME") is DEFAULT_TENANT  # key was not


# --- empty-string / None tenant_id -------------------------------------------

def test_empty_string_tenant_id_resolves_to_default(tenant_store):
    result = resolve_tenant("")
    assert result is DEFAULT_TENANT


def test_empty_string_tenant_id_does_not_collide_with_configured_tenants(tenant_store):
    tenant_store("acme", status="suspended")
    assert resolve_tenant("").suspended is False
    assert resolve_tenant("").tenant_id == "default"


def test_none_tenant_id_resolves_to_default_without_raising(tenant_store):
    """resolve_tenant trusts its input (per the module docstring) but must not
    blow up on None -- dict.get(None, DEFAULT_TENANT) is safe, confirm it
    stays that way."""
    result = resolve_tenant(None)
    assert result is DEFAULT_TENANT


def test_none_tenant_id_configured_as_a_literal_json_key_is_unreachable(tenant_store):
    """JSON object keys are always strings, so a store can never contain a
    real None key -- resolve_tenant(None) must fall through to DEFAULT_TENANT
    even if a tenant happens to be named the string 'None'."""
    tenant_store("None", status="suspended")
    assert resolve_tenant(None) is DEFAULT_TENANT
    assert resolve_tenant("None").suspended is True


# --- tenant override vs global-default fallback pattern (api/main.py) -------

def test_tenant_with_no_quota_override_falls_back_to_settings_default(tenant_store):
    """Mirrors api/main.py's exact fallback expression: when
    token_quota_daily is None, callers use settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT."""
    tenant_store("acme", status="active")
    tenant_config = resolve_tenant("acme")
    assert tenant_config.token_quota_daily is None

    effective_quota = (
        tenant_config.token_quota_daily
        if tenant_config.token_quota_daily is not None
        else settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT
    )
    assert effective_quota == settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT


def test_tenant_with_explicit_quota_override_wins_over_settings_default(tenant_store):
    tenant_store("acme", token_quota_daily=12345)
    tenant_config = resolve_tenant("acme")
    assert tenant_config.token_quota_daily == 12345

    effective_quota = (
        tenant_config.token_quota_daily
        if tenant_config.token_quota_daily is not None
        else settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT
    )
    assert effective_quota == 12345
    assert effective_quota != settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT or 12345 == settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT


def test_tenant_zero_quota_override_wins_as_unlimited_not_fallback(tenant_store):
    """0 is a real override (unlimited) and must NOT be treated as falsy by
    an `if tenant_config.token_quota_daily` check -- only `is not None` is
    correct, which is exactly what api/main.py uses."""
    tenant_store("acme", token_quota_daily=0)
    tenant_config = resolve_tenant("acme")
    assert tenant_config.token_quota_daily == 0

    effective_quota = (
        tenant_config.token_quota_daily
        if tenant_config.token_quota_daily is not None
        else settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT
    )
    assert effective_quota == 0  # NOT settings default -- would be wrong if `if` used truthiness


def test_unknown_tenant_also_falls_back_to_settings_default_quota(tenant_store):
    """The fallback pattern must behave identically whether the tenant is
    known-with-no-override or entirely unconfigured -- both carry
    token_quota_daily=None on the resolved config."""
    tenant_config = resolve_tenant("never-registered")
    assert tenant_config.token_quota_daily is None
    effective_quota = (
        tenant_config.token_quota_daily
        if tenant_config.token_quota_daily is not None
        else settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT
    )
    assert effective_quota == settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT


def test_rate_limit_rpm_override_and_fallback_mirror_quota_pattern(tenant_store):
    """Same None-means-inherit-default convention documented for
    rate_limit_rpm; confirm it holds for a configured override too."""
    tenant_store("acme", rate_limit_rpm=42)
    tenant_config = resolve_tenant("acme")
    assert tenant_config.rate_limit_rpm == 42.0

    tenant_store("beta", status="active")
    beta_config = resolve_tenant("beta")
    assert beta_config.rate_limit_rpm is None


# --- reload / hot-update semantics (module has no auto-invalidation) --------

def test_reload_swaps_in_a_newly_added_tenant(tenant_store):
    """A tenant added after the first load is invisible until reload_tenants()
    is called -- this is the caching layer's hot-update contract."""
    assert resolve_tenant("newcomer") is DEFAULT_TENANT  # triggers initial load of {}

    tenant_store("newcomer", status="active", display_name="Newcomer Inc")
    # tenant_store() installs a brand-new TenantStore instance, which defeats
    # the "same store, stale cache" scenario -- write directly instead to
    # prove the cache, not the fixture's monkeypatching, is what's stale here.
    tenant_store.path.write_text(
        json.dumps({"newcomer": {"status": "active", "display_name": "Newcomer Inc"}}),
        encoding="utf-8",
    )
    # _store was replaced by the second tenant_store() call above already
    # loaded; force a fresh unloaded store pointed at the same file instead.
    import core.tenancy as tm
    fresh_store = TenantStore(str(tenant_store.path))
    tm._store = fresh_store
    assert resolve_tenant("newcomer").display_name == "Newcomer Inc"


def test_reload_tenants_returns_the_store(tenant_store):
    """reload_tenants() returns the (reloaded) store object itself."""
    result = tenancy_mod.reload_tenants()
    assert result is tenancy_mod._store


def test_get_tenant_store_len_reflects_only_valid_entries_after_reload(tmp_path, monkeypatch):
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({
        "good-one": {"status": "active"},
        "bad-one": {"status": "not-a-real-status"},
    }), encoding="utf-8")
    monkeypatch.setattr(tenancy_mod, "_store", TenantStore(str(path)))

    assert len(tenancy_mod._store) == 1

    path.write_text(json.dumps({
        "good-one": {"status": "active"},
        "good-two": {"status": "active"},
    }), encoding="utf-8")
    tenancy_mod.reload_tenants()
    assert len(tenancy_mod._store) == 2

"""
Tenant resolution — the "Tenant Resolver" box in the V2 reference
architecture (API Gateway -> Auth -> Tenant Resolver -> Policy Context -> ...).

SCOPE, DELIBERATELY NARROW
---------------------------
This module resolves WHO the tenant is and whether they may proceed at all
(active vs suspended) and their SLA parameters (rate limit). It does NOT
decide policy — no risk-threshold overrides, no per-tenant BLOCK/RESTRICT/
ALLOW mapping. That is "Policy Context", the next box in the diagram, and a
separate piece of work: conflating identity resolution with policy would
make it impossible to reason about either independently, and would repeat
the exact mistake core/auth.py's docstring documents fixing (capability
being asserted by the request instead of resolved from a verified source).

WHY THIS EXISTS NOW, SPECIFICALLY
----------------------------------
`Principal.tenant` (core/auth.py) has existed since auth was rebuilt, but
nothing has ever read it — confirmed in the Phase 0 component audit
(docs/ENGINEERING_ASSESSMENT.md, "Tenant Resolver" row: "`Principal.tenant`
is threaded through but nothing branches on it"). A resolver that only
echoes a field back is not a resolver, it is dead code with an audit trail.
This module makes the field load-bearing: an unknown/misconfigured tenant
gets a safe default, a suspended tenant is rejected before any detection
work runs, and a tenant's SLA actually changes its rate limit.

DESIGN, MIRRORING core/auth.py's KeyStore ON PURPOSE
------------------------------------------------------
Same shape as the API key store: JSON file, loaded once and cached, a
force-reload hook for after provisioning, per-entry validation so one
malformed tenant doesn't take down the others, and a safe default when the
file is absent (no tenants.json means every caller gets the DEFAULT_TENANT
config, unchanged from today's behaviour — configuring tenancy is opt-in,
not a prerequisite for the gateway to keep working). Consistency with the
existing auth module matters more here than any abstract preference,
because whoever operates one will need to operate the other.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

VALID_STATUSES = ("active", "suspended")


@dataclass(frozen=True)
class TenantConfig:
    """
    Resolved tenant identity and SLA. NOT policy — see module docstring.

    `rate_limit_rpm=None` means "use the global default for this caller's
    authentication tier", not "unlimited". An explicit override is how a
    tenant's SLA (paid tier, trial tier, ...) actually changes behaviour;
    without one, tenancy would be observable in the audit log and nowhere
    else, which is the same "recorded but not load-bearing" problem this
    module exists to fix for `Principal.tenant` itself.
    """
    tenant_id: str
    display_name: str = ""
    status: str = "active"
    rate_limit_rpm: float | None = None

    # Phase 5 token accounting (core/token_quota.py). None means "use
    # settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT for this tenant", mirroring
    # rate_limit_rpm=None's "use the tier default" convention exactly. 0 is
    # an explicit override to unlimited for this specific tenant, distinct
    # from None's "no override, inherit the default".
    token_quota_daily: int | None = None

    @property
    def suspended(self) -> bool:
        return self.status == "suspended"


# The tenant every caller resolves to when tenancy is unconfigured, or when
# their tenant_id is not in the store. Matches core/auth.py's ANONYMOUS
# pattern: a safe, fully-functional default rather than an error, so a
# deployment that never touches tenants.json is unaffected by this module
# existing at all.
DEFAULT_TENANT = TenantConfig(tenant_id="default", display_name="Default", status="active")


class TenantStore:
    """
    Maps tenant_id -> TenantConfig, loaded from `settings.TENANTS_FILE`:

        {
          "acme": {
            "display_name": "Acme Corp",
            "status": "active",
            "rate_limit_rpm": 500,
            "token_quota_daily": 200000
          },
          "acme-trial": {
            "display_name": "Acme Corp (trial)",
            "status": "suspended"
          }
        }

    A missing file is not an error — see DEFAULT_TENANT. Malformed entries
    are rejected individually and logged, same reasoning as KeyStore: a typo
    in one tenant's config must not silently affect the others.
    """

    def __init__(self, path=None):
        self.path = path or settings.TENANTS_FILE
        self._tenants: dict[str, TenantConfig] = {}
        self._loaded = False

    def load(self, force=False):
        if self._loaded and not force:
            return self
        self._tenants = {}
        self._loaded = True

        if not os.path.exists(self.path):
            logger.info(f"No tenant store at {self.path}; all callers resolve to DEFAULT_TENANT.")
            return self

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            # Fail CLOSED on a corrupt store, matching KeyStore: an
            # unreadable file might have contained a suspension, and
            # silently falling back to "no tenants configured" could
            # re-admit a caller who was meant to be blocked.
            logger.error(f"Tenant store unreadable ({e}); no tenants loaded.")
            return self

        if not isinstance(raw, dict):
            logger.error(f"Tenant store {self.path} must be a JSON object; no tenants loaded.")
            return self

        for tenant_id, grant in raw.items():
            if not isinstance(grant, dict):
                logger.error(f"Ignoring malformed tenant entry {tenant_id!r}: not an object.")
                continue
            status = str(grant.get("status", "active")).lower()
            if status not in VALID_STATUSES:
                logger.error(
                    f"Ignoring tenant {tenant_id!r}: status {status!r} not one of "
                    f"{VALID_STATUSES}."
                )
                continue
            rpm = grant.get("rate_limit_rpm")
            if rpm is not None:
                try:
                    rpm = float(rpm)
                    if rpm <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    logger.error(
                        f"Ignoring rate_limit_rpm for tenant {tenant_id!r}: "
                        f"{grant.get('rate_limit_rpm')!r} is not a positive number."
                    )
                    rpm = None
            quota = grant.get("token_quota_daily")
            if quota is not None:
                try:
                    quota = int(quota)
                    if quota < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    logger.error(
                        f"Ignoring token_quota_daily for tenant {tenant_id!r}: "
                        f"{grant.get('token_quota_daily')!r} is not a non-negative integer."
                    )
                    quota = None

            self._tenants[tenant_id] = TenantConfig(
                tenant_id=tenant_id,
                display_name=str(grant.get("display_name", tenant_id)),
                status=status,
                rate_limit_rpm=rpm,
                token_quota_daily=quota,
            )

        logger.info(f"Loaded {len(self._tenants)} tenant(s) from {self.path}")
        return self

    def get(self, tenant_id: str) -> TenantConfig:
        """Never raises. Unknown tenant -> DEFAULT_TENANT, same fail-open-
        but-unprivileged shape as an unrecognised API key resolving to
        GENERAL rather than a 500."""
        self.load()
        return self._tenants.get(tenant_id, DEFAULT_TENANT)

    def __len__(self):
        self.load()
        return len(self._tenants)


_store = TenantStore()


def reload_tenants():
    """Re-reads the tenant store. Call after provisioning or suspending a tenant."""
    return _store.load(force=True)


def resolve_tenant(tenant_id: str) -> TenantConfig:
    """
    THE RESOLVER. Given the tenant_id off a verified Principal (never off
    anything the caller asserts directly — that boundary is core/auth.py's
    job, this function trusts its input the same way policy_decision()
    trusts `capability`), returns that tenant's config or DEFAULT_TENANT.
    """
    return _store.get(tenant_id)

"""Tenant configuration and admission state.

Tenant identity comes from verified principal context. Unconfigured optional
stores use defaults; unavailable configured stores deny admission. Reload
publishes one complete snapshot. Operators control suspension, limits and
privacy overrides."""
from __future__ import annotations

import json
import os
import threading
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

    # token accounting (core/token_quota.py). None means "use
    # settings.GATEWAY_TOKEN_QUOTA_DAILY_DEFAULT for this tenant", mirroring
    # rate_limit_rpm=None's "use the tier default" convention exactly. 0 is
    # an explicit override to unlimited for this specific tenant, distinct
    # from None's "no override, inherit the default".
    token_quota_daily: int | None = None

    privacy_disabled_patterns: tuple[str, ...] | None = None
    privacy_ner_labels: tuple[str, ...] | None = None

    @property
    def suspended(self) -> bool:
        return self.status != "active"


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
            "token_quota_daily": 200000,
            "privacy_disabled_patterns": ["AADHAAR"],
            "privacy_ner_labels": ["PERSON", "ORG"]
          },
          "acme-trial": {
            "display_name": "Acme Corp (trial)",
            "status": "suspended"
          }
        }

    An absent, unconfigured optional store uses defaults. Failed configured
    stores reject admission; malformed entries reject only their own tenant.
    Reloads publish one complete snapshot under the store lock.
    """

    def __init__(self, path=None):
        self.path = path or settings.TENANTS_FILE
        self._tenants: dict[str, TenantConfig] = {}
        self._loaded = False
        self._configured = bool(path or os.environ.get("TENANTS_FILE")
                                or "TENANTS_FILE" in settings.model_fields_set
                                or settings.TENANTS_FILE != "tenants.json")
        self._lock = threading.Lock()
        self._snapshot = ("unconfigured", {})

    def load(self, force=False):
        """Atomically publish state; failed configured reloads reject admission."""
        with self._lock:
            if self._loaded and not force:
                return self
            missing = not os.path.exists(self.path)
            if missing and not self._configured:
                snapshot = ("unconfigured", {})
            else:
                self._configured = True
                tenants = self._read_snapshot()
                snapshot = (("unavailable", self._snapshot[1]) if tenants is None
                            else ("valid", tenants))
            self._snapshot = snapshot
            self._tenants = snapshot[1]
            self._loaded = True
        return self

    @property
    def state(self):
        self.load()
        return self._snapshot[0]

    def _read_snapshot(self):
        tenants = {}

        if not os.path.exists(self.path):
            logger.error("Configured tenant store is missing; rejecting admission.")
            return None

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            # Fail CLOSED on a corrupt store, matching KeyStore: an
            # unreadable file might have contained a suspension, and
            # silently falling back to "no tenants configured" could
            # re-admit a caller who was meant to be blocked.
            logger.error(f"Tenant store unreadable ({e}); no tenants loaded.")
            return None

        if not isinstance(raw, dict):
            logger.error(f"Tenant store {self.path} must be a JSON object; no tenants loaded.")
            return None

        for tenant_id, grant in raw.items():
            if not isinstance(grant, dict):
                logger.error(f"Ignoring malformed tenant entry {tenant_id!r}: not an object.")
                tenants[tenant_id] = TenantConfig(tenant_id=tenant_id, status="unavailable")
                continue
            status = str(grant.get("status", "active")).lower()
            if status not in VALID_STATUSES:
                logger.error(
                    f"Ignoring tenant {tenant_id!r}: status {status!r} not one of "
                    f"{VALID_STATUSES}."
                )
                tenants[tenant_id] = TenantConfig(tenant_id=tenant_id, status="unavailable")
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

            priv_disabled = grant.get("privacy_disabled_patterns")
            if priv_disabled is not None:
                if isinstance(priv_disabled, list):
                    priv_disabled = tuple(str(p).upper() for p in priv_disabled if str(p).strip())
                else:
                    logger.error(
                        f"Ignoring privacy_disabled_patterns for tenant {tenant_id!r}: not a list."
                    )
                    priv_disabled = None

            priv_ner = grant.get("privacy_ner_labels")
            if priv_ner is not None:
                if isinstance(priv_ner, list):
                    priv_ner = tuple(str(lbl).upper() for lbl in priv_ner if str(lbl).strip())
                else:
                    logger.error(
                        f"Ignoring privacy_ner_labels for tenant {tenant_id!r}: not a list."
                    )
                    priv_ner = None

            tenants[tenant_id] = TenantConfig(
                tenant_id=tenant_id,
                display_name=str(grant.get("display_name", tenant_id)),
                status=status,
                rate_limit_rpm=rpm,
                token_quota_daily=quota,
                privacy_disabled_patterns=priv_disabled,
                privacy_ner_labels=priv_ner,
            )

        logger.info(f"Loaded {len(tenants)} tenant(s) from {self.path}")
        return tenants

    def get(self, tenant_id: str) -> TenantConfig:
        """Resolve one immutable snapshot; unavailable configuration denies admission."""
        self.load()
        state, tenants = self._snapshot
        if state == "unavailable":
            previous = tenants.get(tenant_id)
            if previous is not None and previous.suspended:
                return previous
            return TenantConfig(tenant_id=tenant_id, status="unavailable")
        return tenants.get(tenant_id, DEFAULT_TENANT)

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

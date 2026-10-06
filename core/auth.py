"""Credential-derived capability and tenant resolution.

API keys are stored as SHA-256 hashes. Optional authentication resolves
unauthenticated callers to GENERAL; required authentication denies them at
admission. Request bodies cannot grant privilege. Logs use non-secret key IDs."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from dataclasses import dataclass

from core.config import (
    CAPABILITY_ELEVATED,
    CAPABILITY_GENERAL,
    CAPABILITY_INTERNAL,
    settings,
)
from core.logger import get_logger

logger = get_logger(__name__)

VALID_CAPABILITIES = (CAPABILITY_GENERAL, CAPABILITY_ELEVATED, CAPABILITY_INTERNAL)


@dataclass(frozen=True)
class Principal:
    """
    The authenticated identity behind a request.

    `capability` is authoritative for policy decisions. `authenticated` is False
    for anonymous callers, who always receive the least-privilege tier.
    """
    capability: str = CAPABILITY_GENERAL
    tenant: str = "default"
    key_id: str = "anonymous"
    authenticated: bool = False
    reason: str = "no credential presented"

    def to_audit(self) -> dict:
        """Audit representation. Deliberately contains no secret material."""
        return {
            "capability": self.capability,
            "tenant": self.tenant,
            "key_id": self.key_id,
            "authenticated": self.authenticated,
        }


ANONYMOUS = Principal()


# ---------------------------------------------------------------------------
# Key store
# ---------------------------------------------------------------------------

def hash_key(plaintext: str) -> str:
    """SHA-256 of an API key. The only form ever persisted."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_key(prefix: str = "gk") -> str:
    """
    Mints a cryptographically random API key.

    Returned once to the operator and never stored in plaintext by this module.
    """
    return f"{prefix}_{secrets.token_urlsafe(32)}"


class KeyStore:
    """
    Maps SHA-256 key hashes to capability grants.

    Loaded from `settings.API_KEYS_FILE` — a JSON object:

        {
          "<sha256 of key>": {
            "capability": "ELEVATED",
            "tenant": "acme",
            "key_id": "acme-research-01"
          }
        }

    A missing file is not an error: it means no keys are provisioned, so every
    request is anonymous and receives GENERAL. That is a safe default state.
    Malformed entries are rejected individually and logged, because a typo in
    one grant must not silently widen or void the others.
    """

    def __init__(self, path=None):
        self.path = path or settings.API_KEYS_FILE
        self._keys = {}
        self._loaded = False
        self._lock = threading.Lock()

    def load(self, force=False):
        with self._lock:
            if self._loaded and not force:
                return self

            if not os.path.exists(self.path):
                logger.info(f"No API key store at {self.path}; all requests anonymous (GENERAL).")
                self._keys = {}
                self._loaded = True
                return self

            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception as e:
                # Fail CLOSED on a corrupt store: an unreadable key file must not
                # be treated as "no keys, carry on" if it might have contained
                # revocations. Anonymous access still yields GENERAL either way.
                # Existing keys (if any were already loaded) are left in place
                # rather than wiped, for the same reason.
                logger.error(f"API key store unreadable ({e}); no keys loaded.")
                self._loaded = True
                return self

            if not isinstance(raw, dict):
                logger.error(f"API key store {self.path} must be a JSON object; no keys loaded.")
                self._keys = {}
                self._loaded = True
                return self

            new_keys = {}
            for key_hash, grant in raw.items():
                if not isinstance(grant, dict):
                    logger.error(f"Ignoring malformed grant for key ...{key_hash[-8:]}")
                    continue
                capability = str(grant.get("capability", "")).upper()
                if capability not in VALID_CAPABILITIES:
                    logger.error(
                        f"Ignoring key ...{key_hash[-8:]}: capability {capability!r} "
                        f"not one of {VALID_CAPABILITIES}"
                    )
                    continue
                new_keys[key_hash.lower()] = {
                    "capability": capability,
                    "tenant": str(grant.get("tenant", "default")),
                    "key_id": str(grant.get("key_id", f"key-{key_hash[:8]}")),
                }

            # Atomic swap: any concurrent reader sees either the complete
            # old dict or the complete new one, never a partial one.
            self._keys = new_keys
            self._loaded = True
            logger.info(f"Loaded {len(self._keys)} API key(s) from {self.path}")
            return self

    def lookup(self, plaintext: str):
        """Returns the grant for a key, or None. Never logs the key."""
        self.load()
        if not plaintext:
            return None
        candidate = hash_key(plaintext)
        # Snapshot the reference once -- self._keys may be swapped to a new
        # dict by a concurrent load() while this iterates, but a local
        # binding to the dict object itself is unaffected by that swap.
        keys = self._keys
        for stored_hash, grant in keys.items():
            # compare_digest on the hashes; both are fixed-length hex.
            if hmac.compare_digest(candidate, stored_hash):
                return grant
        return None

    def __len__(self):
        self.load()
        return len(self._keys)


_store = KeyStore()


def reload_keys():
    """Re-reads the key store. Call after provisioning or revoking a key."""
    return _store.load(force=True)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _extract_bearer(authorization) -> str:
    """Pulls the token out of an `Authorization: Bearer <token>` header."""
    if not authorization:
        return ""
    parts = authorization.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def resolve_principal(authorization=None, api_key=None) -> Principal:
    """
    Resolves the caller's capability from their credential.

    THE SECURITY BOUNDARY. Nothing the client asserts about its own privilege
    reaches this function — only the credential does.

    Returns ANONYMOUS-equivalent (GENERAL, authenticated=False) whenever the
    credential is absent, malformed, or unknown. Callers that require
    authentication should check `principal.authenticated`.
    """
    token = api_key or _extract_bearer(authorization)
    if not token:
        return Principal(reason="no credential presented")

    grant = _store.lookup(token)
    if grant is None:
        # Deliberately vague: do not reveal whether the key existed but was
        # revoked, expired, or never valid.
        logger.warning("Rejected an unrecognised API key; falling back to GENERAL.")
        return Principal(reason="unrecognised credential")

    return Principal(
        capability=grant["capability"],
        tenant=grant["tenant"],
        key_id=grant["key_id"],
        authenticated=True,
        reason="verified api key",
    )


def auth_required() -> bool:
    """True when anonymous requests should be rejected outright."""
    return settings.AUTH_MODE == "required"


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def get_user_role():
    """
    Interactive capability selector for CLI/demo use ONLY.

    This is a local convenience for the CLI demo where the operator is the
    machine's user. It must never be wired into a network-facing path: it lets
    the caller choose their own privilege, which is exactly the vulnerability
    this module exists to prevent.
    """
    print("\nSelect Role (LOCAL DEMO ONLY - not an authentication mechanism):")
    print("  1. GENERAL  (Default / Public)")
    print("  2. ELEVATED (Researcher)")
    print("  3. INTERNAL (Admin)")
    choice = input("Enter choice [1/2/3]: ").strip()
    if choice == "2":
        return CAPABILITY_ELEVATED
    if choice == "3":
        return CAPABILITY_INTERNAL
    return CAPABILITY_GENERAL

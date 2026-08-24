"""
Policy Context — the box after Tenant Resolver in the V2 reference
architecture (API Gateway -> Auth -> Tenant Resolver -> Policy Context ->
detection -> ... -> Decision Engine).

WHAT THIS OWNS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
This module maps (capability, risk_level) -> action for a given tenant. It
does NOT touch risk scoring — detector thresholds, fusion weights, and the
per-class risk vector stay identical across every tenant. The example in the
V2 planning notes ("Tenant A: Injection > 0.75 -> BLOCK; Tenant B: Injection
> 0.65 -> BLOCK") describes threshold-level policy, which is a materially
bigger change: it would mean re-running fusion per tenant, not remapping an
already-computed risk_level. That is future work if it turns out to be
needed. What ships here is the version that is actually load-bearing today:
the SAME risk assessment can lead to different enforcement decisions for
different tenants, e.g. a stricter tenant escalating MEDIUM to BLOCK where
the default tenant only RESTRICTs.

DESIGN, MIRRORING core/auth.py's KeyStore and core/tenancy.py's TenantStore
------------------------------------------------------------------------------
Load once and cache, a force-reload hook, per-tenant validation so one
malformed entry cannot take down another tenant's policy, and a required
"default" tenant that every unconfigured or unknown tenant_id falls back to.
This is the third module in this codebase built to the same shape (auth,
tenancy, policy) — consistency here is deliberate: an operator who has
learned how to provision an API key or suspend a tenant already knows how to
edit a tenant's policy.

MIGRATION FROM THE PRE-TENANCY FORMAT
---------------------------------------
policy_rules.json used to be a flat {policies, default_action} object — one
policy, global. It is now {default_action, tenants: {tenant_id: {policies}}}
with a required "tenants.default" entry. `default_action` stays a top-level,
tenant-independent fail-safe (not one more thing to configure per tenant) —
it only fires when a capability tier itself is undefined, which should be
rare given there are exactly three tiers, and giving every tenant its own
fail-safe-of-the-fail-safe would add a layer of configuration surface for a
path that is meant to almost never execute.

TWO DIFFERENT "SOMETHING IS WRONG" STATES, KEPT DISTINCT
------------------------------------------------------------
1. Tenant not found / not configured -> resolves to the "default" tenant's
   REAL policy. Normal, expected, silent (matches core/tenancy.py's
   DEFAULT_TENANT — tenancy is opt-in, so is per-tenant policy).
2. No usable policy data AT ALL (file missing, corrupt, or "default" itself
   failed validation) -> BLOCK everything, logged as an error. This is the
   pre-existing fail-safe behaviour, preserved exactly: a system that cannot
   prove what its policy is must not guess ALLOW.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import yaml

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

VALID_ACTIONS = ("BLOCK", "RESTRICT", "ALLOW", "REVIEW")
DEFAULT_TENANT_ID = "default"

YAML_EXTENSIONS = (".yaml", ".yml")


def _parse_policy_file(path: str) -> dict:
    """
    Parses a policy file, dispatching on extension: `.yaml`/`.yml` ->
    yaml.safe_load, anything else -> json.load.

    Phase 3 (Policy-as-Code): YAML support added alongside JSON, not instead
    of it. Existing deployments pointing POLICY_RULES_FILE at a `.json` path
    are unaffected -- only a `.yaml`/`.yml` extension opts into the new
    parser. YAML's actual advantage over JSON here is comments: an operator
    can annotate WHY a tenant's policy is stricter directly in the file that
    defines it, which the JSON format has never been able to express.

    yaml.safe_load, deliberately not yaml.load: safe_load refuses to
    construct arbitrary Python objects from tags like `!!python/object`,
    which full yaml.load would happily do -- a policy file that decides
    ALLOW/BLOCK for every request must not also be a code-execution vector
    for whoever can write to it.
    """
    with open(path, "r", encoding="utf-8") as f:
        if os.path.splitext(path)[1].lower() in YAML_EXTENSIONS:
            return yaml.safe_load(f)
        return json.load(f)


def load_policy_file(path: str = None) -> dict:
    """Public wrapper over `_parse_policy_file`, defaulting to the live
    `settings.POLICY_RULES_FILE` -- for callers (the Policy Editor API)
    that need the raw parsed content itself, not a resolved `PolicySet`.
    Raises the same way `_parse_policy_file` does (missing file, bad
    YAML/JSON) -- callers that need a safe fallback should catch, the
    same discipline `PolicyStore.load` already applies around this exact
    call."""
    return _parse_policy_file(path or settings.POLICY_RULES_FILE)


@dataclass(frozen=True)
class PolicySet:
    """
    One tenant's capability -> risk_level -> action mapping.

    `default_action` here is the FAIL-SAFE STUB value (see FAIL_SAFE below),
    not a per-tenant setting — see the module docstring for why the two are
    kept separate.
    """
    tenant_id: str
    policies: dict = field(default_factory=dict)


# Returned when there is no usable policy data at all (file absent, corrupt,
# or "default" itself failed validation). tenant_id is a sentinel, never a
# real tenant's id, so policy_decision can recognise this state unambiguously
# rather than inferring it from an empty dict (which a legitimately
# misconfigured single tenant could also produce).
FAIL_SAFE = PolicySet(tenant_id="__fail_safe__", policies={})


class PolicyStore:
    """
    Maps tenant_id -> PolicySet, loaded from `settings.POLICY_RULES_FILE`:

        {
          "default_action": "BLOCK",
          "tenants": {
            "default": {
              "policies": {
                "GENERAL": {"HIGH": "BLOCK", "MEDIUM": "RESTRICT", "LOW": "ALLOW"},
                "ELEVATED": {"HIGH": "BLOCK", "MEDIUM": "ALLOW", "LOW": "ALLOW"},
                "INTERNAL": {"HIGH": "ALLOW", "MEDIUM": "ALLOW", "LOW": "ALLOW"}
              }
            },
            "acme": {
              "policies": {
                "GENERAL": {"HIGH": "BLOCK", "MEDIUM": "BLOCK", "LOW": "ALLOW"}
              }
            }
          }
        }

    A missing/corrupt file or a "tenants" object missing "default" all
    resolve to FAIL_SAFE — see the module docstring for why this must not
    degrade to "no tenants configured, allow through", unlike TenantStore's
    equivalent case. Policy is the last gate before a decision; identity
    resolution (TenantStore) is not.
    """

    def __init__(self, path=None):
        self.path = path or settings.POLICY_RULES_FILE
        self._tenants: dict[str, PolicySet] = {}
        self._default_action = "BLOCK"
        self._usable = False
        self._loaded = False

    def load(self, force=False):
        if self._loaded and not force:
            return self
        self._tenants = {}
        self._usable = False
        self._loaded = True

        if not os.path.exists(self.path):
            logger.error(f"No policy store at {self.path}; failing closed (BLOCK) until provisioned.")
            return self

        try:
            raw = _parse_policy_file(self.path)
        except Exception as e:
            logger.error(f"Policy store unreadable ({e}); failing closed (BLOCK).")
            return self

        if not isinstance(raw, dict) or "tenants" not in raw:
            logger.error(
                f"Policy store {self.path} must be a JSON object with a "
                f"'tenants' key; failing closed (BLOCK)."
            )
            return self

        self._default_action = str(raw.get("default_action", "BLOCK")).upper()
        if self._default_action not in VALID_ACTIONS:
            logger.error(
                f"default_action {self._default_action!r} not one of "
                f"{VALID_ACTIONS}; falling back to BLOCK."
            )
            self._default_action = "BLOCK"

        tenants = raw["tenants"]
        if not isinstance(tenants, dict):
            logger.error(f"Policy store {self.path}: 'tenants' must be an object; failing closed (BLOCK).")
            return self

        for tenant_id, entry in tenants.items():
            policy_set = self._parse_tenant_entry(tenant_id, entry)
            if policy_set is not None:
                self._tenants[tenant_id] = policy_set

        if DEFAULT_TENANT_ID not in self._tenants:
            logger.error(
                f"Policy store {self.path} has no usable '{DEFAULT_TENANT_ID}' "
                f"tenant; failing closed (BLOCK) for every unconfigured tenant."
            )
            return self

        self._usable = True
        logger.info(
            f"Loaded policy for {len(self._tenants)} tenant(s) from {self.path} "
            f"(default_action={self._default_action})"
        )
        return self

    def _parse_tenant_entry(self, tenant_id, entry):
        """Returns a PolicySet, or None if this ONE tenant's entry is
        malformed — never raises, never takes down a sibling tenant."""
        if not isinstance(entry, dict):
            logger.error(f"Ignoring policy for tenant {tenant_id!r}: entry is not an object.")
            return None

        raw_policies = entry.get("policies")
        if not isinstance(raw_policies, dict) or not raw_policies:
            logger.error(f"Ignoring policy for tenant {tenant_id!r}: 'policies' missing or empty.")
            return None

        cleaned = {}
        for capability, risk_map in raw_policies.items():
            if not isinstance(risk_map, dict):
                logger.error(
                    f"Ignoring capability {capability!r} in tenant {tenant_id!r}'s "
                    f"policy: not an object."
                )
                continue
            cleaned_risk_map = {}
            for risk_level, action in risk_map.items():
                action_upper = str(action).upper()
                if action_upper not in VALID_ACTIONS:
                    logger.error(
                        f"Ignoring {tenant_id!r}/{capability!r}/{risk_level!r}: "
                        f"action {action!r} not one of {VALID_ACTIONS}."
                    )
                    continue
                cleaned_risk_map[risk_level] = action_upper
            if cleaned_risk_map:
                cleaned[capability] = cleaned_risk_map

        if not cleaned:
            logger.error(f"Ignoring policy for tenant {tenant_id!r}: no valid capability entries.")
            return None

        return PolicySet(tenant_id=tenant_id, policies=cleaned)

    def get(self, tenant_id: str) -> PolicySet:
        """Never raises. Unknown/unconfigured tenant -> the 'default'
        tenant's real policy. No usable policy data at all -> FAIL_SAFE."""
        self.load()
        if not self._usable:
            return FAIL_SAFE
        return self._tenants.get(tenant_id, self._tenants[DEFAULT_TENANT_ID])

    @property
    def default_action(self) -> str:
        self.load()
        return self._default_action

    def __len__(self):
        self.load()
        return len(self._tenants)


_store = PolicyStore()


def validate_policy_file(path: str) -> list[str]:
    """
    Reports EVERY problem in a candidate policy file, for the Phase 3
    (Policy-as-Code) validation step -- `scripts/validate_policy.py`.

    Deliberately NOT the same code path as PolicyStore.load(): the loader's
    job is to keep serving a good tenant's policy when a SIBLING tenant's
    entry is malformed (see the module docstring), so it silently drops one
    bad entry and logs a warning rather than surfacing every problem at
    once. That is the wrong behaviour for an operator validating a file
    BEFORE deploying it -- they want the complete list of what is wrong,
    not one warning at a time discovered by trial and error. Returns an
    empty list when the file is fully valid.
    """
    errors: list[str] = []

    if not os.path.exists(path):
        return [f"File not found: {path}"]

    try:
        raw = _parse_policy_file(path)
    except Exception as e:
        return [f"File is not valid YAML/JSON: {type(e).__name__}: {e}"]

    if not isinstance(raw, dict):
        return ["Top level must be an object/mapping."]
    if "tenants" not in raw:
        errors.append("Missing required top-level key: 'tenants'.")
        return errors

    default_action = str(raw.get("default_action", "BLOCK")).upper()
    if default_action not in VALID_ACTIONS:
        errors.append(
            f"default_action {raw.get('default_action')!r} is not one of {VALID_ACTIONS}."
        )

    tenants = raw["tenants"]
    if not isinstance(tenants, dict):
        errors.append("'tenants' must be an object/mapping of tenant_id -> policy.")
        return errors

    if DEFAULT_TENANT_ID not in tenants:
        errors.append(f"Missing required tenant: {DEFAULT_TENANT_ID!r}.")

    for tenant_id, entry in tenants.items():
        prefix = f"tenants.{tenant_id}"
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: must be an object, got {type(entry).__name__}.")
            continue

        policies = entry.get("policies")
        if not isinstance(policies, dict) or not policies:
            errors.append(f"{prefix}.policies: missing, empty, or not an object.")
            continue

        for capability, risk_map in policies.items():
            cap_prefix = f"{prefix}.policies.{capability}"
            if not isinstance(risk_map, dict) or not risk_map:
                errors.append(f"{cap_prefix}: missing, empty, or not an object.")
                continue
            for risk_level, action in risk_map.items():
                action_str = str(action).upper()
                if action_str not in VALID_ACTIONS:
                    errors.append(
                        f"{cap_prefix}.{risk_level}: action {action!r} is not "
                        f"one of {VALID_ACTIONS}."
                    )

    return errors


def reload_policies():
    """Re-reads the policy store. Call after editing a tenant's policy."""
    return _store.load(force=True)


def resolve_policy_set(tenant_id: str, store: PolicyStore = None) -> PolicySet:
    return (store or _store).get(tenant_id)


def policy_decision(capability: str, risk: str, tenant_id: str = DEFAULT_TENANT_ID,
                    store: PolicyStore = None):
    """
    Determines the enforcement action for (capability, risk) under a
    tenant's policy.

    `tenant_id` defaults to "default" so an existing call site that upgrades
    without passing one keeps getting exactly the single global policy this
    module used to be — the same backward-compatibility shape core/fusion.py
    uses for a v1 policy artifact missing a `per_class` section.

    `store` defaults to the live module-global policy store. Passing a
    different `PolicyStore` instance (Phase 3, Policy-as-Code) is how
    `scripts/simulate_policy.py` evaluates a CANDIDATE policy file against
    historical audit records without touching what the running gateway is
    actually enforcing — the live `_store` is never mutated by a simulation.

    Fail-safe default, preserved exactly from the pre-tenancy version: no
    usable policy data at all -> BLOCK. An undefined capability under an
    otherwise-usable policy -> that policy's own `default_action`, not a
    hardcoded BLOCK — a tenant's default_action is intentionally global
    (see module docstring), so this is the SAME default_action regardless of
    which tenant resolved, which is correct: it is a fail-safe backstop, not
    a per-tenant policy choice.
    """
    active_store = store or _store
    policy_set = resolve_policy_set(tenant_id, store=active_store)

    if policy_set is FAIL_SAFE:
        return "BLOCK", "System Error: Policies not loaded"

    # `policy_set.tenant_id` is whichever tenant's policy ACTUALLY applied,
    # which is "default" when `tenant_id` was unconfigured — different from
    # the requested `tenant_id` in that case. The reason string must say so
    # explicitly: reporting the requested tenant alone would make an audit
    # record claim tenant X has its own policy when X silently fell back to
    # someone else's. This is exactly the class of bug it exists to prevent.
    resolved = policy_set.tenant_id
    tenant_note = resolved if resolved == tenant_id else f"{tenant_id} -> {resolved} (fallback)"

    capability_policy = policy_set.policies.get(capability)
    if not capability_policy:
        return active_store.default_action, f"Role '{capability}' not defined for tenant '{tenant_note}'"

    action = capability_policy.get(risk, active_store.default_action)
    return action, f"Policy applied for {capability} (Risk: {risk}, Tenant: {tenant_note})"

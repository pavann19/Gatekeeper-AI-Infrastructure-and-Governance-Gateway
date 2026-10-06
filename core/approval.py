"""Construct approval identity from trusted admission and exact action inputs."""
import hashlib
import json

from core.cache import security_cache_scope, _CACHE_SCOPE
from core.policy import policy_identity


def approval_context(principal, request_id: str, action: str, arguments, decision_context: str) -> dict:
    identity = json.dumps({'action': action, 'arguments': arguments}, sort_keys=True, separators=(',', ':'), allow_nan=False)
    with security_cache_scope(identity, principal.tenant):
        scope = _CACHE_SCOPE.get()
    return {
        'request_id': request_id, 'tenant': principal.tenant,
        'principal': principal.key_id, 'capability': principal.capability,
        'action': action, 'policy_digest': policy_identity(),
        'security_version': hashlib.sha256(json.dumps(scope, sort_keys=True, default=str).encode()).hexdigest(),
        'decision_context': hashlib.sha256((decision_context + identity).encode()).hexdigest(),
    }

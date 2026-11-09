"""Readiness probes validate the governance dependencies without model loading."""
import os
import sqlite3
import sys
import threading
import time

from core.config import settings

_probe_lock = threading.Lock()
_probe_cache: tuple[float, dict[str, bool]] = (0.0, {})


def cached_readiness_checks():
    """Single-flight dependency probes, cached for at most five seconds."""
    global _probe_cache
    with _probe_lock:
        at, checks = _probe_cache
        if not checks or time.monotonic() - at >= 5:
            checks = readiness_checks()
            _probe_cache = (time.monotonic(), checks)
        return dict(checks)


def readiness_checks() -> dict[str, bool]:
    from core.policy import capture_policy
    from core.policy_loader import get_hard_ban_keywords, get_jailbreak_patterns, get_instruction_override_patterns
    from core.privacy import NLP_MODEL
    checks = {
        'policy': capture_policy()._usable,
        'symbolic_rules': all(value is not None for value in (
            get_hard_ban_keywords(), get_jailbreak_patterns(), get_instruction_override_patterns())),
        'privacy_model': NLP_MODEL is not None,
        'redis': True,
        'persistent_state': True,
    }
    embeddings = sys.modules.get('core.embeddings')
    detectors = sys.modules.get('core.detectors')
    registry = getattr(detectors, '_REGISTRY', None) or {}
    checks['embedding_loaded'] = getattr(embeddings, '_model', None) is not None
    checks['baseline_detectors_loaded'] = all(
        getattr(registry.get(name), '_model', None) is not None
        for name in ('protectai_injection', 'madhurjindal_jailbreak', 'toxic_bert'))
    if os.getenv('REDIS_URL'):
        from core.redis_client import get_redis_client
        try:
            client = get_redis_client()
            checks['redis'] = client is not None and bool(client.ping())
        except Exception:
            checks['redis'] = False
    if settings.STATE_BACKEND == 'postgres':
        from core.postgres_state import get_postgres_store
        checks['persistent_state'] = get_postgres_store().ready()
    else:
        try:
            for path in (settings.AUDIT_LOG_PATH, settings.AUDIT_DB_PATH,
                         settings.REVIEW_QUEUE_FILE, settings.CACHE_EXACT_FILE):
                directory = os.path.dirname(os.path.abspath(path))
                checks['persistent_state'] &= os.access(directory, os.W_OK)
            from core.review_queue import ReviewQueue
            queue = ReviewQueue(settings.REVIEW_QUEUE_FILE).load(force=True)
            if queue._corrupt:
                checks['persistent_state'] = False
            if settings.AUDIT_SQLITE_ENABLED and os.path.exists(settings.AUDIT_DB_PATH):
                with sqlite3.connect(settings.AUDIT_DB_PATH, timeout=0.5) as connection:
                    connection.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()
        except Exception:
            checks['persistent_state'] = False
    return checks

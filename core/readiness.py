"""Readiness probes validate the governance dependencies without model loading."""
import os
import sqlite3
import tempfile
import sys

from core.config import settings


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
    try:
        for path in (settings.AUDIT_LOG_PATH, settings.AUDIT_DB_PATH,
                     settings.REVIEW_QUEUE_FILE, settings.POLICY_RULES_FILE):
            directory = os.path.dirname(os.path.abspath(path))
            with tempfile.TemporaryFile(dir=directory) as handle:
                handle.write(b'probe')
                handle.flush()
                os.fsync(handle.fileno())
        from core.review_queue import ReviewQueue
        queue = ReviewQueue(settings.REVIEW_QUEUE_FILE).load(force=True)
        if queue._corrupt:
            checks['persistent_state'] = False
        if settings.AUDIT_SQLITE_ENABLED and os.path.exists(settings.AUDIT_DB_PATH):
            with sqlite3.connect(settings.AUDIT_DB_PATH, timeout=0.5) as connection:
                checks['persistent_state'] &= connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    except Exception:
        checks['persistent_state'] = False
    return checks

import asyncio
from datetime import datetime, timedelta, timezone
import json
import threading
import time
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api import main
from core.config import settings
from core.review_queue import ReviewCapacityError, ReviewQueue


@pytest.fixture(autouse=True)
def _synthetic_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'AUDIT_LOG_PATH', str(tmp_path / 'audit.jsonl'))
    monkeypatch.setattr(settings, 'AUDIT_DB_PATH', str(tmp_path / 'audit.db'))
    monkeypatch.setattr(settings, 'REVIEW_QUEUE_FILE', str(tmp_path / 'reviews.json'))


def test_accounting_outage_returns_completed_response_without_recalling_provider(monkeypatch):
    from core.llm_providers import LLMResponse
    provider = Mock()
    provider.complete.return_value = LLMResponse(content='safe answer', model='synthetic', provider='ollama',
                                                 usage={'total_tokens': 17})
    quota = Mock()
    quota.record.side_effect = ConnectionError('synthetic accounting outage')
    monkeypatch.setattr(main, 'get_provider', lambda name: provider)
    monkeypatch.setattr(main, 'gateway_token_quota', quota)
    monkeypatch.setattr(main, 'assess_risk', lambda *a, **kw: ('LOW', {'source': 'synthetic'}))
    monkeypatch.setattr('core.output_guardrails.assess_output',
                        lambda *a: ('ALLOW', {'clean_response': 'safe answer'}))
    response = TestClient(main.app).post('/api/v1/gateway/chat', json={'prompt': 'hello'})
    assert response.status_code == 200
    assert response.json()['content'] == 'safe answer'
    assert response.json()['details']['token_accounting_degraded'] is True
    provider.complete.assert_called_once()
    quota.record.assert_called_once_with('default', 17)


def test_full_review_queue_maps_to_backpressure(monkeypatch):
    monkeypatch.setattr(main, 'assess_risk', lambda *a, **kw: ('MEDIUM', {'source': 'synthetic'}))
    monkeypatch.setattr(main, 'policy_decision', lambda *a, **kw: ('REVIEW', 'synthetic'))
    monkeypatch.setattr(main, 'enqueue_review', Mock(side_effect=ReviewCapacityError('full')))
    response = TestClient(main.app).post('/api/v1/assess', json={'prompt': 'hello'})
    assert response.status_code == 503
    assert response.headers['Retry-After'] == '5'


def test_expiry_and_pagination_retain_review_evidence(tmp_path):
    path = tmp_path / 'reviews.json'
    queue = ReviewQueue(str(path))
    first = queue.enqueue('r', 'GENERAL', 'MEDIUM', 'a', 'h', 'r')
    second = queue.enqueue('r', 'GENERAL', 'MEDIUM', 'b', 'h', 'r')
    state = json.loads(path.read_text())
    state[first.review_id]['expires_at'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(state))
    assert queue.list_pending(tenant='a') == []
    assert [r['review_id'] for r in queue.list_pending(limit=1)] == [second.review_id]
    assert queue.expire_pending() == 1
    assert queue.get(first.review_id)['status'] == 'EXPIRED'
    assert len(json.loads(path.read_text())) == 2


def test_audit_sqlite_roundtrip_retains_identity_and_utc(monkeypatch):
    from core import audit_store
    from core.logger import log_event
    monkeypatch.setattr(settings, 'AUDIT_READ_FROM_SQLITE', True)
    log_event('GENERAL', 'synthetic prompt', 'LOW', 'ALLOW')
    record = audit_store.recent()['events'][0]
    assert len(record['policy_digest']) == 64
    assert len(record['event_id']) == 32
    assert record['timestamp'].endswith('+00:00')


def test_audit_schema_is_not_reconciled_on_each_write(tmp_path, monkeypatch):
    from core import audit_store
    path = str(tmp_path / 'audit.db')
    audit_store.write('input_assessment', {'decision': 'BLOCK'}, path)
    monkeypatch.setattr(audit_store, '_SCHEMA', 'INVALID SQL MUST NOT RUN')
    audit_store.write('input_assessment', {'decision': 'BLOCK'}, path)
    assert len(audit_store.recent(path=path)['events']) == 2


def test_readiness_is_cached_and_caller_cannot_mutate_it(monkeypatch):
    from core import readiness
    checks = Mock(return_value={'policy': True})
    monkeypatch.setattr(readiness, 'readiness_checks', checks)
    monkeypatch.setattr(readiness, '_probe_cache', (0., {}))
    first = readiness.cached_readiness_checks()
    first['policy'] = False
    assert readiness.cached_readiness_checks()['policy'] is True
    checks.assert_called_once()


def test_startup_warmup_has_a_bounded_wait(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(settings, 'WARM_MODELS_ON_STARTUP', True)
    monkeypatch.setattr(settings, 'STARTUP_WARMUP_TIMEOUT_SECONDS', .05)
    monkeypatch.setattr(main, '_startup_status', 'not_started')

    def slow_model():
        entered.set()
        release.wait(3)

    monkeypatch.setattr('core.embeddings._get_model', slow_model)
    monkeypatch.setattr('core.threat_centroid._get_malicious_centroid', lambda: None)
    monkeypatch.setattr('core.fusion.warm_up', lambda: (True, 'synthetic'))
    started = time.perf_counter()
    try:
        asyncio.run(main.warm_models())
        assert entered.is_set()
        assert time.perf_counter() - started < 1
        assert TestClient(main.app).get('/startup').status_code == 503
    finally:
        release.set()


@pytest.mark.parametrize('route,payload', [('/api/v1/policy/deploy', {'content': '{}'}),
                                          ('/api/v1/policy/rollback', {'version': 'old.json'})])
def test_production_cannot_self_promote_policy(route, payload, monkeypatch):
    from core.auth import Principal
    monkeypatch.setattr(settings, 'ENVIRONMENT', 'production')
    monkeypatch.setattr(main, '_require_internal', lambda request: Principal(capability='INTERNAL', authenticated=True))
    deploy, rollback = Mock(), Mock()
    monkeypatch.setattr(main, 'deploy_policy', deploy)
    monkeypatch.setattr(main, 'rollback_to', rollback)
    assert TestClient(main.app).post(route, json=payload).status_code == 403
    deploy.assert_not_called()
    rollback.assert_not_called()


def test_tool_output_is_bounded_in_bytes_before_release(monkeypatch):
    from core.tools import ToolRegistry, ToolSpec, execute_tool
    registry = ToolRegistry()
    registry.register(ToolSpec('synthetic', 'bounded output', risk_level='LOW'), handler=lambda: '\u20ac' * 40)
    monkeypatch.setattr(settings, 'TOOL_MAX_OUTPUT_BYTES', 100)
    result = execute_tool('GENERAL', 'synthetic', {}, registry=registry)
    assert result['decision'] == 'BLOCK'
    assert 'output' not in result

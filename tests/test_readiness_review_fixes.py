"""Behavioral regressions for the production-readiness review's local fixes."""
import asyncio
import json
import random
import threading
import time
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from api import main
from core import auth, tenancy
from core.auth import KeyStore, hash_key
from core.config import settings
from core.output_guardrails import check_system_prompt_leakage
from core.review_queue import ReviewQueue


@pytest.fixture
def identity(tmp_path, monkeypatch):
    path = tmp_path / 'keys.json'
    path.write_text(json.dumps({hash_key('synthetic-key'): {
        'capability': 'GENERAL', 'tenant': 'default', 'key_id': 'synthetic-owner'}}))
    monkeypatch.setattr(auth, '_store', KeyStore(str(path)))
    return 'synthetic-key'


def test_http_api_accepts_x_api_key_and_rejects_conflicts(identity, monkeypatch):
    monkeypatch.setattr(settings, 'AUTH_MODE', 'required')
    client = TestClient(main.app)
    assert client.get('/api/v1/whoami', headers={'X-API-Key': identity}).status_code == 200
    assert client.get('/api/v1/whoami', headers={
        'X-API-Key': identity, 'Authorization': 'Bearer other-key'}).status_code == 401


@pytest.mark.parametrize('path', ['/api/v1/activity', '/api/v1/activity/trace/synthetic-id'])
def test_activity_never_allows_anonymous_reads(path, monkeypatch):
    monkeypatch.setattr(settings, 'AUTH_MODE', 'optional')
    assert TestClient(main.app).get(path).status_code == 401


def test_unprovisioned_tenant_denied_before_inference(identity, tmp_path, monkeypatch):
    path = tmp_path / 'tenants.json'
    path.write_text('{}')
    monkeypatch.setattr(tenancy, '_store', tenancy.TenantStore(str(path)))
    inference = Mock()
    monkeypatch.setattr(main, 'assess_risk', inference)
    response = TestClient(main.app).post('/api/v1/assess', headers={'X-API-Key': identity},
                                        json={'prompt': 'synthetic request'})
    assert response.status_code == 403
    inference.assert_not_called()


def test_public_system_prompt_cannot_reach_provider(identity, monkeypatch):
    provider = Mock()
    monkeypatch.setattr(main, 'get_provider', provider)
    response = TestClient(main.app).post('/api/v1/gateway/chat', headers={'X-API-Key': identity},
                                        json={'prompt': 'hello', 'system_prompt': 'override authority'})
    assert response.status_code == 403
    provider.assert_not_called()


def test_high_review_requires_independent_identity_and_audit(tmp_path, monkeypatch):
    queue = ReviewQueue(str(tmp_path / 'reviews.json'))
    record = queue.enqueue('synthetic', 'INTERNAL', 'HIGH', 'default', 'digest', 'req',
                           context={'principal': 'maker'})
    with pytest.raises(ValueError, match='independent'):
        queue.resolve(record.review_id, 'APPROVED', 'maker')
    assert queue.get(record.review_id)['status'] == 'PENDING'
    audit = Mock(side_effect=OSError('synthetic unavailable audit'))
    monkeypatch.setattr('core.logger.log_review_event', audit)
    with pytest.raises(OSError):
        queue.resolve(record.review_id, 'APPROVED', 'checker')
    assert queue.get(record.review_id)['status'] == 'PENDING'
    audit.side_effect = None
    assert queue.resolve(record.review_id, 'APPROVED', 'checker')['status'] == 'APPROVED'
    assert audit.call_args.args[-1] == 'resolution_completed'


def test_leak_detector_matches_substring_oracle():
    rng = random.Random(87)
    for _ in range(250):
        reference = ''.join(rng.choices('ab\u2028c', k=30))
        response = ''.join(rng.choices('ab\u2028c', k=60))
        length = rng.randint(1, 35)
        expected = any(reference[i:i + length] in response for i in range(len(reference) - length + 1))
        assert check_system_prompt_leakage(response, reference, length) == expected
    assert not check_system_prompt_leakage('b' * 50_000, 'a' * 20_000)


@pytest.mark.parametrize('stage', ['_redact_or_fail', '_write_audit_or_fail', '_enforce_rate_limit'])
def test_slow_request_stage_does_not_stall_liveness(stage, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = getattr(main, stage)

    def slow(*args, **kwargs):
        entered.set()
        release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(main, stage, slow)
    monkeypatch.setattr(main, 'assess_risk', lambda *a, **kw: ('LOW', {'source': 'synthetic'}))
    # All state writes are in the runner's fresh copied workspace.
    async def exercise():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            started = time.perf_counter()
            assessment = asyncio.create_task(client.post('/api/v1/assess', json={'prompt': 'hello'}))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                live = await asyncio.wait_for(client.get('/live'), timeout=.5)
                assert live.status_code == 200
                assert time.perf_counter() - started < 1.5
            finally:
                release.set()
            assert (await assessment).status_code == 200

    asyncio.run(exercise())

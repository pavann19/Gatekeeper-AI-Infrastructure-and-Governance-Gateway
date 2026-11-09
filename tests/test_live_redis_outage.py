"""The owned Redis fixture is paused; configured authority must fail closed."""
import os
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(not os.getenv('GATEKEEPER_REDIS_OUTAGE_FIXTURE'),
                                reason='Owned paused Redis fixture required')


def test_real_required_redis_outage_denies_before_assessment(monkeypatch, caplog):
    from api import main
    from fastapi.testclient import TestClient
    inference = Mock()
    monkeypatch.setattr(main, 'assess_risk', inference)
    response = TestClient(main.app).post('/api/v1/assess', json={'prompt': 'synthetic request'})
    assert response.status_code == 429
    assert 'Retry-After' in response.headers
    inference.assert_not_called()
    assert 'synthetic-test-only' not in caplog.text


def test_real_required_cache_and_quota_do_not_create_local_authority():
    from core.cache import lookup_cache
    from core.cache_backend import CacheDependencyUnavailable
    from core.token_quota import gateway_token_quota
    with pytest.raises(CacheDependencyUnavailable):
        lookup_cache('synthetic prompt', [1.] * 768)
    assert gateway_token_quota.would_exceed('synthetic', 100)
    with pytest.raises(ConnectionError):
        gateway_token_quota.record('synthetic', 5)

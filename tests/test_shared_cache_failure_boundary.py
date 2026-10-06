from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.cache import FAISSCache
from core.cache_backend import RedisExactCache, CacheDependencyUnavailable


def test_unavailable_shared_cache_never_falls_through_to_fuzzy_or_fresh_allow():
    client = MagicMock()
    client.get.side_effect = ConnectionError('unavailable')
    cache = FAISSCache.__new__(FAISSCache)
    cache._exact = RedisExactCache(client)
    with pytest.raises(CacheDependencyUnavailable):
        cache.lookup('prompt', None)


def test_shared_cache_failure_is_observable_as_unavailable_not_approval(monkeypatch):
    monkeypatch.setattr('api.main.settings.RATE_LIMIT_ENABLED', False)
    def failed(*args, **kwargs):
        raise CacheDependencyUnavailable('unavailable')
    monkeypatch.setattr('api.main.assess_risk', failed)
    response = TestClient(app).post('/api/v1/assess', json={'prompt': 'hello'})
    assert response.status_code == 503
    assert 'decision' not in response.json()

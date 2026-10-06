"""Real Redis contracts using only a unique, owned key namespace."""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import redis

from core.cache_backend import RedisExactCache
from core.rate_limit import RedisRateLimiter
from core.token_quota import RedisTokenQuotaTracker

pytestmark = pytest.mark.skipif(not os.getenv('GATEKEEPER_REDIS_INTEGRATION_URL'), reason='Live Redis acceptance URL required')


@pytest.fixture
def replicas():
    url = os.environ['GATEKEEPER_REDIS_INTEGRATION_URL']
    clients = [redis.Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1) for _ in range(2)]
    namespace = 'gatekeeper:acceptance:' + uuid.uuid4().hex + ':'
    for client in clients:
        assert client.ping()
    try:
        yield clients, namespace
    finally:
        for key in clients[0].scan_iter(namespace + '*'):
            clients[0].delete(key)
        for client in clients:
            client.close()


def test_real_cache_high_is_monotonic_across_independent_clients(replicas):
    clients, prefix = replicas
    backends = [RedisExactCache(client) for client in clients]
    for backend in backends:
        backend.KEY_PREFIX = prefix
    backends[0].set('same', {'risk': 'HIGH'}, ttl_seconds=60)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: backends[i % 2].set('same', {'risk': 'LOW'}, 60), range(100)))
    assert all(result['risk'] == 'HIGH' for result in results)
    assert backends[1].get('same')['risk'] == 'HIGH'


def test_real_rate_limit_is_one_shared_budget(replicas):
    clients, prefix = replicas
    limiters = [RedisRateLimiter(client, name='shared') for client in clients]
    for limiter in limiters:
        limiter.KEY_PREFIX = prefix
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: limiters[i % 2].check('owner', 5, 0.000001, now=100), range(100)))
    assert sum(allowed for allowed, _ in results) == 5


def test_real_quota_records_have_no_lost_updates(replicas):
    clients, prefix = replicas
    trackers = [RedisTokenQuotaTracker(client) for client in clients]
    for tracker in trackers:
        tracker.KEY_PREFIX = prefix
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: trackers[i % 2].record('tenant', 2), range(100)))
    assert trackers[0].usage_today('tenant') == trackers[1].usage_today('tenant') == 200
    assert trackers[0].would_exceed('tenant', 200)

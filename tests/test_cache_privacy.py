"""Persistent stores must never receive request content or embedding vectors."""
import json
from unittest.mock import Mock

import numpy as np

from core import cache
from core.cache_backend import LocalExactCache, RedisExactCache, PERSISTED_FIELDS


def test_local_backend_strips_content_on_write_and_reload(tmp_path):
    path = str(tmp_path / 'exact.json')
    backend = LocalExactCache(path)
    backend.set('digest', {'risk': 'HIGH', 'score': .99, 'prompt': 'synthetic-private-value',
                           'vector': [1., 2.], 'response': 'synthetic-secret'})
    backend._save_thread.join(timeout=3)
    assert not backend._save_thread.is_alive()
    raw = (tmp_path / 'exact.json').read_text()
    assert 'synthetic-private-value' not in raw
    assert 'synthetic-secret' not in raw
    assert set(json.loads(raw)['digest']) <= PERSISTED_FIELDS
    reloaded = LocalExactCache(path)
    assert reloaded.get('digest')['risk'] == 'HIGH'


def test_redis_receives_metadata_only():
    client = Mock()
    client.eval.side_effect = lambda script, count, key, payload, ttl: payload
    backend = RedisExactCache(client)
    backend.set('digest', {'risk': 'HIGH', 'score': .9, 'prompt': 'private-request', 'vector': [1.]})
    payload = json.loads(client.eval.call_args.args[3])
    assert set(payload) <= PERSISTED_FIELDS
    assert 'private-request' not in str(payload)


def test_fuzzy_never_loads_or_writes_legacy_snapshot(tmp_path, monkeypatch):
    legacy = tmp_path / 'fuzzy.json'
    original = '[{"prompt":"legacy-synthetic-sensitive-value"}]'
    legacy.write_text(original)
    monkeypatch.setattr(cache, 'CACHE_FILE', str(legacy))
    backend = LocalExactCache(str(tmp_path / 'exact.json'))
    monkeypatch.setattr(cache, 'build_exact_cache_backend', lambda **kw: backend)
    store = cache.FAISSCache(dimension=2)
    store.load()
    assert not store.cache_data
    store.add('new-sensitive-value', np.array([1., 0.]), 'HIGH', .99)
    backend._save_thread.join(timeout=3)
    store._save_to_disk()
    store.save_async()
    assert legacy.read_text() == original
    assert 'prompt' not in next(iter(store.cache_data.values()))
    assert 'new-sensitive-value' not in (tmp_path / 'exact.json').read_text()


def test_fuzzy_lookup_keeps_index_ids_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'build_exact_cache_backend',
                        lambda **kw: LocalExactCache(str(tmp_path / 'exact.json')))
    store = cache.FAISSCache(dimension=2)
    monkeypatch.setattr(store._exact, '_save_async', lambda: None)
    monkeypatch.setattr(store._exact, 'get', lambda key: None)
    store.add('first', [1., 0.], 'HIGH', .91)
    store.add('second', [0., 1.], 'HIGH', .99)
    assert store.lookup('first-query', [1., 0.]) == ('HIGH', .91)
    assert store.lookup('first-query-again', [1., 0.]) == ('HIGH', .91)
    store.add('third', [-1., 0.], 'HIGH', .95)
    assert store._index.ntotal == 3
    assert store.lookup('third-query', [-1., 0.]) == ('HIGH', .95)

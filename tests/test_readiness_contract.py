import types

from fastapi.testclient import TestClient

from api.main import app
from core.readiness import readiness_checks


def test_liveness_is_distinct_from_failed_readiness(monkeypatch):
    monkeypatch.setattr('core.readiness.readiness_checks', lambda: {'policy': False})
    client = TestClient(app)
    assert client.get('/live').status_code == 200
    response = client.get('/ready')
    assert response.status_code == 503
    assert response.json()['status'] == 'degraded'


def test_readiness_does_not_load_missing_models(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'core.embeddings', types.SimpleNamespace(_model=None))
    monkeypatch.setitem(sys.modules, 'core.detectors', types.SimpleNamespace(_REGISTRY={}))
    result = readiness_checks()
    assert result['embedding_loaded'] is False
    assert result['baseline_detectors_loaded'] is False


def test_configured_redis_failure_cannot_report_ready(monkeypatch):
    monkeypatch.setenv('REDIS_URL', 'redis://unavailable.invalid:6379')
    monkeypatch.setattr('core.redis_client.get_redis_client', lambda: None)
    assert readiness_checks()['redis'] is False

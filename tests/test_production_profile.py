"""Production mode must reject the demo defaults before application startup."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import Settings


SAFE = dict(ENVIRONMENT='production', AUTH_MODE='required', METRICS_REQUIRE_AUTH=True,
            STATE_BACKEND='postgres', DATABASE_URL='postgresql://synthetic@db.example.test/gk?sslmode=verify-full',
            CORS_ORIGINS='https://gateway.example.test', DETECTOR_VERIFY_HASHES=True,
            WARM_MODELS_ON_STARTUP=True, REGISTER_DEMO_TOOLS=False)


@pytest.mark.parametrize('changes', [dict(AUTH_MODE='optional'), dict(METRICS_REQUIRE_AUTH=False),
                         dict(CORS_ORIGINS='*'), dict(CORS_ORIGINS='https://*.example.test'),
                         dict(CORS_ORIGINS=''), dict(CORS_ORIGINS='http://example.test'),
                         dict(DETECTOR_VERIFY_HASHES=False), dict(REGISTER_DEMO_TOOLS=True),
                         dict(WARM_MODELS_ON_STARTUP=False), dict(STATE_BACKEND='local'),
                         dict(DATABASE_URL='postgresql://synthetic@db.example.test/gk?sslmode=disable')])
def test_unsafe_production_settings_rejected(changes):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **(SAFE | changes))


def test_explicit_hardened_settings_accepted():
    settings = Settings(_env_file=None, **SAFE)
    assert settings.AUTH_MODE == 'required'


def test_pilot_profile_has_no_published_ports_or_default_passwords():
    import yaml
    profile = yaml.safe_load((Path(__file__).resolve().parents[1] / 'deploy/compose.pilot.yml').read_text())
    service = profile['services']['gatekeeper-api']
    assert 'ports' not in service
    assert service['read_only'] is True
    assert service['cap_drop'] == ['ALL']
    assert service['environment']['AUTH_MODE'] == 'required'
    assert service['environment']['HF_HUB_OFFLINE'] == '1'
    assert service['environment']['STATE_BACKEND'] == 'postgres'
    assert all(volume.endswith(':ro') for volume in service['volumes'])


def test_mcp_startup_refuses_unready_postgres(monkeypatch):
    from unittest.mock import Mock
    from fastapi.testclient import TestClient
    from core.config import settings
    from core.mcp_http_server import create_mcp_app
    store = Mock()
    store.ready.return_value = False
    monkeypatch.setattr(settings, 'STATE_BACKEND', 'postgres')
    monkeypatch.setattr('core.postgres_state.get_postgres_store', lambda: store)
    with pytest.raises(RuntimeError, match='Required PostgreSQL'):
        with TestClient(create_mcp_app()):
            pass

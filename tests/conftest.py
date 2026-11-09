"""
Session-wide test fixtures.

autouse fixtures here apply to EVERY test in the suite, not just files that
explicitly request them — used sparingly, only for global state that would
otherwise leak between tests and cause order-dependent failures.
"""
import pytest
import json
import ipaddress
import os
import socket

# Set before importing any application module. Real-model tests opt in to
# installed offline weights; source tests never download missing assets.
for _offline_flag in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[_offline_flag] = '1'

from core.circuit_breaker import llama_guard_breaker, ollama_judge_breaker
from core.rate_limit import assess_rate_limiter


def pytest_addoption(parser):
    parser.addoption('--allow-test-network', action='store_true', default=False,
                     help='Explicitly allow non-loopback network access for isolated integration infrastructure')


@pytest.fixture(autouse=True)
def _deny_external_test_network(monkeypatch, request):
    """Guard the current test process; loopback synthetic servers remain usable."""
    if request.config.getoption('--allow-test-network'):
        return
    connect, connect_ex, resolve = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def local(host):
        if host is None:
            return True
        if isinstance(host, bytes):
            host = host.decode('ascii')
        if str(host).lower().rstrip('.') == 'localhost':
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def guarded_resolve(host, *args, **kwargs):
        if not local(host):
            raise OSError('External network disabled in source tests')
        return resolve(host, *args, **kwargs)

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and not local(address[0]):
            raise OSError('External network disabled in source tests')
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        if isinstance(address, tuple) and not local(address[0]):
            raise OSError('External network disabled in source tests')
        return connect_ex(sock, address)

    monkeypatch.setattr(socket, 'getaddrinfo', guarded_resolve)
    monkeypatch.setattr(socket.socket, 'connect', guarded_connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', guarded_connect_ex)


@pytest.fixture
def provisioned_tenants(tmp_path, monkeypatch):
    """Explicit synthetic tenants for HTTP tests that previously used fallback."""
    from core import tenancy
    path = tmp_path / 'provisioned-tenants.json'
    path.write_text(json.dumps({name: {'status': 'active'} for name in
                               ('default', 'acme', 'acme-corp', 'beta', 'ops', 'strict-acme')}))
    monkeypatch.setattr(tenancy, '_store', tenancy.TenantStore(str(path)))


@pytest.fixture
def authenticated_mcp_transport(monkeypatch):
    from core.auth import Principal
    monkeypatch.setattr('core.mcp_http_server.resolve_request_principal',
                        lambda request: Principal(authenticated=True, key_id='synthetic-transport'))


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """
    Same reasoning as the breakers below: core/rate_limit.py's limiter is a
    process-wide singleton by design, so tokens spent by one test would
    otherwise still be missing in the next. A test suite that makes enough API
    calls would start seeing spurious 429s in whichever test happened to run
    last — an order-dependent failure that is miserable to diagnose.
    """
    assess_rate_limiter.reset()
    yield
    assess_rate_limiter.reset()


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """
    core/circuit_breaker.py's breakers are module-level singletons shared by
    every call in the process — deliberately, so they track a backend's real
    health across requests. That same property means failures recorded by one
    test (e.g. a test that feeds llama_guard_arbitration a raising stub) would
    otherwise accumulate toward the failure_threshold and could trip the
    breaker for a LATER, unrelated test that expects a call to actually be
    attempted. Reset before and after every test so each starts from a known
    closed state, independent of what ran before it.
    """
    ollama_judge_breaker.reset()
    llama_guard_breaker.reset()
    yield
    ollama_judge_breaker.reset()
    llama_guard_breaker.reset()

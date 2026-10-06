"""Security boundary tests: real persistence/transport, controlled inference/network."""
import asyncio
import json
import socket
import io
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.auth import Principal
from core.config import settings
from core.logger import AuditPersistenceError, log_event
from core.mcp_http_server import MCPSession, create_mcp_app
from core.tools import ToolRegistry, ToolSpec, execute_tool


@pytest.fixture(autouse=True)
def reset_mcp_buckets():
    from core.rate_limit import mcp_rate_limiter
    mcp_rate_limiter.reset()
    yield
    mcp_rate_limiter.reset()


def tool_registry(handler):
    registry = ToolRegistry()
    registry.register(ToolSpec(name="probe", description="probe", risk_level="LOW",
                              parameters={"type": "object", "properties": {}},
                              capability_required="GENERAL"), handler)
    return registry


@pytest.mark.parametrize("failure", [PermissionError("private value"), OSError("disk full")])
def test_authoritative_jsonl_write_failure_is_observable(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(settings, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(settings, "AUDIT_READ_FROM_SQLITE", False)
    with patch("builtins.open", side_effect=failure), pytest.raises(AuditPersistenceError):
        log_event("GENERAL", "private prompt", "LOW", "ALLOW")


def test_fsync_failure_is_observable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(settings, "AUDIT_READ_FROM_SQLITE", False)
    with patch("core.logger.os.fsync", side_effect=OSError("disk full")), pytest.raises(AuditPersistenceError):
        log_event("GENERAL", "private prompt", "LOW", "ALLOW")


def test_failed_intent_prevents_tool_side_effect():
    calls = []
    registry = tool_registry(lambda: calls.append("ran"))
    with patch("core.logger.log_tool_event", side_effect=AuditPersistenceError("unavailable")):
        with pytest.raises(AuditPersistenceError):
            execute_tool("GENERAL", "probe", {}, registry=registry)
    assert calls == []


def test_tool_intent_and_completion_are_durable(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(settings, "AUDIT_LOG_PATH", str(path))
    monkeypatch.setattr(settings, "AUDIT_READ_FROM_SQLITE", False)
    monkeypatch.setattr(settings, "AUDIT_SQLITE_ENABLED", False)

    def handler():
        assert json.loads(path.read_text().splitlines()[0])["phase"] == "execution_intent"
        return "done"

    result = execute_tool("GENERAL", "probe", {}, registry=tool_registry(handler))
    assert result["output"] == "done"
    assert [json.loads(line)["phase"] for line in path.read_text().splitlines()] == [
        "execution_intent", "execution_completed"]


def test_completion_failure_does_not_report_tool_success():
    calls = []
    with patch("core.logger.log_tool_event", side_effect=[None, AuditPersistenceError("unavailable")]):
        with pytest.raises(AuditPersistenceError):
            execute_tool("GENERAL", "probe", {}, registry=tool_registry(lambda: calls.append("ran")))
    assert calls == ["ran"]  # The already executed effect cannot be rolled back.


def test_tool_exception_payload_is_not_returned():
    def handler():
        raise RuntimeError("gk_" + "A" * 43)
    with patch("core.logger.log_tool_event"):
        result = execute_tool("GENERAL", "probe", {}, registry=tool_registry(handler))
    assert "gk_" not in json.dumps(result)
    assert "execution failed" in result["error"]


def test_untrusted_tool_names_and_argument_keys_are_not_written_verbatim(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(settings, "AUDIT_LOG_PATH", str(path))
    monkeypatch.setattr(settings, "AUDIT_READ_FROM_SQLITE", False)
    monkeypatch.setattr(settings, "AUDIT_SQLITE_ENABLED", False)
    secret = "gk_" + "A" * 43
    registry = tool_registry(lambda: pytest.fail("Blocked tool must not execute"))
    assert execute_tool("GENERAL", secret, {}, registry=registry)["decision"] == "BLOCK"
    result = execute_tool("GENERAL", "probe", {secret: "value"}, registry=registry)
    assert result["decision"] == "BLOCK" and secret not in result["reason"]
    assert secret not in path.read_text()


def test_stdio_internal_error_does_not_echo_request_or_exception_payload(caplog):
    from core.mcp_server import run_stdio_server
    secret = "gk_" + "A" * 43
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "probe", "arguments": {"private": secret}}}
    output = io.StringIO()
    with patch("core.mcp_server.handle_request", side_effect=RuntimeError(secret)):
        run_stdio_server(input_stream=io.StringIO(json.dumps(request) + "\n"), output_stream=output,
                         registry=ToolRegistry())
    assert json.loads(output.getvalue())["error"]["code"] == -32603
    assert secret not in output.getvalue()
    assert secret not in caplog.text


@pytest.mark.parametrize("provider_name", ["ollama", "openai", "anthropic"])
@pytest.mark.parametrize("failure_mode", ["network", "http_body", "json_shape"])
def test_provider_errors_do_not_echo_private_payload(provider_name, failure_mode):
    from core.llm_providers import OllamaProvider, OpenAICompatibleProvider, AnthropicCompatibleProvider, LLMProviderError
    from unittest.mock import MagicMock
    import requests
    secret = "gk_" + "A" * 43
    provider = {"ollama": OllamaProvider(), "openai": OpenAICompatibleProvider(api_key="fake-test-key"),
                "anthropic": AnthropicCompatibleProvider(api_key="fake-test-key")}[provider_name]
    response = MagicMock(status_code=503 if failure_mode == "http_body" else 200, text=secret)
    response.json.side_effect = ValueError(secret)
    with patch("core.llm_providers.requests.post", return_value=response,
               side_effect=requests.ConnectionError(secret) if failure_mode == "network" else None):
        with pytest.raises(LLMProviderError) as caught:
            provider.complete([{"role": "user", "content": "hello"}], model="test")
    assert secret not in str(caught.value)


@pytest.mark.parametrize("endpoint", ["/mcp/sse", "/mcp/jsonrpc", "/mcp/messages"])
def test_mcp_all_ingress_requires_configured_auth(endpoint):
    client = TestClient(create_mcp_app(ToolRegistry()))
    with patch("core.mcp_http_server.auth_required", return_value=True):
        response = client.get(endpoint) if endpoint.endswith("sse") else client.post(endpoint, json={})
    assert response.status_code == 401


def test_mcp_foreign_session_is_opaque_and_never_dispatches():
    app = create_mcp_app(ToolRegistry())
    app.state.sessions["known"] = MCPSession("another identity", asyncio.Queue(maxsize=1))
    with patch("core.mcp_http_server.handle_request") as dispatch:
        response = TestClient(app).post("/mcp/messages?sessionId=known", json={})
    assert response.status_code == 404
    dispatch.assert_not_called()


def test_session_identity_uses_verified_key_identity_across_header_formats():
    from core.mcp_http_server import _session_identity
    from starlette.requests import Request
    principal = Principal(tenant="owner", key_id="same-label", authenticated=True)

    def request(headers):
        return Request({"type": "http", "headers": headers, "client": ("127.0.0.1", 1)})

    bearer = _session_identity(request([(b"authorization", b"Bearer verified-key")]), principal)
    api_key = _session_identity(request([(b"x-api-key", b"verified-key")]), principal)
    other = _session_identity(request([(b"authorization", b"Bearer other-key")]), principal)
    assert bearer == api_key and bearer != other


def test_mcp_session_limit_blocks_allocation(monkeypatch):
    monkeypatch.setattr("core.mcp_http_server.MAX_SESSIONS", 0)
    app = create_mcp_app(ToolRegistry())
    response = TestClient(app).get("/mcp/sse")
    assert response.status_code == 503
    assert app.state.sessions == {}


def test_review_foreign_tenant_and_missing_are_indistinguishable():
    from api.main import app
    principal = Principal(tenant="owner-a", authenticated=True)
    with patch("api.main.resolve_principal", return_value=principal), \
         patch("api.main._reject_suspended_and_rate_limit"), \
         patch("api.main.get_review", side_effect=[{"tenant": "owner-b"}, None]):
        client = TestClient(app)
        foreign = client.get("/api/v1/review/known")
        missing = client.get("/api/v1/review/missing")
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


def test_review_needs_auth_even_in_optional_auth_mode():
    from api.main import app
    with patch("api.main.resolve_principal", return_value=Principal()), \
         patch("api.main.auth_required", return_value=False):
        response = TestClient(app).get("/api/v1/review/known")
    assert response.status_code == 401


def test_authenticated_internal_can_read_and_resolve_across_tenants(tmp_path):
    from api.main import app
    from core.review_queue import ReviewQueue
    queue = ReviewQueue(str(tmp_path / "reviews.json"))
    record = queue.enqueue(reason="review", capability="GENERAL", risk="HIGH", tenant="victim",
                           prompt_hash="hash", request_id="request")
    admin = Principal(capability="INTERNAL", tenant="admin", key_id="reviewer", authenticated=True)
    with patch("api.main.resolve_principal", return_value=admin), \
         patch("api.main._reject_suspended_and_rate_limit"), \
         patch("api.main.get_review", side_effect=queue.get), \
         patch("api.main.resolve_review", side_effect=queue.resolve):
        client = TestClient(app)
        status = client.get(f"/api/v1/review/{record.review_id}")
        resolved = client.post(f"/api/v1/review/{record.review_id}/resolve", json={"outcome": "REJECTED"})
    assert status.status_code == resolved.status_code == 200
    assert resolved.json()["tenant"] == "victim"
    assert resolved.json()["final_decision"] == "BLOCK"


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "169.254.169.254", "10.0.0.1", "ff02::1"])
def test_pinned_fetch_rejects_nonpublic_addresses_before_socket(address):
    from core.real_tools import DisallowedURLError, _pinned_fetch
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    infos = [(family, socket.SOCK_STREAM, 6, "", (address, 80))]
    with patch("core.real_tools.socket.getaddrinfo", return_value=infos), \
         patch("core.real_tools.socket.socket") as connect:
        with pytest.raises(DisallowedURLError):
            _pinned_fetch("http://allowed.example/", 1)
    connect.assert_not_called()


def test_pinned_fetch_rejects_empty_dns():
    from core.real_tools import DisallowedURLError, _pinned_fetch
    with patch("core.real_tools.socket.getaddrinfo", return_value=[]), pytest.raises(DisallowedURLError):
        _pinned_fetch("https://allowed.example/", 1)


def test_permissive_cache_never_replaces_fresh_assessment(tmp_path):
    import numpy as np
    from core.cache import FAISSCache
    from core.cache_backend import LocalExactCache
    cache = FAISSCache(dimension=384)
    cache._exact = LocalExactCache(str(tmp_path / "exact.json"))
    cache.save_async = lambda: None
    vector = np.ones(384, dtype="float32")
    cache.add("benign", vector, "LOW", 0.0)
    assert cache.lookup("benign", vector) == (None, None)
    assert cache.lookup("benign with an attack suffix", vector) == (None, None)


def test_security_cache_identity_separates_raw_inputs_and_tenants():
    from core.cache import _security_key, security_cache_scope
    keys = []
    for raw, tenant in [("John", "a"), ("Alice", "a"), ("John", "b")]:
        with security_cache_scope(raw, tenant):
            keys.append(_security_key("[REDACTED:PERSON]"))
    assert len(set(keys)) == 3


def test_changed_security_configuration_invalidates_cache_identity(monkeypatch):
    from core.cache import _security_key, security_cache_scope
    with security_cache_scope("original", "tenant"):
        old = _security_key("cleaned")
    monkeypatch.setattr(settings, "SEMANTIC_THRESHOLD_HIGH", settings.SEMANTIC_THRESHOLD_HIGH + 0.01)
    with security_cache_scope("original", "tenant"):
        new = _security_key("cleaned")
    assert old != new


@pytest.mark.parametrize("https,address,family", [
    (False, "1.1.1.1", socket.AF_INET),
    (True, "2606:4700:4700::1111", socket.AF_INET6),
])
def test_http_connection_uses_validated_address_and_original_host(https, address, family, monkeypatch):
    from core.real_tools import _pinned_fetch
    from unittest.mock import MagicMock
    import ssl
    port = 443 if https else 80
    sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
    sock = MagicMock()
    sock.makefile.return_value = io.BytesIO(
        b"HTTP/1.1 302 Found\r\nContent-Length: 3\r\nLocation: http://127.0.0.1/\r\n\r\nabc")
    context = ssl.create_default_context()
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    with patch("core.real_tools.socket.getaddrinfo", return_value=[(family, socket.SOCK_STREAM, 6, "", sockaddr)]) as dns, \
         patch("core.real_tools.socket.socket", return_value=sock), \
         patch("core.real_tools.ssl.create_default_context", return_value=context), \
         patch.object(context, "wrap_socket", return_value=sock) as tls:
        result = _pinned_fetch(("https" if https else "http") + "://approved.example/path", 2)
    dns.assert_called_once()
    sock.connect.assert_called_once_with(sockaddr)
    request = b"".join(call.args[0] for call in sock.sendall.call_args_list)
    assert b"Host: approved.example\r\n" in request
    assert result["status_code"] == 302 and result["content"] == "abc"
    if https:
        tls.assert_called_once_with(sock, server_hostname="approved.example")
    else:
        tls.assert_not_called()


def _stalled_fetch_worker(pipe, url, timeout):
    time.sleep(30)


def test_total_fetch_deadline_terminates_stalled_process():
    from core.real_tools import _isolated_fetch
    import multiprocessing
    before = {process.pid for process in multiprocessing.active_children()}
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        _isolated_fetch("https://approved.example", 0.5, _worker=_stalled_fetch_worker)
    assert time.monotonic() - start < 2.0
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_missing_configured_ner_never_returns_unredacted_content():
    from core.privacy import PrivacyAssessmentUnavailable, redact_pii
    with patch("core.privacy.NLP_MODEL", None), pytest.raises(PrivacyAssessmentUnavailable):
        redact_pii("John Doe works at Example Corp")


def test_missing_ner_blocks_output_even_with_safe_toxicity():
    from core.output_guardrails import assess_output
    with patch("core.privacy.NLP_MODEL", None), patch("core.output_guardrails.output_judge", return_value="SAFE"):
        decision, details = assess_output("John Doe works at Example Corp")
    assert decision == "BLOCK" and details["clean_response"] is None
    assert details["source"] == "privacy_unavailable_fail_closed"


def test_mcp_full_response_queue_rejects_before_dispatch(monkeypatch):
    monkeypatch.setattr("core.mcp_http_server._session_identity", lambda *args: "owner")
    monkeypatch.setattr("core.mcp_http_server.MAX_SESSION_QUEUE", 1)
    app = create_mcp_app(ToolRegistry())
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait({"existing": True})
    app.state.sessions["known"] = MCPSession("owner", queue)
    with patch("core.mcp_http_server.handle_request") as dispatch:
        response = TestClient(app).post("/mcp/messages?sessionId=known", json={})
    assert response.status_code == 503
    dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_slow_tool_does_not_block_event_loop():
    import httpx
    import threading
    started, release = threading.Event(), threading.Event()

    def slow_dispatch(**kwargs):
        started.set()
        assert release.wait(3)
        return {"jsonrpc": "2.0", "id": 1, "result": {}}

    app = create_mcp_app(ToolRegistry())
    with patch("core.mcp_http_server.handle_request", side_effect=slow_dispatch):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/mcp/jsonrpc", json={"id": 1}))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
                assert health.status_code == 200
            finally:
                release.set()
            assert (await pending).status_code == 200


@pytest.mark.asyncio
async def test_mcp_chunked_oversized_body_stops_reading_before_tail(monkeypatch):
    import httpx
    monkeypatch.setattr("core.mcp_http_server.MAX_PAYLOAD_BYTES", 8)
    consumed = []

    async def body():
        for chunk in [b"12345678", b"overflow", b"must not consume"]:
            consumed.append(chunk)
            yield chunk

    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_mcp_app(ToolRegistry())),
                                base_url="http://test") as client:
        response = await client.post("/mcp/jsonrpc", content=body())
    assert response.status_code == 413
    assert consumed == [b"12345678", b"overflow"]


def test_sqlite_authority_failure_is_not_hidden_by_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(settings, "AUDIT_READ_FROM_SQLITE", True)
    monkeypatch.setattr(settings, "AUDIT_SQLITE_ENABLED", True)
    with patch("core.audit_store.write", side_effect=OSError("database unavailable")):
        with pytest.raises(AuditPersistenceError):
            log_event("GENERAL", "private prompt", "LOW", "ALLOW")
    assert not (tmp_path / "audit.jsonl").exists()


@pytest.mark.parametrize("mode", ["standalone", "combined", "gateway"])
def test_unavailable_toxicity_is_blocked_at_every_http_output_boundary(mode, monkeypatch):
    from api.main import app
    from core.llm_providers import LLMResponse
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    response_text = "ordinary response"
    low = ("LOW", {"semantic_score": 0.0, "source": "mock"})
    provider_response = LLMResponse(content=response_text, model="test", provider="ollama", usage={})
    with patch("api.main.assess_risk", return_value=low), \
         patch("core.output_guardrails.output_judge", return_value="JUDGE_OFFLINE"), \
         patch("core.llm_providers.OllamaProvider.complete", return_value=provider_response):
        client = TestClient(app)
        if mode == "standalone":
            response = client.post("/api/v1/assess_output", json={"response_text": response_text})
        elif mode == "combined":
            response = client.post("/api/v1/assess", json={"prompt": "hello", "response_text": response_text})
        else:
            response = client.post("/api/v1/gateway/chat", json={"prompt": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "BLOCK"
    assert body.get("content") is None
    assert body.get("clean_response") is None
    assert response_text not in json.dumps(body)

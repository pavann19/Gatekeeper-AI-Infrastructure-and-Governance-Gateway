"""
Tests for core/mcp_http_server.py (MCP HTTP / SSE transport).
"""
import asyncio
import pytest
from fastapi.testclient import TestClient

from core.auth import Principal
from core.demo_tools import register_demo_tools
from core.mcp_http_server import MAX_PAYLOAD_BYTES, create_mcp_app
from core.rate_limit import assess_rate_limiter
from core.tenancy import TenantConfig
from core.tools import ToolRegistry


@pytest.fixture
def mcp_app():
    registry = ToolRegistry()
    register_demo_tools(registry)
    return create_mcp_app(registry=registry)


@pytest.fixture
def mcp_client(mcp_app):
    return TestClient(mcp_app)


def test_health_endpoint(mcp_client):
    res = mcp_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["transport"] == "mcp_http_sse"


def test_mcp_sse_handshake(mcp_client):
    response = mcp_client.get("/mcp/sse", headers={"X-Test-Stream-Once": "1"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: endpoint" in response.text
    assert "/mcp/messages?sessionId=" in response.text


def test_mcp_sse_session_relay(mcp_app, mcp_client):
    """
    Verifies full SSE session relay:
    1. Client registers an active SSE session queue in app.state.sessions.
    2. Client posts a JSON-RPC request to /mcp/messages?sessionId=<id>, receiving HTTP 202 Accepted.
    3. The JSON-RPC response is enqueued in the active session for SSE delivery.
    """
    session_id = "test-session-42"
    queue = asyncio.Queue()
    mcp_app.state.sessions[session_id] = queue

    msg = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {"name": "demo.echo", "arguments": {"text": "relayed over sse"}},
    }
    res = mcp_client.post(f"/mcp/messages?sessionId={session_id}", json=msg)
    assert res.status_code == 202
    assert res.json() == {"status": "accepted"}

    # Verify the JSON-RPC response was enqueued into the session's SSE delivery queue
    assert not queue.empty()
    queued_msg = queue.get_nowait()
    assert queued_msg["id"] == 101
    assert queued_msg["result"]["isError"] is False
    assert queued_msg["result"]["content"] == [{"type": "text", "text": "relayed over sse"}]


def test_mcp_messages_invalid_session_returns_404(mcp_client):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    res = mcp_client.post("/mcp/messages?sessionId=non-existent-session", json=msg)
    assert res.status_code == 404
    assert "not found or expired" in res.json()["detail"]


def test_mcp_messages_direct_fallback(mcp_client):
    """When sessionId is omitted, returns JSON-RPC response directly with HTTP 200."""
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    res = mcp_client.post("/mcp/messages", json=msg)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == 1
    assert data["result"]["serverInfo"]["name"] == "gatekeeper"


def test_mcp_messages_tools_list(mcp_client):
    msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    res = mcp_client.post("/mcp/messages", json=msg)
    assert res.status_code == 200
    data = res.json()
    tool_names = {t["name"] for t in data["result"]["tools"]}
    assert "demo.echo" in tool_names


def test_mcp_jsonrpc_direct(mcp_client):
    msg = {"jsonrpc": "2.0", "id": 4, "method": "tools/list"}
    res = mcp_client.post("/mcp/jsonrpc", json=msg)
    assert res.status_code == 200
    assert "demo.calculator.add" in {t["name"] for t in res.json()["result"]["tools"]}


def test_mcp_notification_returns_202(mcp_client):
    msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    res = mcp_client.post("/mcp/jsonrpc", json=msg)
    assert res.status_code == 202


# --- Security, pentest, and edge-case hardening tests ---

def test_mcp_oversized_payload_returns_413(mcp_client):
    """Requests exceeding 100KB are rejected with 413 Payload Too Large."""
    oversized_data = "x" * (MAX_PAYLOAD_BYTES + 1024)
    res = mcp_client.post(
        "/mcp/jsonrpc",
        content=f'{{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "padding": "{oversized_data}"}}',
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 413
    assert "exceeds maximum size" in res.json()["detail"]


def test_mcp_malformed_json_returns_400(mcp_client):
    res = mcp_client.post(
        "/mcp/jsonrpc",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 400
    assert "malformed JSON" in res.json()["detail"]


def test_mcp_non_dict_json_returns_400(mcp_client):
    res = mcp_client.post(
        "/mcp/jsonrpc",
        json=[1, 2, 3],
    )
    assert res.status_code == 400
    assert "must be a JSON object" in res.json()["detail"]


def test_mcp_suspended_tenant_returns_403(mcp_client, monkeypatch):
    from core import mcp_http_server as server_mod

    monkeypatch.setattr(
        server_mod,
        "resolve_principal",
        lambda authorization=None: Principal(authenticated=True, tenant="suspended-corp", key_id="susp-key", capability="GENERAL"),
    )
    monkeypatch.setattr(
        server_mod,
        "resolve_tenant",
        lambda tenant: TenantConfig(tenant_id="suspended-corp", status="suspended"),
    )

    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    res = mcp_client.post("/mcp/jsonrpc", json=msg)
    assert res.status_code == 403
    assert "suspended" in res.json()["detail"]


def test_mcp_auth_required_returns_401(mcp_client, monkeypatch):
    from core import mcp_http_server as server_mod

    monkeypatch.setattr(server_mod, "auth_required", lambda: True)
    monkeypatch.setattr(
        server_mod,
        "resolve_principal",
        lambda authorization=None: Principal(authenticated=False, tenant="anonymous", key_id="anonymous", capability="GENERAL"),
    )

    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    res = mcp_client.post("/mcp/jsonrpc", json=msg)
    assert res.status_code == 401
    assert "Authentication required" in res.json()["detail"]


def test_mcp_rate_limit_exceeded_returns_429(mcp_client, monkeypatch):
    # Force rate limiter check to reject
    monkeypatch.setattr(assess_rate_limiter, "check", lambda key, cap, ref: (False, 15.0))

    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    res = mcp_client.post("/mcp/jsonrpc", json=msg)
    assert res.status_code == 429
    assert "Rate limit exceeded" in res.json()["detail"]
    assert res.headers.get("Retry-After") == "16"

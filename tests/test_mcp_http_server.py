"""
Tests for core/mcp_http_server.py (MCP HTTP / SSE transport).
"""
import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from core.demo_tools import register_demo_tools
from core.mcp_http_server import create_mcp_app
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

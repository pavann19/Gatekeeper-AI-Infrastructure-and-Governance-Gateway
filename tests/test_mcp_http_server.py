"""
Tests for core/mcp_http_server.py (MCP HTTP / SSE transport).
"""
import pytest
from fastapi.testclient import TestClient

from core.demo_tools import register_demo_tools
from core.mcp_http_server import create_mcp_app
from core.tools import ToolRegistry


@pytest.fixture
def mcp_client():
    registry = ToolRegistry()
    register_demo_tools(registry)
    app = create_mcp_app(registry=registry)
    return TestClient(app)


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


def test_mcp_messages_initialize(mcp_client):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    res = mcp_client.post("/mcp/messages?sessionId=test-123", json=msg)
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


def test_mcp_messages_tools_call(mcp_client):
    msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "demo.echo", "arguments": {"text": "hello mcp"}},
    }
    res = mcp_client.post("/mcp/messages", json=msg)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == 3
    assert data["result"]["isError"] is False
    assert data["result"]["content"] == [{"type": "text", "text": "hello mcp"}]


def test_mcp_jsonrpc_direct(mcp_client):
    msg = {"jsonrpc": "2.0", "id": 4, "method": "tools/list"}
    res = mcp_client.post("/mcp/jsonrpc", json=msg)
    assert res.status_code == 200
    assert "demo.calculator.add" in {t["name"] for t in res.json()["result"]["tools"]}


def test_mcp_notification_returns_202(mcp_client):
    msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    res = mcp_client.post("/mcp/messages", json=msg)
    assert res.status_code == 202

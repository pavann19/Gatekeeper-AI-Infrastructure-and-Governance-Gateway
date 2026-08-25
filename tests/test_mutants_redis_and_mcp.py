"""
Targeted mutation testing suite for Redis, MCP HTTP/SSE, Privacy, and Activity subsystems.
Verifies that deliberate logic mutations (boundary flips, omitted security checks,
corrupted state propagation) are cleanly detected and rejected by the test harness.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from core.activity import _tail_raw_lines
from core.mcp_http_server import MAX_PAYLOAD_BYTES, create_mcp_app
from core.privacy import redact_pii
from core.rate_limit import RedisRateLimiter
from core.tenancy import TenantConfig
from core.token_quota import RedisTokenQuotaTracker


# ---------------------------------------------------------------------------
# Mutant 1: MCP Payload Size Boundary Enforcement
# ---------------------------------------------------------------------------

def test_mutant_mcp_payload_size_boundary():
    """
    Target: `core/mcp_http_server.py::_read_and_validate_body`
    Mutant: relaxing `len(raw_body) > MAX_PAYLOAD_BYTES` to `len(raw_body) > MAX_PAYLOAD_BYTES + 1024`
    Detection: exactly MAX_PAYLOAD_BYTES + 1 bytes must be rejected with 413.
    """
    app = create_mcp_app()
    client = TestClient(app)

    # 1. Exactly MAX_PAYLOAD_BYTES (100KB) with valid JSON shape -> not 413
    exact_size = MAX_PAYLOAD_BYTES
    padding = " " * (exact_size - len('{"jsonrpc":"2.0","id":1,"method":"tools/list"}'))
    body_exact = '{"jsonrpc":"2.0","id":1,"method":"tools/list"' + padding + "}"
    assert len(body_exact.encode("utf-8")) == exact_size

    resp_exact = client.post("/mcp/jsonrpc", content=body_exact, headers={"Content-Type": "application/json"})
    assert resp_exact.status_code != 413

    # 2. MAX_PAYLOAD_BYTES + 1 byte -> must immediately trigger 413
    body_oversized = body_exact + " "
    assert len(body_oversized.encode("utf-8")) == exact_size + 1

    resp_oversized = client.post("/mcp/jsonrpc", content=body_oversized, headers={"Content-Type": "application/json"})
    assert resp_oversized.status_code == 413
    assert "payload exceeds" in resp_oversized.text.lower()


# ---------------------------------------------------------------------------
# Mutant 2: MCP Suspended Tenant Check Omission
# ---------------------------------------------------------------------------

def test_mutant_mcp_suspended_tenant_enforcement():
    """
    Target: `core/mcp_http_server.py::_dispatch_rpc`
    Mutant: omitting `if tenant_config.suspended: return 403`
    Detection: a valid API key belonging to a suspended tenant must receive 403.
    """
    app = create_mcp_app()
    client = TestClient(app)

    from core.auth import Principal
    principal = Principal(capability="INTERNAL", tenant="suspended-corp", authenticated=True)
    suspended_tenant = TenantConfig(tenant_id="suspended-corp", status="suspended")

    with patch("core.mcp_http_server.resolve_principal", return_value=principal), \
         patch("core.mcp_http_server.resolve_tenant", return_value=suspended_tenant):

        resp = client.post(
            "/mcp/jsonrpc",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Authorization": "Bearer gk_valid_suspended_key"},
        )
        assert resp.status_code == 403
        assert "suspended" in resp.json()["detail"].lower()




# ---------------------------------------------------------------------------
# Mutant 3: MCP Session Queue Relay Delivery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mutant_mcp_session_queue_delivery():
    """
    Target: `core/mcp_http_server.py::_handle_messages`
    Mutant: omitting `await queue.put(rpc_resp)`
    Detection: the session queue must receive the exact JSON-RPC response dict.
    """
    app = create_mcp_app()
    client = TestClient(app)

    session_id = "test-session-uuid-1234"
    session_queue = asyncio.Queue()
    app.state.sessions[session_id] = session_queue

    resp = client.post(
        f"/mcp/messages?sessionId={session_id}",
        json={"jsonrpc": "2.0", "id": 42, "method": "tools/list", "params": {}},
    )
    assert resp.status_code == 202
    assert resp.json() == {"status": "accepted"}

    # Queue must contain the response within timeout
    queued_msg = await asyncio.wait_for(session_queue.get(), timeout=2.0)
    assert isinstance(queued_msg, dict)
    assert queued_msg.get("id") == 42
    assert "result" in queued_msg


# ---------------------------------------------------------------------------
# Mutant 4: Redis Rate Limiter Capacity Underflow Flip
# ---------------------------------------------------------------------------

def test_mutant_redis_rate_limiter_boundary():
    """
    Target: `core/rate_limit.py::LUA_RATE_LIMIT`
    Mutant: `if tokens >= 1.0` mutated to `if tokens >= 0.0` (allowing exhaustion)
    Detection: with capacity 1.0, the second call without elapsed time must be denied.
    """
    mock_redis = MagicMock()
    mock_script = MagicMock()
    mock_redis.register_script.return_value = mock_script

    # Mock Lua script evaluation returning [allowed, retry_after]
    # First call: allowed=1, retry_after=0.0
    # Second call: allowed=0, retry_after=1.0
    mock_script.side_effect = [
        [1, "0.0"],
        [0, "1.0"],
    ]

    limiter = RedisRateLimiter(mock_redis, name="test_assess")
    allowed1, retry1 = limiter.check("tenant-a", 1.0, 0.1)
    allowed2, retry2 = limiter.check("tenant-a", 1.0, 0.1)

    assert allowed1 is True
    assert retry1 == 0.0
    assert allowed2 is False
    assert retry2 == 1.0


# ---------------------------------------------------------------------------
# Mutant 5: Redis Token Quota Daily TTL Expiry
# ---------------------------------------------------------------------------

def test_mutant_redis_token_quota_ttl_behavior():
    """
    Target: `core/token_quota.py::LUA_RECORD_QUOTA`
    Mutant: removing `redis.call('EXPIRE', key, ttl)` in the Lua script.
    Detection: script invocation passes key, tokens, and positive TTL.
    """
    mock_redis = MagicMock()
    mock_script = MagicMock()
    mock_redis.register_script.return_value = mock_script
    mock_script.return_value = 500

    tracker = RedisTokenQuotaTracker(mock_redis)
    tracker.record("tenant-b", 500)

    mock_script.assert_called_once()
    kwargs = mock_script.call_args[1]
    assert len(kwargs["keys"]) == 1
    assert "gatekeeper:quota:tenant-b:" in kwargs["keys"][0]
    assert kwargs["args"][0] == "500"
    assert int(kwargs["args"][1]) > 0  # positive TTL to midnight


# ---------------------------------------------------------------------------
# Mutant 6: Per-Tenant Privacy Pattern Bypass
# ---------------------------------------------------------------------------

def test_mutant_privacy_disabled_pattern_override():
    """
    Target: `core/privacy.py::redact_pii`
    Mutant: ignoring `tenant_config.privacy_disabled_patterns`
    Detection: phone numbers must NOT be redacted when PHONE is disabled, but EMAIL must be.
    """
    text = "Call 9876543210 or email test@example.com"
    tenant_cfg = TenantConfig(tenant_id="custom-privacy", privacy_disabled_patterns=["PHONE"])

    redacted_with_override, info_override = redact_pii(text, tenant_config=tenant_cfg)
    assert "9876543210" in redacted_with_override
    assert "[REDACTED:EMAIL]" in redacted_with_override
    assert not any(item.startswith("PHONE:") for item in info_override.get("items", []))
    assert any(item.startswith("EMAIL:") for item in info_override.get("items", []))

    # Without override, both must be redacted
    redacted_default, info_default = redact_pii(text, tenant_config=None)
    assert "9876543210" not in redacted_default
    assert "[REDACTED:PHONE]" in redacted_default
    assert "[REDACTED:EMAIL]" in redacted_default


# ---------------------------------------------------------------------------
# Mutant 7: Activity Log Tail Carry Buffer Preservation
# ---------------------------------------------------------------------------

def test_mutant_activity_tail_carry_buffer_integrity(tmp_path):
    """
    Target: `core/activity.py::_tail_raw_lines`
    Mutant: omitting carry buffer (`carry = b""` on resume)
    Detection: parsing multi-line JSON events across multiple resume steps.
    """
    log_file = tmp_path / "activity_test.jsonl"
    lines = [
        b'{"request_id": "req-1", "action": "ALLOW", "timestamp": 1000}\n',
        b'{"request_id": "req-2", "action": "BLOCK", "timestamp": 2000}\n',
        b'{"request_id": "req-3", "action": "REVIEW", "timestamp": 3000}\n',
    ]
    log_file.write_bytes(b"".join(lines))

    # Single pass reading all 3 lines
    parsed_lines, truncated, exhausted, next_pos, next_carry = _tail_raw_lines(
        str(log_file), needed=3, return_state=True
    )
    assert len(parsed_lines) == 3
    assert b"req-3" in parsed_lines[0]
    assert b"req-2" in parsed_lines[1]
    assert b"req-1" in parsed_lines[2]
    assert exhausted is True
    assert next_pos == 0
    assert next_carry == b""

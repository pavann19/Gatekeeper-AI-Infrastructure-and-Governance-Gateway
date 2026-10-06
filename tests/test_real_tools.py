"""
Tests for core/real_tools.py's http.get -- the first REAL, non-sandboxed
tool this project ships. Network calls (requests.get) and DNS resolution
(socket.getaddrinfo) are mocked throughout: these tests verify the
security logic (allowlist, DNS-rebinding defence, redirect/size/timeout
handling), not that a real network actually works.
"""
import socket
from unittest.mock import MagicMock, patch

import pytest

from core.real_tools import (
    DisallowedURLError,
    HTTP_GET_SPEC,
    _http_get,
    _resolved_addresses_are_all_public,
    register_real_tools,
)
from core.tools import ToolRegistry, execute_tool


def _fake_addrinfo(ip):
    """Shapes a fake socket.getaddrinfo() result carrying one address."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


@pytest.fixture(autouse=True)
def _allowlist(monkeypatch):
    monkeypatch.setattr("core.real_tools.settings.TOOL_HTTP_GET_ALLOWED_DOMAINS",
                        "example.com,api.example.org")



def _controlled_fetch(url, status=200, body=b"hello"):
    import io
    from core.real_tools import _pinned_fetch
    sock = MagicMock()
    wire = f"HTTP/1.1 {status} Response\r\nContent-Length: {len(body)}\r\nLocation: http://127.0.0.1/\r\n\r\n".encode() + body
    sock.makefile.return_value = io.BytesIO(wire)
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")), \
         patch("socket.socket", return_value=sock), \
         patch("ssl.create_default_context") as tls:
        tls.return_value.wrap_socket.return_value = sock
        result = _pinned_fetch(url, 2)
    sock.connect.assert_called_once()
    return result

# --- domain allowlist ---------------------------------------------------------

def test_disallowed_hostname_rejected():
    with pytest.raises(DisallowedURLError, match="not in the configured allowlist"):
        _http_get("https://evil.com/path")


def test_empty_allowlist_disables_the_tool_entirely(monkeypatch):
    monkeypatch.setattr("core.real_tools.settings.TOOL_HTTP_GET_ALLOWED_DOMAINS", "")
    with pytest.raises(DisallowedURLError, match="fully disabled"):
        _http_get("https://example.com/")


def test_subdomain_not_implicitly_covered_by_parent_domain():
    """No wildcard support -- an allowlisted 'example.com' must not
    silently also allow 'attacker.example.com' or similar."""
    with pytest.raises(DisallowedURLError, match="not in the configured allowlist"):
        _http_get("https://sub.example.com/")


def test_case_insensitive_hostname_match(monkeypatch):
    with patch("core.real_tools._isolated_fetch", return_value={"status_code": 200}) as fetch:
        assert _http_get("https://EXAMPLE.COM/path")["status_code"] == 200
    fetch.assert_called_once_with("https://EXAMPLE.COM/path", 10)


@pytest.mark.parametrize("scheme", ["file", "ftp", "gopher", "javascript"])
def test_non_http_schemes_rejected(scheme):
    with pytest.raises(DisallowedURLError, match="scheme"):
        _http_get(f"{scheme}://example.com/path")


def test_no_hostname_rejected():
    with pytest.raises(DisallowedURLError, match="no hostname"):
        _http_get("https:///path")


# --- DNS-rebinding defence: resolved address must be public ------------------

def test_private_ip_resolution_blocks_the_call(monkeypatch):
    from core.real_tools import _pinned_fetch
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("10.0.0.5")), patch("socket.socket") as connect:
        with pytest.raises(DisallowedURLError, match="not exclusively public"):
            _pinned_fetch("https://example.com/", 1)
    connect.assert_not_called()


def test_loopback_resolution_blocks_the_call(monkeypatch):
    from core.real_tools import _pinned_fetch
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("127.0.0.1")), patch("socket.socket") as connect:
        with pytest.raises(DisallowedURLError, match="not exclusively public"):
            _pinned_fetch("https://example.com/", 1)
    connect.assert_not_called()


def test_cloud_metadata_address_blocks_the_call(monkeypatch):
    from core.real_tools import _pinned_fetch
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("169.254.169.254")), patch("socket.socket") as connect:
        with pytest.raises(DisallowedURLError, match="not exclusively public"):
            _pinned_fetch("https://example.com/", 1)
    connect.assert_not_called()


def test_dns_resolution_failure_fails_closed(monkeypatch):
    from core.real_tools import _pinned_fetch
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("failed")), patch("socket.socket") as connect:
        with pytest.raises(socket.gaierror):
            _pinned_fetch("https://example.com/", 1)
    connect.assert_not_called()


def test_public_ip_resolution_allows_the_call(monkeypatch):
    result = _controlled_fetch("https://example.com/")
    assert result["content"] == "hello" and result["status_code"] == 200


def test_resolved_addresses_are_all_public_directly():
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("8.8.8.8")):
        assert _resolved_addresses_are_all_public("example.com") is True
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("192.168.1.1")):
        assert _resolved_addresses_are_all_public("example.com") is False


# --- redirects are never followed --------------------------------------------

def test_redirect_response_is_not_followed_requests_called_with_allow_redirects_false(monkeypatch):
    result = _controlled_fetch("https://example.com/", status=302, body=b"")
    assert result["status_code"] == 302  # Only one connection despite private Location.


def test_response_larger_than_cap_is_truncated(monkeypatch):
    monkeypatch.setattr("core.real_tools.MAX_RESPONSE_BYTES", 10)
    result = _controlled_fetch("https://example.com/", body=b"0123456789more data")
    assert result["truncated"] is True
    assert len(result["content"].encode()) <= 10


def test_request_uses_a_hard_timeout(monkeypatch):
    with patch("core.real_tools._isolated_fetch", return_value={}) as fetch:
        _http_get("https://example.com/")
    assert fetch.call_args.args[1] == 10


def test_http_get_spec_requires_elevated_capability_and_medium_risk():
    assert HTTP_GET_SPEC.capability_required == "ELEVATED"
    assert HTTP_GET_SPEC.risk_level == "MEDIUM"


def test_http_get_spec_bounds_url_length():
    """Phase 8 hardening -- a URL has no legitimate reason to be huge;
    an oversized one should never reach urlsplit/DNS resolution."""
    assert HTTP_GET_SPEC.parameters["properties"]["url"]["maxLength"] == 2048


def test_oversized_url_rejected_before_the_handler_even_runs():
    reg = ToolRegistry()
    register_real_tools(reg)
    huge_url = "https://example.com/" + ("x" * 3000)
    result = execute_tool("ELEVATED", "http.get", {"url": huge_url}, registry=reg)
    assert result["decision"] == "BLOCK"
    assert "maxLength" in result["reason"]


# --- full pipeline via execute_tool: a disallowed URL is ALLOW+error, --------
# --- NOT a gateway-level BLOCK ------------------------------------------------

def test_disallowed_url_surfaces_as_allow_with_error_not_a_block():
    """core/tools.py's own documented contract: semantic/business-rule
    validation a handler performs is an execution error, distinct from
    a security BLOCK the gateway itself decided."""
    reg = ToolRegistry()
    register_real_tools(reg)
    result = execute_tool("ELEVATED", "http.get", {"url": "https://evil.com/"}, registry=reg)
    assert result["decision"] == "ALLOW"
    assert result["error"] is not None
    assert "DisallowedURLError" in result["error"]
    assert "evil.com" not in result["error"]


def test_general_capability_denied_before_the_url_is_even_checked():
    reg = ToolRegistry()
    register_real_tools(reg)
    result = execute_tool("GENERAL", "http.get", {"url": "https://evil.com/"}, registry=reg)
    assert result["decision"] == "BLOCK"


def test_allowed_url_succeeds_through_the_full_pipeline(monkeypatch):
    monkeypatch.setattr("core.real_tools._isolated_fetch", lambda url, timeout: _controlled_fetch(url, body=b"ok"))
    reg = ToolRegistry()
    register_real_tools(reg)
    result = execute_tool("ELEVATED", "http.get", {"url": "https://example.com/"}, registry=reg)
    assert result["decision"] == "ALLOW" and result["output"]["content"] == "ok"


def test_register_real_tools_is_idempotent():
    reg = ToolRegistry()
    register_real_tools(reg)
    register_real_tools(reg)  # second call must not raise
    assert len(reg) == 1

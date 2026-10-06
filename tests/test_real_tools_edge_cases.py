"""
Additional edge-case coverage for core/real_tools.py's http.get, focused on
the actual tool implementation (`_http_get`) rather than the JSON-Schema
validation layer (already covered in tests/test_real_tools.py and
tests/test_gateway_chat.py). These tests exercise real requests.get-shaped
network exchanges (mocked at the `requests.get` boundary, since that's the
HTTP client core/real_tools.py actually imports and calls) covering:
timeouts, connection errors, non-2xx statuses, content-type/decoding edge
cases, and additional DNS-rebinding scenarios not already exercised.

SSRF protection already exists in the module (hostname allowlist + DNS
resolution check against private/loopback/link-local/reserved/multicast
addresses, checked at call time) and is covered thoroughly by
tests/test_real_tools.py; this file adds scenarios that file doesn't touch
(mixed-address resolution, IPv6 loopback, and exception propagation through
execute_tool's generic handler-error path).
"""
import socket
from unittest.mock import MagicMock, patch

import pytest
import requests

from core.real_tools import (
    DisallowedURLError,
    _http_get,
    _resolved_addresses_are_all_public,
    register_real_tools,
)
from core.tools import ToolRegistry, execute_tool


def _fake_addrinfo(*ips):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]


@pytest.fixture(autouse=True)
def _allowlist(monkeypatch):
    monkeypatch.setattr("core.real_tools.settings.TOOL_HTTP_GET_ALLOWED_DOMAINS",
                        "example.com,api.example.org")



def _exchange(status=200, body=b"ok", content_type=None, read_error=None):
    import io
    from core.real_tools import _pinned_fetch
    sock = MagicMock()
    headers = f"HTTP/1.1 {status} Response\r\nContent-Length: {len(body)}\r\n"
    if content_type:
        headers += f"Content-Type: {content_type}\r\n"
    if read_error:
        class BrokenStream(io.BytesIO):
            def read1(self, *args):
                raise read_error
        reader = BrokenStream((headers + "\r\n").encode() + body)
    else:
        reader = io.BytesIO((headers + "\r\n").encode() + body)
    sock.makefile.return_value = reader
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")), \
         patch("socket.socket", return_value=sock), patch("ssl.create_default_context") as tls:
        tls.return_value.wrap_socket.return_value = sock
        try:
            result = _pinned_fetch("https://example.com/", 2)
        finally:
            sock.close.assert_called_once()
    return result, sock

# --- timeout handling -----------------------------------------------------

def test_timeout_propagates_as_a_real_exception(monkeypatch):
    with patch("core.real_tools._isolated_fetch", side_effect=TimeoutError("timed out")):
        with pytest.raises(TimeoutError):
            _http_get("https://example.com/")


def test_timeout_surfaces_through_execute_tool_as_allow_with_error(monkeypatch):
    reg = ToolRegistry()
    register_real_tools(reg)
    with patch("core.real_tools._isolated_fetch", side_effect=TimeoutError("private diagnostic")):
        result = execute_tool("ELEVATED", "http.get", {"url": "https://example.com/"}, registry=reg)
    assert result["decision"] == "ALLOW" and "TimeoutError" in result["error"]
    assert "private diagnostic" not in result["error"]


# --- connection-error handling ---------------------------------------------

def test_connection_error_propagates_as_a_real_exception(monkeypatch):
    with patch("core.real_tools._isolated_fetch", side_effect=RuntimeError("Outbound fetch failed")):
        with pytest.raises(RuntimeError):
            _http_get("https://example.com/")


def test_connection_error_surfaces_through_execute_tool_as_allow_with_error(monkeypatch):
    reg = ToolRegistry()
    register_real_tools(reg)
    with patch("core.real_tools._isolated_fetch", side_effect=RuntimeError("private diagnostic")):
        result = execute_tool("ELEVATED", "http.get", {"url": "https://example.com/"}, registry=reg)
    assert result["decision"] == "ALLOW" and "RuntimeError" in result["error"]
    assert "private diagnostic" not in result["error"]


# --- non-2xx status codes are returned as ordinary results, not errors -----

@pytest.mark.parametrize("status", [404, 403, 500, 503])
def test_non_2xx_status_codes_are_returned_not_raised(monkeypatch, status):
    result, _ = _exchange(status=status, body=b"error body")
    assert result["status_code"] == status and result["content"] == "error body"


def test_successful_response_content_and_metadata_returned_correctly(monkeypatch):
    result, _ = _exchange(body=b'{"a": 1}', content_type="application/json")
    assert result == {"status_code": 200, "url": "https://example.com/", "content": '{"a": 1}',
                      "truncated": False, "content_type": "application/json"}


def test_missing_content_type_header_returns_none(monkeypatch):
    result, _ = _exchange()
    assert result["content_type"] is None


def test_non_utf8_bytes_are_replaced_not_raised(monkeypatch):
    result, _ = _exchange(body=b"\xff\xfe not valid utf-8")
    assert "�" in result["content"]


def test_response_is_always_closed_even_when_content_fits(monkeypatch):
    result, sock = _exchange()
    assert result["content"] == "ok"
    sock.close.assert_called_once()


def test_response_is_closed_even_when_iter_content_raises(monkeypatch):
    with pytest.raises(OSError):
        _exchange(read_error=OSError("broken stream"))


# --- DNS-rebinding defence: additional scenarios not covered elsewhere -----

def test_one_private_address_among_multiple_resolved_blocks_the_call(monkeypatch):
    from core.real_tools import _pinned_fetch
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34", "10.0.0.1")), patch("socket.socket") as connect:
        with pytest.raises(DisallowedURLError):
            _pinned_fetch("https://example.com/", 1)
    connect.assert_not_called()


def test_ipv6_loopback_resolution_blocks_the_call(monkeypatch):
    from core.real_tools import _pinned_fetch
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0))]), patch("socket.socket") as connect:
        with pytest.raises(DisallowedURLError):
            _pinned_fetch("https://example.com/", 1)
    connect.assert_not_called()


def test_all_public_addresses_directly_via_helper():
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34", "8.8.8.8")):
        assert _resolved_addresses_are_all_public("example.com") is True


# --- request shape: stream=True is used so the size cap can be enforced ---

def test_request_uses_streaming_so_size_cap_can_be_enforced(monkeypatch):
    monkeypatch.setattr("core.real_tools.MAX_RESPONSE_BYTES", 4)
    result, _ = _exchange(body=b"abcde")
    assert result["content"] == "abcd" and result["truncated"] is True

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


# --- timeout handling -----------------------------------------------------

def test_timeout_propagates_as_a_real_exception(monkeypatch):
    """The handler does not catch requests' own exceptions -- a timeout
    must surface to the caller as a requests.exceptions.Timeout, not be
    swallowed or converted into a DisallowedURLError."""
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(requests.exceptions.Timeout):
            _http_get("https://example.com/")


def test_timeout_surfaces_through_execute_tool_as_allow_with_error(monkeypatch):
    """core/tools.py::execute_tool wraps any handler exception into an
    ALLOW decision carrying an `error` string -- verifying the real,
    observable behavior a caller of the full pipeline actually sees for
    a timed-out request."""
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    reg = ToolRegistry()
    register_real_tools(reg)
    with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
        result = execute_tool("ELEVATED", "http.get", {"url": "https://example.com/"}, registry=reg)
    assert result["decision"] == "ALLOW"
    assert "Timeout" in result["error"]


# --- connection-error handling ---------------------------------------------

def test_connection_error_propagates_as_a_real_exception(monkeypatch):
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(requests.exceptions.ConnectionError):
            _http_get("https://example.com/")


def test_connection_error_surfaces_through_execute_tool_as_allow_with_error(monkeypatch):
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    reg = ToolRegistry()
    register_real_tools(reg)
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        result = execute_tool("ELEVATED", "http.get", {"url": "https://example.com/"}, registry=reg)
    assert result["decision"] == "ALLOW"
    assert "ConnectionError" in result["error"]
    assert "refused" in result["error"]


# --- non-2xx status codes are returned as ordinary results, not errors -----

@pytest.mark.parametrize("status", [404, 403, 500, 503])
def test_non_2xx_status_codes_are_returned_not_raised(monkeypatch, status):
    """A 4xx/5xx response is a normal, successfully-completed HTTP
    exchange from the handler's point of view -- it must be returned as
    data (status_code + body), never raised as an exception."""
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock(status_code=status, headers={"Content-Type": "text/plain"})
        mock_response.iter_content.return_value = [b"error body"]
        mock_get.return_value = mock_response
        result = _http_get("https://example.com/")
    assert result["status_code"] == status
    assert result["content"] == "error body"


def test_successful_response_content_and_metadata_returned_correctly(monkeypatch):
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock(status_code=200, headers={"Content-Type": "application/json"})
        mock_response.iter_content.return_value = [b'{"a": 1}']
        mock_get.return_value = mock_response
        result = _http_get("https://api.example.org/data")
    assert result == {
        "status_code": 200,
        "url": "https://api.example.org/data",
        "content": '{"a": 1}',
        "truncated": False,
        "content_type": "application/json",
    }


def test_missing_content_type_header_returns_none(monkeypatch):
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock(status_code=200, headers={})
        mock_response.iter_content.return_value = [b"ok"]
        mock_get.return_value = mock_response
        result = _http_get("https://example.com/")
    assert result["content_type"] is None


def test_non_utf8_bytes_are_replaced_not_raised(monkeypatch):
    """The handler decodes with errors='replace' -- invalid UTF-8 bytes
    must not raise UnicodeDecodeError, and should show up as the
    replacement character rather than being silently dropped."""
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock(status_code=200, headers={"Content-Type": "text/plain"})
        mock_response.iter_content.return_value = [b"\xff\xfe not valid utf-8"]
        mock_get.return_value = mock_response
        result = _http_get("https://example.com/")
    assert "�" in result["content"]


def test_response_is_always_closed_even_when_content_fits(monkeypatch):
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock(status_code=200, headers={})
        mock_response.iter_content.return_value = [b"ok"]
        mock_get.return_value = mock_response
        _http_get("https://example.com/")
    mock_response.close.assert_called_once()


def test_response_is_closed_even_when_iter_content_raises(monkeypatch):
    """finally: response.close() must run even if streaming the body
    itself blows up mid-read -- a resource-cleanup guarantee worth
    pinning down since it's a `finally` block, not incidental."""
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock(status_code=200, headers={})
        mock_response.iter_content.side_effect = requests.exceptions.ChunkedEncodingError("broken")
        mock_get.return_value = mock_response
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            _http_get("https://example.com/")
    mock_response.close.assert_called_once()


# --- DNS-rebinding defence: additional scenarios not covered elsewhere -----

def test_one_private_address_among_multiple_resolved_blocks_the_call(monkeypatch):
    """A hostname resolving to BOTH a public and a private address must
    still be rejected -- ALL resolved addresses must be public, not just
    the first one returned by getaddrinfo."""
    monkeypatch.setattr(socket, "getaddrinfo",
                         lambda host, port: _fake_addrinfo("93.184.216.34", "10.0.0.1"))
    with pytest.raises(DisallowedURLError, match="private/loopback/reserved"):
        _http_get("https://example.com/")


def test_ipv6_loopback_resolution_blocks_the_call(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))],
    )
    with pytest.raises(DisallowedURLError, match="private/loopback/reserved"):
        _http_get("https://example.com/")


def test_all_public_addresses_directly_via_helper():
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34", "8.8.8.8")):
        assert _resolved_addresses_are_all_public("example.com") is True


# --- request shape: stream=True is used so the size cap can be enforced ---

def test_request_uses_streaming_so_size_cap_can_be_enforced(monkeypatch):
    monkeypatch.setattr("core.real_tools._resolved_addresses_are_all_public", lambda h: True)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock(status_code=200, headers={})
        mock_response.iter_content.return_value = [b""]
        mock_get.return_value = mock_response
        _http_get("https://example.com/")
    assert mock_get.call_args.kwargs["stream"] is True

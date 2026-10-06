"""
The first REAL (non-sandboxed) tool this project ships: `http.get`, an
outbound HTTP GET restricted to an operator-configured domain allowlist.
Everything in `core/demo_tools.py` operates on an in-memory fake dataset
by design; this one makes an actual network call to a real destination,
which is why it gets real security engineering around it rather than
the same treatment a demo tool needs.

WHY DOMAIN ALLOWLISTING ISN'T ENOUGH ON ITS OWN, AND WHAT ELSE IS HERE
--------------------------------------------------------------------------
A hostname allowlist alone is vulnerable to DNS rebinding: an attacker
who controls DNS for an allowlisted domain (or gets a TTL-expired lookup
to re-resolve mid-attack) can point it at `127.0.0.1` or an internal
address, turning "fetch this approved external URL" into an internal
port scan or cloud-metadata-endpoint read (`169.254.169.254`, the
classic SSRF target). This tool therefore also resolves the hostname
itself and rejects the call if ANY resolved address is private,
loopback, link-local, or otherwise non-global — checked at CALL TIME,
not once at startup. The connection uses the validated socket address
directly, avoiding a second DNS lookup. Host and TLS verification retain
the original hostname. A killable worker bounds DNS and streaming together.

Redirects are not followed (`allow_redirects=False`): a 3xx response
pointing somewhere off the allowlist would otherwise be a clean bypass
of the entire allowlist check, and a same-allowlist redirect gains
nothing worth the added complexity of re-validating a second URL. The
caller sees the redirect as a normal (non-2xx) response instead.

Response size is capped (`MAX_RESPONSE_BYTES`) and the request carries a
hard timeout (`REQUEST_TIMEOUT_SECONDS`) — an allowlisted domain serving
an enormous or slow response is still a resource-exhaustion vector this
tool's caller (Gatekeeper's own bounded worker pools) shouldn't have to
absorb unbounded.

WHY A DISALLOWED DOMAIN IS AN EXECUTION ERROR, NOT A BLOCK
------------------------------------------------------------
`core/tools.py`'s own architecture draws this line already: JSON-Schema
structural validation (does `url` look like a string) is generic and
gateway-level; "is this SPECIFIC url allowed" is semantic, tool-specific
business logic that only the handler can enforce, and `execute_tool`'s
docstring documents exactly this shape — a handler that raises reports
an execution error ("bad input the schema didn't catch"), distinct from
a security BLOCK the gateway itself decided. A disallowed domain is
precisely that: the schema correctly validated `url` as a string; this
handler is what knows which strings are actually acceptable.
"""
from __future__ import annotations

import http.client
import multiprocessing
import ssl
import time

import ipaddress
import socket
from urllib.parse import urlsplit

from core.config import settings
from core.tools import ToolSpec

MAX_RESPONSE_BYTES = 1_000_000
REQUEST_TIMEOUT_SECONDS = 10


class DisallowedURLError(ValueError):
    """Raised for any URL this tool refuses to fetch — wrong scheme, an
    unlisted hostname, or a hostname that resolves to a non-public
    address. A subclass of ValueError so it still lands exactly where
    `core/tools.py::execute_tool`'s docstring says a handler's own
    validation failure should: an ALLOW decision with `error` set, not a
    gateway-level BLOCK."""


def _allowed_domains() -> set:
    return {d.strip().lower() for d in settings.TOOL_HTTP_GET_ALLOWED_DOMAINS.split(",") if d.strip()}


def _resolved_addresses_are_all_public(hostname: str) -> bool:
    """Resolves `hostname` and returns False if ANY result is private,
    loopback, link-local, reserved, or a multicast address — the DNS-
    rebinding defence described in the module docstring. A hostname that
    fails to resolve at all is treated as not-public (fail closed): a
    resolution failure here should surface as "we could not verify this
    is safe to fetch," never as an implicit pass."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    if not infos:
        return False
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False
    return True


def _http_get(url: str) -> dict:
    parts = urlsplit(url)

    if parts.scheme not in ("http", "https"):
        raise DisallowedURLError(f"scheme {parts.scheme!r} is not allowed; only http/https")

    hostname = (parts.hostname or "").lower()
    if not hostname:
        raise DisallowedURLError("URL has no hostname")

    allowed = _allowed_domains()
    if not allowed:
        raise DisallowedURLError(
            "TOOL_HTTP_GET_ALLOWED_DOMAINS is empty -- this tool is fully disabled "
            "until an operator explicitly allowlists at least one hostname"
        )
    if hostname not in allowed:
        raise DisallowedURLError(f"hostname {hostname!r} is not in the configured allowlist")

    if parts.username is not None or parts.password is not None:
        raise DisallowedURLError("URL credentials are forbidden")
    return _isolated_fetch(url, REQUEST_TIMEOUT_SECONDS)


def _pinned_fetch(url, timeout):
    deadline = time.monotonic() + timeout
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not infos or any(not ipaddress.ip_address(info[4][0]).is_global
                        or ipaddress.ip_address(info[4][0]).is_multicast for info in infos):
        raise DisallowedURLError("Destination is not exclusively public")
    family, socktype, proto, _, sockaddr = infos[0]

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Fetch deadline exceeded")
        return seconds

    class PinnedConnection(http.client.HTTPConnection):
        default_port = 443 if parts.scheme == "https" else 80

        def connect(self):
            self.sock = socket.socket(family, socktype, proto)
            try:
                self.sock.settimeout(remaining())
                self.sock.connect(sockaddr)
                if parts.scheme == "https":
                    # Default context validates the certificate against the ORIGINAL hostname.
                    self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=host)
            except Exception:
                self.sock.close()
                raise

    connection = PinnedConnection(host, port, timeout=remaining())
    try:
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        connection.request("GET", target, headers={"Connection": "close"})
        response = connection.getresponse()
        chunks = []
        size = 0
        while size <= MAX_RESPONSE_BYTES:
            remaining()
            chunk = response.read1(min(8192, MAX_RESPONSE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        remaining()
        body = b"".join(chunks)
        return {"status_code": response.status, "url": url,
                "content": body[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace"),
                "truncated": size > MAX_RESPONSE_BYTES,
                "content_type": response.getheader("Content-Type")}
    finally:
        connection.close()


def _fetch_worker(pipe, url, timeout):
    try:
        pipe.send(("ok", _pinned_fetch(url, timeout)))
    except DisallowedURLError:
        pipe.send(("disallowed", None))
    except Exception:
        pipe.send(("failed", None))
    finally:
        pipe.close()


def _isolated_fetch(url, timeout, _worker=None):
    # A killable process bounds DNS, connect, TLS and slow streaming together.
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker or _fetch_worker, args=(writer, url, timeout), daemon=True)
    try:
        process.start()
        writer.close()
        if not reader.poll(max(0, deadline - time.monotonic())):
            raise TimeoutError("Fetch deadline exceeded")
        status, result = reader.recv()
        if time.monotonic() >= deadline:
            raise TimeoutError("Fetch deadline exceeded")
        if status == "disallowed":
            raise DisallowedURLError("Destination is not exclusively public")
        if status != "ok":
            raise RuntimeError("Outbound fetch failed")
        return result
    finally:
        reader.close()
        writer.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            process.close()


HTTP_GET_SPEC = ToolSpec(
    name="http.get",
    description="Fetches a URL via HTTP GET, restricted to an operator-configured "
                "domain allowlist (TOOL_HTTP_GET_ALLOWED_DOMAINS). The first REAL, "
                "non-sandboxed tool -- makes an actual outbound network call.",
    parameters={
        "type": "object",
        # maxLength=2048: comfortably covers any real URL (common browser/
        # server limits sit around 2000-8000 chars) while bounding the
        # worst case -- a deliberately oversized url doing real work
        # (urlsplit, a DNS lookup, string formatting into log/error
        # messages) before the allowlist check even gets a chance to
        # reject it on hostname grounds (core/tools.py's validate_arguments
        # now enforces JSON-Schema maxLength; this is the one caller of it).
        "properties": {"url": {"type": "string", "maxLength": 2048}},
        "required": ["url"],
    },
    risk_level="MEDIUM",
    capability_required="ELEVATED",
)


def register_real_tools(registry=None) -> None:
    """Explicit opt-in, same convention `core/demo_tools.py::register_
    demo_tools` already uses — importing this module must not silently
    grant every deployment a live network-calling tool."""
    from core.tools import get_tool_registry

    reg = registry if registry is not None else get_tool_registry()
    if HTTP_GET_SPEC.name not in reg:
        reg.register(HTTP_GET_SPEC, handler=_http_get)

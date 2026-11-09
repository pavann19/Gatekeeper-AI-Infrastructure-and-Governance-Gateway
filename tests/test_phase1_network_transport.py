"""Local TCP/TLS and killable-worker tests; no public network is used."""
import socket
import ssl
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from core.real_tools import _isolated_fetch, _pinned_fetch


@pytest.mark.parametrize("trusted,certificate_host", [(False, "approved.example"), (True, "wrong.example"),
                                                      (True, "approved.example")])
def test_actual_tls_checks_chain_and_original_hostname(tmp_path, trusted, certificate_host):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from datetime import datetime, timedelta, timezone

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, certificate_host)])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(datetime.now(timezone.utc) - timedelta(hours=1))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(certificate_host)]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False)
            .add_extension(x509.KeyUsage(True, False, True, False, False, True, True, False, False), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    sni, requests = [], []
    server_context.set_servername_callback(lambda sock, name, context: sni.append(name))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(3)
    port = listener.getsockname()[1]

    def serve():
        try:
            client, _ = listener.accept()
            with client:
                try:
                    with server_context.wrap_socket(client, server_side=True) as tls:
                        requests.append(tls.recv(4096))
                        tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                except ssl.SSLError:
                    pass  # Expected handshake rejection for negative controls.
        finally:
            listener.close()

    original_socket = socket.socket
    connected = []

    class RoutedSocket(original_socket):
        def connect(self, address):
            connected.append(address)
            assert address == ("1.1.1.1", port)
            return super().connect(("127.0.0.1", port))

    context = ssl.create_default_context(cafile=str(cert_path) if trusted else None)
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with patch("core.real_tools.socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))]) as dns, \
             patch("core.real_tools.socket.socket", RoutedSocket), \
             patch("core.real_tools.ssl.create_default_context", return_value=context):
            if trusted and certificate_host == "approved.example":
                assert _pinned_fetch(f"https://approved.example:{port}/", 2)["content"] == "ok"
            else:
                with pytest.raises(ssl.SSLCertVerificationError):
                    _pinned_fetch(f"https://approved.example:{port}/", 2)
                assert requests == []
        dns.assert_called_once()
        assert connected == [("1.1.1.1", port)]
        assert sni == ["approved.example"]
    finally:
        thread.join(timeout=3)
        listener.close()
    assert not thread.is_alive()


def _dns_stall_worker(pipe, marker, timeout):
    from core.real_tools import _fetch_worker

    def stalled_dns(*args, **kwargs):
        Path(marker).write_text("entered")
        time.sleep(30)

    with patch("core.real_tools.socket.getaddrinfo", side_effect=stalled_dns):
        _fetch_worker(pipe, "https://approved.example/", timeout)


def test_total_deadline_covers_actual_dns_stage_and_reaps_worker(tmp_path):
    import multiprocessing
    marker = tmp_path / "dns-entered"
    children = {p.pid for p in multiprocessing.active_children()}
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        _isolated_fetch(str(marker), 3, _worker=_dns_stall_worker)
    assert marker.exists(), "Worker did not reach DNS; timeout is not a DNS-stage proof"
    assert time.monotonic() - start < 4.5
    assert {p.pid for p in multiprocessing.active_children()} == children


def _slow_stream_worker(pipe, url, timeout):
    from core.real_tools import _fetch_worker
    from urllib.parse import urlsplit
    port = urlsplit(url).port
    original_socket = socket.socket

    class RoutedSocket(original_socket):
        def connect(self, address):
            assert address == ("1.1.1.1", port)
            return super().connect(("127.0.0.1", port))

    with patch("core.real_tools.socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))]), \
         patch("core.real_tools.socket.socket", RoutedSocket):
        _fetch_worker(pipe, url, timeout)


def test_total_deadline_stops_real_slow_drip_stream():
    import multiprocessing
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    reached = threading.Event()

    class SlowHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            reached.set()
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            try:
                for _ in range(1000):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    children = {p.pid for p in multiprocessing.active_children()}
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            _isolated_fetch(f"http://approved.example:{server.server_port}/", 3, _worker=_slow_stream_worker)
        assert reached.is_set(), "Worker did not reach stream; this is not a streaming-stage proof"
        assert time.monotonic() - start < 4.5
        assert {p.pid for p in multiprocessing.active_children()} == children
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

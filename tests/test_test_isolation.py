"""The default test boundary must stop accidental external calls before I/O."""
import socket

import pytest


def test_external_dns_is_denied_by_default(request):
    if request.config.getoption('--allow-test-network'):
        pytest.skip('Network guard was explicitly disabled')
    with pytest.raises(OSError, match='External network disabled'):
        socket.getaddrinfo('external-service.invalid', 443)


def test_external_socket_is_denied_even_with_literal_address(request):
    if request.config.getoption('--allow-test-network'):
        pytest.skip('Network guard was explicitly disabled')
    with socket.socket() as client:
        with pytest.raises(OSError, match='External network disabled'):
            client.connect(('198.51.100.27', 443))


def test_loopback_server_is_permitted():
    with socket.socket() as server, socket.socket() as client:
        server.bind(('127.0.0.1', 0))
        server.listen()
        client.settimeout(1)
        client.connect(server.getsockname())
        connection, _ = server.accept()
        connection.close()

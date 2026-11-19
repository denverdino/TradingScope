"""Offline test boundaries; probes always have a synthetic network backstop."""

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpcore
import httpx
import pytest
import requests
from curl_cffi import Curl
from curl_cffi.requests import AsyncSession, Session

URL = "https://offline.invalid/data?api_key=synthetic-secret"
BLOCKED = "Network access is disabled for this test"


def test_requests_is_blocked_before_transport(monkeypatch):
    network = Mock(side_effect=AssertionError("Synthetic network backstop reached"))
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", network)

    with pytest.raises(pytest.fail.Exception, match=BLOCKED) as failure:
        requests.get(URL)

    network.assert_not_called()
    assert "synthetic-secret" not in str(failure.value)
    assert not isinstance(failure.value, Exception)


def test_curl_session_is_blocked_before_transport(monkeypatch):
    network = Mock(side_effect=AssertionError("Synthetic network backstop reached"))
    monkeypatch.setattr(Curl, "perform", network)

    with Session() as session, pytest.raises(pytest.fail.Exception, match=BLOCKED):
        session.get(URL)

    network.assert_not_called()


@pytest.mark.asyncio
async def test_async_curl_session_is_blocked_before_transport():
    network = Mock(side_effect=AssertionError("Synthetic network backstop reached"))
    async_curl = Mock(add_handle=network, close=AsyncMock())

    async with AsyncSession(async_curl=async_curl) as session:
        with pytest.raises(pytest.fail.Exception, match=BLOCKED):
            await session.get(URL)

    network.assert_not_called()


@pytest.mark.parametrize("direct_transport", [False, True])
def test_httpx_real_transport_is_blocked(monkeypatch, direct_transport):
    network = Mock(side_effect=AssertionError("Synthetic network backstop reached"))
    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", network)

    with pytest.raises(pytest.fail.Exception, match=BLOCKED):
        if direct_transport:
            with httpx.HTTPTransport() as transport:
                transport.handle_request(httpx.Request("GET", URL))
        else:
            with httpx.Client(trust_env=False) as client:
                client.get(URL)

    network.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("direct_transport", [False, True])
async def test_async_httpx_real_transport_is_blocked(monkeypatch, direct_transport):
    network = AsyncMock(side_effect=AssertionError("Synthetic network backstop reached"))
    monkeypatch.setattr(httpcore.AsyncConnectionPool, "handle_async_request", network)

    with pytest.raises(pytest.fail.Exception, match=BLOCKED):
        if direct_transport:
            async with httpx.AsyncHTTPTransport() as transport:
                await transport.handle_async_request(httpx.Request("GET", URL))
        else:
            async with httpx.AsyncClient(trust_env=False) as client:
                await client.get(URL)

    network.assert_not_called()


def test_httpx_mock_transport_remains_local():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"offline": True}))
    with httpx.Client(transport=transport) as client:
        assert client.get(URL).json() == {"offline": True}


@pytest.mark.asyncio
async def test_async_httpx_mock_transport_remains_local():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"offline": True}))
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await client.get(URL)).json() == {"offline": True}


# Capture public methods during collection, before the autouse fixture is installed.
# The identity assertion must pass before any socket/DNS probe can be called.
SOCKET_ORIGINALS = {name: getattr(socket.socket, name) for name in ("connect", "connect_ex")}
DNS_PROBES = {
    "create_connection": (("offline.invalid", 443),),
    "getaddrinfo": ("offline.invalid", 443),
    "gethostbyname": ("offline.invalid",),
    "gethostbyname_ex": ("offline.invalid",),
    "gethostbyaddr": ("192.0.2.1",),
    "getnameinfo": (("192.0.2.1", 443), 0),
}
DNS_ORIGINALS = {name: getattr(socket, name) for name in DNS_PROBES}


@pytest.mark.parametrize("method", SOCKET_ORIGINALS)
def test_socket_connection_is_blocked(method):
    assert getattr(socket.socket, method) is not SOCKET_ORIGINALS[method], "No guard installed; do not attempt a connection"
    with socket.socket() as connection, pytest.raises(pytest.fail.Exception, match=BLOCKED):
        getattr(connection, method)(("192.0.2.1", 443))


@pytest.mark.parametrize("method", DNS_PROBES)
def test_connection_helper_and_dns_are_blocked(method):
    boundary = getattr(socket, method)
    assert boundary is not DNS_ORIGINALS[method], "No guard installed; do not attempt DNS or a connection"
    with pytest.raises(pytest.fail.Exception, match=BLOCKED):
        boundary(*DNS_PROBES[method])


def test_asyncio_loop_and_socketpair_remain_local():
    async def local_work():
        loop = asyncio.get_running_loop()
        reader, writer = socket.socketpair()
        try:
            reader.setblocking(False)
            writer.setblocking(False)
            await loop.sock_sendall(writer, b"offline")
            assert await loop.sock_recv(reader, 7) == b"offline"
            assert await asyncio.to_thread(lambda: "local") == "local"
        finally:
            reader.close()
            writer.close()

    asyncio.run(local_work())


def _test_item(*, integration, run_live):
    return SimpleNamespace(
        get_closest_marker=lambda name: pytest.mark.integration if integration else None,
        config=SimpleNamespace(getoption=lambda name: run_live),
    )


def test_run_live_option_is_registered(pytestconfig):
    assert isinstance(pytestconfig.getoption("--run-live"), bool)


@pytest.mark.parametrize(
    ("run_live", "enabled", "credential", "allowed"),
    [
        (False, "1", "synthetic-key", False),
        (True, "0", "synthetic-key", False),
        (True, "true", "synthetic-key", False),
        (True, "1", "", False),
        (True, "1", "synthetic-key", True),
    ],
)
def test_integration_requires_every_live_gate(monkeypatch, run_live, enabled, credential, allowed):
    from tests import conftest

    monkeypatch.setenv("RUN_DASHSCOPE_INTEGRATION", enabled)
    monkeypatch.setenv("DASHSCOPE_API_KEY", credential)
    item = _test_item(integration=True, run_live=run_live)

    if allowed:
        conftest.pytest_runtest_setup(item)
    else:
        with pytest.raises(pytest.skip.Exception, match="--run-live"):
            conftest.pytest_runtest_setup(item)


@pytest.mark.parametrize("run_live", [False, True])
def test_ordinary_tests_stay_blocked_with_keys_and_live_environment(monkeypatch, run_live):
    from tests import conftest

    monkeypatch.setenv("RUN_DASHSCOPE_INTEGRATION", "1")
    for credential in ("DASHSCOPE_API_KEY", "FINHUB_API_KEY", "PERPLEXITY_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.setenv(credential, "synthetic-key")
    item = _test_item(integration=False, run_live=run_live)
    conftest.pytest_runtest_setup(item)
    network = Mock(side_effect=AssertionError("Synthetic network backstop reached"))
    monkeypatch.setattr(requests.sessions.Session, "request", network)
    conftest.offline_network_guard.__wrapped__(SimpleNamespace(node=item), monkeypatch)

    with pytest.raises(pytest.fail.Exception, match=BLOCKED):
        requests.get(URL)

    network.assert_not_called()


def test_opted_in_integration_leaves_synthetic_transport_untouched(monkeypatch):
    from tests import conftest

    monkeypatch.setenv("RUN_DASHSCOPE_INTEGRATION", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "synthetic-key")
    item = _test_item(integration=True, run_live=True)
    network = Mock(return_value="synthetic-response")
    monkeypatch.setattr(requests.sessions.Session, "request", network)
    conftest.offline_network_guard.__wrapped__(SimpleNamespace(node=item), monkeypatch)

    assert requests.get(URL) == "synthetic-response"


def test_denied_request_cannot_reach_vendor_fallback(monkeypatch):
    from tradingscope.dataflows import interface

    network = Mock(side_effect=AssertionError("Synthetic network backstop reached"))
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", network)
    fallback = Mock(return_value="Unexpected paid fallback")
    fallback.__name__ = "synthetic_fallback"

    def denied_vendor(*args, **kwargs):
        return requests.get(URL)

    monkeypatch.setitem(interface.VENDOR_METHODS, "get_news", {"finnhub": denied_vendor, "perplexity": fallback})
    monkeypatch.setattr(interface, "get_config", lambda: {"tool_vendors": {"get_news": "finnhub"}})

    with pytest.raises(pytest.fail.Exception, match=BLOCKED):
        interface.route_to_vendor("get_news", "AAPL", "2025-01-01", "2025-01-02")

    network.assert_not_called()
    fallback.assert_not_called()

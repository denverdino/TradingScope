"""Configuration file for pytest."""

import os
import socket
import sys
from unittest.mock import patch

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MockModel:
    """Mock model for testing."""

    def __init__(self, *args, **kwargs):
        self.stream = False

    def __call__(self, *args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.content = "Mock response"

        return MockResponse()


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Allow explicitly marked integration tests with RUN_DASHSCOPE_INTEGRATION=1 and credentials",
    )


def _live_network_allowed(item):
    # All current integration cases use DashScope credentials.
    return (
        item.get_closest_marker("integration") is not None
        and item.config.getoption("--run-live")
        and os.getenv("RUN_DASHSCOPE_INTEGRATION") == "1"
        and bool(os.getenv("DASHSCOPE_API_KEY"))
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("integration") is not None and not _live_network_allowed(item):
        pytest.skip("Live integration requires --run-live, RUN_DASHSCOPE_INTEGRATION=1, and DASHSCOPE_API_KEY")


def _deny_network(*args, **kwargs):
    # Failed inherits BaseException: vendor fallback handlers must not swallow it.
    pytest.fail("Network access is disabled for this test; mock the external boundary", pytrace=False)


async def _deny_async_network(*args, **kwargs):
    _deny_network()


@pytest.fixture(autouse=True)
def offline_network_guard(request, monkeypatch):
    """Keep ordinary tests offline even when API credentials are available."""
    if _live_network_allowed(request.node):
        return

    import httpx
    import requests
    from curl_cffi.requests import AsyncSession, Session

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _deny_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _deny_async_network)
    monkeypatch.setattr(requests.sessions.Session, "request", _deny_network)
    monkeypatch.setattr(Session, "request", _deny_network)
    monkeypatch.setattr(AsyncSession, "request", _deny_async_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny_network)
    for method in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        monkeypatch.setattr(socket, method, _deny_network)
    # Leave socketpair untouched: asyncio uses it for its local wake-up pipe.


@pytest.fixture
def mock_model():
    """Fixture that provides a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_context():
    """Fixture that provides a mock AgentContext without requiring API keys."""
    from tradingscope.agents.utils.context import AgentContext

    with (
        patch("tradingscope.agents.utils.context.get_last_completed_us_session", return_value="2025-01-01"),
        patch("tradingscope.agents.utils.context.OpenAICredential"),
        patch(
            "tradingscope.agents.utils.context.DashScopeResponseModel",
            return_value=MockModel(),
        ),
    ):
        context = AgentContext()
    return context


@pytest.fixture
def mock_agent():
    """Fixture that provides a mock agent for testing."""

    class MockAgent:
        def __init__(self, name="MockAgent"):
            self.name = name

        async def __call__(self, *args, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.content = f"Mock response from {self.name}"

            return MockResponse()

    return MockAgent()


@pytest.fixture
def sample_company_data():
    """Fixture that provides sample company data for testing."""
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "investment_plan": "Test investment plan",
        "market_research_report": "Market research report",
        "sentiment_report": "Sentiment report",
        "news_report": "News report",
        "fundamentals_report": "Fundamentals report",
        "analysis_date": "2025-01-01",
    }

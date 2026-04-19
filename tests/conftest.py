"""Configuration file for pytest."""

import os
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


@pytest.fixture
def mock_model():
    """Fixture that provides a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_context():
    """Fixture that provides a mock AgentContext without requiring API keys."""
    from tradingscope.agents.utils.context import AgentContext

    with patch("tradingscope.agents.utils.context.OpenAIChatModel", return_value=MockModel()):
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
        "company_of_interest": "AAPL",
        "company_name": "Apple Inc.",
        "investment_plan": "Test investment plan",
        "market_research_report": "Market research report",
        "sentiment_report": "Sentiment report",
        "news_report": "News report",
        "fundamentals_report": "Fundamentals report",
        "trade_date": "2025-01-01",
    }

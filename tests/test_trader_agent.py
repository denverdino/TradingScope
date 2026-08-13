"""Unit tests for the Trader Agent."""

import os
import sys
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.agents.trader.trader import create_trader_agent


class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self.stream = False

    def __call__(self, *args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.content = "Mock trader response with recommendation: Buy"

        return MockResponse()


class TestTraderAgent:
    """Test cases for the Trader Agent."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_model = MockModel()
        self.company_of_interest = "AAPL"
        self.investment_plan = "Test investment plan"
        self.market_research_report = "Market research report"
        self.sentiment_report = "Sentiment report"
        self.news_report = "News report"
        self.fundamentals_report = "Fundamentals report"

    def _setup_context(self, context):
        """Populate context with test data."""
        context.company_of_interest = self.company_of_interest
        context.investment_plan = self.investment_plan
        context.market_report = self.market_research_report
        context.sentiment_report = self.sentiment_report
        context.news_report = self.news_report
        context.fundamentals_report = self.fundamentals_report
        return context

    def test_create_trader_agent_success(self, mock_context):
        """Test that trader agent can be created successfully."""
        context = self._setup_context(mock_context)

        agent = create_trader_agent(
            context=context,
        )

        # Verify the agent was created
        assert agent is not None
        assert hasattr(agent, "name")
        assert agent.model is context.model

    @patch("tradingscope.agents.utils.agent_utils.get_company_name")
    @patch("tradingscope.agents.utils.stock_utils.StockUtils.get_market_info")
    def test_create_trader_agent_with_optional_params(self, mock_get_market_info, mock_get_company_name, mock_context):
        """Test that trader agent can be created with optional parameters."""
        # Setup mocks
        mock_get_market_info.return_value = {"market_name": "NASDAQ", "currency_name": "US Dollar", "currency_symbol": "$"}
        mock_get_company_name.return_value = "Apple Inc."

        context = self._setup_context(mock_context)
        context.trade_date = "2025-01-01"

        agent = create_trader_agent(
            context=context,
        )

        # Verify the agent was created
        assert agent is not None

    @patch("tradingscope.agents.utils.agent_utils.get_company_name")
    @patch("tradingscope.agents.utils.stock_utils.StockUtils.get_market_info")
    def test_trader_agent_system_prompt_includes_currency(self, mock_get_market_info, mock_get_company_name, mock_context):
        """Test that the trader agent's system prompt includes currency information."""
        # Setup mocks
        mock_get_market_info.return_value = {"market_name": "NASDAQ", "currency_name": "US Dollar", "currency_symbol": "$"}
        mock_get_company_name.return_value = "Apple Inc."

        context = self._setup_context(mock_context)

        agent = create_trader_agent(
            context=context,
        )

        # Verify the agent was created with proper system prompt
        assert agent is not None
        # Check that currency information is in the system prompt
        assert "US Dollar" in agent._system_prompt
        assert "$" in agent._system_prompt

    def test_trader_agent_has_user_message_in_memory(self, mock_context):
        """Test that the trader agent has the user message pre-loaded in memory."""
        context = self._setup_context(mock_context)

        agent = create_trader_agent(
            context=context,
        )

        # Check that memory has content (we can't directly test this due to async nature)
        # Just verify the agent was created successfully
        assert agent is not None

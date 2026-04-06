"""Unit tests for the Portfolio Manager Agent."""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent


class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self.stream = False

    def __call__(self, *args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.content = "Mock portfolio manager response with recommendation: Hold"

        return MockResponse()


class TestPortfolioManagerAgent:
    """Test cases for the Portfolio Manager Agent."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_model = MockModel()
        self.agent_name = "TestPortfolioManager"

    def test_create_portfolio_manager_agent_success(self):
        """Test that portfolio manager agent can be created successfully."""
        # Create AgentContext
        from tradingscope.agents.utils.context import AgentContext
        context = AgentContext()
        context.company_of_interest = "AAPL"
        context.market_report = "Market data"
        context.sentiment_report = "Sentiment data"
        context.news_report = "News data"
        context.fundamentals_report = "Fundamentals data"
        context.trader_investment_plan = "Trader plan"

        agent = create_portfolio_manager_agent(
            model=self.mock_model,
            context=context,
            name=self.agent_name,
        )

        # Verify the agent was created
        assert agent is not None
        assert hasattr(agent, "name")
        assert agent.name == self.agent_name
        assert hasattr(agent, "memory")

    def test_portfolio_manager_agent_system_prompt_content(self):
        """Test that the portfolio manager agent has the correct system prompt content."""
        # Create AgentContext
        from tradingscope.agents.utils.context import AgentContext
        context = AgentContext()
        context.company_of_interest = "AAPL"
        context.market_report = "Market data"
        context.sentiment_report = "Sentiment data"
        context.news_report = "News data"
        context.fundamentals_report = "Fundamentals data"
        context.trader_investment_plan = "Trader plan"

        agent = create_portfolio_manager_agent(
            model=self.mock_model,
            context=context,
            name=self.agent_name,
        )

        # Verify the agent was created with proper system prompt
        assert agent is not None
        # Check that key elements are in the system prompt
        assert "风险管理委员会主席" in agent.sys_prompt
        assert "买入、卖出或持有" in agent.sys_prompt
        assert "辩论" in agent.sys_prompt

    def test_portfolio_manager_agent_configuration(self):
        """Test that the portfolio manager agent has the correct configuration."""
        # Create AgentContext
        from tradingscope.agents.utils.context import AgentContext
        context = AgentContext()
        context.company_of_interest = "AAPL"
        context.market_report = "Market data"
        context.sentiment_report = "Sentiment data"
        context.news_report = "News data"
        context.fundamentals_report = "Fundamentals data"
        context.trader_investment_plan = "Trader plan"

        agent = create_portfolio_manager_agent(
            model=self.mock_model,
            context=context,
            name=self.agent_name,
        )

        # Verify the agent configuration
        assert agent is not None
        assert hasattr(agent, "max_iters")
        assert agent.max_iters == 8  # As defined in the implementation
        assert hasattr(agent, "parallel_tool_calls")
        assert agent.parallel_tool_calls is False  # As defined in the implementation

"""Test script to verify that the PortfolioManagerAgent works with ReAct pattern."""

from agentscope.agent._config import ReActConfig


def test_react_agent_creation(mock_model, mock_context):
    """Test that PortfolioManagerAgent can be created with ReAct pattern."""
    from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent

    context = mock_context
    context.company_of_interest = "AAPL"
    context.market_report = "Market data"
    context.sentiment_report = "Sentiment data"
    context.news_report = "News data"
    context.fundamentals_report = "Fundamentals data"
    context.trader_investment_plan = "Trader plan"

    portfolio_manager = create_portfolio_manager_agent(context, "TestPortfolioManager")

    assert portfolio_manager is not None
    assert portfolio_manager.name == "TestPortfolioManager"
    assert isinstance(portfolio_manager.react_config, ReActConfig)

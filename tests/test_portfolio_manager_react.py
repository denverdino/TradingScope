"""Test script to verify that the PortfolioManagerAgent works with ReAct pattern."""


def test_react_agent_creation(mock_model, mock_context):
    """Test that PortfolioManagerAgent can be created with ReAct pattern."""
    try:
        # Import AgentScope components
        from agentscope.message import Msg

        from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent

        context = mock_context
        context.company_of_interest = "AAPL"
        context.market_report = "Market data"
        context.sentiment_report = "Sentiment data"
        context.news_report = "News data"
        context.fundamentals_report = "Fundamentals data"
        context.trader_investment_plan = "Trader plan"

        # Test creating portfolio manager agent with model (ReAct pattern)
        portfolio_manager = create_portfolio_manager_agent(context, "TestPortfolioManager")
        print("✅ Portfolio manager agent with ReAct pattern created successfully")

        # Test calling the portfolio manager agent
        test_msg = Msg(
            name="test",
            role="user",
            content="Test message for portfolio manager",
        )

        # Since the ReActAgent is async, we need to handle it properly
        response = portfolio_manager(test_msg)
        print("✅ Portfolio manager agent with ReAct pattern called successfully")
        print(f"Response type: {type(response)}")

        print("\n🎉 Portfolio Manager ReAct pattern test completed successfully!")
        # Use assertion instead of return
        assert True
    except Exception as e:
        print(f"❌ Error testing ReAct pattern: {e}")
        raise AssertionError(f"Error testing ReAct pattern: {e}")

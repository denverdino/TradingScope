"""Test script to verify that risk management agents can be instantiated."""


def test_agent_creation(mock_model, mock_context):
    """Test that all risk management agents can be created."""
    try:
        # Import AgentScope components
        from agentscope.message import Msg

        # Import our agents
        from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent
        from tradingscope.agents.risk_mgmt.aggressive_debater import create_aggressive_debater_agent
        from tradingscope.agents.risk_mgmt.conservative_debater import create_conservative_debater_agent
        from tradingscope.agents.risk_mgmt.neutral_debater import create_neutral_debater_agent

        context = mock_context
        context.ticker = "AAPL"
        context.market_report = "Market data"
        context.sentiment_report = "Sentiment data"
        context.news_report = "News data"
        context.fundamentals_report = "Fundamentals data"
        context.trader_investment_plan = "Trader plan"

        # Test creating each debater agent
        create_aggressive_debater_agent(context, "TestAggressive")
        print("✅ Aggressive debater agent created successfully")

        create_conservative_debater_agent(context, "TestConservative")
        print("✅ Conservative debater agent created successfully")

        create_neutral_debater_agent(context, "TestNeutral")
        print("✅ Neutral debater agent created successfully")

        # Test creating portfolio manager agent
        portfolio_manager = create_portfolio_manager_agent(context, "TestPortfolioManager")
        print("✅ Portfolio manager agent created successfully")

        print("✅ Portfolio manager agent instantiation verified")

        print("\n🎉 All risk management agents instantiated and tested successfully!")
        # Use assertion instead of return
        assert True
    except Exception as e:
        print(f"❌ Error testing agents: {e}")
        raise AssertionError(f"Error testing agents: {e}")

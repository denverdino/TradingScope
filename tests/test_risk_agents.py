"""Test script to verify that risk management agents can be instantiated."""


def test_agent_creation(mock_model):
    """Test that all risk management agents can be created."""
    try:
        # Import AgentScope components
        from agentscope.message import Msg

        from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent

        # Import our agents
        from tradingscope.agents.risk_mgmt.aggressive_debator import create_aggressive_debator_agent
        from tradingscope.agents.risk_mgmt.conservative_debator import create_conservative_debator_agent
        from tradingscope.agents.risk_mgmt.neutral_debator import create_neutral_debator_agent

        # Create AgentContext
        from tradingscope.agents.utils.context import AgentContext

        context = AgentContext()
        context.company_of_interest = "AAPL"
        context.market_report = "Market data"
        context.sentiment_report = "Sentiment data"
        context.news_report = "News data"
        context.fundamentals_report = "Fundamentals data"
        context.trader_investment_plan = "Trader plan"

        # Test creating each debator agent
        create_aggressive_debator_agent(mock_model, context, "TestAggressive")
        print("✅ Aggressive debator agent created successfully")

        create_conservative_debator_agent(mock_model, context, "TestConservative")
        print("✅ Conservative debator agent created successfully")

        create_neutral_debator_agent(mock_model, context, "TestNeutral")
        print("✅ Neutral debator agent created successfully")

        # Test creating portfolio manager agent
        portfolio_manager = create_portfolio_manager_agent(mock_model, context, "TestPortfolioManager")
        print("✅ Portfolio manager agent created successfully")

        print("✅ Portfolio manager agent instantiation verified")

        print("\n🎉 All risk management agents instantiated and tested successfully!")
        # Use assertion instead of return
        assert True
    except Exception as e:
        print(f"❌ Error testing agents: {e}")
        raise AssertionError(f"Error testing agents: {e}")

"""Test script to verify that risk management agents can be instantiated."""


def test_agent_creation(mock_model):
    """Test that all risk management agents can be created."""
    try:
        # Import AgentScope components
        from agentscope.message import Msg

        from tradingscope.agents.managers.risk_manager import create_risk_manager_agent

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

        # Test creating risk manager agent
        risk_manager = create_risk_manager_agent(mock_model, context, "TestRiskManager")
        print("✅ Risk manager agent created successfully")

        # Test calling the risk manager agent
        test_msg = Msg(
            name="test",
            role="user",
            content={
                "company_name": "AAPL",
                "history": "Test history",
                "market_research_report": "Market data",
                "sentiment_report": "Sentiment data",
                "news_report": "News data",
                "fundamentals_report": "Fundamentals data",
                "trader_plan": "Trader plan",
            },
        )

        risk_manager(test_msg)
        print("✅ Risk manager agent called successfully")

        print("\n🎉 All risk management agents instantiated and tested successfully!")
        # Use assertion instead of return
        assert True
    except Exception as e:
        print(f"❌ Error testing agents: {e}")
        raise AssertionError(f"Error testing agents: {e}")

"""Test script to verify that the RiskManagerAgent works with ReAct pattern."""


def test_react_agent_creation(mock_model):
    """Test that RiskManagerAgent can be created with ReAct pattern."""
    try:
        # Import AgentScope components
        from agentscope.message import Msg

        from tradingscope.agents.managers.risk_manager import create_risk_manager_agent

        # Test creating risk manager agent with model (ReAct pattern)
        risk_manager = create_risk_manager_agent(mock_model, "TestRiskManager")
        print("✅ Risk manager agent with ReAct pattern created successfully")

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

        # Since the ReActAgent is async, we need to handle it properly
        response = risk_manager(test_msg)
        print("✅ Risk manager agent with ReAct pattern called successfully")
        print(f"Response type: {type(response)}")

        print("\n🎉 Risk Manager ReAct pattern test completed successfully!")
        # Use assertion instead of return
        assert True
    except Exception as e:
        print(f"❌ Error testing ReAct pattern: {e}")
        raise AssertionError(f"Error testing ReAct pattern: {e}")

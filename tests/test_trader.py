"""Test script to verify that the trader agent can be instantiated."""

from unittest.mock import patch


def test_trader_agent_creation(mock_model, sample_company_data):
    """Test that the trader agent can be created."""
    try:
        # Import our agent
        from tradingscope.agents.trader.trader import create_trader_agent

        # Create AgentContext
        from tradingscope.agents.utils.context import AgentContext
        context = AgentContext()
        context.company_of_interest = sample_company_data["company_of_interest"]
        context.investment_plan = sample_company_data["investment_plan"]
        context.market_report = sample_company_data["market_research_report"]
        context.sentiment_report = sample_company_data["sentiment_report"]
        context.news_report = sample_company_data["news_report"]
        context.fundamentals_report = sample_company_data["fundamentals_report"]

        # Test creating the trader agent
        agent = create_trader_agent(
            model=mock_model,
            context=context,
        )

        assert agent is not None
        print("✅ Trader agent created successfully")
        print("\n🎉 Trader agent instantiated successfully!")
        # Use assertion instead of return
        assert True

    except Exception as e:
        print(f"❌ Error testing trader agent: {e}")
        raise AssertionError(f"Error testing trader agent: {e}")


@patch("tradingscope.agents.utils.agent_utils.get_company_name")
@patch("tradingscope.agents.utils.stock_utils.StockUtils.get_market_info")
def test_trader_agent_with_china_stock(mock_get_market_info, mock_get_company_name, mock_model, sample_company_data):
    """Test that the trader agent works with China stocks."""
    # Setup mocks for China stock
    mock_get_market_info.return_value = {"market_name": "Shanghai Stock Exchange", "currency_name": "Chinese Yuan", "currency_symbol": "¥"}
    mock_get_company_name.return_value = "Ping An Insurance"

    try:
        # Import our agent
        from tradingscope.agents.trader.trader import create_trader_agent

        # Create AgentContext
        from tradingscope.agents.utils.context import AgentContext
        context = AgentContext()
        context.company_of_interest = "600000"  # China stock code
        context.investment_plan = sample_company_data["investment_plan"]
        context.market_report = sample_company_data["market_research_report"]
        context.sentiment_report = sample_company_data["sentiment_report"]
        context.news_report = sample_company_data["news_report"]
        context.fundamentals_report = sample_company_data["fundamentals_report"]

        # Test creating the trader agent with China stock
        agent = create_trader_agent(
            model=mock_model,
            context=context,
        )

        assert agent is not None
        # Check that currency information is in the system prompt
        assert "Chinese Yuan" in agent.sys_prompt
        assert "¥" in agent.sys_prompt
        print("✅ Trader agent with China stock created successfully")
        # Use assertion instead of return
        assert True

    except Exception as e:
        print(f"❌ Error testing trader agent with China stock: {e}")
        raise AssertionError(f"Error testing trader agent with China stock: {e}")

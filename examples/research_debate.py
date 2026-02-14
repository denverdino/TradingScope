#!/usr/bin/env python3
"""Example script demonstrating the research multi-agent debate system with long-term memory."""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentscope.model import OpenAIChatModel

from tradingscope.agents.managers.research_manager import create_research_manager_agent
from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent
from tradingscope.agents.researchers.bull_researcher import create_bull_researcher_agent
from tradingscope.agents.researchers.debate_orchestrator import create_research_debate_orchestrator
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager


async def main():
    """Example usage of the research debate system with long-term memory."""

    # Initialize the model
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    # Create sample data for testing
    company_of_interest = "AAPL"
    market_research_report = "Technical indicators show a bullish trend with strong support at $180."
    sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    news_report = "Recent news indicates strong quarterly earnings and product launches."
    fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."
    trade_date = "2025-10-05"

    # Create AgentContext
    context = AgentContext()
    context.company_of_interest = company_of_interest
    context.market_report = market_research_report
    context.sentiment_report = sentiment_report
    context.news_report = news_report
    context.fundamentals_report = fundamentals_report
    context.trade_date = trade_date

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the research agents with long-term memory
        bull_researcher = create_bull_researcher_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.bull_researcher_memory,
            long_term_memory_mode="static_control",
        )

        bear_researcher = create_bear_researcher_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.bear_researcher_memory,
            long_term_memory_mode="static_control",
        )

        research_manager = create_research_manager_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.research_manager_memory,
            long_term_memory_mode="static_control",
        )

        # Create the debate orchestrator
        orchestrator = create_research_debate_orchestrator(
            bull_researcher=bull_researcher, bear_researcher=bear_researcher, research_manager=research_manager, max_rounds=3
        )

        print(f"Starting research debate for {company_of_interest}")
        print("=" * 60)

        # Run the debate
        final_decision = await orchestrator.run_debate(
            company_name=company_of_interest,
        )

        print("=" * 60)
        print("Final Decision from Research Manager:")
        print(final_decision.content)
        print("=" * 60)

    except Exception as e:
        print(f"Error during debate: {e}")
        print("Please ensure you have a valid DASHSCOPE_API_KEY set.")
    finally:
        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(main())

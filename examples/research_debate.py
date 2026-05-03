#!/usr/bin/env python3
"""Example script demonstrating the research multi-agent debate system with long-term memory."""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.agents.managers.research_manager import create_research_manager_agent
from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent
from tradingscope.agents.researchers.bull_researcher import create_bull_researcher_agent
from tradingscope.agents.researchers.debate_orchestrator import create_research_debate_orchestrator
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager


async def main():
    """Example usage of the research debate system with long-term memory."""

    # Create AgentContext
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = "Technical indicators show a bullish trend with strong support at $180."
    context.sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    context.news_report = "Recent news indicates strong quarterly earnings and product launches."
    context.fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the research agents with long-term memory
        bull_researcher = create_bull_researcher_agent(
            context=context,
            long_term_memory=memory_manager.get_readonly_memory(),
            long_term_memory_mode="static_control",
        )

        bear_researcher = create_bear_researcher_agent(
            context=context,
            long_term_memory=memory_manager.get_readonly_memory(),
            long_term_memory_mode="static_control",
        )

        research_manager = create_research_manager_agent(
            context=context,
            long_term_memory=memory_manager.get_readonly_memory(),
            long_term_memory_mode="static_control",
        )

        # Create the debate orchestrator
        orchestrator = create_research_debate_orchestrator(
            bull_researcher=bull_researcher, bear_researcher=bear_researcher, research_manager=research_manager, max_rounds=3
        )

        print(f"Starting research debate for {context.company_of_interest}")
        print("=" * 60)

        # Run the debate
        final_decision = await orchestrator.run_debate(
            company_name=context.company_of_interest,
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

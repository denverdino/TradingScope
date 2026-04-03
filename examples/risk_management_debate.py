"""Example script demonstrating the risk management multi-agent debate system with long-term memory."""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentscope.model import OpenAIChatModel

from tradingscope.agents.managers.risk_manager import create_risk_manager_agent
from tradingscope.agents.risk_mgmt.aggressive_debator import create_aggressive_debator_agent
from tradingscope.agents.risk_mgmt.conservative_debator import create_conservative_debator_agent
from tradingscope.agents.risk_mgmt.debate_orchestrator import create_debate_orchestrator
from tradingscope.agents.risk_mgmt.neutral_debator import create_neutral_debator_agent
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager
from tradingscope.default_config import DEFAULT_CONFIG


async def main():
    """Example usage of the risk management debate system with long-term memory."""

    # Initialize the model
    model = OpenAIChatModel(
        model_name=DEFAULT_CONFIG["deep_think_llm"],
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
        generate_kwargs={"temperature": 0}
    )

    # Example data for a stock analysis
    company_name = "TSLA"
    trader_plan = "计划买入TSLA股票200股，目标价位250美元，止损价位200美元"

    # Example reports
    market_research_report = """特斯拉(TSLA)股价近期表现强劲，电动汽车市场需求持续增长。
技术分析显示股价处于上升通道中，相对强弱指数(RSI)处于健康水平。
宏观经济环境有利于新能源汽车发展，政策支持力度不断加大。"""

    sentiment_report = """社交媒体对特斯拉持积极态度，主要关注点包括：
1. 新车型发布预期强烈
2. 自动驾驶技术进展受到关注
3. 马斯克的言论对股价有显著影响
负面情绪主要来自对产能和交付量的担忧。"""

    news_report = """最新世界 affairs 报告：
1. 全球电动汽车补贴政策变化
2. 传统汽车制造商加大电动车投入
3. 电池原材料价格波动
4. 自动驾驶法规更新进展"""

    fundamentals_report = """特斯拉公司基本面分析：
- 营收持续增长，但增速有所放缓
- 利润率保持在较高水平
- 现金流状况良好
- 研发投入占比稳定
- 市场份额在电动车领域领先"""

    # Create AgentContext
    context = AgentContext()
    context.company_of_interest = company_name
    context.market_report = market_research_report
    context.sentiment_report = sentiment_report
    context.news_report = news_report
    context.fundamentals_report = fundamentals_report
    context.trader_investment_plan = trader_plan

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the risk management agents
        # Debators don't use long-term memory, only the risk manager does
        aggressive_agent = create_aggressive_debator_agent(
            model=model,
            context=context
        )
        conservative_agent = create_conservative_debator_agent(
            model=model,
            context=context
        )
        neutral_agent = create_neutral_debator_agent(
            model=model,
            context=context
        )

        # Risk manager with long-term memory
        risk_manager = create_risk_manager_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.risk_manager_memory,
            long_term_memory_mode="static_control",
        )

        # Create the debate orchestrator
        orchestrator = create_debate_orchestrator(
            aggressive_agent=aggressive_agent, conservative_agent=conservative_agent, neutral_agent=neutral_agent, risk_manager=risk_manager, max_rounds=3
        )

        print(f"Starting risk management debate for {company_name}")
        print(f"Trader plan: {trader_plan}")
        print("=" * 60)

        # Run the debate
        final_decision = await orchestrator.run_debate(
            company_name=company_name,
        )

        print("=" * 60)
        print("Final Decision from Risk Manager:")
        print(final_decision.content)
        print("=" * 60)

    except Exception as e:
        print(f"Error during debate: {e}")
        print("Please ensure you have a valid DASHSCOPE_API_KEY set.")
    finally:
        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(main())

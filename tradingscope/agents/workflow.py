import asyncio

#from agentscope.embedding import EmbeddingModelBase
#from agentscope.memory import Mem0LongTermMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from tradingscope.agents.managers.risk_manager import create_risk_manager_agent

from .analysts.fundamentals_analyst import create_fundamentals_analyst_agent
from .analysts.market_analyst import create_market_analyst_agent
from .analysts.news_analyst import create_news_analyst_agent
from .analysts.social_media_analyst import create_social_media_analyst_agent
from .managers.research_manager import create_research_manager_agent
from .researchers.bear_researcher import create_bear_researcher_agent
from .researchers.bull_researcher import create_bull_researcher_agent
from .researchers.debate_orchestrator import create_research_debate_orchestrator
from .risk_mgmt.aggressive_debator import create_aggressive_debator_agent
from .risk_mgmt.conservative_debator import create_conservative_debator_agent

# Risk management imports
from .risk_mgmt.debate_orchestrator import create_debate_orchestrator
from .risk_mgmt.neutral_debator import create_neutral_debator_agent
from .trader.trader import create_trader_agent

# def create_long_term_memory(model: OpenAIChatModel, embedding_model: EmbeddingModelBase, agent_name: str) -> Mem0LongTermMemory:
#     """创建长期记忆实例。"""

#     return Mem0LongTermMemory(
#         agent_name=agent_name,
#         user_name="long_term_memory",
#         model=model,
#         embedding_model=embedding_model,
#         on_disk=False,
#     )

async def analyze(model: OpenAIChatModel, ticker: str, trade_date: str) -> None:
    """运行并发智能体并执行多轮辩论。"""
    # 创建分析师代理
    #china_market_analyst = create_china_market_analyst_agent(model=model)
    fundamentals_analyst = create_fundamentals_analyst_agent(model=model, ticker=ticker, current_date=trade_date)
    market_analyst = create_market_analyst_agent(model=model, ticker=ticker, trade_date=trade_date)
    news_analyst = create_news_analyst_agent(model=model, ticker=ticker, trade_date=trade_date)
    social_media_analyst = create_social_media_analyst_agent(model=model, ticker=ticker, trade_date=trade_date)

    # 创建消息
    user_message = Msg(name="user", role="user", content=f"请分析股票 {ticker}，交易日期为 {trade_date}。")

    # 并发运行分析师代理并获取结果
    analyst_results = await asyncio.gather(
        market_analyst(None),
        fundamentals_analyst(None),
        news_analyst(None),
        social_media_analyst(None),
        #china_market_analyst(None),
        return_exceptions=True,
    )

    # 提取分析师报告内容
    market_research_report = getattr(analyst_results[0], "content", "") if not isinstance(analyst_results[0], Exception) else ""
    fundamentals_report = getattr(analyst_results[1], "content", "") if not isinstance(analyst_results[1], Exception) else ""
    news_report = getattr(analyst_results[2], "content", "") if not isinstance(analyst_results[2], Exception) else ""
    sentiment_report = getattr(analyst_results[3], "content", "") if not isinstance(analyst_results[3], Exception) else ""
    #china_market_report = getattr(analyst_results[4], "content", "") if not isinstance(analyst_results[4], Exception) else ""


    # 创建研究员代理
    bear_researcher = create_bear_researcher_agent(
        model=model,
        company_of_interest=ticker,
        market_research_report=market_research_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        trade_date=trade_date,
    )

    bull_researcher = create_bull_researcher_agent(
        model=model,
        company_of_interest=ticker,
        market_research_report=market_research_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        trade_date=trade_date,
    )

    # 创建研究经理代理
    research_manager = create_research_manager_agent(model=model)

    # 创建研究辩论协调器
    research_orchestrator = create_research_debate_orchestrator(
        bull_researcher=bull_researcher, bear_researcher=bear_researcher, research_manager=research_manager, max_rounds=1
    )

    # 运行研究辩论
    manager_response = await research_orchestrator.run_debate(
        company_name=ticker,
    )

    # Extract manager content
    investment_plan = getattr(manager_response, "content", "") if not isinstance(manager_response, Exception) else ""
    print(f"投资决策:\n{investment_plan}")

    # 交易员基于研究经理的决策做出最终交易决策
    print("\n=== 交易员最终决策 ===")
    # 创建交易员代理
    trader = create_trader_agent(
        model=model,
        company_of_interest=ticker,
        investment_plan=investment_plan,  # 研究经理的投资计划
        market_research_report=market_research_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        trade_date=trade_date,
    )

    # 交易员做出最终决策
    trader_prompt = Msg(
        name="主持人",
        role="user",
        content=f"作为交易员，请基于研究经理的投资决策和所有研究报告，做出最终的交易决策：\n\n研究经理决策：\n{investment_plan}\n\n请给出明确的买入、卖出或持有建议，并提供具体的目标价位和风险评估。",
    )

    trader_response = await trader(trader_prompt)
    trader_plan = getattr(trader_response, "content", "") if not isinstance(trader_response, Exception) else ""
    print(f"交易决策:\n{trader_plan}")

    # 风险管理团队对交易员决策进行辩论和评估
    print("\n=== 风险管理团队辩论 ===")

    aggressive_agent = create_aggressive_debator_agent(model)
    conservative_agent = create_conservative_debator_agent(model)
    neutral_agent = create_neutral_debator_agent(model)
    risk_manager = create_risk_manager_agent(model)

    # 创建风险辩论协调器
    risk_orchestrator = create_debate_orchestrator(aggressive_agent, conservative_agent, neutral_agent, risk_manager, max_rounds=1)

    # 运行风险辩论
    risk_decision = await risk_orchestrator.run_debate(
        company_name=ticker,
        trader_plan=trader_plan,
    )

    # Extract manager content
    risk_content = getattr(risk_decision, "content", "") if not isinstance(risk_decision, Exception) else ""
    print(f"风险管理决策:\n{risk_content}")

import asyncio
import json
from dataclasses import dataclass

from agentscope import logger
from agentscope.message import Msg

from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent
from tradingscope.agents.output import (
    AnalysisResult,
)
from tradingscope.utils.oss_report_uploader import upload_reports
from tradingscope.utils.oss_structured_output_uploader import upload_structured_outputs

# Analyst imports
from .analysts.fundamentals_analyst import create_fundamentals_analyst_agent
from .analysts.market_analyst import create_market_analyst_agent
from .analysts.news_analyst import create_news_analyst_agent
from .analysts.social_media_analyst import create_social_media_analyst_agent

# Researcher imports
from .managers.research_manager import create_research_manager_agent
from .researchers.bear_researcher import create_bear_researcher_agent
from .researchers.bull_researcher import create_bull_researcher_agent
from .researchers.debate_orchestrator import create_research_debate_orchestrator
from .risk_mgmt.aggressive_debator import create_aggressive_debator_agent
from .risk_mgmt.conservative_debator import create_conservative_debator_agent

# Risk management imports
from .risk_mgmt.debate_orchestrator import create_debate_orchestrator
from .risk_mgmt.neutral_debator import create_neutral_debator_agent

# Trader imports
from .trader.trader import create_trader_agent
from .utils.agent_utils import call_agent_with_retry

# Import AgentContext
from .utils.context import AgentContext


def get_content(result: Msg | Exception) -> str:
    """从消息或异常中提取内容。"""
    if isinstance(result, Msg):
        return result.get_text_content()
    return str(result)


def get_structured_output(result: Msg | Exception) -> dict | None:
    """从 AgentScope 响应中提取结构化输出。

    AgentScope ReActAgent 将结构化输出直接存储在 Msg.metadata 中
    （而非嵌套在 metadata["structured_output"] 下），因此需要检查
    metadata 是否包含结构化输出的特征字段（direction/action）。
    """
    if not isinstance(result, Msg):
        return None
    if not result.metadata:
        return None
    # AgentScope stores structured output directly in metadata (flat dict)
    # with keys like "direction", "action", "confidence"
    if isinstance(result.metadata, dict) and ("direction" in result.metadata or "action" in result.metadata):
        return result.metadata
    # Also support nested "structured_output" key for backward compatibility
    return result.metadata.get("structured_output")


@dataclass
class AnalysisOutput:
    """Combined output from the analysis workflow.

    Contains both the Markdown report (for human consumption)
    and the structured AnalysisResult (for downstream systems),
    plus individual agent structured outputs for independent persistence.
    """

    report_md: str
    structured: AnalysisResult
    individual_structured: dict | None = None


async def analyze(ticker: str, trade_date: str | None = None) -> AnalysisOutput:
    """运行并发智能体并执行多轮辩论，返回综合报告。"""
    context = AgentContext()
    context.company_of_interest = ticker
    if trade_date:
        context.trade_date = trade_date

    # 创建分析师代理
    fundamentals_analyst = create_fundamentals_analyst_agent(context=context)
    market_analyst = create_market_analyst_agent(context=context)
    news_analyst = create_news_analyst_agent(context=context)
    social_media_analyst = create_social_media_analyst_agent(context=context)

    # 并发运行分析师代理并获取结果
    analyst_results = await asyncio.gather(
        call_agent_with_retry(market_analyst, None),
        call_agent_with_retry(fundamentals_analyst, None),
        call_agent_with_retry(news_analyst, None),
        call_agent_with_retry(social_media_analyst, None),
        return_exceptions=True,
    )

    # 提取分析师报告内容
    market_research_report = get_content(analyst_results[0])
    fundamentals_report = get_content(analyst_results[1])
    news_report = get_content(analyst_results[2])
    sentiment_report = get_content(analyst_results[3])

    # 提取分析师结构化输出
    market_structured = get_structured_output(analyst_results[0])
    fundamentals_structured = get_structured_output(analyst_results[1])
    news_structured = get_structured_output(analyst_results[2])
    social_media_structured = get_structured_output(analyst_results[3])

    # 更新context中的报告内容
    context.market_report = market_research_report
    context.fundamentals_report = fundamentals_report
    context.news_report = news_report
    context.sentiment_report = sentiment_report

    logger.info("分析师报告已生成，开始创建决策Agent")

    # Upload analyst reports to OSS
    await upload_reports(
        context.trade_date,
        ticker,
        {
            "market_analyst": market_research_report,
            "fundamentals_analyst": fundamentals_report,
            "news_analyst": news_report,
            "social_media_analyst": sentiment_report,
        },
    )

    # Upload analyst structured outputs to OSS
    analyst_structured_outputs = {}
    for name, data in [
        ("market_analyst", market_structured),
        ("fundamentals_analyst", fundamentals_structured),
        ("news_analyst", news_structured),
        ("social_media_analyst", social_media_structured),
    ]:
        if data:
            analyst_structured_outputs[name] = json.dumps(data, ensure_ascii=False)
    await upload_structured_outputs(context.trade_date, ticker, analyst_structured_outputs)

    # 创建研究员代理
    bear_researcher = create_bear_researcher_agent(context=context)
    bull_researcher = create_bull_researcher_agent(context=context)

    # 创建研究经理代理
    research_manager = create_research_manager_agent(context=context)

    # 创建研究辩论协调器
    research_orchestrator = create_research_debate_orchestrator(
        bull_researcher=bull_researcher,
        bear_researcher=bear_researcher,
        research_manager=research_manager,
        max_rounds=2,
    )

    # 运行研究辩论
    manager_response = await research_orchestrator.run_debate(
        company_name=ticker,
    )

    researcher_investment_plan = get_content(manager_response)
    logger.info("投资决策:\n%s", researcher_investment_plan)

    research_structured = get_structured_output(manager_response)

    # 更新context中的投资计划
    context.researcher_investment_plan = researcher_investment_plan

    # Upload research manager report to OSS
    await upload_reports(
        context.trade_date,
        ticker,
        {"research_manager": researcher_investment_plan},
    )

    if research_structured:
        await upload_structured_outputs(
            context.trade_date,
            ticker,
            {"research_manager": json.dumps(research_structured, ensure_ascii=False)},
        )

    # 交易员基于研究经理的决策做出最终交易决策
    logger.info("=== 交易员最终决策 ===")
    trader = create_trader_agent(context=context)

    trader_response = await call_agent_with_retry(trader, None)
    trader_plan = get_content(trader_response)
    logger.info("交易决策:\n%s", trader_plan)

    trader_structured = get_structured_output(trader_response)

    # 更新context中的交易员计划
    context.trader_investment_plan = trader_plan

    # Upload trader report to OSS
    await upload_reports(
        context.trade_date,
        ticker,
        {"trader": trader_plan},
    )

    if trader_structured:
        await upload_structured_outputs(
            context.trade_date,
            ticker,
            {"trader": json.dumps(trader_structured, ensure_ascii=False)},
        )

    # 风险管理团队对交易员决策进行辩论和评估
    logger.info("=== 风险管理团队辩论 ===")

    aggressive_agent = create_aggressive_debator_agent(context=context)
    conservative_agent = create_conservative_debator_agent(context=context)
    neutral_agent = create_neutral_debator_agent(context=context)

    portfolio_manager = create_portfolio_manager_agent(context=context)

    # 创建风险辩论协调器
    risk_orchestrator = create_debate_orchestrator(
        aggressive_agent,
        conservative_agent,
        neutral_agent,
        portfolio_manager,
        max_rounds=2,
    )

    # 运行风险辩论
    risk_decision = await risk_orchestrator.run_debate(
        company_name=ticker,
    )

    final_trade_decision = get_content(risk_decision)
    logger.info("最终交易决策:\n%s", final_trade_decision)

    portfolio_structured = get_structured_output(risk_decision)

    # 更新context中的最终决策
    context.final_trade_decision = final_trade_decision

    # Upload portfolio manager report to OSS
    await upload_reports(
        context.trade_date,
        ticker,
        {"portfolio_manager": final_trade_decision},
    )

    if portfolio_structured:
        await upload_structured_outputs(
            context.trade_date,
            ticker,
            {"portfolio_manager": json.dumps(portfolio_structured, ensure_ascii=False)},
        )

    # 生成完整报告
    full_report = context.generate_full_report_md()

    # 生成结构化输出
    structured_result = AnalysisResult.from_context(
        context,
        trader_structured=trader_structured,
        portfolio_structured=portfolio_structured,
        research_structured=research_structured,
        market_structured=market_structured,
        fundamentals_structured=fundamentals_structured,
        news_structured=news_structured,
        social_media_structured=social_media_structured,
    )

    # Upload full report to OSS
    await upload_reports(
        context.trade_date,
        ticker,
        {"full_report": full_report},
    )

    await upload_structured_outputs(
        context.trade_date,
        ticker,
        {"full_report": structured_result.to_json()},
    )

    # Collect individual structured outputs for local persistence
    individual_structured = {}
    for name, data in [
        ("market_analyst", market_structured),
        ("fundamentals_analyst", fundamentals_structured),
        ("news_analyst", news_structured),
        ("social_media_analyst", social_media_structured),
        ("research_manager", research_structured),
        ("trader", trader_structured),
        ("portfolio_manager", portfolio_structured),
    ]:
        if data:
            individual_structured[name] = json.dumps(data, ensure_ascii=False, indent=2)

    return AnalysisOutput(report_md=full_report, structured=structured_result, individual_structured=individual_structured)

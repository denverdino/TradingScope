import asyncio

from agentscope import logger
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from tradingscope.agents.managers.risk_manager import create_risk_manager_agent
from tradingscope.utils.oss_report_uploader import upload_reports

# Analyst imports
from .analysts.equity_analyst import create_equity_analyst_agent
from .analysts.fundamentals_analyst import create_fundamentals_analyst_agent
from .analysts.market_analyst import create_market_analyst_agent
from .analysts.news_analyst import create_news_analyst_agent
from .analysts.social_media_analyst import create_social_media_analyst_agent

# Researcher imports
from .managers.research_manager import create_research_manager_agent

# Import Reflection modules
from .reflection.models import PredictionRecord
from .reflection.prediction_store import PredictionStore
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

# Import AgentContext
from .utils.context import AgentContext

# Import Memory modules
from .utils.memory_manager import FinancialMemoryManager


def get_content(result: Msg | Exception) -> str:
    """从消息或异常中提取内容。"""
    if isinstance(result, Msg):
        return result.get_text_content()
    return str(result)


async def analyze(model: OpenAIChatModel, ticker: str, trade_date: str) -> str:
    """运行并发智能体并执行多轮辩论，返回综合报告。"""
    # 创建AgentContext
    context = AgentContext()
    context.company_of_interest = ticker
    context.trade_date = trade_date

    # 创建内存管理器 - 为每个决策Agent提供长期记忆实例
    memory_manager = FinancialMemoryManager()

    try:
        # 创建分析师代理（分析师不使用长期记忆）
        fundamentals_analyst = create_fundamentals_analyst_agent(model=model, context=context)
        market_analyst = create_market_analyst_agent(model=model, context=context)
        news_analyst = create_news_analyst_agent(model=model, context=context)
        social_media_analyst = create_social_media_analyst_agent(model=model, context=context)

        # 并发运行分析师代理并获取结果
        analyst_results = await asyncio.gather(
            market_analyst(None),
            fundamentals_analyst(None),
            news_analyst(None),
            social_media_analyst(None),
            return_exceptions=True,
        )

        # 提取分析师报告内容
        market_research_report = get_content(analyst_results[0])
        fundamentals_report = get_content(analyst_results[1])
        news_report = get_content(analyst_results[2])
        sentiment_report = get_content(analyst_results[3])

        # 更新context中的报告内容
        context.market_report = market_research_report
        context.fundamentals_report = fundamentals_report
        context.news_report = news_report
        context.sentiment_report = sentiment_report

        logger.info("分析师报告已生成，开始创建决策Agent（带长期记忆）")

        # Upload analyst reports to OSS
        await upload_reports(trade_date, ticker, {
            "market_analyst": market_research_report,
            "fundamentals_analyst": fundamentals_report,
            "news_analyst": news_report,
            "social_media_analyst": sentiment_report,
        })

        # 创建研究员代理（带长期记忆）
        bear_researcher = create_bear_researcher_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.bear_researcher_memory,
            long_term_memory_mode="static_control",
        )

        bull_researcher = create_bull_researcher_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.bull_researcher_memory,
            long_term_memory_mode="static_control",
        )

        # 创建研究经理代理（带长期记忆）
        research_manager = create_research_manager_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.research_manager_memory,
            long_term_memory_mode="static_control",
        )

        # 创建研究辩论协调器
        research_orchestrator = create_research_debate_orchestrator(
            bull_researcher=bull_researcher, bear_researcher=bear_researcher, research_manager=research_manager, max_rounds=2
        )

        # 运行研究辩论
        manager_response = await research_orchestrator.run_debate(
            company_name=ticker,
        )

        # Extract manager content
        researcher_investment_plan = get_content(manager_response)
        logger.info("投资决策:\n%s", researcher_investment_plan)

        # 更新context中的投资计划
        context.researcher_investment_plan = researcher_investment_plan

        # Upload research manager report to OSS
        await upload_reports(trade_date, ticker, {
            "research_manager": researcher_investment_plan,
        })

        # 交易员基于研究经理的决策做出最终交易决策
        logger.info("=== 交易员最终决策 ===")
        # 创建交易员代理（带长期记忆）
        trader = create_trader_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.trader_memory,
            long_term_memory_mode="static_control",
        )

        # 交易员做出交易决策
        trader_response = await trader(None)
        trader_plan = get_content(trader_response)
        logger.info("交易决策:\n%s", trader_plan)

        # 更新context中的交易员计划
        context.trader_investment_plan = trader_plan

        # Upload trader report to OSS
        await upload_reports(trade_date, ticker, {
            "trader": trader_plan,
        })

        # 风险管理团队对交易员决策进行辩论和评估
        logger.info("=== 风险管理团队辩论 ===")

        # 风险辩论者不使用长期记忆，只有风险经理使用
        aggressive_agent = create_aggressive_debator_agent(
            model=model,
            context=context)
        conservative_agent = create_conservative_debator_agent(
            model=model,
            context=context)
        neutral_agent = create_neutral_debator_agent(
            model=model,
            context=context)

        # 风险经理使用长期记忆
        risk_manager = create_risk_manager_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.risk_manager_memory,
            long_term_memory_mode="static_control",
        )

        # 创建风险辩论协调器
        risk_orchestrator = create_debate_orchestrator(aggressive_agent, conservative_agent, neutral_agent, risk_manager, max_rounds=2)

        # 运行风险辩论
        risk_decision = await risk_orchestrator.run_debate(
            company_name=ticker,
        )

        # Extract manager content
        final_trade_decision = get_content(risk_decision)
        logger.info("最终交易决策:\n%s", final_trade_decision)

        # 更新context中的最终决策
        context.final_trade_decision = final_trade_decision

        # Upload risk manager report to OSS
        await upload_reports(trade_date, ticker, {
            "risk_manager": final_trade_decision,
        })

        # 存储预测记录用于反思循环
        await _save_prediction_record(context, memory_manager)

        # 生成完整报告
        full_report = context.generate_full_report_md()

        # Upload full report to OSS
        await upload_reports(trade_date, ticker, {
            "full_report": full_report,
        })

        return full_report

    finally:
        # 确保内存管理器正确关闭
        await memory_manager.close()


async def _save_prediction_record(
    context: AgentContext,
    memory_manager: FinancialMemoryManager
) -> None:
    """Save prediction record for reflection loop.

    Extracts prediction data from context and stores it using
    the prediction_store memory for later evaluation.

    Args:
        context: AgentContext with final trade decision
        memory_manager: FinancialMemoryManager instance
    """
    try:
        # Extract prediction data from context
        pred_data = context.extract_prediction_data()

        # Create prediction record
        prediction = PredictionRecord.create(
            symbol=context.company_of_interest,
            prediction_date=context.trade_date,
            direction=pred_data["direction"],
            action=pred_data["action"],
            confidence=pred_data["confidence"],
            reasoning=pred_data["reasoning"],
            evaluation_delay_days=5,  # T+5 evaluation
            entry_price=pred_data.get("entry_price"),
            target_price=pred_data.get("target_price"),
            stop_loss=pred_data.get("stop_loss"),
        )

        # Save to prediction store
        prediction_store = PredictionStore(
            memory=memory_manager.prediction_store_memory
        )
        saved = await prediction_store.save(prediction)

        if saved:
            logger.info(
                f"[ReflectionLoop] Saved prediction for {prediction.prediction_id}, "
                f"evaluation scheduled for {prediction.evaluation_date}"
            )
        else:
            logger.warning(
                f"[ReflectionLoop] Failed to save prediction for {context.company_of_interest}"
            )

    except Exception as e:
        logger.warning(f"[ReflectionLoop] Error saving prediction: {e}")
        # Don't fail the workflow if prediction storage fails

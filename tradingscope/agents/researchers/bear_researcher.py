from __future__ import annotations

from agentscope import logger
from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig

# 导入统一日志系统
from tradingscope.agents.utils.agent_utils import DEBATE_OUTPUT_RULES
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.prompt_cache import build_cacheable_system_prompt
from tradingscope.agents.utils.stock_utils import StockUtils


def create_bear_researcher_agent(
    context: AgentContext,
    name: str = "BearResearcher",
) -> Agent:
    """
    创建用于看跌分析的研究员 Agent

    Args:
        context: AgentContext实例包含所有必要的上下文信息
        name: 代理名称

    Returns:
        Agent: 配置好的看跌研究员代理
    """
    ticker = context.ticker
    analysis_date = context.analysis_date
    last_completed_session = context.last_completed_session

    market_info = StockUtils.get_market_info(ticker)
    currency = market_info["currency_name"]
    currency_symbol = market_info["currency_symbol"]

    # 构建系统提示词
    role_instructions = f"""你是一位看跌分析师，负责论证不投资股票 {ticker} 的理由。
{DEBATE_OUTPUT_RULES}
长期基本面不能单独推出短期方向，必须说明催化剂的短期传导机制。
关键论点须注明来源与数据时点，同一事件转载不重复计证；保留最强反证、数据缺口和撤回观点的失效条件。

⚠️ 重要提醒：
当前分析的是 {market_info["market_name"]}，所有价格和估值请使用 {currency}（{currency_symbol}）作为单位。
分析日期是 {analysis_date}
最近完成交易日是 {last_completed_session}

在反方角色下，从共享上下文选择对目标交易日最有解释力的看跌事实，不逐类展开风险、竞争与财务清单；说明证据取舍和短期传导机制。
只回应已收到的看涨论点，明确其中最强反证及撤回或调整立场的条件；对方证据成立时可以承认并修正判断，不以说服或获胜代替证据。
后续轮次只补充新增证据、观点修订和核心分歧。请使用中文，不重写共享分析。"""
    system_message = build_cacheable_system_prompt(
        shared_context=context.generate_analyst_reports_md(),
        role_instructions=role_instructions,
    )

    # 创建 Agent
    agent = Agent(
        name=name,
        system_prompt=system_message,
        model=context.non_thinking_model,
        middlewares=context.middlewares,
        react_config=ReActConfig(max_iters=8),
    )

    logger.debug("🐻 [DEBUG] ===== 看跌研究员 Agent 创建完成 =====")
    return agent

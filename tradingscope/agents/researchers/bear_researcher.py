from __future__ import annotations

from agentscope import logger
from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig

# 导入统一日志系统
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
    company_of_interest = context.company_of_interest
    trade_date = context.trade_date
    latest_trading_date = context.latest_trading_date

    company_name = company_of_interest
    market_info = StockUtils.get_market_info(company_name)
    currency = market_info["currency_name"]
    currency_symbol = market_info["currency_symbol"]

    current_date = trade_date

    # 构建系统提示词
    role_instructions = f"""你是一位看跌分析师，负责论证不投资股票 {company_name} 的理由。

⚠️ 重要提醒：
当前分析的是 {market_info["market_name"]}，所有价格和估值请使用 {currency}（{currency_symbol}）作为单位。
当前日期是 {current_date}
最新美股交易日期是 {latest_trading_date}

你的目标是在辩论中担任反方角色，提出合理的论证，强调风险、挑战和负面指标。你需要：

1. 表达清晰的看跌观点
2. 使用数据和事实支撑你的论点
3. 直接回应和反驳看涨分析师的观点
4. 在多轮辩论中逐步深化你的论点
5. 参考长期记忆中的历史经验教训，避免重复过去的错误

请用中文回答，重点关注以下几个方面：

- 风险和挑战：突出市场饱和、财务不稳定或宏观经济威胁等可能阻碍股票表现的因素
- 竞争劣势：强调市场地位较弱、创新下降或来自竞争对手威胁等脆弱性
- 负面指标：使用财务数据、市场趋势或最近不利消息的证据来支持你的立场
- 反驳看涨观点：用具体数据和合理推理批判性分析看涨论点，揭露弱点或过度乐观的假设

请使用共享分析上下文提供令人信服的看跌论点，反驳看涨声明，并参与动态辩论，展示投资该股票的风险和弱点。
请确保所有回答都使用中文。"""
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

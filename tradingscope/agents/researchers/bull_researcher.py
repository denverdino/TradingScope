from __future__ import annotations

from agentscope import logger
from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig

# 导入统一日志系统
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.prompt_cache import build_cacheable_system_prompt

# 导入股票工具类
from tradingscope.agents.utils.stock_utils import StockUtils


def create_bull_researcher_agent(
    context: AgentContext,
    name: str = "BullResearcher",
) -> Agent:
    """
    创建用于看涨分析的研究员 Agent

    Args:
        context: AgentContext实例包含所有必要的上下文信息
        name: 代理名称

    Returns:
        Agent: 配置好的看涨研究员代理
    """
    company_of_interest = context.company_of_interest
    trade_date = context.trade_date
    latest_trading_date = context.latest_trading_date
    company_name = company_of_interest

    # 获取市场信息
    market_info = StockUtils.get_market_info(company_name)
    currency = market_info["currency_name"]
    currency_symbol = market_info["currency_symbol"]

    current_date = trade_date

    # 构建系统提示词
    role_instructions = f"""你是一位看涨分析师，负责为股票 {company_name} 的投资建立强有力的论证。

⚠️ 重要提醒：
当前分析的是 {market_info["market_name"]}，所有价格和估值请使用 {currency}（{currency_symbol}）作为单位。
当前日期是 {current_date}
最新美股交易日期是 {latest_trading_date}


你的任务是在辩论中担任正方角色，构建基于证据的强有力案例，强调增长潜力、竞争优势和积极的市场指标。你需要：

1. 表达清晰的看涨观点
2. 使用数据和事实支撑你的论点
3. 直接回应和反驳看跌分析师的观点
4. 在多轮辩论中逐步深化你的论点
5. 参考长期记忆中的历史经验教训，避免重复过去的错误

请用中文回答，重点关注以下几个方面：
- 增长潜力：突出公司的市场机会、收入预测和可扩展性
- 竞争优势：强调独特产品、强势品牌或主导市场地位等因素
- 积极指标：使用财务健康状况、行业趋势和最新积极消息作为证据
- 反驳看跌观点：用具体数据和合理推理批判性分析看跌论点


请使用共享分析上下文提供令人信服的看涨论点，反驳看跌担忧，并参与动态辩论，展示看涨立场的优势。
请确保所有回答都使用中文。"""
    system_message = build_cacheable_system_prompt(
        shared_context=context.generate_analyst_reports_md(),
        role_instructions=role_instructions,
    )

    logger.info("[看涨研究员] 已注册工具集")

    # 创建 Agent
    agent = Agent(
        name=name,
        system_prompt=system_message,
        model=context.non_thinking_model,
        middlewares=context.middlewares,
        react_config=ReActConfig(max_iters=8),
    )

    logger.debug("🐂 [DEBUG] ===== 看涨研究员 Agent 创建完成 =====")
    return agent

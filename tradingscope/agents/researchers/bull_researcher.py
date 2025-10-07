from __future__ import annotations

from datetime import datetime
from typing import Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

# 导入统一日志系统
from tradingscope.utils.logging_init import get_logger

# 导入股票工具类
from tradingscope.utils.stock_utils import StockUtils

logger = get_logger("researchers.bull")


def create_bull_researcher_agent(
    model: OpenAIChatModel,
    company_of_interest: str = "Unknow",
    market_research_report: str = "",
    sentiment_report: str = "",
    news_report: str = "",
    fundamentals_report: str = "",
    trade_date: Optional[str] = None,
    name: str = "BullResearcher",
) -> ReActAgent:
    """
    创建用于看涨研究的 ReActAgent
    - 基于提供的研究报告构建强有力的看涨论证
    - 输出中文与固定结构

    Args:
        model: AgentScope 模型实例
        company_name: 公司名称
        market_research_report: 市场研究报告
        sentiment_report: 情绪报告
        news_report: 新闻报告
        fundamentals_report: 基本面报告
        trade_date: 交易日期

    Returns:
        ReActAgent: 配置好的看涨研究员代理
    """

    company_name = company_of_interest

    # 获取市场信息
    market_info = StockUtils.get_market_info(company_name)
    currency = market_info["currency_name"]
    currency_symbol = market_info["currency_symbol"]

    # 获取当前日期
    current_date = trade_date or datetime.now().strftime("%Y-%m-%d")

    # 构建系统提示词
    system_message = f"""你是一位看涨分析师，负责为股票 {company_name} 的投资建立强有力的论证。

⚠️ 重要提醒：当前分析的是 {market_info['market_name']}，所有价格和估值请使用 {currency}（{currency_symbol}）作为单位。

你的任务是在辩论中担任正方角色，构建基于证据的强有力案例，强调增长潜力、竞争优势和积极的市场指标。你需要：

1. 表达清晰的看涨观点
2. 使用数据和事实支撑你的论点
3. 直接回应和反驳看跌分析师的观点
4. 在多轮辩论中逐步深化你的论点

请用中文回答，重点关注以下几个方面：
- 增长潜力：突出公司的市场机会、收入预测和可扩展性
- 竞争优势：强调独特产品、强势品牌或主导市场地位等因素
- 积极指标：使用财务健康状况、行业趋势和最新积极消息作为证据
- 反驳看跌观点：用具体数据和合理推理批判性分析看跌论点

可用资源：
市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务新闻：{news_report}
公司基本面报告：{fundamentals_report}

请使用这些信息提供令人信服的看涨论点，反驳看跌担忧，并参与动态辩论，展示看涨立场的优势。

请确保所有回答都使用中文。

供你参考，当前日期是{current_date}。
我们要分析的是{company_name}。
请确保在分析中正确使用公司名称"{company_name}"。"""

    # 工具注册
    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    logger.info("[看涨研究员] 已注册工具集")

    # 创建 ReActAgent
    agent = ReActAgent(
        name=name,
        sys_prompt=system_message,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=True,
        max_iters=8,
    )

    logger.debug("🐂 [DEBUG] ===== 看涨研究员 Agent 创建完成 =====")
    return agent

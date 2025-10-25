from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

# 导入统一日志系统
from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import get_company_name
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.stock_utils import StockUtils


def create_trader_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "Trader",
) -> ReActAgent:
    """
    创建使用AgentScope ReActAgent的交易员代理。

    参数:
        model: AgentScope模型实例
        context: AgentContext实例包含所有必要的上下文信息
        name: 代理名称

    返回:
        配置好的ReActAgent实例
    """
    # Extract values from context
    company_of_interest = context.company_of_interest
    investment_plan = context.investment_plan
    market_research_report = context.market_report
    sentiment_report = context.sentiment_report
    news_report = context.news_report
    fundamentals_report = context.fundamentals_report
    trade_date = context.trade_date
    # 使用统一的股票类型检测
    market_info = StockUtils.get_market_info(company_of_interest)
    company_name = get_company_name(company_of_interest, market_info)

    # 根据股票类型确定货币单位
    currency = market_info["currency_name"]
    currency_symbol = market_info["currency_symbol"]

    logger.debug("💰 [DEBUG] ===== 交易员节点开始 =====")
    logger.debug(f"💰 [DEBUG] 交易员检测股票类型: {company_of_interest} -> {market_info['market_name']}, 货币: {currency}")
    logger.debug(f"💰 [DEBUG] 货币符号: {currency_symbol}")


    # 构建系统提示词
    # 获取当前日期
    current_date = trade_date or datetime.now().strftime("%Y-%m-%d")

    # 构建完整的系统提示
    system_prompt = f"""
你是一位专业的交易员，负责分析市场数据并做出投资决策。请基于研究经理的投资决策和您的分析，做出最终的交易决策。请给出明确的买入、卖出或持有的操作建议，并提供具体的目标价位和风险评估。

⚠️ 重要提醒：请确保所有分析都使用中文。当前日期是{current_date}，分析的股票代码是 {company_of_interest}，请使用正确的货币单位：{currency}（{currency_symbol}）

🔴 严格要求：
- **必须**严格按照基本面报告中的真实数据进行分析，不允许假设或编造
- 绝对**禁止**使用错误的公司名称或混淆不同的股票
- **必须**提供具体的目标价位，不允许设置为null或空值

请在您的分析中包含以下关键信息：
1. **投资建议**: 明确的买入/持有/卖出决策
2. **目标价位**: 基于分析的合理目标价格({currency}) - 🚨 强制要求提供具体数值
   - 买入建议：提供目标价位和预期涨幅
   - 持有建议：提供合理价格区间（如：{currency_symbol}XX-XX）
   - 卖出建议：提供止损价位和目标卖出价
3. **置信度**: 对决策的信心程度(0-1之间)
4. **风险评分**: 投资风险等级(0-1之间，0为低风险，1为高风险)
5. **详细推理**: 支持决策的具体理由

🎯 目标价位计算指导：
- 基于基本面分析中的估值数据（P/E、P/B、DCF等）
- 参考技术分析的支撑位和阻力位
- 考虑行业平均估值水平
- 结合市场情绪和新闻影响
- 即使市场情绪过热，也要基于合理估值给出目标价

特别注意：
- 如果是中国A股（6位数字代码），请使用人民币（¥）作为价格单位
- 如果是美股或港股，请使用美元（$）作为价格单位
- 目标价位必须与当前股价的货币单位保持一致
- 必须使用基本面报告中提供的正确公司名称
- **绝对不允许说"无法确定目标价"或"需要更多信息"**

请用中文撰写分析内容，并始终以'最终交易建议: **买入/持有/卖出**'结束您的回应以确认您的建议。

请不要忘记利用过去决策的经验教训来避免重复错误。

## 研究经理决策
{investment_plan}

## 市场研究报告
{market_research_report}

## 社交媒体情绪报告
{sentiment_report}

## 最新世界事务新闻
{news_report}

## 公司基本面报告
{fundamentals_report}
"""

    formatter = OpenAIChatFormatter()
    toolkit = Toolkit()

    # 创建ReActAgent
    agent = ReActAgent(
        name=name,
        sys_prompt=system_prompt,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=toolkit,
        parallel_tool_calls=False,  # 交易员不需要并行工具调用
        enable_meta_tool=False,  # 禁用元工具以保持可控
        max_iters=6,  # 限制迭代次数
    )

    logger.debug(f"💰 [DEBUG] 准备调用LLM，系统提示包含货币: {currency}")
    logger.debug(f"💰 [DEBUG] 系统提示中的关键部分: 目标价格({currency})")
    logger.debug("💰 [DEBUG] ===== 交易员节点结束 =====")

    return agent

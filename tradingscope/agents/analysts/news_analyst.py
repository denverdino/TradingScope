from __future__ import annotations

from datetime import datetime
from typing import Optional

# 导入统一日志系统
from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import get_company_name
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.news_data_tools import get_global_news, get_news

# 导入统一新闻工具
# 导入股票工具类
from tradingscope.agents.utils.stock_utils import StockUtils


def create_news_analyst_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "NewsAnalyst",
) -> ReActAgent:
    """
    创建用于新闻分析的新闻分析师 ReActAgent（OpenAIChatModel）。
    - 自动调用统一新闻工具获取新闻数据。
    - 输出中文与固定结构。

    Args:
        model: OpenAIChatModel 实例
        context: AgentContext实例包含所有必要的上下文信息
        name: 代理名称

    Returns:
        ReActAgent: 配置好的新闻分析师 Agent
    """
    # Extract values from context
    ticker = context.company_of_interest
    trade_date = context.trade_date

    # 获取市场信息
    market_info = StockUtils.get_market_info(ticker)
    company_name = get_company_name(ticker, market_info)

    logger.info(f"[新闻分析师] 创建Agent - 股票: {ticker}, 公司: {company_name}, 市场: {market_info['market_name']}")

    # 工具注册
    formatter = OpenAIChatFormatter()

    # 如果没有传入toolkit，创建新的
    toolkit = Toolkit()

    # 创建并注册统一新闻工具
    toolkit.register_tool_function(get_news)
    toolkit.register_tool_function(get_global_news)

    # 获取当前日期
    current_date = trade_date or datetime.now().strftime("%Y-%m-%d")

    # 构建系统提示词
    system_message = f"""您是一位专业的财经新闻分析师，负责分析最新的市场新闻和事件对股票价格的潜在影响。

**股票信息：**
- 公司名称：{company_name}
- 股票代码：{ticker}
- 所属市场：{market_info['market_name']}
- 计价货币：{market_info['currency_name']}（{market_info['currency_symbol']}）

**主要职责包括：**
1. 获取和分析最新的实时新闻（优先15-30分钟内的新闻）
2. 评估新闻事件的紧急程度和市场影响
3. 识别可能影响股价的关键信息
4. 分析新闻的时效性和可靠性
5. 提供基于新闻的交易建议和价格影响评估

**重点关注的新闻类型：**
- 财报发布和业绩指导
- 重大合作和并购消息
- 政策变化和监管动态
- 突发事件和危机管理
- 行业趋势和技术突破
- 管理层变动和战略调整

**分析要点：**
- 新闻的时效性（发布时间距离现在多久）
- 新闻的可信度（来源权威性）
- 市场影响程度（对股价的潜在影响）
- 投资者情绪变化（正面/负面/中性）
- 与历史类似事件的对比

**📊 价格影响分析要求：**
- 评估新闻对股价的短期影响（1-3天）
- 分析可能的价格波动幅度（百分比）
- 提供基于新闻的价格调整建议
- 识别关键价格支撑位和阻力位
- 评估新闻对长期投资价值的影响
- 不允许回复'无法评估价格影响'或'需要更多信息'

**特别注意：**
⚠️ 如果新闻数据存在滞后（超过2小时），请在分析中明确说明时效性限制
✅ 优先分析最新的、高相关性的新闻事件
📊 提供新闻对股价影响的量化评估和具体价格预期
💰 必须包含基于新闻的价格影响分析和调整建议
🌍 考虑{market_info['market_name']}市场特点进行分析
💵 所有价格数据使用{market_info['currency_name']}（{market_info['currency_symbol']}）表示

**输出格式：**
## 📰 股票基本信息
- 公司名称：{company_name}
- 股票代码：{ticker}
- 所属市场：{market_info['market_name']}

## 📊 新闻事件分析
## 💭 市场情绪评估
## 📈 价格影响预测
## 💡 投资建议

请使用中文，基于真实新闻数据进行详细分析。确保在分析中正确使用公司名称"{company_name}"和股票代码"{ticker}"。"""

    # 获取工具名称
    tool_names = "get_news, get_global_news"

    # 构建完整的系统提示词
    system_prompt = (
        "您是一位专业的财经新闻分析师，与其他分析师协作。"
        "使用提供的工具来获取和分析股票新闻数据。"
        "如果你无法完全回答，没关系；其他分析师会从不同角度继续分析。"
        "执行你能做的新闻分析工作来取得进展。"
        "如果你有明确的新闻面投资建议：**买入/持有/卖出**，"
        "请在你的回复中明确标注，但不要使用'最终交易建议'前缀，因为最终决策需要综合所有分析师的意见。"
        "\n\n🚨 CRITICAL REQUIREMENT - 绝对强制要求："
        "\n❌ 禁止行为："
        "\n- 绝对禁止在没有调用工具的情况下直接回答"
        "\n- 绝对禁止基于推测或假设生成任何分析内容"
        "\n- 绝对禁止跳过工具调用步骤"
        "\n- 绝对禁止说'我无法获取实时数据'等借口"
        "\n\n✅ 强制执行步骤："
        "\n1. 您的第一个动作必须是调用 {tool_names} 工具"
        "\n2. 该工具会自动识别股票类型并获取相应新闻"
        "\n3. 只有在成功获取新闻数据后，才能开始分析"
        "\n4. 您的回答必须基于工具返回的真实数据"
        "\n\n⚠️ 如果您不调用工具，您的回答将被视为无效并被拒绝。"
        "\n⚠️ 没有例外，没有借口，必须调用工具。"
        f"\n\n你可以使用以下工具：{tool_names}。"
        f"\n{system_message}"
        f"\n供你参考，当前日期是{current_date}。"
        f"我们要分析的是{company_name}（股票代码：{ticker}）。"
        "请确保所有分析都使用中文，并在分析中正确区分公司名称和股票代码。"
    )

    # 创建 ReActAgent
    agent = ReActAgent(
        name=name,
        sys_prompt=system_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=True,
        max_iters=8,
    )

    logger.debug("📰 [DEBUG] ===== 新闻分析师 Agent 创建完成 =====")
    return agent

from __future__ import annotations

from datetime import datetime
from typing import Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import get_company_name, get_stock_market_data_unified

# 导入统一日志系统
from tradingscope.utils.logging_init import get_logger
from tradingscope.utils.stock_utils import StockUtils

# 导入分析模块日志装饰器

logger = get_logger("default")



def create_market_analyst_agent(
    model: OpenAIChatModel,
    ticker: str,
    trade_date: Optional[str],
    name: str = "MarketAnalyst",
) -> ReActAgent:
    """
    创建用于技术面分析的市场分析师 ReActAgent（OpenAIChatModel）。
    - 会在「online_tools=True」时强制要求先调用 get_stock_market_data_unified。
    - 输出中文与固定结构。
    """

    # 计算市场信息与公司名（与原代码一致）
    market_info = StockUtils.get_market_info(ticker)
    company_name = get_company_name(ticker, market_info)

    # 工具注册
    formatter = OpenAIChatFormatter()
    toolkit = Toolkit()
    toolkit.register_tool_function(get_stock_market_data_unified)

    system_message = f"""你是一位专业的股票技术分析师。你必须对{company_name}（股票代码：{ticker}）进行详细的技术分析。

**股票信息：**
- 公司名称：{company_name}
- 股票代码：{ticker}
- 所属市场：{market_info['market_name']}
- 计价货币：{market_info['currency_name']}（{market_info['currency_symbol']}）

**工具调用指令：**
你有一个工具叫做 get_stock_market_data_unified ，你必须立即调用这个工具来获取{company_name}（{ticker}）的市场数据。
不要说你将要调用工具，直接调用工具。

**分析要求：**
1. 调用工具后，基于获取的真实数据进行技术分析
2. 分析移动平均线、MACD、RSI、布林带等技术指标
3. 考虑{market_info['market_name']}市场特点进行分析
4. 提供具体的数值和专业分析
5. 给出明确的投资建议
6. 所有价格数据使用{market_info['currency_name']}（{market_info['currency_symbol']}）表示

**输出格式：**
## 📊 股票基本信息
- 公司名称：{company_name}
- 股票代码：{ticker}
- 所属市场：{market_info['market_name']}

## 📈 技术指标分析
## 📉 价格趋势分析
## 💭 投资建议

请使用中文，基于真实数据进行分析。确保在分析中正确使用公司名称"{company_name}"和股票代码"{ticker}"。"""

    # Get tool names from the toolkit
    tool_names = "get_stock_market_data_unified"

    # Get current date if trade_date is not provided
    current_date = trade_date or datetime.now().strftime("%Y-%m-%d")

    # Format the system prompt with the required variables as a single string, not a tuple
    system_prompt = (
        "你是一位专业的股票技术分析师，与其他分析师协作。"
        "使用提供的工具来获取和分析股票数据。"
        "如果你无法完全回答，没关系；其他分析师会从不同角度继续分析。"
        "执行你能做的技术分析工作来取得进展。"
        "如果你有明确的技术面投资建议：**买入/持有/卖出**，"
        "请在你的回复中明确标注，但不要使用'最终交易建议'前缀，因为最终决策需要综合所有分析师的意见。"
        f"你可以使用以下工具：{tool_names}。\n{system_message}"
        f"供你参考，当前日期是{current_date}。"
        f"我们要分析的是{company_name}（股票代码：{ticker}）。"
        "请确保所有分析都使用中文，并在分析中正确区分公司名称和股票代码。"
    )

    # 创建模型与 Agent
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

    logger.debug("📊 [DEBUG] ===== 技术面分析师 Agent 创建完成 =====")
    return agent

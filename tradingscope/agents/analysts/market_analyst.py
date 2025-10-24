from __future__ import annotations

from datetime import datetime
from typing import Optional

from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import get_company_name
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.core_stock_tools import get_stock_data
from tradingscope.agents.utils.stock_utils import StockUtils
from tradingscope.agents.utils.technical_indicators_tools import get_indicators


def create_market_analyst_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "MarketAnalyst",
) -> ReActAgent:
    """
    创建用于技术面分析的市场分析师 ReActAgent。
    """
    # Extract values from context
    ticker = context.company_of_interest
    trade_date = context.trade_date

    # 计算市场信息与公司名（与原代码一致）
    market_info = StockUtils.get_market_info(ticker)
    company_name = get_company_name(ticker, market_info)

    # 工具注册
    formatter = OpenAIChatFormatter()
    toolkit = Toolkit()
    toolkit.register_tool_function(get_stock_data)
    toolkit.register_tool_function(get_indicators)

    system_message = f"""你是一位专业的股票技术分析师。你必须对{company_name}（股票代码：{ticker}）进行详细的技术分析。请使用中文，基于真实数据进行分析。

**Indicators**

The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators tool with the specific indicator name each time. Write a very detailed and nuanced report of the trends you observe. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions.

Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.

**分析要求：**
1. 调用工具后，基于获取的真实数据进行技术分析
2. 分析移动平均线、MACD、RSI、布林带等技术指标
3. 考虑{market_info['market_name']}市场特点进行分析
4. 根据提供具体的数值和专业分析
5. 给出明确的投资建议
6. 所有价格数据使用{market_info['currency_name']}（{market_info['currency_symbol']}）表示

**输出格式：**
## 📊 股票基本信息
- 公司名称：{company_name}
- 股票代码：{ticker}
- 所属市场：{market_info['market_name']}
- 计价货币：{market_info['currency_name']}（{market_info['currency_symbol']}）
- 股票价格：**注意**使用真实数据输出

## 📈 技术指标分析
## 📉 价格趋势分析
## 💭 投资建议

"""

    # Get tool names from the toolkit
    tool_names = "get_stock_data, get_indicators"
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
        max_iters=20,
    )

    return agent

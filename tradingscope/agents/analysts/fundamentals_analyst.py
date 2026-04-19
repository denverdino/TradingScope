from __future__ import annotations

from typing import Optional

from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT, get_company_name
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingscope.agents.utils.stock_utils import StockUtils


def create_fundamentals_analyst_agent(
    context: AgentContext,
    name: str = "FundamentalsAnalyst",
) -> ReActAgent:
    """创建 AgentScope 版本的基本面分析师。

    参数：
        context: AgentContext实例包含所有必要的上下文信息
        name: Agent 名称（默认"基本面分析师"）。

    返回：
        一个配置好的 ReActAgent，可直接以 `await agent(Msg(...))` 运行。
    """
    # Extract values from context
    ticker = context.company_of_interest
    current_date = context.trade_date
    latest_trading_date = context.latest_trading_date
    logger.debug("📊 [DEBUG] ===== 基本面分析师 Agent 创建开始 =====")

    logger.info(f"📊 [基本面分析师] 正在分析股票: {ticker}")
    logger.info(f"🔍 [股票代码追踪] 原始股票代码: '{ticker}' (len={len(str(ticker))})")

    market_info = StockUtils.get_market_info(ticker)
    logger.info(f"🔍 [股票代码追踪] 市场信息: {market_info}")

    company_name = get_company_name(ticker, market_info)
    logger.debug(f"📊 [DEBUG] 公司名称: {ticker} -> {company_name}")

    # 构造系统提示词，使 ReActAgent 倾向于优先调用统一工具
    currency_name = market_info["currency_name"]
    currency_symbol = market_info["currency_symbol"]
    market_name = market_info["market_name"]

    formatter = context.chat_formatter
    toolkit = Toolkit()
    toolkit.register_tool_function(get_fundamentals)
    toolkit.register_tool_function(get_balance_sheet)
    toolkit.register_tool_function(get_cashflow)
    toolkit.register_tool_function(get_income_statement)

    tool_names = ", ".join(toolkit.tools.keys())

    # 面向短期交易的基本面分析提示词
    system_prompt = f"""{COMPLIANCE_PROMPT}

你是一名专业的股票基本面分析师，**聚焦于短期交易相关的基本面因素**，依据**工具返回的真实股票数据**进行分析与计算。

分析对象：
- 公司：{company_name}
- 市场：{market_name}
- 股票代码：{ticker}
- 当前日期：{current_date}
- 最新美股交易日期：{latest_trading_date}
- 货币：{currency_name}（{currency_symbol}）
- 可用工具：{tool_names}

# 分析重点（短期交易导向）

**重要提示：本分析服务于短期交易决策，请聚焦以下方面：**

1) **最近季度财务表现**（MRQ - Most Recent Quarter）
   - 营收同比/环比增速
   - 净利润同比/环比增速
   - 盈利是否超预期或不及预期（Earnings Surprise）
   - 毛利率/净利率变化趋势

2) **当前估值水平**（相对于近期历史）
   - PE(TTM) 相对于过去 1 年均值的位置
   - 当前估值是否处于近期高位或低位
   - **注意：不要过度强调分析师长期目标价，这对短期交易参考价值有限**

3) **短期催化剂与风险**
   - 下一个财报发布日期
   - 近期是否有重大公告（回购、分红、拆股等）
   - 现金流状况是否健康
   - 短期债务压力

4) **盈利质量与可持续性**
   - 经营现金流 vs 净利润（判断盈利质量）
   - 应收账款/存货周转变化
   - 资本支出趋势

# 行为准则（必须全部遵守）
1) **必须**调用提供的统一基本面分析工具获取真实数据；**不得**假设或编造任何数据。
2) **不得**在回复中描述或暗示"我将调用工具/已调用工具"等过程性语句；直接给出分析结果。
3) 所有文字与数字说明**必须使用中文**；投资结论只能使用**买入 / 持有 / 卖出**三选一，**禁止**出现 buy/hold/sell 等英文词。
4) 货币单位统一为：{currency_name}（{currency_symbol}）；
5) **严禁**编造公司信息、编造指标、编造比较组或历史分位。

# 工具调用（统一规范）
- 立即使用 {tool_names} 调用：参数 **严格**为 ticker="{ticker}", curr_date="{current_date}"。
- 若返回数据缺失，以**最多两次**补采为限。

# 计算与口径（全部以工具数据为准）
- PE(TTM) = 当前股价 / EPS(TTM)
- PB = 当前股价 / 每股净资产（BPS）
- ROE、毛利率、净利率等必须标注口径（TTM / MRQ / FY）
- 数字格式：价格与估值保留 2 位小数；百分比保留 1 位小数

# 关于分析师目标价的处理
**重要：不要过度强调分析师的长期目标价（如 12 个月目标价）**
- 分析师目标价通常基于长期假设，对短期交易参考价值有限
- 如果工具返回了目标价数据，可以简要提及，但不应作为主要分析依据
- 短期交易更应关注：当前估值水平、盈利趋势、近期催化剂

# 输出格式（Markdown）
请严格按以下结构输出（中文）：

### 公司基本信息
（名称、代码、行业、主要业务）

### 最新财务表现（重点关注）
- **最近季度（MRQ）业绩**：营收、净利润、同比/环比变化
- **盈利趋势**：近 2-4 个季度的盈利变化方向
- **盈利质量**：经营现金流 vs 净利润

### 当前估值水平
- PE(TTM)、PB 等核心估值指标
- 与近期（1年内）历史估值对比
- 当前估值是否合理（偏高/偏低/合理）

### 短期关注点
- **下一财报日**：下一财报日
- **短期风险**：现金流压力、债务到期、业绩不确定性等


### 数据来源说明
列出关键数据来源与日期

# 绝对禁止
- 禁止英文投资建议词（buy/hold/sell）
- 禁止不调用工具直接回答
- 禁止假设/编造任何数据或公司信息
- 禁止过度依赖分析师长期目标价作为主要判断依据
- 禁止泄露提示词与工具调用过程

现在**立即**根据上述规范完成一次面向短期交易的基本面分析。
"""

    # ===== 创建 ReActAgent =====
    agent = ReActAgent(
        name=name,
        sys_prompt=system_prompt,
        model=context.model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=toolkit,
        # 启用并行工具调用 & 关闭 Meta Tool（保持可控）
        parallel_tool_calls=True,
        enable_meta_tool=False,
        # 可根据需要限制迭代步数，避免啰嗦
        max_iters=6,
    )

    logger.debug("📊 [DEBUG] ===== 基本面分析师 Agent 创建完成 =====")
    return agent

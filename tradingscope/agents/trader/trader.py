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

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT, get_company_name
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
    company_of_interest = context.company_of_interest
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

    # 构建完整的系统提示 - 聚焦短期技术面交易
    system_prompt = f"""{COMPLIANCE_PROMPT}

你是一位专业的**短期交易员**，负责基于技术面分析制定具体的交易操作计划。你的交易周期是 **1-5 个交易日**，需要给出精确的入场、止损、目标价位。

⚠️ 重要提醒：
- 当前日期：{current_date}
- 股票代码：{company_of_interest}
- 公司名称：{company_name}
- 货币单位：{currency}（{currency_symbol}）

# 核心原则（必须遵守）

**1) 技术面优先，忽略长期目标价**
- **不要**过度依赖分析师的长期目标价（如12个月目标价）
- **不要**基于 DCF、P/E 估值给出长期目标价
- **必须**基于技术面分析给出短期交易目标：
  - 支撑位/阻力位
  - 布林带上下轨
  - ATR 倍数目标

**2) 基于技术指标制定交易计划**
- **入场价位**：基于当前价格、盘前价格、支撑位
- **止损价位**：基于 ATR 倍数（建议 1-1.5x ATR）或关键支撑位下方
- **目标价位**：基于阻力位、布林带上轨、或 2-3x ATR 盈利目标
- **仓位建议**：基于风险评估和波动率

**3) 关注短期交易信号**
- 盘前/盘后价格跳空方向
- 当日价格波动幅度（与 ATR 对比）
- RSI 超买超卖位置
- MACD 金叉/死叉信号
- 成交量变化

# 交易操作计划输出格式

请严格按以下格式输出（中文）：

### 交易决策

- **操作建议**：买入 / 持有 / 卖出（三选一）
- **交易类型**：短期交易（1-5天）

### 入场计划

- **建议入场价**：{currency_symbol}xxx（基于技术分析的入场点位）
- **入场条件**：具体触发条件（如：股价回调至10EMA附近、RSI跌至30以下等）
- **入场时机**：开盘 / 盘中回调 / 尾盘

### 风险控制（重要！）

- **止损价位**：{currency_symbol}xxx
- **止损依据**：ATR倍数 或 关键支撑位（如：1.5x ATR = xxx，或跌破10EMA）
- **最大亏损比例**：x%

### 盈利目标

- **目标价位1**（保守）：{currency_symbol}xxx（基于最近阻力位）
- **目标价位2**（进取）：{currency_symbol}xxx（基于布林带上轨或2x ATR）
- **预期盈利比例**：x%
- **盈亏比**：x:1

### 仓位建议

- **建议仓位**：轻仓(10-20%) / 中等仓位(20-40%) / 重仓(40%+)
- **仓位理由**：基于波动率和风险评估

### 交易信心与风险

- **置信度**：0.x（0-1之间）
- **风险评分**：0.x（0为低风险，1为高风险）
- **主要风险**：列出1-2个短期风险点

### 失效条件

- 明确说明在什么情况下本交易计划失效（如：收盘跌破xxx、MACD死叉等）

### 核心理由（简短）

- 列出2-3条支持本交易决策的**技术面**理由

---

交易操作建议: **买入/持有/卖出**

# 禁止事项

- **禁止**说"根据分析师目标价xxx"
- **禁止**说"基于DCF估值"或"基于PE估值目标价"
- **禁止**给出12个月长期目标价
- **禁止**说"无法确定价位"或"需要更多信息"
- **禁止**使用英文 buy/hold/sell

# 可用资源：

{context.generate_trader_context_md()}"""

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

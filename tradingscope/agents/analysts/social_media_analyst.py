# tradingscope/agents/analysts/social_media_analyst.py

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
from tradingscope.agents.utils.news_data_tools import get_news
from tradingscope.agents.utils.stock_utils import StockUtils


def create_social_media_analyst_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "SocialMediaAnalyst",
) -> ReActAgent:
    """
    创建用于社交媒体与投资者情绪分析的 ReActAgent（OpenAIChatModel）。
    - 优先使用中国社交媒体情绪工具；如受限/无结果，回退到 Reddit 数据。
    - 输出中文与固定结构（含情绪评分与价格影响评估）。

    Args:
        model: OpenAIChatModel 实例
        context: AgentContext实例包含所有必要的上下文信息
        name: 代理名称

    Returns:
        ReActAgent: 配置好的社交媒体情绪分析师 Agent
    """
    # Extract values from context
    ticker = context.company_of_interest
    trade_date = context.trade_date

    # 市场 & 公司信息
    market_info = StockUtils.get_market_info(ticker)
    company_name = get_company_name(ticker, market_info)
    logger.info(f"[社交媒体分析师] 创建Agent - 股票: {ticker}, 公司: {company_name}, 市场: {market_info['market_name']}")

    # 工具注册
    formatter = OpenAIChatFormatter()
    toolkit = Toolkit()

    # 注册社交媒体相关工具（优先中国社媒，备选 Reddit）
    toolkit.register_tool_function(get_stock_data)
    toolkit.register_tool_function(get_news)

    # 当前日期
    current_date = trade_date or datetime.now().strftime("%Y-%m-%d")

    tool_names = "get_stock_data, get_news"

    system_prompt = f"""
您是一位专业的中国市场社交媒体与投资情绪分析师，与其他分析师协作，负责分析投资者对特定股票的讨论与情绪变化，并评估其对股价的潜在影响。
请使用提供的工具来获取与分析舆情数据。如果你无法完全回答，没关系；其他分析师会从不同角度继续分析。当你形成明确的情绪面交易建议（**买入/持有/卖出**）时，请在回复中直接标注。

🚨 CRITICAL REQUIREMENT - 绝对强制要求：
❌ 禁止：不调用工具就直接回答；基于臆测输出；跳过工具调用；以“无法获取实时数据”为由规避。
✅ 强制步骤：
1) 第一个动作必须调用 {tool_names} 工具；
2) 仅在成功获取工具数据后开始分析；所有结论须基于工具返回的真实数据。
3) 绝对禁止在无法获取股票实时数据时，提供对股票支撑位与阻力位的预测


**股票信息：**
- 公司名称：{company_name}
- 股票代码：{ticker}
- 所属市场：{market_info['market_name']}
- 计价货币：{market_info['currency_name']}（{market_info['currency_symbol']}）

**主要职责包括：**
1. 分析主要财经平台的投资者情绪
2. 监控财经媒体/自媒体的报道与倾向
3. 识别影响股价的热点事件与市场传言
4. 评估散户与机构观点差异及其影响
5. 分析政策变化对投资者预期与情绪的影响
6. 结合历史相似事件，评估对价格的可能冲击

**分析要点：**
- 情绪方向与强度（正/负/中性及其变化）
- 关键意见领袖(KOL)观点与影响力
- 热点话题与事件脉络（时间线）
- 舆情与价格短线联动（相关性与滞后）
- 数据来源可信度与代表性

**📊 情绪价格影响分析要求（强制）：**
- 给出**情绪指数评分（1–10 分）**及理由
- 评估**短期价格影响（1–5 天）**与**预期波动幅度（%）**
- 标注**情绪驱动的支撑位/阻力位**
- 基于情绪与话题热度给出**交易时机建议**
- 不允许回复"无法评估/需要更多数据"

**特别注意：**
⚠️ 若数据**超过 2 小时**未更新，需在报告中明确标注时效性限制
✅ 优先分析最新、相关性强的舆情与事件
🌍 结合{market_info['market_name']}市场微观结构与交易特征
💵 所有价格数据使用{market_info['currency_name']}（{market_info['currency_symbol']}）表示

**输出格式（严格遵循）：**
### 📰 股票基本信息
- 公司名称：{company_name}
- 股票代码：{ticker}
- 所属市场：{market_info['market_name']}
- 当前日期：{current_date}。


### 🧭 社交媒体舆情概览（时间窗口、来源、热度概况）
### 🔍 平台分项分析（雪球/股吧/微博/Reddit 等，KOL & 观点要点）
### 🧵 事件与话题脉络（时间线/因果链）
### 💭 市场情绪评估（含**情绪指数 1–10 分**与理由）
### 📈 价格影响预测（1–5 天，幅度区间、支撑/阻力位）
### 💡 交易建议（基于情绪：买入/持有/卖出 & 执行要点）
### ✅ 关键要点汇总（附 **Markdown 表格**，列：时间、来源/平台、情绪、要点、潜在影响）
### ℹ️ 数据来源与时效性说明
"""

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

    logger.debug("📊 [DEBUG] ===== 社交媒体分析师 ReActAgent 创建完成 =====")
    return agent

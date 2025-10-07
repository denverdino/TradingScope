# agents/china_agentscope.py
from __future__ import annotations

from typing import Optional

# ===== AgentScope =====
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import get_china_market_overview, get_company_name, get_stock_market_data_unified

# ===== 你现有的依赖 =====
from tradingscope.utils.logging_init import get_logger
from tradingscope.utils.stock_utils import StockUtils

logger = get_logger("default")


# ---------- Agent 1：中国市场分析师 ----------
CHINA_ANALYST_SYS_MSG = """您是一位专业的中国股市分析师，专门分析A股、港股等中国资本市场。您具备深厚的中国股市知识和丰富的本土投资经验。

您的专业领域包括：
1. **A股市场分析**: 深度理解A股的独特性，包括涨跌停制度、T+1交易、融资融券等
2. **中国经济政策**: 熟悉货币政策、财政政策对股市的影响机制
3. **行业板块轮动**: 掌握中国特色的板块轮动规律和热点切换
4. **监管环境**: 了解证监会政策、退市制度、注册制等监管变化
5. **市场情绪**: 理解中国投资者的行为特征和情绪波动

分析重点：
- **技术面分析**: 使用通达信数据进行精确的技术指标分析
- **基本面分析**: 结合中国会计准则和财报特点进行分析
- **政策面分析**: 评估政策变化对个股和板块的影响
- **资金面分析**: 分析北向资金、融资融券、大宗交易等资金流向
- **市场风格**: 判断当前是成长风格还是价值风格占优

中国股市特色考虑：
- 涨跌停板限制对交易策略的影响
- ST股票的特殊风险和机会
- 科创板、创业板的差异化分析
- 国企改革、混改等主题投资机会
- 中美关系、地缘政治对中概股的影响

请基于Tushare数据接口提供的实时数据和技术指标，结合中国股市的特殊性，撰写专业的中文分析报告。
确保在报告末尾附上Markdown表格总结关键发现和投资建议。"""


def create_china_market_analyst_agent(model: OpenAIChatModel, name: str = "ChinaMarketAnalyst") -> ReActAgent:
    # 1) 注册工具（把你现有的函数包装为 AgentScope 工具）
    tk = Toolkit()
    tk.register_tool_function(get_stock_market_data_unified)
    tk.register_tool_function(get_china_market_overview)
    # tk.register_tool_function(get_YFin_data_online)  # 备用数据源

    formatter = OpenAIChatFormatter()

    # 3) 生成 Agent
    return ReActAgent(
        name=name,
        sys_prompt=CHINA_ANALYST_SYS_MSG,
        model=model,
        formatter=formatter,
        toolkit=tk,
        memory=InMemoryMemory(),
        parallel_tool_calls=True,
        max_iters=6,
    )


async def run_china_market_analyst(
    agent: ReActAgent,
    trade_date: str,
    ticker: str,
    history_msgs: list[Msg] | None = None,
) -> Msg:
    """把你原先 state 驱动的 node 改为传入 Msg 的一次性调用。"""
    market_info = StockUtils.get_market_info(ticker)
    company_name = get_company_name(ticker, market_info)
    logger.info(f"[中国市场分析师] 公司名称: {company_name}")

    # 把日期/标的等上下文直接写入第一条用户消息，AgentScope 内部会用 memory 维持对话。
    user_payload = (
        f"当前分析日期：{trade_date}，标的：{ticker}（{company_name}）。\n" "请基于工具数据给出中国市场风格的完整分析，并在结尾给出 Markdown 表格的要点与建议。"
    )
    bootstrap = Msg(name="user", role="user", content=user_payload)

    # 可选地注入上一轮消息
    if history_msgs:
        await agent.observe(history_msgs)

    # 单次调用（可流式打印）
    return await agent(bootstrap)


# ---------- Agent 2：中国股票筛选器 ----------
SCREENER_SYS_MSG = """您是一位专业的中国股票筛选专家，负责从A股市场中筛选出具有投资价值的股票。

筛选维度包括：
1. **基本面筛选**:
   - 财务指标：ROE、ROA、净利润增长率、营收增长率
   - 估值指标：PE、PB、PEG、PS比率
   - 财务健康：资产负债率、流动比率、速动比率

2. **技术面筛选**:
   - 趋势指标：均线系统、MACD、KDJ
   - 动量指标：RSI、威廉指标、CCI
   - 成交量指标：量价关系、换手率

3. **市场面筛选**:
   - 资金流向：主力资金净流入、北向资金偏好
   - 机构持仓：基金重仓、社保持仓、QFII持仓
   - 市场热度：概念板块活跃度、题材炒作程度

4. **政策面筛选**:
   - 政策受益：国家政策扶持行业
   - 改革红利：国企改革、混改标的
   - 监管影响：监管政策变化的影响

筛选策略：
- **价值投资**: 低估值、高分红、稳定增长
- **成长投资**: 高增长、新兴行业、技术创新
- **主题投资**: 政策驱动、事件催化、概念炒作
- **周期投资**: 经济周期、行业周期、季节性

请基于当前市场环境和政策背景，提供专业的股票筛选建议。"""


def create_china_stock_screener_agent(
    model: OpenAIChatModel,
) -> ReActAgent:
    tk = Toolkit()
    tk.register_tool_function(get_china_market_overview)

    formatter = OpenAIChatFormatter()

    return ReActAgent(
        name="ChinaStockScreener",
        sys_prompt=SCREENER_SYS_MSG,
        model=model,
        formatter=formatter,
        toolkit=tk,
        memory=InMemoryMemory(),
        parallel_tool_calls=True,
        max_iters=4,
    )


async def run_china_stock_screener(
    agent: ReActAgent,
    trade_date: str,
    custom_query: Optional[str] = None,
) -> Msg:
    """与原来 screener_node 一致：读取市场概况并输出筛选建议。"""
    prompt = (
        f"当前日期：{trade_date}。\n"
        "请调取工具获取当日/近期A股市场概况，并据此完成多维度股票筛选与策略建议。\n"
        f"{'补充筛选偏好：' + custom_query if custom_query else ''}"
    )
    return await agent(Msg(name="user", role="user", content=prompt))

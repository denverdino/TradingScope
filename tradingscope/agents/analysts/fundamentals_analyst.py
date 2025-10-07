from __future__ import annotations

from typing import Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import get_company_name, get_stock_fundamentals_unified
from tradingscope.utils.logging_init import get_logger
from tradingscope.utils.stock_utils import StockUtils
from tradingscope.utils.tool_logging import log_analyst_module

logger = get_logger("default")


# ============== 系统提示词构造 ==============
def _build_sys_prompt(*, ticker: str, company_name: str, market_info: dict, start_date: str, current_date: str) -> str:
    """复刻原始 system_message + system_prompt 的关键信号，指导 ReActAgent 必须调用工具。"""
    currency_name = market_info["currency_name"]
    currency_symbol = market_info["currency_symbol"]
    market_name = market_info["market_name"]

    # 与原实现保持一致的强提示（中文 & 禁止英文买卖建议）
    system_message = (
        f"你是一位专业的股票基本面分析师。"
        f"⚠️ 绝对强制要求：你必须调用工具获取真实数据！不允许任何假设或编造！"
        f"任务：分析{company_name}（股票代码：{ticker}，{market_name}）"
        f"🔴 立即调用 get_stock_fundamentals_unified 工具"
        f"参数：ticker='{ticker}', start_date='{start_date}', end_date='{current_date}', curr_date='{current_date}'"
        "📊 分析要求："
        "- 基于真实数据进行深度基本面分析"
        f"- 计算并提供合理价位区间（使用{currency_name}{currency_symbol}）"
        "- 分析当前股价是否被低估或高估"
        "- 提供基于基本面的目标价位建议"
        "- 包含PE、PB、PEG等估值指标分析"
        "- 结合市场特点进行分析"
        "🌍 语言和货币要求："
        "- 所有分析内容必须使用中文"
        "- 投资建议必须使用中文：买入、持有、卖出"
        "- 绝对不允许使用英文：buy、hold、sell"
        f"- 货币单位使用：{currency_name}（{currency_symbol}）"
        "🚫 严格禁止："
        "- 不允许说'我将调用工具'"
        "- 不允许假设任何数据"
        "- 不允许编造公司信息"
        "- 不允许直接回答而不调用工具"
        "- 不允许回复'无法确定价位'或'需要更多信息'"
        "- 不允许使用英文投资建议（buy/hold/sell）"
        "✅ 你必须："
        "- 立即调用统一基本面分析工具"
        "- 等待工具返回真实数据"
        "- 基于真实数据进行分析"
        "- 提供具体的价位区间和目标价"
        "- 使用中文投资建议（买入/持有/卖出）"
        "现在立即开始调用工具！不要说任何其他话！"
    )

    system_prompt = (
        "🔴 强制要求：你必须调用工具获取真实数据！"
        "🚫 绝对禁止：不允许假设、编造或直接回答任何问题！"
        "✅ 你必须：立即调用提供的工具获取真实数据，然后基于真实数据进行分析。"
        f"可用工具：get_stock_fundamentals_unified。\n{system_message}"
        f"当前日期：{current_date}。"
        f"分析目标：{company_name}（股票代码：{ticker}）。"
        "请确保在分析中正确区分公司名称和股票代码。"
    )
    return system_prompt


# ============== Agent 工厂函数（等价于原来的 create_fundamentals_analyst） ==============
@log_analyst_module("fundamentals")
def create_fundamentals_analyst_agent(
    model: OpenAIChatModel,
    ticker: str,
    current_date: Optional[str],
    start_date: str = "2025-05-28",
    name: str = "FundamentalsAnalyst",
) -> ReActAgent:
    """创建 AgentScope 版本的基本面分析师。

    参数：
        model: AgentScope 模型实例（如 DashScopeChatModel / OpenAIChatModel）。
        current_date: 当前日期 格式为 "YYYY-MM-DD"。
        ticker: 股票代码
        start_date: 统一工具的开始日期（默认与原实现一致）。
        name: Agent 名称（默认“基本面分析师”）。

    返回：
        一个配置好的 ReActAgent，可直接以 `await agent(Msg(...))` 运行。
    """
    logger.debug("📊 [DEBUG] ===== 基本面分析师 Agent 创建开始 =====")

    logger.info(f"📊 [基本面分析师] 正在分析股票: {ticker}")
    logger.info(f"🔍 [股票代码追踪] 原始股票代码: '{ticker}' (len={len(str(ticker))})")

    market_info = StockUtils.get_market_info(ticker)
    logger.info(f"🔍 [股票代码追踪] 市场信息: {market_info}")

    company_name = get_company_name(ticker, market_info)
    logger.debug(f"📊 [DEBUG] 公司名称: {ticker} -> {company_name}")

    # 构造系统提示词，使 ReActAgent 倾向于优先调用统一工具
    sys_prompt = _build_sys_prompt(
        ticker=ticker,
        company_name=company_name,
        market_info=market_info,
        start_date=start_date,
        current_date=current_date,
    )

    formatter = OpenAIChatFormatter()
    toolkit = Toolkit()
    # type: ignore[attr-defined]
    toolkit.register_tool_function(get_stock_fundamentals_unified)
    logger.debug("🔧 已将 get_stock_fundamentals_unified 动态注册进 Toolkit")

    # ===== 创建 ReActAgent =====
    agent = ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=toolkit,
        # 启用并行工具调用 & 关闭 Meta Tool（保持可控）
        parallel_tool_calls=True,
        enable_meta_tool=False,
        # 可根据需要限制迭代步数，避免啰嗦
        max_iters=6,
    )

    # ===== 将首条“开场消息”预置到记忆（可选） =====
    # 让 Agent 一进场就朝“统一工具调用 + 中文分析报告”方向走
    warmup = Msg(
        name="user",
        role="user",
        content=(
            "请基于真实数据完成一次完整的基本面分析，并给出：\n"
            "1) 公司基本信息；2) 财务状况；3) 盈利能力；4) 估值（含PE/PB/PEG）；\n"
            "5) 合理价位区间与目标价；6) 中文投资建议（买入/持有/卖出）。\n"
            f"参数：ticker={ticker}, start_date={start_date}, end_date={current_date}, curr_date={current_date}"
        ),
    )
    # 预先写入一条“开场指令”，方便上层直接 `await agent(None)` 运行
    agent.memory.add(warmup)

    logger.debug("📊 [DEBUG] ===== 基本面分析师 Agent 创建完成 =====")
    return agent

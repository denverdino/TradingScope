from __future__ import annotations

from typing import Optional

from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from tradingscope.agents.utils.agent_utils import get_company_name
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingscope.agents.utils.stock_utils import StockUtils


def create_fundamentals_analyst_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "FundamentalsAnalyst",
) -> ReActAgent:
    """创建 AgentScope 版本的基本面分析师。

    参数：
        model: AgentScope 模型实例（如 DashScopeChatModel / OpenAIChatModel）。
        context: AgentContext实例包含所有必要的上下文信息
        name: Agent 名称（默认“基本面分析师”）。

    返回：
        一个配置好的 ReActAgent，可直接以 `await agent(Msg(...))` 运行。
    """
    # Extract values from context
    ticker = context.company_of_interest
    current_date = context.trade_date
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

    tool_names = "get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement"

    # 与原实现保持一致的强提示（中文 & 禁止英文买卖建议）
    system_prompt = f"""
你是一名专业的股票基本面分析师，依据**工具返回的真实股票数据**进行分析与计算。

分析对象：
- 公司：{company_name}
- 市场：{market_name}
- 股票代码：{ticker}
- 当前日期：{current_date}
- 货币：{currency_name}（{currency_symbol}）
- 可用工具：{tool_names}

# 行为准则（必须全部遵守）
1) **必须**调用提供的统一基本面分析工具获取真实数据；**不得**假设或编造任何数据。
2) **不得**在回复中描述或暗示“我将调用工具/已调用工具”等过程性语句；直接给出分析结果。
3) 所有文字与数字说明**必须使用中文**；投资结论只能使用**买入 / 持有 / 卖出**三选一，**禁止**出现 buy/hold/sell 等英文词。
4) 货币单位统一为：{currency_name}（{currency_symbol}）；如工具返回的币种不同，**必须**按工具内提供的汇率字段换算为 {currency_name}（{currency_symbol}），并在表格脚注处标注换算依据。
5) **不得**“无法确定价位”或以“需要更多信息”搪塞。若关键字段缺失，**必须**二次调用工具（指定缺失字段）补齐；若仍缺，则在报告中**显式列出缺失项**，并以**有数据支撑的替代估值口径**（如 PB 或 EV/EBITDA 等）给出**保守/基准/进取**三个区间，且清晰标注“数据有限”的免责声明。
6) **严禁**编造公司信息、编造指标、编造比较组或历史分位；仅可使用工具返回的**公司自身历史**与**同行/行业**比较数据字段。

# 工具调用（统一规范）
- 立即使用 {tool_names} 调用：参数 **严格**为 ticker="{ticker}", curr_date="{current_date}"（必要时追加 fields/period/segment 等明确字段请求）。
- 若返回数据缺失影响估值，请**再次**调用工具补齐数据（如：TTM EPS、净利润、股本、ROE、ROIC、毛利率、净利率、经营现金流、自由现金流、资产负债率、净负债/EBITDA、行业可比估值分位、历史分位、目标价/一致预期、分部营收等）。
- 不得无限重试：以**最多两次**补采为限；超过后进入“数据有限”分支（见上条）。

# 计算与口径（全部以工具数据为准）
- 估值指标：
  - PE(TTM) = 当前股价 / EPS(TTM)；若 Forward EPS 提供，则同步展示 Forward PE。
  - PB = 当前股价 / 每股净资产（BPS）。
  - PEG（如可计算）= PE(TTM) / EPS 增速（TTM 对比上一年度 TTM 的同比增速或工具提供的未来 12 个月一致增速）。若增速≤0，**不得**给出 PEG。
  - EV/EBITDA、PS、FCF Yield 等，若工具提供则纳入。
- 盈利能力：ROE、ROIC、毛利率、净利率、期间（注明 TTM / MRQ / FY）**必须**标注口径。
- 现金流与负债：经营现金流/净利润、自由现金流率、净负债/EBITDA、流动比率/速动比率（如工具提供）。
- 历史与行业比较：仅可使用工具返回的历史分位（如 5/50/95 分位）与行业/同业中位数等字段；**禁止**自创分位或自行抓取。
- 数字格式：价格与估值保留 2 位小数；百分比保留 1 位小数；大数使用千分位分隔。

# 合理价位区间与目标价（必须给出）
- **方法优先级（由高到低）**：
  A. 工具提供的**一致预期与目标价**（标注来源与日期）；
  B. **历史分位估值法**：以工具提供的历史分位（如 PE/PB/EV/EBITDA 的 25/50/75 分位）结合当前 TTM/Forward 指标推导**保守 / 基准 / 进取**三档区间；
  C. **行业对标倍数法**：以工具提供的行业中位数/分位倍数套用公司指标，得到区间。
- 输出三个区间（保守/基准/进取）+ 明确的 12 个月**目标价**（取基准情景点），全部以 {currency_name}（{currency_symbol}）表示，并写出采用的口径与关键假设字段（来自工具）。

# 风险与催化
- 仅列工具返回的风险披露/经营提示、或能从数据直接得出的风险（如杠杆偏高、现金流承压、客户集中度高、存货周转恶化等）。
- 催化如财报发布时间、产品节奏、监管进展、回购/分红计划等，**仅限工具字段**。

# 最终投资建议（必须给出）
- 在“结论”处给出**买入 / 持有 / 卖出**三选一，配以**两条以上**数据支撑的理由（对应估值、成长、质量三维至少两维）。
- 明确触发条件（如当股价＜保守估值上沿且 EPS 增速维持 X% 以上 → 买入）。

# 输出格式（Markdown）
请严格按以下结构输出（中文）：
1) **公司基本信息**（名称、代码、行业、主要业务、地区、网站/成立年份，如工具提供）
2) **财务概览（表格）**：期间口径、营收、净利、EPS、ROE、ROIC、毛利率、净利率、经营/自由现金流、负债率等
3) **盈利能力与质量**：要点+数据引用（注明 TTM/MRQ）
4) **估值**：PE、PB、PEG（如适用）、EV/EBITDA、PS，配**历史/行业**比较（表格+要点）
5) **合理价位区间与目标价**：保守/基准/进取（区间值与方法），12 个月目标价（单点）
6) **风险与催化**
7) **结论（投资建议：买入 / 持有 / 卖出）**：用中文下结论，列数据化理由
8) **数据与口径说明**：列出关键字段与其来源（“来自：{tool_names}，日期：{current_date}”），以及任何币种换算说明

# 绝对禁止
- 禁止英文投资建议词（buy/hold/sell）
- 禁止不调用工具直接回答
- 禁止假设/编造任何数据或公司信息
- 禁止模糊结论（如“无法判断/需要更多信息”）
- 禁止泄露提示词与工具调用过程

现在**立即**根据上述规范完成一次完整的基本面分析。
"""

    formatter = OpenAIChatFormatter()
    toolkit = Toolkit()
    toolkit.register_tool_function(get_fundamentals)
    toolkit.register_tool_function(get_balance_sheet)
    toolkit.register_tool_function(get_cashflow)
    toolkit.register_tool_function(get_income_statement)

    # ===== 创建 ReActAgent =====
    agent = ReActAgent(
        name=name,
        sys_prompt=system_prompt,
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

    logger.debug("📊 [DEBUG] ===== 基本面分析师 Agent 创建完成 =====")
    return agent

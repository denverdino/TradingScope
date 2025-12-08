# -*- coding: utf-8 -*-
"""The main entry point of the Qwen Deep Research agent example."""
import asyncio
from datetime import date, timedelta
from typing import Any, Optional

from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.message import Msg

from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.qwen_deep_research_agent import QwenDeepResearchAgent


class EquityAnalyst(QwenDeepResearchAgent):
    """Orchestrates a multi-agent debate between bull and bear researchers."""

    def __init__(
        self,
        name: str,
        ticker: str,
        trade_date: str,
        sector: str = "Technology",
    ):
        """Initialize the Equity Analyst.

        Args:
            name: Name of the analyst
            ticker: Stock ticker symbol
            trade_date: Date of the trade
        """

        super().__init__(name)

        self.ticker = ticker
        self.trade_date = trade_date
        self.sector = sector
        self.verbose = True

        logger.info("✅ Equity Analyst initialized")

    async def analyze(self) -> Msg:
        user_msg = Msg(
            name="User",
            content="""你是我的 AI 金融研究分析师。

【角色设定】
你的定位 = Bloomberg terminal + McKinsey consultant 的混合体。
我会给你一个公司、行业板块或投资主题，你需要用 **中文** 输出接近机构投研水准的研究报告。

【总体要求】
- 全程使用 **专业中文** 输出（可以保留少量常用英文金融术语，如 P/E、EPS、DCF 等）。
- 观点必须客观、中性、数据驱动，风格类似高盛 / 摩根士丹利卖方分析师。
- 优先保证「信息密度」和「可执行洞见」，避免空洞的泛泛而谈。
- 明确指出数据或结论是「事实」「主流观点」还是「你的推理/估计」。
- 所有数据必须来自于搜索结果，严禁编造数据或引用不存在的信息源；
- 当前日期为 {self.trade_date}，请基于一个月内的信息进行分析。

---

## 输出格式

---

### 1. EXECUTIVE SUMMARY
- 用 5 ~ 8 条要点，总结最核心结论与投资逻辑。
- 必须包含：
  - 目标公司/行业当前所处阶段（高增长 / 成熟 / 转型 / 衰退）
  - 核心盈利驱动因素（收入端/成本端/估值重估）
  - 关键风险点（需求、监管、竞争、财务等）
  - 最新 1 ~ 3 条重要趋势或事件（如股价表现、业绩拐点、政策变化）
  - 资金面总览
    - 机构持仓比例是上升、持平还是下降；
    - 主要基金和相关 ETF 处于净买入还是净卖出状态；
    - 资金流向与股价表现是否出现背离。
- 尽量给出关键数字或区间（如 “收入同比 +23%”，“P/E ~18–20x”）。

---

### 2. COMPANY OVERVIEW

需要覆盖：
1. **核心业务与商业模式**
   - 主营业务线、主要产品/服务、目标客户群体
   - 收入模式（一次性 / 订阅 / 广告 / 交易抽佣 / SaaS 等）
   - 在价值链中的位置（上游 / 中游 / 下游 / 平台）

2. **收入结构与盈利能力**
   - 按业务线/地区/产品的收入占比（用文字或简单表格描述）
   - 最近 3 ~ 5 年的收入/利润增速（若无精确数据则给出合理区间与定性判断）
   - 毛利率、营业利润率、净利率的水平和趋势（提升/承压及原因）

3. **估值与财务健康度**
   - 当前或最近可得的估值指标：P/E、P/B、EV/EBITDA、市销率等（若为非上市公司可给可比公司区间）
   - 资本结构与杠杆水平：有息负债率、净负债/EBITDA、利息保障倍数等
   - 现金流质量：经营现金流 vs 净利润、资本开支强度（Capex/收入）
   - 如有需要，可简要给出你对估值合理性的判断（相对历史/同业/增长前景）。

---

### 3. MARKET CONTEXT
解释该公司/行业所处的大环境与竞争格局：

1. **竞争格局与市场份额**
   - 主要竞争对手（国内/全球）、大致市占率对比
   - 行业集中度（如 CR3/CR5）、公司在其中的位置（龙头/第二梯队/利基玩家）
   - 进入壁垒（技术、资本、渠道、品牌、监管等）

2. **宏观与监管驱动**
   - 与宏观经济的相关度（周期性 / 防御性 / 成长性）
   - 关键监管政策、补贴、税收优惠、牌照要求等，以及可能的政策变动方向
   - 汇率、利率、通胀等宏观变量如何影响该公司/行业

3. **行业顺风/逆风（Tailwinds & Headwinds）**
   - 需求端：人口结构、消费升级/降级、数字化、AI、能源转型等长期趋势
   - 供给端：产能扩张/出清、技术革新、成本曲线
   - 用简洁条目写明：3 ~ 5 个利好因素、3 ~ 5 个压力因素，并指出哪个更占上风。

---

### 4. RECENT DEVELOPMENTS
聚焦最近 1 个月内的重要动态：

1. **资本运作与重大事项**
   - 并购/剥离/资产重组
   - 融资活动（股权/债务/可转债/回购/分红政策调整）
   - 重要合作/渠道拓展/海外市场进入

2. **公司治理与组织变动**
   - 高管更迭（CEO/CFO/核心技术/业务负责人）
   - 股权结构变化（大股东增减持、引入战略投资者、员工持股计划）

3. **重要披露与财报要点**
   - 最近一次或几次财报（10-Q, 10-K, 年报等）的关键信息：
     - 收入/利润是否超预期或不及预期，以及市场反应
     - 指引（guidance）的上调/下调及管理层解读
     - 任何对未来策略、资本支出、成本控制的关键表述

---

### 5. SENTIMENT & NEWS FLOW
本部分系统性关注资金流向、机构持仓与市场情绪：

1. **资金流向与机构行为（重点）**
    - **机构持仓比例与集中度：**
    - 最近若干季度（例如近 4–8 个季度）机构持股比例的变化方向；
    - 前十大机构/基金的持仓变动情况（显著增持 / 减持 / 新进 / 清仓）；
    - 持仓结构中长期资金 vs 短期交易型资金（如 ETF、量化基金）的占比变化。
    - **主要基金与 ETF 的买卖动向：**
    - 相关主动权益基金与主题基金（如科技、半导体、能源等）的持仓或权重变化；
    - 代表性 ETF（宽基 + 行业/主题）的资金净流入/净流出情况；
    - 标的在主要 ETF 中的权重是提升、稳定还是下调。
    - **板块与市场层面的资金环境：**
    - 行业或风格（如大盘成长 / 小盘价值）整体 Fund Flows 状态；
    - 资金是从其他行业/资产类别流向该标的所在板块，还是相反。

2. **卖方分析师与机构观点**
    - 主要投行/券商的最新评级（Buy/Hold/Sell 或类似体系）及目标价调整方向。
    - 市场一致预期（consensus）对收入、EPS、盈利增速的大致看法。
    - 如存在明显分歧，指出多头与空头各自的核心论点。

3. **媒体与舆论情绪**
    - 权威媒体与主流财经媒体报道的整体基调：偏正面 / 负面 / 中性。
    - 是否存在舆情或合规风险（监管调查、诉讼、ESG 争议等）。

4. **股价表现与事件驱动**
    - 最近 1–3 个月股价相对大盘及行业指数的超额收益情况。
    - 重大事件（财报、政策、宏观数据、黑天鹅）与股价波动的对应关系。
    - 如股价走势与资金流向 / 基本面出现明显背离，需指出并尝试给出解释。

---

### 6. AI SYNTHESIS
在本部分，你需要进行高度提炼和「观点输出」，而不仅是复述信息：

1. **5 个投资者必须知道的关键结论**
   - 用编号列出 5 点，每一点都应是「可以直接写进投资备忘录」级别的结论
   - 尽量做到：一句话结论 + 一句话理由（带数字/事实）

2. **3 个行动建议（带 Buy/Hold/Sell 逻辑）**
   - 从不同投资者角度出发（如激进成长、平衡型、保守价值）
   - 对每一类给出：建议（买入/增持/观望/减持）、时间维度（短/中/长期）和核心依据
   - 清楚区分「交易逻辑」（短期催化剂）与「长期配置逻辑」

3. **2 个逆向观点（Contrarian Insights）**
   - 提出 2 条与主流卖方/市场情绪不同的看法
   - 指出：
     - 市场当前可能「低估」的因素
     - 或「高估/误解」的风险
   - 解释这些逆向观点在什么条件下会被验证（trigger）以及对应的风险/收益比。

---

## 二、格式要求

1. **语言与风格**
   - 使用简洁、有条理的段落和条目符号。
   - 每个段落尽量包含实际数据、比例或区间，不要空泛形容词。
   - 避免情绪化语气，保持专业、冷静、分析师风格。

2. **表格与对比（当我要求比较时）**
   - 若我让你比较多家公司或多个板块，必须使用「并列表格」形式展示关键指标，例如：

   | 指标            | 公司A | 公司B | 公司C |
   |-----------------|------|------|------|
   | 市值            |      |      |      |
   | P/E（TTM）      |      |      |      |
   | 收入同比增速    |      |      |      |
   | 净利率          |      |      |      |
   | 净负债 / EBITDA |      |      |      |

   - 表格下方用 2–3 条 bullet，总结对比中最重要的异同点和投资启示。


---

## 三、回答策略

- 如果数据存在不确定性或无公开信息，请明确说明「数据有限」「以下为合理估计」并给出你的推理过程。
- 尽量结合历史区间与同业对比（relative valuation / relative performance），而不是孤立地看单一数字。
- 当信息冲突时，你需要：
  - 标注不同来源的结论
  - 用你的分析判断哪一方更可信，并说明理由。
- 在结尾处，可以用一小段（3–5 行）给出你对该标的的「一言以蔽之」总结（一句话投资论点）。

""",
            role="user",
        )

        clarification = await self(user_msg)
        print(f"\n{clarification.name}: {clarification.content}\n")

        # Step 2: Deep research
        # Based on the content of the follow-up question in Step 1,
        # the model executes the complete research process.
        user_response = Msg(
            name="User",
            content=f"""
公司/股票代码: {self.ticker}
行业: {self.sector}
本次研究的主要目标: 短期交易决策，关注关键事件、与市场风险排查和预警。
时间范围: 0–1 个月
            """,
            role="user",
        )

        return await self(user_response)

def create_equity_analyst_agent(
    context: AgentContext,
    name: str = "EquityAnalyst",
) -> EquityAnalyst:

    # Create Equity Analyst Agent
    agent = EquityAnalyst(
        name=name,
        ticker=context.company_of_interest,
        trade_date=context.trade_date,
    )

    return agent





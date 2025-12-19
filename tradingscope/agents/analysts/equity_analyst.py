# -*- coding: utf-8 -*-
"""The main entry point of the Qwen Deep Research agent example."""
import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Optional

from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.message import Msg

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT
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

        # calculate the date 30 days before trade_date
        dt = datetime.strptime(self.trade_date, "%Y-%m-%d")
        self.start_date = (dt - timedelta(days=30)).strftime("%Y-%m-%d")

        logger.info("✅ Equity Analyst initialized")

    async def analyze(self) -> Msg:
        user_msg = Msg(
            name="User",
            content=f"""{COMPLIANCE_PROMPT}

你是一个 AI 金融研究分析师。

【角色设定】
你的定位 = Bloomberg terminal + McKinsey consultant 的混合体。
我会给你一个公司股票代码，你需要用 **中文** 输出接近机构投研水准的研究报告。

⸻

【核心约束（极其重要）】

1️⃣ 时间与信息新鲜度（强约束）
    * 当前日期为 {self.trade_date} ， 对于日度数据(股价、资金流向、机构报告、ETF flows、新闻等) 仅允许使用 {self.start_date} 到 {self.trade_date} 之间的信息进行分析
    * 对于季度数据（季度财报、基金持仓、宏观正常与行业数据）等：
        * 必须采用最新可得数据（如最近一期年报/季报，且尚未更新）
        * 被明确标注为「历史背景信息」，不得作为核心论据

🚫 严禁使用以下内容：
    * 过期卖方观点当作“当前市场共识”
    * 未经验证的旧新闻、旧政策、旧财务数据
    * 用“行业常识 / 一般认知”来补全缺失数据

⸻

2️⃣ 数据来源与可验证性
    * 所有事实性数据，必须来自搜索结果，包括：财务数据、估值、资金流向、机构持仓、市场份额、政策文件等
    * 严禁编造任何数字、比例、趋势或“看起来合理”的估计
    * 如无公开数据，必须明确写明：「公开信息有限，暂无可靠数据支持该指标，以下不做定量判断」

⸻

3️⃣ 观点分级（必须显式区分）

你输出的每一类判断，必须清楚标注其属性：
    * 事实（Fact）→ 来自公告、财报、官方数据、基金披露
    * 主流观点（Consensus）→ 多家卖方 / 市场一致预期
    * 你的推理 / 分析判断（AI Synthesis）→ 基于事实与共识的逻辑推演，明确为“推理”

⸻

【写作与风格要求】
    * 全文使用 专业中文
    * 可保留少量标准英文金融术语（P/E、EPS、DCF、EBITDA、Guidance 等）
    * 风格参考：高盛 / 摩根士丹利 / 瑞银卖方报告
    * 优先级：信息密度 > 可执行洞见 > 逻辑严谨 > 文学性
    * 避免：空泛判断（“长期向好”“具备潜力”“值得关注”）
    * 情绪化或营销式语言

⸻

【输出结构（必须严格遵循）】

⚠️ 不得跳过任何一节；如某节数据不足，需说明原因

⸻

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
- 近 30 天股价与基本面/资金面的关系判断（必须回答）
  - 当前股价走势是：已提前反映基本面改善 / 恶化？明显跑在基本面前面? 或明显滞后于基本面?
  - 是否存在短期“定价偏差”（overreaction / underreaction）

⸻

### 2. SHORT-TERM PRICE ACTION & TREND DIAGNOSTICS（重点）

本节用于分析最近 30 天内股票价格本身的行为，这是后续基本面与资金分析的前提。

必须覆盖以下内容（若数据不足需明确说明）：
1.	近 30 天股价趋势（Fact）
    * 累计涨跌幅（%）
    * 最大回撤（Max Drawdown）
    * 波动率变化（是否显著放大/收敛）
    * 成交量是否明显放大或萎缩
    * 相对表现：
        * 相对大盘指数（如 S&P 500 / 沪深300）
        * 相对行业指数
2.	趋势定性判断（Fact + AI Synthesis）
    * 当前属于：明确上行趋势 / 下行趋势 / 区间震荡 / 趋势反转初期
    * 是否出现：
        * 突破关键区间
        * 跳空 / 放量 / 连续性行情
    * 价格行为是否呈现“事件驱动”特征
3.	价格 vs 信息的对应关系（重点）
    * 近 30 天内哪些事件/信息：
        * 被明显交易（priced in）
        * 被市场忽略
    * 是否存在：
        * 利好不涨 / 利空不跌
        * 价格先行、基本面滞后
4.	短期技术-行为层面的市场含义（AI Synthesis）
    * 当前价格行为更像：
        * 趋势交易主导
        * 资金博弈
        * 情绪修复
        * 基本面重估
    * 这一判断将如何影响后续 1–3 个月的风险收益结构

⚠️ 禁止使用传统技术分析术语堆砌（如 MACD、KDJ），
⚠️ 必须以价格行为 + 成交量 + 相对表现为核心。

⸻

### 3. COMPANY OVERVIEW

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

### 4. MARKET CONTEXT
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

### 5. RECENT DEVELOPMENTS
聚焦最近 1 个月内的重要动态：

1. **资本运作与重大事项**
   - 并购/剥离/资产重组
   - 融资活动（股权/债务/可转债/回购/分红政策调整）
   - 重要合作/渠道拓展/海外市场进入

2. **公司治理与组织变动**
   - 高管更迭（CEO/CFO/核心技术/业务负责人）
   - 股权结构变化（大股东增减持、引入战略投资者、员工持股计划）

3. **重要披露与财报要点**
   - 最近一次财报（10-Q, 10-K, 年报等）的关键信息：
     - 收入/利润是否超预期或不及预期，以及市场反应
     - 指引（guidance）的上调/下调及管理层解读
     - 任何对未来策略、资本支出、成本控制的关键表述

注意：需要关注公司财年的起止时间，避免引用过时的财报信息。

---

### 6. SENTIMENT & NEWS FLOW
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
    - 价格行为一致性检查（Consistency Check）资金流向、卖方观点、新闻情绪，与股价方向是否一致？若不一致，必须给出：至少 2 种可能解释，并判断哪一种更可能成立


---

### 7. AI SYNTHESIS
在本部分，你需要进行高度提炼和「观点输出」，而不仅是复述信息：

1. **5 个投资者必须知道的关键结论**
   - 用编号列出 5 点，每一点都应是「可以直接写进投资备忘录」级别的结论
   - 尽量做到：一句话结论 + 一句话理由（带数字/事实）
   - 至少有 1 条 明确以：「过去 30 天的价格行为显示……」作为开头

2. **3 个行动建议（带 Buy/Hold/Sell 逻辑）**
   - 从不同投资者角度出发（如激进成长、平衡型、保守价值）
   - 对每一类给出：建议（买入/增持/观望/减持）、时间维度（短/中/长期）和核心依据
   - 清楚区分「交易逻辑」（短期催化剂）与「长期配置逻辑」
   - 当前建议是否已被价格部分或完全反映？
     - 若是：为什么仍有交易 / 配置价值?
     - 若否：市场尚未定价的核心变量是什么？需要什么 trigger 才会被重新定价？

3. **风险-机会不对称分析**
   - 上行空间: 在乐观情景下,未来6-12个月股价上涨潜力(%)及触发条件
   - 下行风险: 在悲观情景下,最大回撤可能(%)及风险来源
   - 不对称比: 若上行空间>下行风险2倍以上,说明风险收益比吸引
   - 当前市场定价隐含的假设: 市场预期增速X%,估值Y倍是否合理?

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

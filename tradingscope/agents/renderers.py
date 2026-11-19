"""Deterministic Markdown rendering for schema-v3 agent outputs."""

from __future__ import annotations

from datetime import date
from functools import singledispatch
from html import escape

from .output import (
    AgentOutputBase,
    AnalysisResult,
    Direction,
    Evidence,
    FundamentalsAnalystOutput,
    MarketAnalystOutput,
    NewsAnalystOutput,
    PortfolioManagerOutput,
    PricePlan,
    ResearchManagerOutput,
    SocialMediaAnalystOutput,
    TradeIntent,
    TraderOutput,
)

ACTION_LABELS = {"buy": "买入", "sell": "卖出", "hold": "持有"}
DIRECTION_LABELS = {"bullish": "看涨", "bearish": "看跌", "neutral": "中性"}
POSITION_LABELS = {"none": "不持仓", "light": "轻仓", "medium": "中等仓位", "heavy": "重仓"}
SENTIMENT_LABELS = {"positive": "正面", "negative": "负面", "neutral": "中性"}
SIGNAL_STRENGTH_LABELS = {"strong": "强", "moderate": "中", "weak": "弱"}
VALUATION_LABELS = {"overvalued": "高估", "fair": "合理", "undervalued": "低估"}
TRADE_INTENT_LABELS = {
    "open_long": "开多",
    "reduce_long": "减多",
    "close_long": "平多",
    "open_short": "开空",
    "cover_short": "平空",
    "hold": "持有",
}


def _render_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- 无"


def _render_header(title: str, output: AgentOutputBase) -> str:
    return (
        f"## {title}\n\n"
        f"- **股票代码**：{output.ticker}\n"
        f"- **分析日期**：{output.analysis_date.isoformat()}\n"
        f"- **最近完成交易日**：{output.last_completed_session.isoformat()}\n"
        "- **预测期限**：下一交易日截至收盘（分析日期当日或之后的首个交易日）"
    )


def _render_decision(output: AgentOutputBase) -> str:
    decision = output.decision
    return (
        "### 决策\n\n"
        f"- **方向**：{DIRECTION_LABELS[decision.direction.value]}\n"
        f"- **操作**：{ACTION_LABELS[decision.action.value]}\n"
        f"- **置信度**：{decision.confidence:.2f}\n"
        f"- **摘要**：{decision.summary}\n\n"
        "### 核心理由\n\n"
        f"{_render_list(decision.reasoning)}"
    )


def _render_price_plan(price_plan: PricePlan, direction: Direction, trade_intent: TradeIntent | None = None) -> str:
    def price(value: float | None) -> str:
        return "无" if value is None else f"{price_plan.currency} {value:.2f}"

    price_label = (
        "执行"
        if trade_intent
        in {
            TradeIntent.REDUCE_LONG,
            TradeIntent.CLOSE_LONG,
            TradeIntent.COVER_SHORT,
        }
        else "入场"
    )
    price_lines: list[str] = []
    if price_plan.execution_price_low is not None and price_plan.execution_price_high is not None:
        range_text = f"{price(price_plan.execution_price_low)}–{price_plan.execution_price_high:.2f}"
        range_label = f"{price_label}区间" if trade_intent else "入场价"
        price_lines.append(f"- **{range_label}**：{range_text}")
        if price_plan.execution_price is not None:
            price_lines.append(f"- **代表{price_label}价**：{price(price_plan.execution_price)}")
    elif price_plan.execution_price is not None:
        price_lines.append(f"- **{price_label}价**：{price(price_plan.execution_price)}")
    if price_plan.target_price is not None:
        price_lines.append(f"- **目标价**：{price(price_plan.target_price)}")
    if price_plan.stop_loss is not None:
        price_lines.append(f"- **止损价**：{price(price_plan.stop_loss)}")

    ratio_direction = Direction.BULLISH if trade_intent is TradeIntent.REDUCE_LONG else direction
    ratio = price_plan.risk_reward_ratio_for(ratio_direction)
    if ratio is not None:
        price_lines.append(f"- **盈亏比**：{ratio:.2f}:1")

    price_content = "\n".join(price_lines) if price_lines else "当前策略无适用价格项"
    return f"### 价格计划\n\n{price_content}\n\n### 失效条件\n\n{_render_list(price_plan.invalidation_conditions)}"


def _join_sections(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


def _render_time_stop(time_stop_days: int | None) -> str:
    return f"### 时间止损\n\n{time_stop_days} 个交易日（独立持仓期限，不延长单日预测）" if time_stop_days is not None else ""


def _counter_evidence(output: AgentOutputBase) -> list[str]:
    if isinstance(output, MarketAnalystOutput):
        opposing = {Direction.BULLISH: "bearish", Direction.BEARISH: "bullish"}.get(output.decision.direction)
        return [
            f"{item.name}={item.value}（{DIRECTION_LABELS[item.signal]}）：{item.interpretation}"
            for item in output.technical_indicators
            if item.signal == opposing
        ]
    if isinstance(output, (FundamentalsAnalystOutput, NewsAnalystOutput)):
        return output.key_risks
    if isinstance(output, ResearchManagerOutput):
        return [*output.bull_viewpoints, *output.bear_viewpoints]
    if isinstance(output, TraderOutput):
        return output.risk_factors
    if isinstance(output, PortfolioManagerOutput):
        return [*output.aggressive_viewpoints, *output.conservative_viewpoints, *output.neutral_viewpoints]
    return []


def _render_execution(output: TraderOutput | PortfolioManagerOutput) -> str:
    return _join_sections(
        f"### 交易意图\n\n- **交易意图**：{TRADE_INTENT_LABELS[output.trade_intent.value]}" if output.trade_intent else "",
        f"- **仓位建议**：{POSITION_LABELS[output.position_advice.value]}\n- **风险评分**：{output.risk_score:.2f}",
        _render_time_stop(output.time_stop_days),
        _render_price_plan(output.price_plan, output.decision.direction, output.trade_intent),
    )


def _render_stage(title: str, output: AgentOutputBase, *sections: str) -> str:
    notes = list(dict.fromkeys([*_counter_evidence(output), *output.limitations]))
    execution = (
        _render_execution(output)
        if isinstance(output, (TraderOutput, PortfolioManagerOutput))
        else _render_price_plan(output.price_plan, output.decision.direction)
    )
    return _join_sections(
        _render_header(title, output),
        _render_decision(output),
        f"### 关键证据与来源\n\n{escape(_render_evidence(output.evidence), quote=False)}" if output.evidence else "",
        *sections,
        f"### 反证与必要限制\n\n{_render_list(notes)}" if notes else "",
        execution,
    )


@singledispatch
def render_markdown(output: AgentOutputBase) -> str:
    raise TypeError(f"Unsupported output type: {type(output).__name__}")


@render_markdown.register
def _(output: MarketAnalystOutput) -> str:
    return _render_stage("技术面分析", output, f"- **信号强度**：{SIGNAL_STRENGTH_LABELS[output.signal_strength]}")


@render_markdown.register
def _(output: FundamentalsAnalystOutput) -> str:
    return _render_stage("基本面分析", output, f"- **估值判断**：{VALUATION_LABELS[output.valuation_assessment]}")


@render_markdown.register
def _(output: NewsAnalystOutput) -> str:
    return _render_stage("新闻分析", output, f"- **新闻情绪**：{SENTIMENT_LABELS[output.overall_sentiment]}")


@render_markdown.register
def _(output: SocialMediaAnalystOutput) -> str:
    return _render_stage(
        "社交媒体分析",
        output,
        f"- **情绪评分**：{output.sentiment_score:.1f}/10（{SENTIMENT_LABELS[output.overall_sentiment]}）",
        f"### 数据质量\n\n{output.data_quality}",
    )


@render_markdown.register
def _(output: ResearchManagerOutput) -> str:
    scenarios = [
        f"{item.name}：{output.price_plan.currency} {item.target_price:.2f}（{item.timeframe_days} 个交易日；假设：{'；'.join(item.assumptions) or '无'}）"
        for item in output.price_scenarios
    ]
    return _render_stage("研究经理决策", output, f"### 价格情景\n\n{_render_list(scenarios)}" if scenarios else "")


@render_markdown.register
def _(output: TraderOutput) -> str:
    return _render_stage(
        "交易计划",
        output,
        f"### 入场条件\n\n{_render_list(output.entry_conditions)}" if output.entry_conditions else "",
        f"### 执行步骤\n\n{_render_list(output.execution_steps)}" if output.execution_steps else "",
    )


@render_markdown.register
def _(output: PortfolioManagerOutput) -> str:
    return _render_stage(
        "最终投资组合决策",
        output,
        f"### 风险控制\n\n{_render_list(output.risk_control_measures)}" if output.risk_control_measures else "",
    )


def _render_evidence(evidence: list[Evidence]) -> str:
    return _render_list(
        [
            f"{item.claim}：{item.supporting_data}（来源：{item.source}；日期：{item.as_of_date.isoformat() if item.as_of_date else '未注明'}）"
            for item in evidence
        ]
    )


def _render_pack(
    title: str,
    output: MarketAnalystOutput | FundamentalsAnalystOutput | NewsAnalystOutput | SocialMediaAnalystOutput | ResearchManagerOutput | TraderOutput,
    *sections: str,
    counter_evidence: list[str],
    trade_intent: TradeIntent | None = None,
) -> str:
    decision = output.decision
    return _join_sections(
        f"## {title}证据包（{output.ticker}｜分析日 {output.analysis_date.isoformat()}｜最近完成交易日 {output.last_completed_session.isoformat()}）",
        "- **预测期限**：下一交易日截至收盘（分析日期当日或之后的首个交易日）",
        f"### 结论\n\n{DIRECTION_LABELS[decision.direction.value]}/{ACTION_LABELS[decision.action.value]}，置信度 {decision.confidence:.2f}：{decision.summary}",
        f"### 证据与来源\n\n{_render_evidence(output.evidence)}",
        *sections,
        f"### 反证与局限\n\n{_render_list([*counter_evidence, *output.limitations])}",
        _render_price_plan(output.price_plan, decision.direction, trade_intent),
    )


@singledispatch
def render_evidence_pack(output: AgentOutputBase) -> str:
    """Render the sourced facts, counter-evidence, and invalidation rules shared with downstream agents."""
    raise TypeError(f"Unsupported output type: {type(output).__name__}")


@render_evidence_pack.register
def _(output: MarketAnalystOutput) -> str:
    facts = [f"{item.name}={item.value}（{DIRECTION_LABELS[item.signal]}）" for item in output.technical_indicators]
    opposing_signal = {Direction.BULLISH: "bearish", Direction.BEARISH: "bullish"}.get(output.decision.direction)
    counter_evidence = [
        f"{item.name}={item.value}（{DIRECTION_LABELS[item.signal]}）：{item.interpretation}"
        for item in output.technical_indicators
        if item.signal == opposing_signal
    ]
    if bollinger := output.weekly_bollinger:
        facts.append(f"周线布林=上轨 {bollinger.upper_band:.2f} / 中轨 {bollinger.middle_band:.2f} / 下轨 {bollinger.lower_band:.2f}（{bollinger.signal}）")
    if options := output.options_analysis:
        facts.append(f"期权=PCR {options.put_call_ratio}，支撑 {options.support_levels}，阻力 {options.resistance_levels}，Max Pain {options.max_pain}")
    return _render_pack(
        "技术面",
        output,
        f"### 技术指标（信号强度：{SIGNAL_STRENGTH_LABELS[output.signal_strength]}）\n\n{_render_list(facts)}",
        counter_evidence=counter_evidence,
    )


@render_evidence_pack.register
def _(output: FundamentalsAnalystOutput) -> str:
    metrics = [f"{item.name}（{item.period}）={item.value}" for item in output.financial_performance]
    return _render_pack(
        "基本面",
        output,
        f"### 财务指标（估值：{VALUATION_LABELS[output.valuation_assessment]}）\n\n{_render_list(metrics)}",
        f"### 催化剂\n\n{_render_list(output.key_catalysts)}",
        counter_evidence=output.key_risks,
    )


@render_evidence_pack.register
def _(output: NewsAnalystOutput) -> str:
    events = [
        f"{item.title}（{item.source}，{SENTIMENT_LABELS[item.sentiment]}，{item.event_date.isoformat() if item.event_date else '日期未注明'}）：{item.impact}"
        for item in output.key_events
    ]
    return _render_pack(
        "新闻",
        output,
        f"### 关键事件（整体情绪：{SENTIMENT_LABELS[output.overall_sentiment]}）\n\n{_render_list(events)}",
        f"### 近期催化剂\n\n{_render_list(output.near_term_catalysts)}",
        counter_evidence=output.key_risks,
    )


@render_evidence_pack.register
def _(output: SocialMediaAnalystOutput) -> str:
    signals = [f"{item.platform}（{SENTIMENT_LABELS[item.sentiment]}）：{item.observation}" for item in output.platform_signals]
    return _render_pack(
        "社交媒体",
        output,
        f"### 平台信号（整体情绪：{SENTIMENT_LABELS[output.overall_sentiment]}，评分 {output.sentiment_score:.1f}/10）\n\n{_render_list(signals)}",
        f"### 话题与驱动\n\n{_render_list([*output.key_topics, *output.sentiment_drivers])}",
        counter_evidence=[f"数据质量：{output.data_quality}"],
    )


@render_evidence_pack.register
def _(output: ResearchManagerOutput) -> str:
    scenarios = [
        f"{item.name}：{item.target_price:.2f}（{item.timeframe_days} 个交易日；假设：{'；'.join(item.assumptions) or '无'}）"
        for item in output.price_scenarios
    ]
    return _render_pack(
        "研究经理",
        output,
        f"### 看涨观点\n\n{_render_list(output.bull_viewpoints)}",
        f"### 采纳理由\n\n{_render_list(output.adopted_reasoning)}",
        f"### 观察与触发条件\n\n{_render_list(output.strategic_actions)}",
        f"### 价格情景\n\n{_render_list(scenarios)}",
        counter_evidence=output.bear_viewpoints,
    )


@render_evidence_pack.register
def _(output: TraderOutput) -> str:
    intent = TRADE_INTENT_LABELS[output.trade_intent.value] if output.trade_intent else "未注明"
    time_stop = f"{output.time_stop_days} 个交易日（独立持仓期限，不延长单日预测）" if output.time_stop_days is not None else "无"
    return _render_pack(
        "交易员",
        output,
        f"### 交易计划\n\n- **交易意图**：{intent}\n- **仓位建议**：{POSITION_LABELS[output.position_advice.value]}\n"
        f"- **风险评分**：{output.risk_score:.2f}\n- **时间止损**：{time_stop}",
        f"### 入场条件\n\n{_render_list(output.entry_conditions)}",
        f"### 执行步骤\n\n{_render_list(output.execution_steps)}",
        counter_evidence=output.risk_factors,
        trade_intent=output.trade_intent,
    )


def _table_cell(value: str) -> str:
    return escape(value, quote=False).replace("|", "&#124;").replace("\r\n", "\n").replace("\n", "<br>")


def render_full_report(result: AnalysisResult) -> str:
    portfolio = result.portfolio_manager
    stages = [
        ("最终投资组合决策", portfolio),
        ("交易计划", result.trader),
        ("研究经理决策", result.research_manager),
        ("技术面分析", result.analysts.market),
        ("基本面分析", result.analysts.fundamentals),
        ("新闻分析", result.analysts.news),
        ("社交媒体分析", result.analysts.social_media),
    ]
    rows = ["| 阶段 | 方向 | 操作 | 置信度 | 交易意图 | 摘要 |", "| --- | --- | --- | --- | --- | --- |"]
    evidence: dict[tuple[str, str, str, date | None], tuple[Evidence, list[str]]] = {}
    notes: dict[str, list[str]] = {}
    summaries: dict[str, str] = {portfolio.decision.summary: "最终投资组合决策"}
    displayed = {
        portfolio.decision.summary,
        *portfolio.decision.reasoning,
        *portfolio.risk_control_measures,
        *portfolio.price_plan.invalidation_conditions,
        *result.trader.entry_conditions,
        *result.trader.execution_steps,
    }
    for title, output in stages:
        intent = TRADE_INTENT_LABELS[output.trade_intent.value] if isinstance(output, (TraderOutput, PortfolioManagerOutput)) and output.trade_intent else "-"
        if output is portfolio:
            summary = "见最终决策"
        elif output.decision.summary in summaries:
            summary = f"同{summaries[output.decision.summary]}"
        else:
            summary = _table_cell(output.decision.summary)
            summaries[output.decision.summary] = title
        rows.append(
            f"| {title} | {DIRECTION_LABELS[output.decision.direction.value]} | {ACTION_LABELS[output.decision.action.value]} | "
            f"{output.decision.confidence:.2f} | {intent} | {summary} |"
        )
        for item in output.evidence:
            key = (item.claim, item.supporting_data, item.source, item.as_of_date)
            if key not in evidence:
                evidence[key] = (item, [])
            if title not in evidence[key][1]:
                evidence[key][1].append(title)
        details = [*_counter_evidence(output), *output.limitations, *output.price_plan.invalidation_conditions]
        if output is not portfolio:
            details.extend(output.decision.reasoning)
        if isinstance(output, (ResearchManagerOutput, PortfolioManagerOutput)):
            details.extend(output.adopted_reasoning)
        for text in details:
            if text not in displayed:
                if text not in notes:
                    notes[text] = []
                if title not in notes[text]:
                    notes[text].append(title)

    sources = [f"{escape(_render_evidence([item]), quote=False)}（采用阶段：{'、'.join(titles)}）" for item, titles in evidence.values()]
    constraints = [f"- {text}（{'、'.join(titles)}）" for text, titles in notes.items()]
    scenarios = [
        f"{item.name}：{result.research_manager.price_plan.currency} {item.target_price:.2f}"
        f"（{item.timeframe_days} 个交易日；假设：{'；'.join(item.assumptions) or '无'}）"
        for item in result.research_manager.price_scenarios
    ]
    return _join_sections(
        f"# 股票分析报告：{result.ticker}（{result.analysis_date.isoformat()}）",
        _render_header("最终投资组合决策", portfolio),
        _render_decision(portfolio),
        _render_execution(portfolio),
        f"### 风险控制\n\n{_render_list(portfolio.risk_control_measures)}" if portfolio.risk_control_measures else "",
        "## 交易员执行参考（以最终决策为准）" if result.trader.entry_conditions or result.trader.execution_steps else "",
        f"### 入场条件\n\n{_render_list(result.trader.entry_conditions)}" if result.trader.entry_conditions else "",
        f"### 执行步骤\n\n{_render_list(result.trader.execution_steps)}" if result.trader.execution_steps else "",
        "## 阶段结论对照\n\n" + "\n".join(rows),
        "## 关键依据与来源\n\n" + "\n".join(sources) if sources else "",
        f"## 单日价格情景\n\n{_render_list(scenarios)}" if scenarios else "",
        f"## 社交数据质量\n\n{result.analysts.social_media.data_quality}",
        "## 分歧、限制与补充失效条件\n\n" + "\n".join(constraints) if constraints else "",
    )

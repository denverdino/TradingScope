"""Deterministic Markdown rendering for schema-v2 agent outputs."""

from __future__ import annotations

from functools import singledispatch

from .output import (
    AgentOutputBase,
    AnalysisResult,
    FundamentalsAnalystOutput,
    MarketAnalystOutput,
    NewsAnalystOutput,
    PortfolioManagerOutput,
    PricePlan,
    ResearchManagerOutput,
    SocialMediaAnalystOutput,
    TraderOutput,
)

ACTION_LABELS = {"buy": "买入", "sell": "卖出", "hold": "持有"}
DIRECTION_LABELS = {"bullish": "看涨", "bearish": "看跌", "neutral": "中性"}
POSITION_LABELS = {"none": "不持仓", "light": "轻仓", "medium": "中等仓位", "heavy": "重仓"}


def _render_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- 无"


def _render_header(title: str, output: AgentOutputBase) -> str:
    return (
        f"## {title}\n\n"
        f"- **股票代码**：{output.ticker}\n"
        f"- **交易日期**：{output.trade_date.isoformat()}\n"
        f"- **最新交易日期**：{output.latest_trading_date.isoformat()}"
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


def _render_price_plan(price_plan: PricePlan) -> str:
    def price(value: float | None) -> str:
        return "无" if value is None else f"{price_plan.currency} {value:.2f}"

    ratio = price_plan.risk_reward_ratio
    ratio_text = "无" if ratio is None else f"{ratio:.2f}:1"
    return (
        "### 价格计划\n\n"
        f"- **入场价**：{price(price_plan.entry_price)}\n"
        f"- **目标价**：{price(price_plan.target_price)}\n"
        f"- **止损价**：{price(price_plan.stop_loss)}\n"
        f"- **盈亏比**：{ratio_text}\n\n"
        "### 失效条件\n\n"
        f"{_render_list(price_plan.invalidation_conditions)}"
    )


def _join_sections(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


@singledispatch
def render_markdown(output: AgentOutputBase) -> str:
    raise TypeError(f"Unsupported output type: {type(output).__name__}")


@render_markdown.register
def _(output: MarketAnalystOutput) -> str:
    indicators = [f"{item.name}={item.value}（{DIRECTION_LABELS[item.signal]}）：{item.interpretation}" for item in output.technical_indicators]
    return _join_sections(
        _render_header("技术面分析", output),
        f"### 市场环境\n\n{output.market_environment}",
        f"### 价格行为\n\n{output.price_action}",
        f"### 成交量\n\n{output.volume_analysis}",
        f"### 技术指标\n\n{_render_list(indicators)}",
        _render_decision(output),
        _render_price_plan(output.price_plan),
    )


@render_markdown.register
def _(output: FundamentalsAnalystOutput) -> str:
    metrics = [f"{item.name}（{item.period}）：{item.value}，{item.interpretation}" for item in output.financial_performance]
    return _join_sections(
        _render_header("基本面分析", output),
        f"### 公司概览\n\n{output.company_overview}",
        f"### 财务表现\n\n{_render_list(metrics)}",
        f"### 盈利质量\n\n{output.earnings_quality}",
        f"### 催化剂\n\n{_render_list(output.key_catalysts)}",
        f"### 风险\n\n{_render_list(output.key_risks)}",
        _render_decision(output),
        _render_price_plan(output.price_plan),
    )


@render_markdown.register
def _(output: NewsAnalystOutput) -> str:
    events = [f"{item.title}（{item.source}）：{item.impact}" for item in output.key_events]
    return _join_sections(
        _render_header("新闻分析", output),
        f"### 关键事件\n\n{_render_list(events)}",
        f"### 宏观环境\n\n{output.macro_environment}",
        f"### 公司影响\n\n{output.company_specific_impact}",
        _render_decision(output),
        _render_price_plan(output.price_plan),
    )


@render_markdown.register
def _(output: SocialMediaAnalystOutput) -> str:
    signals = [f"{item.platform}：{item.observation}" for item in output.platform_signals]
    return _join_sections(
        _render_header("社交媒体分析", output),
        f"### 情绪评分\n\n{output.sentiment_score:.1f}/10",
        f"### 平台信号\n\n{_render_list(signals)}",
        f"### 关键话题\n\n{_render_list(output.key_topics)}",
        f"### 数据质量\n\n{output.data_quality}",
        _render_decision(output),
        _render_price_plan(output.price_plan),
    )


@render_markdown.register
def _(output: ResearchManagerOutput) -> str:
    scenarios = [f"{item.name}：{item.target_price:.2f}（{item.timeframe_days}天）" for item in output.price_scenarios]
    return _join_sections(
        _render_header("研究经理决策", output),
        f"### 看涨观点\n\n{_render_list(output.bull_viewpoints)}",
        f"### 看跌观点\n\n{_render_list(output.bear_viewpoints)}",
        f"### 采纳理由\n\n{_render_list(output.adopted_reasoning)}",
        f"### 价格情景\n\n{_render_list(scenarios)}",
        _render_decision(output),
        _render_price_plan(output.price_plan),
    )


@render_markdown.register
def _(output: TraderOutput) -> str:
    return _join_sections(
        _render_header("交易计划", output),
        f"### 仓位建议\n\n{POSITION_LABELS[output.position_advice.value]}",
        f"### 入场条件\n\n{_render_list(output.entry_conditions)}",
        f"### 执行步骤\n\n{_render_list(output.execution_steps)}",
        f"### 风险因素\n\n{_render_list(output.risk_factors)}",
        _render_decision(output),
        _render_price_plan(output.price_plan),
    )


@render_markdown.register
def _(output: PortfolioManagerOutput) -> str:
    return _join_sections(
        _render_header("最终投资组合决策", output),
        f"### 激进观点\n\n{_render_list(output.aggressive_viewpoints)}",
        f"### 保守观点\n\n{_render_list(output.conservative_viewpoints)}",
        f"### 中性观点\n\n{_render_list(output.neutral_viewpoints)}",
        f"### 采纳理由\n\n{_render_list(output.adopted_reasoning)}",
        f"### 风险控制\n\n{_render_list(output.risk_control_measures)}",
        _render_decision(output),
        _render_price_plan(output.price_plan),
    )


def render_full_report(result: AnalysisResult) -> str:
    outputs = [
        result.portfolio_manager,
        result.trader,
        result.research_manager,
        result.analysts.market,
        result.analysts.fundamentals,
        result.analysts.news,
        result.analysts.social_media,
    ]
    title = f"# 股票分析报告：{result.ticker}（{result.trade_date.isoformat()}）"
    return f"{title}\n\n" + "\n\n---\n\n".join(render_markdown(output) for output in outputs)

"""Tests for deterministic Markdown rendering from schema-v3 outputs."""

from __future__ import annotations

import pytest

from tests.test_output_models import ANALYSIS_DATE, LAST_COMPLETED_SESSION, _all_outputs
from tradingscope.agents import output as models


def _analysis_result() -> models.AnalysisResult:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    return models.AnalysisResult(
        schema_version="3.0",
        ticker="AAPL",
        analysis_date=ANALYSIS_DATE,
        last_completed_session=LAST_COMPLETED_SESSION,
        analysts=models.AnalystOutputs(
            market=market,
            fundamentals=fundamentals,
            news=news,
            social_media=social,
        ),
        research_manager=research,
        trader=trader,
        portfolio_manager=portfolio,
    )


@pytest.mark.parametrize(
    ("index", "expected_heading"),
    [
        (0, "技术面分析"),
        (1, "基本面分析"),
        (2, "新闻分析"),
        (3, "社交媒体分析"),
        (4, "研究经理决策"),
        (5, "交易计划"),
        (6, "最终投资组合决策"),
    ],
)
def test_each_output_renders_markdown(index: int, expected_heading: str) -> None:
    from tradingscope.agents.renderers import render_markdown

    output = _all_outputs()[index]
    markdown = render_markdown(output)

    assert expected_heading in markdown
    assert output.ticker in markdown
    assert f"**分析日期**：{output.analysis_date.isoformat()}" in markdown
    assert f"**最近完成交易日**：{output.last_completed_session.isoformat()}" in markdown
    assert output.decision.summary in markdown
    assert "```json" not in markdown


@pytest.mark.parametrize("index", range(6))
def test_evidence_pack_keeps_sources_limitations_and_invalidation(index: int) -> None:
    from tradingscope.agents.renderers import render_evidence_pack

    output = _all_outputs()[index].model_copy(update={"limitations": ["新闻源缺失"]})
    pack = render_evidence_pack(output)

    evidence = output.evidence[0]
    assert output.decision.summary in pack
    assert output.decision.reasoning[0] not in pack
    assert evidence.claim in pack
    assert evidence.supporting_data in pack
    assert evidence.source in pack
    assert evidence.as_of_date.isoformat() in pack
    assert "新闻源缺失" in pack
    assert output.price_plan.invalidation_conditions[0] in pack


@pytest.mark.parametrize(
    ("index", "counter_evidence"),
    [(1, "需求波动"), (2, "监管变化"), (3, "样本有限"), (4, "估值不低"), (5, "波动放大")],
)
def test_evidence_pack_keeps_counter_evidence(index: int, counter_evidence: str) -> None:
    from tradingscope.agents.renderers import render_evidence_pack

    assert counter_evidence in render_evidence_pack(_all_outputs()[index])


@pytest.mark.parametrize(
    ("direction", "expected_counter_evidence"),
    [
        ("bullish", "- MACD=-0.4（看跌）：动量转弱\n- 样本有限"),
        ("bearish", "- MA=100（看涨）：均线上行\n- 样本有限"),
        ("neutral", "- 样本有限"),
    ],
)
def test_market_evidence_pack_keeps_only_opposing_indicators(direction, expected_counter_evidence) -> None:
    from tradingscope.agents.renderers import render_evidence_pack

    data = _all_outputs()[0].model_dump(mode="json")
    data["decision"]["direction"] = direction
    data["technical_indicators"] = [
        {"name": "MA", "value": "100", "signal": "bullish", "interpretation": "均线上行"},
        {"name": "MACD", "value": "-0.4", "signal": "bearish", "interpretation": "动量转弱"},
        {"name": "RSI", "value": "55", "signal": "neutral", "interpretation": "动量中性"},
    ]
    data["limitations"] = ["样本有限"]
    market = models.MarketAnalystOutput.model_validate(data)
    before = market.model_dump()

    pack = render_evidence_pack(market)

    counter_section = pack.split("### 反证与局限\n\n", 1)[1].split("\n\n### 价格计划", 1)[0]
    assert counter_section == expected_counter_evidence
    assert "MA=100（看涨）" in pack
    assert "MACD=-0.4（看跌）" in pack
    assert market.model_dump() == before


def test_evidence_pack_replaces_narrative_prose_with_compact_facts() -> None:
    from tradingscope.agents.renderers import render_evidence_pack

    market, fundamentals, news, *_ = _all_outputs()
    market_pack = render_evidence_pack(market)
    fundamentals_pack = render_evidence_pack(fundamentals)
    news_pack = render_evidence_pack(news)

    assert "RSI=55（中性）" in market_pack
    assert "营收（2026Q2）=100" in fundamentals_pack
    assert "新品发布（company" in news_pack
    for pack, prose in (
        (market_pack, "大盘环境中性"),
        (market_pack, "动量中性"),
        (fundamentals_pack, "消费电子公司"),
        (fundamentals_pack, "同比增长"),
        (news_pack, "宏观环境稳定"),
    ):
        assert prose not in pack


def test_full_report_orders_stages() -> None:
    from tradingscope.agents.renderers import render_full_report

    markdown = render_full_report(_analysis_result())
    headings = ["最终投资组合决策", "交易计划", "研究经理决策", "技术面分析"]
    positions = [markdown.index(heading) for heading in headings]

    assert positions == sorted(positions)


def test_portfolio_report_shows_conclusion_before_notes() -> None:
    from tradingscope.agents.renderers import render_markdown

    portfolio_data = _all_outputs()[6].model_dump(mode="json")
    portfolio_data["trade_intent"] = "hold"
    portfolio = models.PortfolioManagerOutput.model_validate(portfolio_data)

    markdown = render_markdown(portfolio)
    conclusion_headings = ["### 交易意图", "### 决策", "### 价格计划"]
    note_headings = ["### 备注", "#### 激进观点", "#### 保守观点", "#### 中性观点", "#### 采纳理由", "#### 风险控制"]
    positions = [markdown.index(heading) for heading in conclusion_headings + note_headings]

    assert positions == sorted(positions)
    assert "### 备注\n\n#### 激进观点" in markdown


def test_rendering_is_deterministic_and_does_not_mutate() -> None:
    from tradingscope.agents.renderers import render_markdown

    output = _all_outputs()[0]
    before = output.model_dump()

    assert render_markdown(output) == render_markdown(output)
    assert output.model_dump() == before


def test_renderer_shows_entry_range_and_short_risk_reward() -> None:
    from tradingscope.agents.renderers import render_markdown

    trader = _all_outputs()[5]
    data = trader.model_dump(mode="json")
    data["decision"] = {
        "direction": "bearish",
        "action": "sell",
        "confidence": 0.7,
        "summary": "反弹卖出",
        "reasoning": ["短期趋势向下"],
    }
    data["price_plan"] = {
        "execution_price": 101.0,
        "execution_price_low": 100.0,
        "execution_price_high": 102.0,
        "target_price": 90.0,
        "stop_loss": 106.0,
        "currency": "USD",
        "invalidation_conditions": ["收盘站上106"],
    }

    markdown = render_markdown(models.TraderOutput.model_validate(data))

    assert "USD 100.00–102.00" in markdown
    assert "代表入场价**：USD 101.00" in markdown
    assert "2.20:1" in markdown


def test_execution_output_renders_trade_intent_price_semantics() -> None:
    from tradingscope.agents.renderers import render_markdown

    trader_data = _all_outputs()[5].model_dump(mode="json")
    trader_data["trade_intent"] = "open_short"
    trader_data["price_plan"].update(
        {
            "execution_price": 101.0,
            "execution_price_low": 100.0,
            "execution_price_high": 102.0,
            "target_price": 90.0,
            "stop_loss": 106.0,
        },
    )
    portfolio_data = _all_outputs()[6].model_dump(mode="json")
    portfolio_data["trade_intent"] = "close_long"
    portfolio_data["price_plan"].update(
        {
            "execution_price": 101.0,
            "execution_price_low": 100.0,
            "execution_price_high": 102.0,
        },
    )

    trader_markdown = render_markdown(models.TraderOutput.model_validate(trader_data))
    portfolio_markdown = render_markdown(models.PortfolioManagerOutput.model_validate(portfolio_data))

    assert "交易意图**：开空" in trader_markdown
    assert "入场区间**：USD 100.00–102.00" in trader_markdown
    assert "交易意图**：平多" in portfolio_markdown
    assert "执行区间**：USD 100.00–102.00" in portfolio_markdown


def test_reduce_long_uses_long_price_semantics_and_renders_time_stops() -> None:
    from tradingscope.agents.renderers import render_markdown

    trader_data = _all_outputs()[5].model_dump(mode="json")
    trader_data.update({"trade_intent": "reduce_long", "position_advice": "light", "time_stop_days": 3})
    trader_data["decision"] = {
        "direction": "bearish",
        "action": "sell",
        "confidence": 0.7,
        "summary": "减仓控制风险",
        "reasoning": ["趋势转弱"],
    }
    trader_data["price_plan"].update(
        {
            "execution_price": 100.0,
            "execution_price_low": 99.0,
            "execution_price_high": 101.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
        }
    )
    portfolio_data = _all_outputs()[6].model_dump(mode="json")
    portfolio_data.update({"trade_intent": "hold", "position_advice": "light", "time_stop_days": 4})

    trader_markdown = render_markdown(models.TraderOutput.model_validate(trader_data))
    portfolio_markdown = render_markdown(models.PortfolioManagerOutput.model_validate(portfolio_data))

    assert "2.00:1" in trader_markdown
    assert "### 时间止损\n\n3 个交易日" in trader_markdown
    assert "### 时间止损\n\n4 个交易日" in portfolio_markdown


def test_positioned_hold_omits_inapplicable_empty_price_fields() -> None:
    from tradingscope.agents.renderers import render_markdown

    portfolio_data = _all_outputs()[6].model_dump(mode="json")
    portfolio_data.update({"trade_intent": "hold", "position_advice": "light", "time_stop_days": 5})
    portfolio_data["price_plan"].update(
        {
            "execution_price": None,
            "execution_price_low": None,
            "execution_price_high": None,
            "target_price": 129.2,
            "stop_loss": 120.3,
        },
    )

    markdown = render_markdown(models.PortfolioManagerOutput.model_validate(portfolio_data))

    assert "入场价**：无" not in markdown
    assert "代表入场价**：无" not in markdown
    assert "盈亏比**：无" not in markdown
    assert "目标价**：USD 129.20" in markdown
    assert "止损价**：USD 120.30" in markdown

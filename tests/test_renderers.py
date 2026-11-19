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
    assert "**预测期限**：下一交易日截至收盘" in markdown


@pytest.mark.parametrize("index", range(7))
def test_stage_report_keeps_sourced_numbers_and_starts_with_decision(index: int) -> None:
    from tradingscope.agents.renderers import render_markdown

    data = _all_outputs()[index].model_dump(mode="json")
    data["evidence"] = [
        {
            "claim": "收盘价较前收盘上涨",
            "supporting_data": "收盘价 103.25 USD，涨幅 3.25%，量比 0.78",
            "source": "合成材料（https://www.anthropic.com/research/components-of-agent-memory）",
            "as_of_date": "2026-07-13",
        },
    ]
    data["limitations"] = ["报价延迟15分钟"]
    output = type(_all_outputs()[index]).model_validate(data)
    before = output.model_dump()

    markdown = render_markdown(output)

    assert markdown.index("### 决策") < markdown.index("### 关键证据与来源")
    assert "103.25 USD，涨幅 3.25%，量比 0.78" in markdown
    assert "合成材料（https://www.anthropic.com/research/components-of-agent-memory）" in markdown
    assert "日期：2026-07-13" in markdown
    assert "报价延迟15分钟" in markdown
    assert output.price_plan.invalidation_conditions[0] in markdown
    assert output.model_dump() == before


@pytest.mark.parametrize("index", range(6))
def test_evidence_pack_keeps_sources_limitations_and_invalidation(index: int) -> None:
    from tradingscope.agents.renderers import render_evidence_pack

    output = _all_outputs()[index].model_copy(update={"limitations": ["新闻源缺失"]})
    pack = render_evidence_pack(output)

    evidence = output.evidence[0]
    assert output.decision.summary in pack
    assert "**预测期限**：下一交易日截至收盘" in pack
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


def test_full_report_leads_with_final_decision_then_compares_stages() -> None:
    from tradingscope.agents.renderers import render_full_report

    markdown = render_full_report(_analysis_result())
    headings = ["## 最终投资组合决策", "## 交易员执行参考", "## 阶段结论对照", "## 关键依据与来源"]
    positions = [markdown.index(heading) for heading in headings]

    assert positions == sorted(positions)
    for title in ("交易计划", "研究经理决策", "技术面分析", "基本面分析", "新闻分析", "社交媒体分析"):
        assert f"\n## {title}\n" not in markdown
        assert f"| {title} |" in markdown
    assert markdown.count("**预测期限**") == 1
    assert markdown.count("**分析日期**") == 1
    assert "以最终决策为准" in markdown


def test_portfolio_report_shows_decision_sources_and_risk_without_debate_appendix() -> None:
    from tradingscope.agents.renderers import render_markdown

    portfolio_data = _all_outputs()[6].model_dump(mode="json")
    portfolio_data["trade_intent"] = "hold"
    portfolio = models.PortfolioManagerOutput.model_validate(portfolio_data)

    markdown = render_markdown(portfolio)

    assert markdown.index("### 决策") < markdown.index("### 关键证据与来源")
    assert "**仓位建议**：不持仓" in markdown
    assert "**风险评分**：0.50" in markdown
    assert "### 风险控制" in markdown
    assert "### 备注" not in markdown
    assert "#### 激进观点" not in markdown
    for item in ("可轻仓试探", "等待确认", "维持观察", "不追高"):
        assert item in markdown


def test_full_report_deduplicates_only_identical_evidence_without_losing_conflicts() -> None:
    from datetime import date

    from tradingscope.agents.renderers import render_full_report

    result = _analysis_result()
    shared = models.Evidence(
        claim="合成价格证据",
        supporting_data="收盘 103.25 USD，涨幅 3.25%，量比 0.78",
        source="合成材料（https://www.anthropic.com/research/components-of-agent-memory）",
        as_of_date=date(2026, 7, 13),
    )
    result.portfolio_manager.evidence = [shared]
    result.trader.evidence = [shared.model_copy()]
    result.analysts.market.evidence = [
        shared.model_copy(),
        shared.model_copy(update={"as_of_date": date(2026, 7, 10)}),
        shared.model_copy(update={"supporting_data": "收盘 98.50 USD，跌幅 1.50%，量比 1.20"}),
        shared.model_copy(update={"source": "另一独立来源"}),
    ]
    before = result.model_dump()

    report = render_full_report(result)
    sources = report.split("## 关键依据与来源\n\n", 1)[1].split("\n\n## ", 1)[0]

    assert sources.count("合成价格证据：") == 4
    assert "103.25 USD，涨幅 3.25%，量比 0.78" in sources
    assert "98.50 USD，跌幅 1.50%，量比 1.20" in sources
    assert shared.source in sources
    assert "日期：2026-07-10" in sources and "日期：2026-07-13" in sources
    assert "最终投资组合决策、交易计划、技术面分析" in sources
    assert render_full_report(result) == report
    assert result.model_dump() == before


def test_full_report_references_identical_summaries_instead_of_repeating_them() -> None:
    from tradingscope.agents.renderers import render_full_report

    result = _analysis_result()
    report = render_full_report(result)

    assert report.count(result.portfolio_manager.decision.summary) == 1
    assert "同最终投资组合决策" in report
    result.analysts.news.decision.summary = "新闻提供独立结论"
    result.analysts.social_media.decision.summary = "新闻提供独立结论"
    report = render_full_report(result)
    assert report.count("新闻提供独立结论") == 1
    assert "同新闻分析" in report


def test_full_report_omits_empty_execution_reference_section() -> None:
    from tradingscope.agents.renderers import render_full_report

    result = _analysis_result()
    result.trader.entry_conditions = []
    result.trader.execution_steps = []

    assert "## 交易员执行参考" not in render_full_report(result)


def test_full_report_preserves_all_safety_conditions_and_role_disagreements() -> None:
    from tradingscope.agents.renderers import render_full_report

    result = _analysis_result()
    portfolio = result.portfolio_manager
    portfolio.trade_intent = models.TradeIntent.REDUCE_LONG
    portfolio.decision.direction = models.Direction.BULLISH
    portfolio.decision.action = models.Action.SELL
    portfolio.position_advice = models.PositionAdvice.LIGHT
    portfolio.risk_score = 0.25
    portfolio.time_stop_days = 5
    portfolio.price_plan.execution_price = 100.25
    portfolio.price_plan.execution_price_low = 99.50
    portfolio.price_plan.execution_price_high = 101.00
    portfolio.price_plan.target_price = 110.75
    portfolio.price_plan.stop_loss = 95.25
    portfolio.price_plan.invalidation_conditions = ["共同失效条件", *[f"PM安全条件{index}" for index in range(5)]]
    portfolio.risk_control_measures = [f"风险控制步骤{index}" for index in range(4)]
    result.trader.trade_intent = models.TradeIntent.OPEN_SHORT
    result.trader.decision.direction = models.Direction.BEARISH
    result.trader.decision.action = models.Action.SELL
    result.trader.entry_conditions = [f"入场条件{index}" for index in range(4)]
    result.trader.execution_steps = [f"执行步骤{index}" for index in range(4)]
    result.analysts.news.price_plan.invalidation_conditions = ["共同失效条件", "新闻独有失效条件"]
    result.analysts.news.limitations = ["新闻事件未核验"]
    result.analysts.fundamentals.limitations = ["财报数据滞后"]
    result.research_manager.bear_viewpoints = ["最强反证：估值高于历史中位数"]
    result.analysts.market.evidence = [
        models.Evidence(claim=f"独立证据{index}", supporting_data=f"指标值{index}.25", source=f"来源{index}", as_of_date=None) for index in range(5)
    ]

    report = render_full_report(result)

    assert "**交易意图**：减多" in report
    assert "**仓位建议**：轻仓" in report
    assert "**风险评分**：0.25" in report
    assert "5 个交易日（独立持仓期限" in report
    assert "代表执行价**：USD 100.25" in report
    assert "USD 99.50–101.00" in report
    assert "目标价**：USD 110.75" in report
    assert "止损价**：USD 95.25" in report
    assert "| 最终投资组合决策 | 看涨 | 卖出 | 0.70 | 减多 |" in report
    assert "| 交易计划 | 看跌 | 卖出 | 0.70 | 开空 |" in report
    assert report.count("共同失效条件") == 1
    assert "新闻独有失效条件（新闻分析）" in report
    assert "新闻事件未核验（新闻分析）" in report
    assert "财报数据滞后（基本面分析）" in report
    assert "最强反证：估值高于历史中位数（研究经理决策）" in report
    for text in [
        *portfolio.price_plan.invalidation_conditions,
        *portfolio.risk_control_measures,
        *result.trader.entry_conditions,
        *result.trader.execution_steps,
    ]:
        assert text in report
    for evidence in result.analysts.market.evidence:
        assert evidence.claim in report
        assert evidence.supporting_data in report


def test_stage_table_preserves_summary_line_breaks_without_breaking_columns() -> None:
    from tradingscope.agents.renderers import render_full_report

    result = _analysis_result()
    result.analysts.news.decision.summary = "事件影响 | 正向\n反证仍成立 <script>alert(1)</script>"

    report = render_full_report(result)

    assert "事件影响 &#124; 正向<br>反证仍成立 &lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "<script>alert(1)</script>" not in report


def test_report_evidence_is_text_not_executable_html_and_handoff_stays_unchanged() -> None:
    from tradingscope.agents.renderers import render_evidence_pack, render_full_report, render_markdown

    result = _analysis_result()
    item = result.analysts.news.evidence[0].model_copy(update={"source": "原始来源 <script>alert(1)</script>"})
    result.analysts.news.evidence = [item]

    for report in (render_markdown(result.analysts.news), render_full_report(result)):
        assert "<script>alert(1)</script>" not in report
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert item.source in render_evidence_pack(result.analysts.news)
    assert result.analysts.news.evidence[0] == item


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

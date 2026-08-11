"""Tests for deterministic Markdown rendering from schema-v2 outputs."""

from __future__ import annotations

import pytest

from tests.test_output_models import LATEST_TRADING_DATE, TRADE_DATE, _all_outputs
from tradingscope.agents import output as models


def _analysis_result() -> models.AnalysisResult:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    return models.AnalysisResult(
        schema_version="2.0",
        ticker="AAPL",
        trade_date=TRADE_DATE,
        latest_trading_date=LATEST_TRADING_DATE,
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
    assert output.decision.summary in markdown
    assert "```json" not in markdown


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
        "entry_price": 101.0,
        "entry_price_low": 100.0,
        "entry_price_high": 102.0,
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
            "entry_price": 101.0,
            "entry_price_low": 100.0,
            "entry_price_high": 102.0,
            "target_price": 90.0,
            "stop_loss": 106.0,
        },
    )
    portfolio_data = _all_outputs()[6].model_dump(mode="json")
    portfolio_data["trade_intent"] = "close_long"
    portfolio_data["price_plan"].update(
        {
            "entry_price": 101.0,
            "entry_price_low": 100.0,
            "entry_price_high": 102.0,
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
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
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

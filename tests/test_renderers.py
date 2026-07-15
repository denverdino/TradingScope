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


def test_rendering_is_deterministic_and_does_not_mutate() -> None:
    from tradingscope.agents.renderers import render_markdown

    output = _all_outputs()[0]
    before = output.model_dump()

    assert render_markdown(output) == render_markdown(output)
    assert output.model_dump() == before

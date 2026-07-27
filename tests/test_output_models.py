"""Tests for schema-v2 agent output models."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from tradingscope.agents import output as models

TRADE_DATE = date(2026, 7, 14)
LATEST_TRADING_DATE = date(2026, 7, 13)


def _decision(action: str = "hold", direction: str = "neutral"):
    return models.Decision(
        direction=direction,
        action=action,
        confidence=0.7,
        summary="测试结论",
        reasoning=["测试依据"],
    )


def _price_plan(*, complete: bool = True):
    return models.PricePlan(
        entry_price=100.0 if complete else None,
        target_price=110.0 if complete else None,
        stop_loss=95.0 if complete else None,
        currency="USD",
        invalidation_conditions=["收盘跌破关键位置"],
    )


def _base(agent_name: str, *, action: str = "hold", direction: str = "neutral") -> dict:
    return {
        "schema_version": "2.0",
        "agent_name": agent_name,
        "ticker": "AAPL",
        "trade_date": TRADE_DATE,
        "latest_trading_date": LATEST_TRADING_DATE,
        "decision": _decision(action=action, direction=direction),
        "evidence": [
            models.Evidence(
                claim="价格维持强势",
                supporting_data="收盘价高于10日均线",
                source="market_data",
                as_of_date=LATEST_TRADING_DATE,
            ),
        ],
        "limitations": [],
    }


def _all_outputs() -> list:
    market = models.MarketAnalystOutput(
        **_base("market_analyst"),
        market_environment="大盘环境中性",
        price_action="价格震荡",
        volume_analysis="成交量正常",
        technical_indicators=[
            models.TechnicalIndicator(
                name="RSI",
                value="55",
                signal="neutral",
                interpretation="动量中性",
            ),
        ],
        weekly_bollinger=None,
        options_analysis=None,
        signal_strength="moderate",
        price_plan=_price_plan(complete=False),
    )
    fundamentals = models.FundamentalsAnalystOutput(
        **_base("fundamentals_analyst"),
        company_overview="消费电子公司",
        financial_performance=[
            models.FinancialMetric(
                name="营收",
                value="100",
                period="2026Q2",
                interpretation="同比增长",
            ),
        ],
        valuation_assessment="fair",
        earnings_quality="现金流健康",
        key_catalysts=["新品发布"],
        key_risks=["需求波动"],
        price_plan=_price_plan(complete=False),
    )
    news = models.NewsAnalystOutput(
        **_base("news_analyst"),
        overall_sentiment="neutral",
        key_events=[
            models.NewsEvent(
                title="新品发布",
                event_date=TRADE_DATE,
                source="company",
                sentiment="positive",
                impact="短期利好",
            ),
        ],
        macro_environment="宏观环境稳定",
        company_specific_impact="影响有限",
        near_term_catalysts=["发布会"],
        key_risks=["监管变化"],
        price_plan=_price_plan(complete=False),
    )
    social = models.SocialMediaAnalystOutput(
        **_base("social_media_analyst"),
        overall_sentiment="neutral",
        sentiment_score=5.5,
        platform_signals=[
            models.PlatformSignal(
                platform="Reddit",
                sentiment="neutral",
                observation="讨论热度稳定",
            ),
        ],
        key_topics=["新品"],
        sentiment_drivers=["发布会预期"],
        data_quality="样本有限",
        price_plan=_price_plan(complete=False),
    )
    research = models.ResearchManagerOutput(
        **_base("research_manager"),
        bull_viewpoints=["增长稳定"],
        bear_viewpoints=["估值不低"],
        adopted_reasoning=["等待更好价格"],
        strategic_actions=["保持观察"],
        price_scenarios=[
            models.PriceScenario(
                name="基准",
                target_price=110.0,
                timeframe_days=5,
                assumptions=["市场稳定"],
            ),
        ],
        price_plan=_price_plan(complete=False),
    )
    trader = models.TraderOutput(
        **_base("trader"),
        trade_type="short_term",
        position_advice="none",
        risk_score=0.5,
        time_stop_days=None,
        entry_conditions=["放量突破"],
        execution_steps=["等待确认"],
        risk_factors=["波动放大"],
        price_plan=_price_plan(complete=False),
    )
    portfolio = models.PortfolioManagerOutput(
        **_base("portfolio_manager"),
        aggressive_viewpoints=["可轻仓试探"],
        conservative_viewpoints=["等待确认"],
        neutral_viewpoints=["维持观察"],
        adopted_reasoning=["风险收益暂不理想"],
        position_advice="none",
        risk_score=0.5,
        risk_control_measures=["不追高"],
        price_plan=_price_plan(complete=False),
    )
    return [market, fundamentals, news, social, research, trader, portfolio]


def test_decision_requires_prediction_fields() -> None:
    with pytest.raises(ValidationError):
        models.Decision()


def test_decision_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        models.Decision(
            direction="bullish",
            action="buy",
            confidence=0.8,
            summary="趋势向上",
            reasoning=["突破确认"],
            unsupported=True,
        )


@pytest.mark.parametrize(
    ("action", "direction"),
    [("buy", "bearish"), ("sell", "bullish")],
)
def test_decision_rejects_direction_action_conflicts(action: str, direction: str) -> None:
    with pytest.raises(ValidationError, match="direction"):
        _decision(action=action, direction=direction)


def test_historical_buy_with_incomplete_plan_remains_readable() -> None:
    historical = _all_outputs()[5].model_dump(mode="json")
    historical["decision"] = {
        "direction": "bullish",
        "action": "buy",
        "confidence": 0.8,
        "summary": "历史买入记录",
        "reasoning": ["历史 schema 没有完整执行计划"],
    }
    historical["price_plan"] = {
        "entry_price": 100.0,
        "target_price": None,
        "stop_loss": 95.0,
        "currency": "USD",
        "invalidation_conditions": ["历史记录缺少目标价"],
    }

    restored = models.TraderOutput.model_validate(historical)

    assert restored.decision.action is models.Action.BUY
    assert restored.price_plan.target_price is None


def test_price_plan_computes_long_risk_reward_ratio() -> None:
    assert _price_plan().risk_reward_ratio == pytest.approx(2.0)


def test_price_plan_computes_direction_aware_risk_reward_ratio() -> None:
    long_plan = _price_plan()
    short_plan = models.PricePlan(
        entry_price=100.0,
        target_price=90.0,
        stop_loss=105.0,
        currency="USD",
        invalidation_conditions=["收盘站上105"],
    )

    assert long_plan.risk_reward_ratio_for(models.Direction.BULLISH) == pytest.approx(2.0)
    assert short_plan.risk_reward_ratio_for(models.Direction.BEARISH) == pytest.approx(2.0)
    assert short_plan.risk_reward_ratio_for(models.Direction.NEUTRAL) is None


def test_price_plan_accepts_entry_range_surrounding_representative_price() -> None:
    plan = models.PricePlan(
        entry_price=101.0,
        entry_price_low=100.0,
        entry_price_high=102.0,
        target_price=110.0,
        stop_loss=95.0,
        currency="USD",
        invalidation_conditions=["收盘跌破95"],
    )

    assert plan.entry_price_low == 100.0
    assert plan.entry_price_high == 102.0


@pytest.mark.parametrize(
    ("entry_price_low", "entry_price_high", "message"),
    [
        (100.0, None, "together"),
        (102.0, 100.0, "ordered"),
    ],
)
def test_price_plan_rejects_incomplete_or_reversed_entry_range(
    entry_price_low: float | None,
    entry_price_high: float | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        models.PricePlan(
            entry_price=101.0,
            entry_price_low=entry_price_low,
            entry_price_high=entry_price_high,
            target_price=110.0,
            stop_loss=95.0,
            currency="USD",
            invalidation_conditions=["收盘跌破95"],
        )


def test_price_plan_rejects_representative_price_outside_entry_range() -> None:
    with pytest.raises(ValidationError, match="within"):
        models.PricePlan(
            entry_price=99.0,
            entry_price_low=100.0,
            entry_price_high=102.0,
            target_price=110.0,
            stop_loss=95.0,
            currency="USD",
            invalidation_conditions=["收盘跌破95"],
        )


def test_price_plan_keeps_historical_json_without_entry_range_readable() -> None:
    historical = {
        "entry_price": 100.0,
        "target_price": 110.0,
        "stop_loss": 95.0,
        "currency": "USD",
        "invalidation_conditions": ["收盘跌破95"],
    }

    plan = models.PricePlan.model_validate(historical)

    assert plan.entry_price == 100.0
    assert plan.entry_price_low is None
    assert plan.entry_price_high is None


def test_historical_sell_output_with_incomplete_plan_remains_readable() -> None:
    historical = _all_outputs()[6].model_dump(mode="json")
    historical["decision"] = {
        "direction": "bearish",
        "action": "sell",
        "confidence": 0.85,
        "summary": "历史卖出记录",
        "reasoning": ["财报风险"],
    }
    historical["price_plan"] = {
        "entry_price": 606.0,
        "target_price": None,
        "stop_loss": None,
        "currency": "USD",
        "invalidation_conditions": ["历史记录没有完整价格计划"],
    }

    restored = models.PortfolioManagerOutput.model_validate(historical)

    assert restored.decision.action is models.Action.SELL
    assert restored.price_plan.target_price is None
    assert restored.price_plan.stop_loss is None


def test_historical_execution_outputs_remain_readable_without_trade_intent() -> None:
    outputs = _all_outputs()
    historical_trader = outputs[5].model_dump(mode="json")
    historical_portfolio = outputs[6].model_dump(mode="json")
    historical_trader.pop("trade_intent")
    historical_portfolio.pop("trade_intent")
    historical_portfolio.pop("time_stop_days")

    assert models.TraderOutput.model_validate(historical_trader).trade_intent is None
    restored_portfolio = models.PortfolioManagerOutput.model_validate(historical_portfolio)
    assert restored_portfolio.trade_intent is None
    assert restored_portfolio.time_stop_days is None


def test_trade_intent_exposes_six_distinct_execution_semantics() -> None:
    assert {item.value for item in models.TradeIntent} == {
        "open_long",
        "reduce_long",
        "close_long",
        "open_short",
        "cover_short",
        "hold",
    }


def test_market_output_decodes_json_encoded_optional_objects() -> None:
    data = _all_outputs()[0].model_dump(mode="json")
    data["weekly_bollinger"] = json.dumps(
        {
            "upper_band": 410.0,
            "middle_band": 400.0,
            "lower_band": 390.0,
            "signal": "无信号",
        },
    )
    data["options_analysis"] = json.dumps(
        {
            "put_call_ratio": 0.7,
            "support_levels": [390.0],
            "resistance_levels": [410.0],
            "max_pain": 400.0,
            "interpretation": "情绪中性",
        },
    )

    output = models.MarketAnalystOutput.model_validate(data)

    assert output.weekly_bollinger.middle_band == 400.0
    assert output.options_analysis.max_pain == 400.0


def test_market_output_rejects_malformed_encoded_optional_object() -> None:
    data = _all_outputs()[0].model_dump(mode="json")
    data["weekly_bollinger"] = "not-json"

    with pytest.raises(ValidationError, match="weekly_bollinger"):
        models.MarketAnalystOutput.model_validate(data)


def test_all_outputs_round_trip_through_json() -> None:
    for output in _all_outputs():
        restored = type(output).model_validate_json(output.model_dump_json())
        assert restored == output


def test_analysis_result_is_fully_typed() -> None:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    result = models.AnalysisResult(
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

    restored = models.AnalysisResult.model_validate_json(result.model_dump_json())
    assert restored == result

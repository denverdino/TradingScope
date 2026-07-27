"""Regression tests for short-horizon decision policy wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _capture_agent(monkeypatch, module):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(_system_prompt=kwargs["system_prompt"])

    monkeypatch.setattr(module, "Agent", factory)
    return captured


def _manager_context() -> SimpleNamespace:
    return SimpleNamespace(
        company_of_interest="AAPL",
        trade_date="2026-07-24",
        latest_trading_date="2026-07-24",
        model=object(),
        non_thinking_model=object(),
        code_interpreter_model=object(),
        middlewares=[],
        generate_analyst_reports_md=lambda: "analyst reports",
        generate_trader_context_md=lambda: "trader context",
        generate_risk_evaluation_context_md=lambda: "risk context",
    )


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("tradingscope.agents.managers.research_manager", "create_research_manager_agent"),
        ("tradingscope.agents.trader.trader", "create_trader_agent"),
        ("tradingscope.agents.managers.portfolio_manager", "create_portfolio_manager_agent"),
        ("tradingscope.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator_agent"),
        ("tradingscope.agents.risk_mgmt.conservative_debator", "create_conservative_debator_agent"),
        ("tradingscope.agents.risk_mgmt.neutral_debator", "create_neutral_debator_agent"),
    ],
)
def test_downstream_agents_share_short_horizon_calibration_policy(
    monkeypatch,
    module_name: str,
    factory_name: str,
) -> None:
    from importlib import import_module

    from tradingscope.agents.utils.decision_policy import MARKET_REGIME_DECISION_POLICY, SHORT_HORIZON_DECISION_POLICY

    module = import_module(module_name)
    captured = _capture_agent(monkeypatch, module)
    if factory_name == "create_trader_agent":
        monkeypatch.setattr(
            module.StockUtils,
            "get_market_info",
            lambda _ticker: {
                "market_name": "US",
                "currency_name": "US Dollar",
                "currency_symbol": "$",
            },
        )
        monkeypatch.setattr(module, "get_company_name", lambda _ticker, _info: "Apple")

    getattr(module, factory_name)(_manager_context())

    prompt = captured["system_prompt"]
    assert SHORT_HORIZON_DECISION_POLICY in prompt
    assert MARKET_REGIME_DECISION_POLICY in prompt
    assert "头部科技股" in prompt
    assert "来源去重" in prompt
    assert "反证" in prompt


def test_market_prompt_requires_symmetric_regime_interpretation(monkeypatch) -> None:
    from tradingscope.agents.analysts import market_analyst
    from tradingscope.agents.utils.decision_policy import MARKET_REGIME_DECISION_POLICY

    captured = _capture_agent(monkeypatch, market_analyst)
    monkeypatch.setattr(
        market_analyst.StockUtils,
        "get_market_info",
        lambda _ticker: {
            "market_name": "US",
            "currency_name": "US Dollar",
            "currency_symbol": "$",
        },
    )
    monkeypatch.setattr(market_analyst, "get_company_name", lambda _ticker, _info: "Apple")
    context = SimpleNamespace(
        company_of_interest="AAPL",
        trade_date="2026-07-24",
        latest_trading_date="2026-07-24",
        model=object(),
        middlewares=[],
    )

    market_analyst.create_market_analyst_agent(context)

    assert MARKET_REGIME_DECISION_POLICY in captured["system_prompt"]
    assert "隐含波动率" in captured["system_prompt"]
    assert "对冲回补" in captured["system_prompt"]


def test_execution_agents_require_ranges_and_complete_directional_plans(monkeypatch) -> None:
    from tradingscope.agents.managers import portfolio_manager
    from tradingscope.agents.trader import trader
    from tradingscope.agents.utils.decision_policy import EXECUTION_PLAN_DECISION_POLICY

    trader_capture = _capture_agent(monkeypatch, trader)
    portfolio_capture = _capture_agent(monkeypatch, portfolio_manager)
    monkeypatch.setattr(
        trader.StockUtils,
        "get_market_info",
        lambda _ticker: {
            "market_name": "US",
            "currency_name": "US Dollar",
            "currency_symbol": "$",
        },
    )
    monkeypatch.setattr(trader, "get_company_name", lambda _ticker, _info: "Apple")

    trader.create_trader_agent(_manager_context())
    portfolio_manager.create_portfolio_manager_agent(_manager_context())

    trader_prompt = trader_capture["system_prompt"]
    portfolio_prompt = portfolio_capture["system_prompt"]
    for prompt in (trader_prompt, portfolio_prompt):
        assert EXECUTION_PLAN_DECISION_POLICY in prompt
        assert "trade_intent" in prompt
        assert "time_stop_days" in prompt
        assert "按 trade_intent 只填写适用字段" in prompt
        assert "light/medium/heavy" in prompt
        assert "无剩余仓位时 `time_stop_days` 留空" in prompt
        for intent in ("open_long", "reduce_long", "close_long", "open_short", "cover_short", "hold"):
            assert intent in prompt

    assert "## 当操作建议为" not in trader_prompt
    assert "备用计划：条件触发" not in trader_prompt
    assert "多头盈亏比" in trader_prompt
    assert "空头盈亏比" in trader_prompt
    assert "\n  - 盈亏比 = (" not in trader_prompt
    assert "短期交易必须设定时间限制" not in trader_prompt
    assert "禁止**在自然语言操作建议中使用英文 buy/hold/sell" in trader_prompt
    assert "禁止**使用英文 buy/hold/sell" not in trader_prompt
    for uncalibrated_gate in ("2:1", "0.3x ATR", "盈亏比≥2:1", "<0.3x ATR"):
        assert uncalibrated_gate not in trader_prompt
    assert "- **止损价位**：xxx（含依据）" not in portfolio_prompt


def test_shared_policy_uses_evidence_rules_instead_of_sample_specific_thresholds() -> None:
    from tradingscope.agents.utils.decision_policy import (
        EXECUTION_PLAN_DECISION_POLICY,
        MARKET_REGIME_DECISION_POLICY,
        SHORT_HORIZON_DECISION_POLICY,
    )

    policy = SHORT_HORIZON_DECISION_POLICY + MARKET_REGIME_DECISION_POLICY + EXECUTION_PLAN_DECISION_POLICY

    assert "来源去重" in policy
    assert "反证" in policy
    assert "事件驱动" in policy
    for forbidden in ("0.65", "0.60", "半个初始风险", "只有放量破位"):
        assert forbidden not in policy


def test_research_manager_produces_scenarios_instead_of_complete_order(monkeypatch) -> None:
    from tradingscope.agents.managers import research_manager

    captured = _capture_agent(monkeypatch, research_manager)

    research_manager.create_research_manager_agent(_manager_context())

    prompt = captured["system_prompt"]
    assert "情景分析" in prompt
    assert "不制定完整订单" in prompt

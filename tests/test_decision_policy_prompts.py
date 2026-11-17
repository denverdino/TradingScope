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


def _stock_agent_prompt(monkeypatch, module, factory_name, *, ticker="AAPL", market_name="US"):
    captured = _capture_agent(monkeypatch, module)
    monkeypatch.setattr(
        module.StockUtils,
        "get_market_info",
        lambda _ticker: {
            "market_name": market_name,
            "currency_name": "US Dollar",
            "currency_symbol": "$",
        },
    )
    monkeypatch.setattr(module, "get_company_name", lambda _ticker, _info: "Example Co")
    context = _manager_context()
    context.ticker = ticker
    getattr(module, factory_name)(context)
    return captured["system_prompt"]


def _manager_context() -> SimpleNamespace:
    return SimpleNamespace(
        ticker="AAPL",
        analysis_date="2026-07-24",
        last_completed_session="2026-07-24",
        model=object(),
        non_thinking_model=object(),
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
        ("tradingscope.agents.risk_mgmt.aggressive_debater", "create_aggressive_debater_agent"),
        ("tradingscope.agents.risk_mgmt.conservative_debater", "create_conservative_debater_agent"),
        ("tradingscope.agents.risk_mgmt.neutral_debater", "create_neutral_debater_agent"),
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
        ticker="AAPL",
        analysis_date="2026-07-24",
        last_completed_session="2026-07-24",
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
        assert "减仓不等于看跌" in prompt
        assert "bullish/neutral/bearish" in prompt
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


def test_market_prompt_compares_price_range_to_atr_and_labels_rsi_consistently(monkeypatch):
    from tradingscope.agents.analysts import market_analyst

    prompt = _stock_agent_prompt(monkeypatch, market_analyst, "create_market_analyst_agent")

    assert "尚未超卖（RSI=31.11 > 30" in prompt
    assert "超卖（RSI=31.11 > 30" not in prompt.replace("尚未超卖（", "")
    assert "日内价差 = 最高价 - 最低价" in prompt
    assert "日内价差 > 1.5x ATR" in prompt
    assert "日内价差 < 0.5x ATR" in prompt
    assert "当日振幅 > 1.5x ATR" not in prompt
    assert "当日振幅 vs ATR" not in prompt


def test_news_prompt_allows_supported_scenarios_without_forced_price_numbers(monkeypatch):
    from tradingscope.agents.analysts import news_analyst

    prompt = _stock_agent_prompt(monkeypatch, news_analyst, "create_news_analyst_agent")

    assert "区分已报道事实、分析推断与条件情景" in prompt
    assert "来源输入、计算依据与情景假设" in prompt
    assert "无法量化" in prompt
    assert "原始链接" in prompt
    for forbidden in (
        "绝对禁止基于推测或假设生成任何分析内容",
        "不允许回复'无法评估价格影响'",
        "提供新闻对股价影响的量化评估和具体价格预期",
    ):
        assert forbidden not in prompt


@pytest.mark.parametrize(("ticker", "market_name"), [("AAPL", "US"), ("0700.HK", "Hong Kong")])
def test_social_prompt_labels_news_proxy_and_uses_the_tickers_market(monkeypatch, ticker, market_name):
    from tradingscope.agents.analysts import social_media_analyst

    prompt = _stock_agent_prompt(
        monkeypatch,
        social_media_analyst,
        "create_social_media_analyst_agent",
        ticker=ticker,
        market_name=market_name,
    )

    assert f"{market_name}市场社交媒体与投资情绪分析师" in prompt
    assert "新闻情绪代理信号" in prompt
    assert "不是社交平台采样结果" in prompt
    assert "启发式评分" in prompt
    assert "缺少平台、KOL、散户或机构样本时标注不可用" in prompt
    assert "不得编造百分比或支撑/阻力位" in prompt
    assert '不允许回复"无法评估/需要更多数据"' not in prompt
    assert "预期波动幅度（%）" not in prompt


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("market_analyst", "create_market_analyst_agent"),
        ("fundamentals_analyst", "create_fundamentals_analyst_agent"),
        ("news_analyst", "create_news_analyst_agent"),
        ("social_media_analyst", "create_social_media_analyst_agent"),
    ],
)
def test_analyst_drafts_supply_decision_evidence_and_nullable_price_material(monkeypatch, module_name, factory_name):
    from importlib import import_module

    module = import_module(f"tradingscope.agents.analysts.{module_name}")
    prompt = _stock_agent_prompt(monkeypatch, module, factory_name)

    for required in (
        "唯一方向判断（看涨/看跌/中性）",
        "唯一操作建议（买入/持有/卖出）",
        "置信度（0-1）",
        "来源、原始链接（如有）、数据日期",
        "最强反证",
        "价格无依据时留空（null）",
        "失效条件",
        "数据缺失与适用限制",
        "不要输出 JSON、JSON 代码块或固定 Markdown 报告模板",
    ):
        assert required in prompt


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [("research_manager", "create_research_manager_agent"), ("portfolio_manager", "create_portfolio_manager_agent")],
)
def test_manager_drafts_require_material_without_mandating_a_report_template(monkeypatch, module_name, factory_name):
    from importlib import import_module

    module = import_module(f"tradingscope.agents.managers.{module_name}")
    captured = _capture_agent(monkeypatch, module)
    getattr(module, factory_name)(_manager_context())
    prompt = captured["system_prompt"]

    assert "无需遵循固定展示格式" in prompt
    assert "置信度（0-1）" in prompt
    assert "股票代码：AAPL" in prompt
    assert "分析日期：2026-07-24" in prompt
    assert "最近完成交易日：2026-07-24" in prompt
    assert "数据缺失与适用限制" in prompt
    assert "来源" in prompt and "反证" in prompt and "失效条件" in prompt
    for forbidden in ("您的回复必须以如下格式开头", "然后按以下格式提供详细分析", "不要回复'无法确定'"):
        assert forbidden not in prompt


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [("bull_researcher", "create_bull_researcher_agent"), ("bear_researcher", "create_bear_researcher_agent")],
)
def test_researchers_require_short_horizon_transmission_and_counter_evidence(monkeypatch, module_name, factory_name):
    from importlib import import_module

    module = import_module(f"tradingscope.agents.researchers.{module_name}")
    captured = _capture_agent(monkeypatch, module)
    getattr(module, factory_name)(_manager_context())
    prompt = captured["system_prompt"]

    assert "1-5 个交易日" in prompt
    assert "短期传导机制" in prompt
    assert "长期基本面不能单独推出短期方向" in prompt
    assert "来源与数据时点" in prompt
    assert "最强反证" in prompt and "失效条件" in prompt


def test_trader_distinguishes_executing_existing_plan_from_changing_direction(monkeypatch):
    from tradingscope.agents.trader import trader

    prompt = _stock_agent_prompt(monkeypatch, trader, "create_trader_agent")

    assert "没有新事件不等于必须等待" in prompt
    assert "原有证据仍有效且具备可执行优势" in prompt
    assert "新增或被遗漏证据" in prompt
    assert "证据不足或缺乏可执行优势时选择 `hold`" in prompt
    assert "若没有新的状态变化证据，则用 `hold`" not in prompt
    assert "减仓不等于看跌" in prompt


def test_research_manager_scenarios_use_sessions_not_calendar_days(monkeypatch):
    from tradingscope.agents.managers import research_manager

    captured = _capture_agent(monkeypatch, research_manager)
    research_manager.create_research_manager_agent(_manager_context())
    prompt = captured["system_prompt"]

    assert "1个交易日" in prompt and "5个交易日" in prompt
    assert "1天: 价格情景" not in prompt
    assert "5天: 价格情景" not in prompt


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("tradingscope.agents.risk_mgmt.aggressive_debater", "create_aggressive_debater_agent"),
        ("tradingscope.agents.risk_mgmt.conservative_debater", "create_conservative_debater_agent"),
        ("tradingscope.agents.risk_mgmt.neutral_debater", "create_neutral_debater_agent"),
    ],
)
def test_risk_debaters_focus_on_key_disagreements_and_allow_evidence_based_concessions(monkeypatch, module_name: str, factory_name: str) -> None:
    from importlib import import_module

    module = import_module(module_name)
    captured = _capture_agent(monkeypatch, module)
    getattr(module, factory_name)(_manager_context())
    prompt = captured["system_prompt"]

    assert "每个观点" not in prompt
    assert "每个反驳点" not in prompt
    assert "最关键的分歧" in prompt
    assert "证据成立时可以承认" in prompt


def test_fundamentals_prompt_marks_unsupported_fields_missing_without_identical_reacquisition(monkeypatch) -> None:
    from tradingscope.agents.analysts import fundamentals_analyst

    prompt = _stock_agent_prompt(monkeypatch, fundamentals_analyst, "create_fundamentals_analyst_agent")

    assert "若返回数据缺失，以**最多两次**补采为限" not in prompt
    assert "不得以相同参数重复调用" in prompt
    assert "超时、限流或连接错误等临时错误可重试" in prompt
    for field in ("盈利超预期/不及预期", "PE(TTM) 相对过去 1 年均值", "下一财报日"):
        assert f"{field}：仅在工具返回相应数据时分析，否则标注数据缺失" in prompt

"""Regression tests for short-horizon decision policy wiring."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest


def _assert_next_session_forecast_contract(prompt: str) -> None:
    """Every stage forecasts one actual session, including the overnight move."""
    assert "下一实际交易日" in prompt
    assert "分析日期当日或之后的首个交易日" in prompt
    assert "最近完成交易日收盘" in prompt and "目标交易日收盘" in prompt
    assert "隔夜跳空" in prompt
    assert "重新分析" in prompt
    assert not re.search(r"1\s*[-–]\s*5\s*个交易日", prompt)


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
    _assert_next_session_forecast_contract(prompt)
    assert "头部科技股" in prompt
    assert "来源去重" in prompt
    assert "反证" in prompt


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("analysts.market_analyst", "create_market_analyst_agent"),
        ("analysts.fundamentals_analyst", "create_fundamentals_analyst_agent"),
        ("analysts.news_analyst", "create_news_analyst_agent"),
        ("analysts.social_media_analyst", "create_social_media_analyst_agent"),
        ("researchers.bull_researcher", "create_bull_researcher_agent"),
        ("researchers.bear_researcher", "create_bear_researcher_agent"),
        ("managers.research_manager", "create_research_manager_agent"),
        ("trader.trader", "create_trader_agent"),
        ("managers.portfolio_manager", "create_portfolio_manager_agent"),
        ("risk_mgmt.aggressive_debater", "create_aggressive_debater_agent"),
        ("risk_mgmt.conservative_debater", "create_conservative_debater_agent"),
        ("risk_mgmt.neutral_debater", "create_neutral_debater_agent"),
    ],
)
def test_active_stage_prompts_select_concise_traceable_evidence(monkeypatch, module_name, factory_name):
    from importlib import import_module

    module = import_module(f"tradingscope.agents.{module_name}")
    if hasattr(module, "get_company_name"):
        prompt = _stock_agent_prompt(monkeypatch, module, factory_name)
    else:
        captured = _capture_agent(monkeypatch, module)
        getattr(module, factory_name)(_manager_context())
        prompt = captured["system_prompt"]

    for required in (
        "结论先行",
        "一句话摘要",
        "最多3条独立关键证据",
        "原始URL不得改写",
        "关键数字与必要计算依据",
        "最强反证",
        "撤回或调整结论的条件",
        "不在摘要、理由、事件、话题和驱动因素中重复同一事实",
        "下游只解释证据取舍与新增调整",
        "不限制执行步骤、风险控制和失效条件",
        "保留所有公共JSON字段",
    ):
        assert required in prompt


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
        _assert_next_session_forecast_contract(prompt)
        assert "入场" in prompt and "目标" in prompt and "下一实际交易日" in prompt
        assert "持仓管理期限" in prompt and "独立" in prompt
        assert "可以超过1个交易日" in prompt
        assert "收盘估值" in prompt and "不代表强制平仓" in prompt
        assert "time_stop_days=1" not in prompt
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


@pytest.mark.parametrize(
    ("module_name", "factory_name", "exhaustive_output"),
    [
        ("market_analyst", "create_market_analyst_agent", "逐项列出工具返回的指标"),
        ("fundamentals_analyst", "create_fundamentals_analyst_agent", "### 最新财务表现"),
        ("news_analyst", "create_news_analyst_agent", "### 信息来源引用"),
        ("social_media_analyst", "create_social_media_analyst_agent", "每项判断就近注明来源与时间"),
    ],
)
def test_analysts_present_selected_evidence_without_repeated_display_sections(monkeypatch, module_name, factory_name, exhaustive_output):
    from importlib import import_module

    module = import_module(f"tradingscope.agents.analysts.{module_name}")
    prompt = _stock_agent_prompt(monkeypatch, module, factory_name)

    assert "只呈现影响结论" in prompt
    assert "不得按工具字段逐项展示" in prompt
    assert exhaustive_output not in prompt
    assert "### 关键要点速览" not in prompt
    assert "在分析结尾给出明确" not in prompt
    assert "2-3 句话" not in prompt


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
    assert "近5日表现" in prompt
    assert "5日/10日/20日均量趋势" in prompt
    assert "周线布林带信号是中期参考信号" in prompt


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
    assert "只在 data_quality 中集中说明一次" in prompt
    assert "limitations 只写其他具体限制" in prompt
    assert "没有其他限制时为空列表 []" in prompt
    assert "代理信号的身份说明不再改写为 limitations" in prompt
    assert "未取到样本的平台不建立条目或小节" in prompt
    assert "platform_signals 仅保留实际取得材料的来源" in prompt
    assert "不得推断其观点、热度、群体差异或影响力" in prompt
    assert "### 平台分项分析" not in prompt
    assert "### 社交媒体舆情概览（时间窗口、来源、热度概况）" not in prompt
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

    _assert_next_session_forecast_contract(prompt)
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
    for required in (
        "原始链接（如有）",
        "数据日期",
        "货币",
        "价格无依据时留空（null）",
        "核心理由最多3点",
        "不得重抄共享报告",
    ):
        assert required in prompt
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

    _assert_next_session_forecast_contract(prompt)
    assert "短期传导机制" in prompt
    assert "长期基本面不能单独推出短期方向" in prompt
    assert "来源与数据时点" in prompt
    assert "最强反证" in prompt and "失效条件" in prompt


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("researchers.bull_researcher", "create_bull_researcher_agent"),
        ("researchers.bear_researcher", "create_bear_researcher_agent"),
        ("managers.research_manager", "create_research_manager_agent"),
        ("trader.trader", "create_trader_agent"),
        ("managers.portfolio_manager", "create_portfolio_manager_agent"),
        ("risk_mgmt.aggressive_debater", "create_aggressive_debater_agent"),
        ("risk_mgmt.conservative_debater", "create_conservative_debater_agent"),
        ("risk_mgmt.neutral_debater", "create_neutral_debater_agent"),
    ],
)
def test_decision_layer_selects_evidence_instead_of_exhaustive_argument_or_intent_lists(monkeypatch, module_name, factory_name):
    from importlib import import_module

    module = import_module(f"tradingscope.agents.{module_name}")
    if hasattr(module, "get_company_name"):
        prompt = _stock_agent_prompt(monkeypatch, module, factory_name)
    else:
        captured = _capture_agent(monkeypatch, module)
        getattr(module, factory_name)(_manager_context())
        prompt = captured["system_prompt"]

    if "researcher" in factory_name or "debater" in factory_name:
        assert "最多2个关键论点" in prompt
        assert "最强反证" in prompt
        assert "证据成立时可以承认" in prompt
        assert "重点关注以下几个方面" not in prompt
        assert "专注于辩论和说服" not in prompt
        assert "优先考虑全面的方法" not in prompt
    elif factory_name == "create_research_manager_agent":
        assert "提供全面的目标价格区间" not in prompt
        assert "核心理由最多3点" in prompt
        assert "基于采纳的证据" in prompt
    else:
        # Two shared directional rules, not a second per-intent display checklist.
        assert prompt.count("完整入场区间") == 2
        assert "执行步骤" in prompt
        assert "减仓比例" in prompt
        assert "交易前持仓假设" in prompt
        assert "调整" in prompt
        assert "列出2-3条具体的风险控制措施" not in prompt


def test_trader_distinguishes_executing_existing_plan_from_changing_direction(monkeypatch):
    from tradingscope.agents.trader import trader

    prompt = _stock_agent_prompt(monkeypatch, trader, "create_trader_agent")

    assert "没有新事件不等于必须等待" in prompt
    assert "原有证据仍有效且具备可执行优势" in prompt
    assert "新增或被遗漏证据" in prompt
    assert "证据不足或缺乏可执行优势时选择 `hold`" in prompt
    assert "若没有新的状态变化证据，则用 `hold`" not in prompt
    assert "减仓不等于看跌" in prompt


def test_research_manager_keeps_three_scenarios_for_only_the_next_actual_session(monkeypatch):
    from tradingscope.agents.managers import research_manager

    captured = _capture_agent(monkeypatch, research_manager)
    research_manager.create_research_manager_agent(_manager_context())
    prompt = captured["system_prompt"]

    _assert_next_session_forecast_contract(prompt)
    for scenario in ("保守", "基准", "乐观"):
        assert scenario in prompt
    assert "timeframe_days=1" in prompt
    assert "5个交易日" not in prompt
    assert "timeframe_days=5" not in prompt
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
    if factory_name == "create_conservative_debater_agent":
        assert "长期收益" not in prompt
        assert "持仓管理期限" in prompt and "独立" in prompt
        assert "下一实际交易日" in prompt


def test_fundamentals_prompt_marks_unsupported_fields_missing_without_identical_reacquisition(monkeypatch) -> None:
    from tradingscope.agents.analysts import fundamentals_analyst

    prompt = _stock_agent_prompt(monkeypatch, fundamentals_analyst, "create_fundamentals_analyst_agent")

    assert "若返回数据缺失，以**最多两次**补采为限" not in prompt
    assert "不得以相同参数重复调用" in prompt
    assert "超时、限流或连接错误等临时错误可重试" in prompt
    assert "未来 5 个交易日" in prompt
    assert "下一实际交易日" in prompt and "优先" in prompt
    assert "更晚事件" in prompt and "背景" in prompt
    assert "近 2-4 个季度" in prompt
    assert "过去 1 年均值" in prompt
    for field in ("盈利超预期/不及预期", "PE(TTM) 相对过去 1 年均值", "下一财报日"):
        assert f"{field}：仅在工具返回相应数据时分析，否则标注数据缺失" in prompt


def test_fundamentals_prompt_requires_period_evidence_before_generalizing_buybacks(monkeypatch):
    from tradingscope.agents.analysts import fundamentals_analyst

    prompt = _stock_agent_prompt(monkeypatch, fundamentals_analyst, "create_fundamentals_analyst_agent", ticker="MSFT")

    for required in (
        "MRQ（单季）、FY（全年）、TTM（滚动十二个月）必须区分，不得混用",
        "回购、现金流等金额",
        "每季度",
        "持续",
        "稳定",
        "逐期核验各期数据",
        "缺失期间不得概括",
        "实际覆盖的期间与数值范围、例外",
        "不得将最近一季外推为所有季度",
    ):
        assert required in prompt


def test_news_prompt_separates_retrieval_freshness_from_publication_and_event_status(monkeypatch):
    from tradingscope.agents.analysts import news_analyst

    prompt = _stock_agent_prompt(monkeypatch, news_analyst, "create_news_analyst_agent")

    for required in (
        "供应商检索时间",
        "文章发布时间",
        "事件发生时间及当前状态",
        "检索时间不能代替发布时间或证明事件仍在持续",
        "最新可用公司新闻",
        "该事件最新相关文章",
        "分别注明标题、发布时间和来源",
        "供应商返回的最新文章不等于全市场最新",
        "没有恢复更新不等于事件仍在持续",
        "缺少更新证据时写明“恢复情况未核实”",
    ):
        assert required in prompt


def test_news_prompt_maps_each_claim_to_its_own_article_title_and_exact_url(monkeypatch):
    from tradingscope.agents.analysts import news_analyst

    prompt = _stock_agent_prompt(monkeypatch, news_analyst, "create_news_analyst_agent")

    assert "每项事实或判断须就近引用对应文章的标题与原始URL" in prompt
    assert "多条报道须逐条映射到各自的标题与链接" in prompt
    assert "不得用另一篇报道的链接替换该项依据" in prompt
    assert "不得改写原始URL或仅给来源主页" in prompt


def test_market_prompt_aligns_return_and_obv_windows_without_changing_volume_average_samples(monkeypatch):
    from tradingscope.agents.analysts import market_analyst

    prompt = _stock_agent_prompt(monkeypatch, market_analyst, "create_market_analyst_agent")

    for required in (
        "N 个交易日收益率包含 N 个收盘到收盘区间，需要 N+1 个收盘价",
        "(期末收盘价 / 起始收盘价 - 1) × 100%",
        "跨工具收益率比较或价格与 OBV 背离判断",
        "起止交易日期一致",
        "窗口不一致或起止日期缺失时，标注不可直接比较",
        "不得混合窗口得出相对强弱或背离结论",
        "N 日均量仍使用 N 个交易日的成交量样本，不要求 N+1 个样本",
    ):
        assert required in prompt


def test_market_prompt_leaves_missing_or_rejected_oi_levels_empty_without_technical_substitution(monkeypatch):
    from tradingscope.agents.analysts import market_analyst

    prompt = _stock_agent_prompt(monkeypatch, market_analyst, "create_market_analyst_agent")

    for required in (
        "OI 支撑/阻力位缺失或被拒绝采用时",
        "options_analysis.support_levels",
        "options_analysis.resistance_levels",
        "对应字段必须为空数组 []",
        "技术面推导的价位只能放在技术面的 price_plan 或 evidence",
        "不得填入期权字段",
        "仅在对应 OI 价位可用且被采用时填写，否则留空",
    ):
        assert required in prompt
    assert "- **期权面支撑位**：$xxx（Put OI 集中区）" not in prompt
    assert "- **期权面阻力位**：$xxx（Call OI 集中区）" not in prompt


def test_market_prompt_reports_nonunique_max_pain_as_interval_with_nullable_single_price(monkeypatch):
    from tradingscope.agents.analysts import market_analyst

    prompt = _stock_agent_prompt(monkeypatch, market_analyst, "create_market_analyst_agent")

    for required in (
        "Max Pain 不可用、不唯一或被拒绝采用时，options_analysis.max_pain 必须为 null",
        "不唯一时仅在解释中报告区间参考",
        "不得任选一点作为单一方向性目标",
        "**Max Pain 价格（可选）**",
        "仅在可用、唯一且被采用时填写",
    ):
        assert required in prompt
    assert "- **Max Pain 价格**：$xxx" not in prompt


def test_market_prompt_preserves_valid_pcr_without_treating_oi_counts_as_verified_coverage(monkeypatch):
    from tradingscope.agents.analysts import market_analyst

    prompt = _stock_agent_prompt(monkeypatch, market_analyst, "create_market_analyst_agent")

    for required in (
        "OI 数值有效仅表示计数可用",
        "不代表覆盖范围、更新时间或样本代表性已核实",
        "执行价远离现价本身不说明合约无效",
        "支撑/阻力位或 Max Pain 不可用时仍保留有效 PCR",
        "更新时间或覆盖范围未知时",
        "同时在 options_analysis.interpretation 与 limitations 中说明",
        "不得据此作确定性方向判断",
    ):
        assert required in prompt

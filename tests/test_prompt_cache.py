from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from agentscope.message import SystemMsg

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT
from tradingscope.agents.utils.prompt_cache import (
    CACHE_PREFIX_BREAK,
    CachedDashScopeChatFormatter,
    build_cacheable_system_prompt,
)


def test_build_cacheable_system_prompt_places_shared_context_first() -> None:
    prompt = build_cacheable_system_prompt(
        shared_context="共享分析证据",
        role_instructions="看涨角色指令",
    )

    shared, role = prompt.split(CACHE_PREFIX_BREAK)
    assert shared == f"{COMPLIANCE_PROMPT}\n\n# 共享分析上下文\n\n共享分析证据"
    assert role == "看涨角色指令"


def test_cached_formatter_marks_only_the_shared_prefix() -> None:
    formatter = CachedDashScopeChatFormatter()
    prompt = build_cacheable_system_prompt(
        shared_context="共享分析证据",
        role_instructions="看跌角色指令",
    )

    messages = asyncio.run(
        formatter.format([SystemMsg(name="system", content=prompt)]),
    )

    assert messages == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{COMPLIANCE_PROMPT}\n\n# 共享分析上下文\n\n共享分析证据",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
        {
            "role": "system",
            "content": [{"type": "text", "text": "看跌角色指令"}],
        },
    ]


def test_cached_formatter_moves_marker_after_confirmed_shared_prefix() -> None:
    formatter = CachedDashScopeChatFormatter()
    first_prompt = build_cacheable_system_prompt(
        shared_context="共享分析证据",
        role_instructions="看涨角色指令",
    )
    second_prompt = build_cacheable_system_prompt(
        shared_context="共享分析证据",
        role_instructions="看跌角色指令",
    )

    formatter.confirm_cache_prefixes(
        [SystemMsg(name="system", content=first_prompt)],
        confirmed_at=100.0,
    )
    with patch("tradingscope.agents.utils.prompt_cache.monotonic", return_value=200.0):
        messages = asyncio.run(
            formatter.format([SystemMsg(name="system", content=second_prompt)]),
        )

    assert messages == [
        {
            "role": "system",
            "content": f"{COMPLIANCE_PROMPT}\n\n# 共享分析上下文\n\n共享分析证据",
        },
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "看跌角色指令",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]


def test_cached_formatter_recreates_expired_prefix() -> None:
    formatter = CachedDashScopeChatFormatter()
    prompt = build_cacheable_system_prompt(
        shared_context="共享分析证据",
        role_instructions="看涨角色指令",
    )

    with patch("tradingscope.agents.utils.prompt_cache.monotonic", return_value=401.0):
        formatter.confirm_cache_prefixes(
            [SystemMsg(name="system", content=prompt)],
            confirmed_at=100.0,
        )
        messages = asyncio.run(
            formatter.format([SystemMsg(name="system", content=prompt)]),
        )

    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[1]["content"][0]


def test_cached_formatter_leaves_regular_prompts_unchanged() -> None:
    formatter = CachedDashScopeChatFormatter()

    messages = asyncio.run(
        formatter.format([SystemMsg(name="system", content="普通提示词")]),
    )

    assert messages == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "普通提示词"}],
        },
    ]


def test_research_agents_share_identical_cache_prefix(monkeypatch) -> None:
    from tradingscope.agents.managers import research_manager
    from tradingscope.agents.researchers import bear_researcher, bull_researcher

    context = SimpleNamespace(
        company_of_interest="BABA",
        trade_date="2026-08-11",
        latest_trading_date="2026-08-10",
        model=object(),
        non_thinking_model=object(),
        middlewares=[],
        generate_analyst_reports_md=lambda: "相同的分析师证据包",
    )
    market_info = {
        "market_name": "US",
        "currency_name": "US Dollar",
        "currency_symbol": "$",
    }
    monkeypatch.setattr(bull_researcher.StockUtils, "get_market_info", lambda _: market_info)
    monkeypatch.setattr(bear_researcher.StockUtils, "get_market_info", lambda _: market_info)
    for module in (bull_researcher, bear_researcher, research_manager):
        monkeypatch.setattr(
            module,
            "Agent",
            lambda **kwargs: SimpleNamespace(_system_prompt=kwargs["system_prompt"]),
        )

    prompts = [
        bull_researcher.create_bull_researcher_agent(context)._system_prompt,
        bear_researcher.create_bear_researcher_agent(context)._system_prompt,
        research_manager.create_research_manager_agent(context)._system_prompt,
    ]

    prefixes = [prompt.split(CACHE_PREFIX_BREAK)[0] for prompt in prompts]
    assert len(set(prefixes)) == 1
    assert "相同的分析师证据包" in prefixes[0]


def test_risk_debators_share_identical_cache_prefix(monkeypatch) -> None:
    from tradingscope.agents.risk_mgmt import aggressive_debator, conservative_debator, neutral_debator

    context = SimpleNamespace(
        company_of_interest="BABA",
        non_thinking_model=object(),
        middlewares=[],
        generate_risk_evaluation_context_md=lambda: "相同的风险证据包",
    )
    modules = (aggressive_debator, conservative_debator, neutral_debator)
    for module in modules:
        monkeypatch.setattr(
            module,
            "Agent",
            lambda **kwargs: SimpleNamespace(_system_prompt=kwargs["system_prompt"]),
        )

    prompts = [
        aggressive_debator.create_aggressive_debator_agent(context)._system_prompt,
        conservative_debator.create_conservative_debator_agent(context)._system_prompt,
        neutral_debator.create_neutral_debator_agent(context)._system_prompt,
    ]

    prefixes = [prompt.split(CACHE_PREFIX_BREAK)[0] for prompt in prompts]
    assert len(set(prefixes)) == 1
    assert "相同的风险证据包" in prefixes[0]

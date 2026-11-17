from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from agentscope.message import SystemMsg, TextBlock, ToolCallBlock, UserMsg

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT
from tradingscope.agents.utils.prompt_cache import (
    CACHE_PREFIX_BREAK,
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


def test_research_agents_share_identical_cache_prefix(monkeypatch) -> None:
    from tradingscope.agents.managers import research_manager
    from tradingscope.agents.researchers import bear_researcher, bull_researcher

    context = SimpleNamespace(
        ticker="BABA",
        analysis_date="2026-08-11",
        last_completed_session="2026-08-10",
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


def test_risk_debaters_share_identical_cache_prefix(monkeypatch) -> None:
    from tradingscope.agents.risk_mgmt import aggressive_debater, conservative_debater, neutral_debater

    context = SimpleNamespace(
        ticker="BABA",
        non_thinking_model=object(),
        middlewares=[],
        generate_risk_evaluation_context_md=lambda: "相同的风险证据包",
    )
    modules = (aggressive_debater, conservative_debater, neutral_debater)
    for module in modules:
        monkeypatch.setattr(
            module,
            "Agent",
            lambda **kwargs: SimpleNamespace(_system_prompt=kwargs["system_prompt"]),
        )

    prompts = [
        aggressive_debater.create_aggressive_debater_agent(context)._system_prompt,
        conservative_debater.create_conservative_debater_agent(context)._system_prompt,
        neutral_debater.create_neutral_debater_agent(context)._system_prompt,
    ]

    prefixes = [prompt.split(CACHE_PREFIX_BREAK)[0] for prompt in prompts]
    assert len(set(prefixes)) == 1
    assert "相同的风险证据包" in prefixes[0]


def test_downstream_stage_prompts_reuse_layered_cache_prefixes(monkeypatch) -> None:
    from tradingscope.agents.managers import portfolio_manager, research_manager
    from tradingscope.agents.risk_mgmt import aggressive_debater
    from tradingscope.agents.trader import trader

    analysts, research, trader_plan = "分析师证据包", "研究经理证据包", "交易员证据包"
    context = SimpleNamespace(
        ticker="BABA",
        analysis_date="2026-08-11",
        last_completed_session="2026-08-10",
        model=object(),
        non_thinking_model=object(),
        middlewares=[],
        generate_analyst_reports_md=lambda: analysts,
        generate_trader_context_md=lambda: f"{analysts}{CACHE_PREFIX_BREAK}{research}",
        generate_risk_evaluation_context_md=lambda: f"{analysts}{CACHE_PREFIX_BREAK}{research}{CACHE_PREFIX_BREAK}{trader_plan}",
    )
    market_info = {"market_name": "US", "currency_name": "US Dollar", "currency_symbol": "$"}
    monkeypatch.setattr(trader.StockUtils, "get_market_info", lambda _: market_info)
    monkeypatch.setattr(trader, "get_company_name", lambda *_: "Alibaba")
    modules = (research_manager, trader, aggressive_debater, portfolio_manager)
    for module in modules:
        monkeypatch.setattr(module, "Agent", lambda **kwargs: SimpleNamespace(_system_prompt=kwargs["system_prompt"]))

    research_segments, trader_segments, risk_segments, portfolio_segments = (
        prompt.split(CACHE_PREFIX_BREAK)
        for prompt in (
            research_manager.create_research_manager_agent(context)._system_prompt,
            trader.create_trader_agent(context)._system_prompt,
            aggressive_debater.create_aggressive_debater_agent(context)._system_prompt,
            portfolio_manager.create_portfolio_manager_agent(context)._system_prompt,
        )
    )

    assert research_segments[0].endswith(analysts)
    assert trader_segments[:2] == [research_segments[0], research]
    assert risk_segments[:3] == [research_segments[0], research, trader_plan]
    assert portfolio_segments[:3] == risk_segments[:3]
    assert len(trader_segments) == 3
    assert len(portfolio_segments) == 4


def test_layered_prefixes_become_explicit_cache_blocks() -> None:
    from tradingscope.agents.utils.prompt_cache import DashScopeResponseFormatter

    prompt = build_cacheable_system_prompt(
        shared_context=f"分析师{CACHE_PREFIX_BREAK}研究{CACHE_PREFIX_BREAK}交易",
        role_instructions="风控角色",
    )
    items = asyncio.run(DashScopeResponseFormatter().format([SystemMsg(name="risk", content=prompt)]))
    blocks = items[0]["content"]

    assert "".join(block["text"] for block in blocks) == prompt
    assert [block.get("prompt_cache_breakpoint") for block in blocks] == [{"mode": "explicit"}] * 3 + [None]
    assert blocks[0]["text"].endswith("分析师")
    assert [block["text"] for block in blocks[1:]] == [f"{CACHE_PREFIX_BREAK}研究", f"{CACHE_PREFIX_BREAK}交易", f"{CACHE_PREFIX_BREAK}风控角色"]


def test_shared_prefix_is_an_explicit_cache_content_block() -> None:
    from tradingscope.agents.utils.prompt_cache import DashScopeResponseFormatter

    formatter = DashScopeResponseFormatter()
    inputs = []
    for role in ("看涨角色", "看跌角色"):
        prompt = build_cacheable_system_prompt(shared_context="相同素材", role_instructions=role)
        items = asyncio.run(formatter.format([SystemMsg(name=role, content=prompt)]))
        inputs.append(items[0]["content"])
        assert "".join(block["text"] for block in items[0]["content"]) == prompt
    assert inputs[0][0] == inputs[1][0]
    assert inputs[0][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert inputs[0][1] != inputs[1][1]
    assert "prompt_cache_breakpoint" not in inputs[0][1]


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_cache_separator_in_non_system_text_is_not_interpreted(role: str) -> None:
    from agentscope.message import Msg

    from tradingscope.agents.utils.prompt_cache import DashScopeResponseFormatter

    content = f"材料{CACHE_PREFIX_BREAK}正文"
    items = asyncio.run(DashScopeResponseFormatter().format([Msg(name="user", role=role, content=[TextBlock(text=content)])]))
    assert len(items[0]["content"]) == 1
    assert items[0]["content"][0]["text"] == content
    assert "prompt_cache_breakpoint" not in items[0]["content"][0]


def test_cache_formatter_preserves_unmarked_messages_and_function_calls() -> None:
    from agentscope.formatter import OpenAIResponseFormatter
    from agentscope.message import AssistantMsg

    from tradingscope.agents.utils.prompt_cache import DashScopeResponseFormatter

    messages = [
        SystemMsg(name="system", content="普通系统提示词"),
        UserMsg(name="user", content="问题"),
        AssistantMsg(
            name="agent",
            content=[TextBlock(text="调用工具"), ToolCallBlock(id="call-1", name="tool", input="{}")],
        ),
    ]
    expected = asyncio.run(OpenAIResponseFormatter().format(messages))
    assert asyncio.run(DashScopeResponseFormatter().format(messages)) == expected

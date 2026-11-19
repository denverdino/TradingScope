"""Offline replay of synthetic audit captures through the real agent and structured-output paths."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.tool import FunctionTool, Toolkit

from tests.support.replay import ReplayCapture, ReplayChatModel, ReplayFormatter, read_records, replay_tool
from tests.test_options_analysis import _make_options_chain
from tradingscope.agents.output import MarketAnalystOutput, OptionsAnalysis
from tradingscope.agents.utils import core_stock_tools
from tradingscope.agents.utils.agent_utils import call_agent_with_retry
from tradingscope.agents.utils.local_audit import AUDIT_VERSION
from tradingscope.agents.utils.structured_output import StructuredAgentRunner, StructuredOutputValidationError

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "market_options.jsonl"
AGENT = "MarketAnalyst"
PROMPT = Msg(name="user", content=[TextBlock(text="分析 RPLY 期权面")], role="user")


@pytest.fixture
def vendor(monkeypatch):
    route = Mock(side_effect=AssertionError("Replay must not reach a vendor"))
    monkeypatch.setattr(core_stock_tools, "route_to_vendor", route)
    return route


def build_agent(capture: ReplayCapture) -> Agent:
    tool = FunctionTool(replay_tool(core_stock_tools.get_options_analysis, capture, AGENT))
    return Agent(name=AGENT, system_prompt="合成回放", model=ReplayChatModel(capture, AGENT), toolkit=Toolkit(tools=[tool]))


def replayable(records: list[dict]) -> list[tuple]:
    projected = []
    for record in records:
        if record["kind"] == "tool_call":
            projected.append((record["kind"], record["call_id"], record["tool"], record["model_turn"], record["arguments"]))
        elif record["kind"] == "tool_result":
            projected.append((record["kind"], record["call_id"], record["state"], record["text"]))
        elif record["kind"] in ("agent_response", "draft"):
            projected.append((record["kind"], record["text"]))
        elif record["kind"] == "structured_response":
            projected.append((record["kind"], record["attempt"], json.dumps(record["content"], sort_keys=True)))
        elif record["kind"] == "structured_error":
            projected.append((record["kind"], record["attempt"], json.dumps(record["errors"], sort_keys=True)))
    return projected


def fixture_records() -> list[dict]:
    return read_records(FIXTURE)


def synthetic(kind: str, **fields) -> dict:
    return {"audit_version": AUDIT_VERSION, "run_id": "synthetic", "kind": kind, **fields}


@pytest.mark.asyncio
async def test_capture_replays_schema_rejection_and_structured_retry(monkeypatch, tmp_path, vendor):
    monkeypatch.setenv("TOOL_AUDIT_DIR", str(tmp_path))
    capture = ReplayCapture(fixture_records())
    agent = build_agent(capture)

    output = await StructuredAgentRunner(ReplayFormatter(capture, AGENT)).run(agent, MarketAnalystOutput, PROMPT)

    assert output.options_analysis == OptionsAnalysis(
        put_call_ratio=0.88,
        support_levels=[],
        resistance_levels=[],
        max_pain=None,
        interpretation="选用 2026-05-01 到期；PCR 为 OI 口径；Max Pain 非唯一，最小区间 $120–$155；OI 稀疏，价位弃用。",
    )
    capture.assert_exhausted()
    vendor.assert_not_called()
    # Draft runs once: two tool turns plus the final text, despite the structured retry.
    assert agent.model._turn == 3
    rerecorded = read_records(next(tmp_path.glob("*.jsonl")))
    assert replayable(rerecorded) == replayable(fixture_records())
    rejection = next(r for r in rerecorded if r["kind"] == "tool_result" and r["call_id"] == "call-missing-symbol")
    assert rejection["state"] == "error"
    assert "'symbol' is a required property" in rejection["text"]


@pytest.mark.asyncio
async def test_parallel_results_match_by_arguments_and_keep_full_text(monkeypatch, tmp_path, vendor):
    monkeypatch.setenv("TOOL_AUDIT_DIR", str(tmp_path))
    long_text = "AAPL options " + "x" * 700
    records = [
        synthetic(
            "tool_call",
            agent=AGENT,
            call_id="call-a",
            tool="get_options_analysis",
            attempt=0,
            model_turn=0,
            arguments={"symbol": "AAPL"},
            arguments_raw='{"symbol": "AAPL"}',
        ),
        synthetic(
            "tool_call",
            agent=AGENT,
            call_id="call-b",
            tool="get_options_analysis",
            attempt=0,
            model_turn=0,
            arguments={"symbol": "MSFT"},
            arguments_raw='{"symbol": "MSFT"}',
        ),
        synthetic("tool_result", agent=AGENT, call_id="call-b", tool="get_options_analysis", state="success", text="MSFT options"),
        synthetic("tool_result", agent=AGENT, call_id="call-a", tool="get_options_analysis", state="success", text=long_text),
        synthetic("agent_response", agent=AGENT, phase="draft", attempt=0, text="两只标的期权面均已读取"),
    ]
    capture = ReplayCapture(records)

    result = await call_agent_with_retry(build_agent(capture), PROMPT)

    assert result.get_text_content() == "两只标的期权面均已读取"
    capture.assert_exhausted()
    vendor.assert_not_called()
    outputs = {r["call_id"]: r["text"] for r in read_records(next(tmp_path.glob("*.jsonl"))) if r["kind"] == "tool_result"}
    assert outputs == {"call-a": long_text, "call-b": "MSFT options"}


def _mismatched_arguments(records):
    next(r for r in records if r.get("call_id") == "call-options" and r["kind"] == "tool_call")["arguments_raw"] = '{"symbol": "MSFT"}'


def _missing_final_turn(records):
    records[:] = [r for r in records if r["kind"] != "agent_response"]


def _extra_model_turn(records):
    records.append(
        synthetic(
            "tool_call",
            agent=AGENT,
            call_id="call-extra",
            tool="get_options_analysis",
            attempt=0,
            model_turn=5,
            arguments={"symbol": "X"},
            arguments_raw='{"symbol": "X"}',
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_mismatched_arguments, "Unrecorded tool call"),
        (_missing_final_turn, "No recorded model turn 2"),
        (_extra_model_turn, "No recorded model turn 2"),
    ],
)
async def test_divergent_replay_fails_without_reaching_vendor(vendor, mutate, message):
    records = fixture_records()
    mutate(records)
    capture = ReplayCapture(records)

    with pytest.raises(pytest.fail.Exception, match=message):
        await call_agent_with_retry(build_agent(capture), PROMPT)

    vendor.assert_not_called()


@pytest.mark.asyncio
async def test_unconsumed_records_fail_exhaustion_check(vendor):
    records = fixture_records()
    extra = copy.deepcopy(next(r for r in records if r["kind"] == "structured_response" and r["attempt"] == 2))
    records.append({**extra, "attempt": 3})
    capture = ReplayCapture(records)

    await StructuredAgentRunner(ReplayFormatter(capture, AGENT)).run(build_agent(capture), MarketAnalystOutput, PROMPT)

    with pytest.raises(pytest.fail.Exception, match="not replayed"):
        capture.assert_exhausted()


@pytest.mark.parametrize(
    "record",
    [
        synthetic("tool_result", call_id="call-chart", state="success", text="", media=[{"media_type": "image/png"}], replay_supported=False),
        synthetic("tool_incomplete", call_id="call-cut", replay_supported=False),
        {**synthetic("run"), "audit_version": "0.1"},
    ],
)
def test_unsupported_records_are_rejected_on_load(record):
    with pytest.raises(pytest.fail.Exception):
        ReplayCapture([*fixture_records(), record])


@pytest.mark.asyncio
async def test_structured_replay_still_fails_when_every_attempt_is_invalid(vendor):
    records = fixture_records()
    second = next(r for r in records if r["kind"] == "structured_response" and r["attempt"] == 2)
    second["content"]["options_analysis"]["max_pain"] = 0
    capture = ReplayCapture(records)
    agent = build_agent(capture)

    with pytest.raises(StructuredOutputValidationError, match="max_pain"):
        await StructuredAgentRunner(ReplayFormatter(capture, AGENT), max_validation_attempts=2).run(agent, MarketAnalystOutput, PROMPT)

    assert agent.model._turn == 3


def test_recorded_options_output_matches_current_tool_for_fixed_chain(monkeypatch):
    from datetime import datetime

    from tradingscope.dataflows import y_finance

    ticker = Mock()
    ticker.options = ("2026-04-23", "2026-04-24", "2026-05-01")
    chains = {
        "2026-04-24": _make_options_chain(
            {"strike": [180.0, 190.0], "openInterest": [0, 0], "volume": [5, 7]},
            {"strike": [170.0, 180.0], "openInterest": [0, 0], "volume": [3, 2]},
        ),
        "2026-05-01": _make_options_chain(
            {"strike": [155.0, 130.0, 560.0], "openInterest": [47, 0, 121], "volume": [1, 0, 2]},
            {"strike": [120.0, 140.0, 350.0], "openInterest": [147, 0, 0], "volume": [3, 0, 0]},
        ),
    }
    ticker.option_chain.side_effect = lambda expiry: chains[expiry]
    ticker.info = {"currentPrice": 192.50}
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: ticker)
    instant = datetime.fromisoformat("2026-04-23T16:00:00+00:00")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    monkeypatch.setattr(y_finance, "datetime", Clock)
    recorded = next(r["text"] for r in fixture_records() if r["kind"] == "tool_result" and r["call_id"] == "call-options")

    # __wrapped__ bypasses the process-wide tool cache without clearing it.
    assert core_stock_tools.get_options_analysis.__wrapped__("RPLY") == recorded
    assert "- 2026-04-23: expired or same-day expiry excluded by forward-window policy" in recorded
    assert "- 2026-04-24: combined OI = 0" in recorded
    assert "PCR (Open Interest): 0.88" in recorded
    assert "Max Pain Minimum Interval: $120.00–$155.00" in recorded
    assert "Max Pain Price: $" not in recorded
    assert [call.args[0] for call in ticker.option_chain.call_args_list] == ["2026-04-24", "2026-05-01"]

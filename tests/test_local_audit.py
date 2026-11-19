from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentscope.event import ToolCallDeltaEvent, ToolCallEndEvent, ToolCallStartEvent, ToolResultEndEvent, ToolResultStartEvent, ToolResultTextDeltaEvent
from agentscope.message import Msg, TextBlock, ToolResultState
from agentscope.tool import FunctionTool

from tradingscope.agents.utils.agent_utils import call_agent_with_retry


def records(directory: Path) -> list[dict]:
    return [json.loads(line) for path in directory.glob("*.jsonl") for line in path.read_text().splitlines()]


def test_redactor_masks_nested_fields_urls_and_known_secret_text(monkeypatch):
    from tradingscope.agents.utils.local_audit import redact

    secret = "synthetic-secret-12345"
    monkeypatch.setenv("DASHSCOPE_API_KEY", secret)
    payload = {
        "api_key": "not-in-environment",
        "data": [{"Authorization": "Bearer unknown-credential", "password": "other-password"}],
        "message": f"provider failed with {secret}",
        "source": "https://example.invalid/path?symbol=AAPL&token=unknown-token",
        "url": "https://user:other-password@example.invalid/path",
        "input_tokens": 123,
        "date": "2026-11-18",
    }
    masked = redact(payload)
    text = json.dumps(masked)
    assert secret not in text
    assert "other-password" not in text
    assert "unknown-token" not in text
    assert "unknown-credential" not in text
    assert masked["input_tokens"] == 123
    assert masked["date"] == "2026-11-18"
    assert payload["api_key"] == "not-in-environment"
    assert masked["api_key"] == "<REDACTED>"


def test_audit_is_opt_in_and_uses_private_unique_files(monkeypatch, tmp_path):
    from tradingscope.agents.utils.local_audit import LocalAuditRecorder, get_audit_recorder

    monkeypatch.delenv("TOOL_AUDIT_DIR", raising=False)
    assert get_audit_recorder() is None
    directory = tmp_path / "audit"
    first = LocalAuditRecorder(directory)
    second = LocalAuditRecorder(directory)
    assert first.path != second.path
    assert first.path.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert records(directory)[0]["kind"] == "run"
    assert "agentscope_version" in records(directory)[0]


@pytest.mark.asyncio
async def test_interleaved_calls_preserve_full_results_and_redact_split_secrets(monkeypatch, tmp_path):
    secret = "synthetic-secret-12345"
    monkeypatch.setenv("DASHSCOPE_API_KEY", secret)
    monkeypatch.setenv("TOOL_AUDIT_DIR", str(tmp_path))

    def quote(symbol: str):
        return symbol

    tool = FunctionTool(quote)
    events = [
        ToolCallStartEvent(reply_id="r", tool_call_id="a", tool_call_name="quote"),
        ToolCallDeltaEvent(reply_id="r", tool_call_id="a", delta='{"symbol":"AAPL","token":"synthetic-'),
        ToolCallStartEvent(reply_id="r", tool_call_id="b", tool_call_name="quote"),
        ToolCallDeltaEvent(reply_id="r", tool_call_id="b", delta='{"ticker":"MSFT"}'),
        ToolCallDeltaEvent(reply_id="r", tool_call_id="a", delta='secret-12345"}'),
        ToolCallEndEvent(reply_id="r", tool_call_id="a"),
        ToolCallEndEvent(reply_id="r", tool_call_id="b"),
        ToolResultStartEvent(reply_id="r", tool_call_id="a", tool_call_name="quote"),
        ToolResultStartEvent(reply_id="r", tool_call_id="b", tool_call_name="quote"),
        ToolResultTextDeltaEvent(reply_id="r", tool_call_id="a", delta="AAPL " + "x" * 600 + " synthetic-"),
        ToolResultTextDeltaEvent(reply_id="r", tool_call_id="b", delta="Input validation failed: missing symbol"),
        ToolResultTextDeltaEvent(reply_id="r", tool_call_id="a", delta="secret-12345"),
        ToolResultEndEvent(reply_id="r", tool_call_id="b", state=ToolResultState.ERROR),
        ToolResultEndEvent(reply_id="r", tool_call_id="a", state=ToolResultState.SUCCESS, metadata={"cache_hit": True}),
    ]

    class Agent:
        name = "MarketAnalyst"
        toolkit = SimpleNamespace(get_tool=AsyncMock(return_value=tool))

        async def _reply(self, prompt):
            for event in events:
                yield event
            yield Msg(name=self.name, content=[TextBlock(text="finished")], role="assistant")

    result = await call_agent_with_retry(Agent(), None)
    assert result.get_text_content() == "finished"
    saved = records(tmp_path)
    calls = {entry["call_id"]: entry for entry in saved if entry["kind"] == "tool_call"}
    outputs = {entry["call_id"]: entry for entry in saved if entry["kind"] == "tool_result"}
    assert calls["a"]["arguments"] == {"symbol": "AAPL", "token": "<REDACTED>"}
    assert calls["b"]["arguments"] == {"ticker": "MSFT"}
    assert calls["a"]["model_turn"] == calls["b"]["model_turn"] == 0
    assert calls["b"]["schema"]["required"] == ["symbol"]
    assert outputs["b"]["state"] == "error"
    assert outputs["a"]["text"] == "AAPL " + "x" * 600 + " <REDACTED>"
    assert outputs["a"]["metadata"]["cache_hit"] is True
    assert outputs["a"]["elapsed_seconds"] >= 0
    assert secret not in "".join(path.read_text() for path in tmp_path.glob("*.jsonl"))


def test_debug_cache_and_errors_are_redacted_without_changing_content(monkeypatch):
    from tradingscope.agents.utils import tool_response_helper

    secret = "synthetic-secret-12345"
    monkeypatch.setenv("DASHSCOPE_API_KEY", secret)
    monkeypatch.setenv("TOOL_DEBUG", "1")
    logged = []
    monkeypatch.setattr(tool_response_helper.logger, "info", lambda *args: logged.append(str(args)))
    monkeypatch.setattr(tool_response_helper.logger, "error", lambda *args: logged.append(str(args)))

    @tool_response_helper.agentscope_tool
    def echo(value):
        return value

    first = echo(secret)
    second = echo(secret)
    assert first.content[0].text == secret
    assert second.content[0].text == secret
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True

    @tool_response_helper.agentscope_tool
    def fail():
        raise RuntimeError(secret)

    with pytest.raises(RuntimeError, match=secret):
        fail()
    assert secret not in "\n".join(logged)


def test_audit_write_failure_does_not_print_raw_record(monkeypatch, tmp_path, caplog):
    from tradingscope.agents.utils.local_audit import LocalAuditRecorder

    recorder = LocalAuditRecorder(tmp_path)

    def denied(*args, **kwargs):
        raise OSError("untrusted-raw-secret")

    monkeypatch.setattr(Path, "open", denied)
    recorder.record("model_response", text="untrusted-raw-secret")
    assert "untrusted-raw-secret" not in caplog.text
    assert "Local audit record could not be written" in caplog.text


@pytest.mark.asyncio
async def test_structured_audit_preserves_attempts_schema_and_feedback(monkeypatch, tmp_path):
    from tests.test_output_models import _all_outputs
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    monkeypatch.setenv("TOOL_AUDIT_DIR", str(tmp_path))
    expected = _all_outputs()[0]
    invalid = {**expected.model_dump(mode="json"), "decision": {}}
    formatter = SimpleNamespace(
        generate_structured_output=AsyncMock(side_effect=[SimpleNamespace(content=invalid), SimpleNamespace(content=expected.model_dump(mode="json"))])
    )
    runner = StructuredAgentRunner(formatter)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "素材"))
    result = await runner.run(SimpleNamespace(name="MarketAnalyst"), type(expected))
    assert result == expected
    saved = records(tmp_path)
    requests = [entry for entry in saved if entry["kind"] == "structured_request"]
    responses = [entry for entry in saved if entry["kind"] == "structured_response"]
    assert [entry["attempt"] for entry in responses] == [1, 2]
    assert responses[0]["content"]["decision"] == {}
    assert requests[0]["schema"]["properties"]["decision"]
    assert "decision.direction" in requests[1]["text"]
    assert any(entry["kind"] == "structured_error" for entry in saved)
    assert responses[1]["analysis_date"] == expected.analysis_date.isoformat()
    runner._run_analysis.assert_awaited_once()

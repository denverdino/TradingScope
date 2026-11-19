"""Deterministic replay of local audit captures for offline tests."""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentscope.credential import DashScopeCredential
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import TextBlock, ToolCallBlock
from agentscope.model import ChatModelBase, ChatResponse, ChatUsage
from pydantic import BaseModel

from tradingscope.agents.utils.local_audit import AUDIT_VERSION


def canonical(arguments) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def read_records(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class ReplayCapture:
    """Index audit records by agent/attempt/turn for models and by name/arguments for tools."""

    def __init__(self, records: list[dict]) -> None:
        self.turns: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        self.usage: dict[tuple, dict] = {}
        self.final_text: dict[tuple, str] = {}
        self.tool_results: dict[tuple, list[dict]] = defaultdict(list)
        self.structured: dict[tuple, dict] = {}
        self.requests: dict[tuple, dict] = {}
        self.divergences: list[str] = []
        calls: dict[str, dict] = {}
        for record in records:
            if record.get("audit_version") != AUDIT_VERSION:
                pytest.fail(f"Unsupported audit version: {record.get('audit_version')}", pytrace=False)
            if record["kind"] == "tool_incomplete" or record.get("replay_supported") is False:
                pytest.fail(f"Capture record is not replayable: {record['kind']} {record.get('call_id')}", pytrace=False)
            kind = record["kind"]
            if kind == "tool_call":
                calls[record["call_id"]] = record
                self.turns[(record["agent"], record["attempt"])][record["model_turn"]].append(record)
            elif kind == "tool_result":
                call = calls.get(record["call_id"])
                if call is None:
                    pytest.fail(f"Tool result without a recorded call: {record['call_id']}", pytrace=False)
                self.tool_results[(call["agent"], call["tool"], canonical(call["arguments"]))].append(record)
            elif kind == "model_usage":
                self.usage[(record["agent"], record["attempt"], record["model_turn"])] = record
            elif kind == "agent_response":
                self.final_text[(record["agent"], record["attempt"])] = record["text"]
            elif kind == "structured_request":
                self.requests[(record["agent"], record["attempt"])] = record
            elif kind == "structured_response":
                self.structured[(record["agent"], record["attempt"])] = record

    @classmethod
    def load(cls, path: str | Path) -> ReplayCapture:
        return cls(read_records(path))

    def model_response(self, agent: str, attempt: int, turn: int) -> ChatResponse:
        if self.divergences:
            pytest.fail(self.divergences[0], pytrace=False)
        turns = self.turns.get((agent, attempt), {})
        if turn in turns:
            content = [ToolCallBlock(id=call["call_id"], name=call["tool"], input=call["arguments_raw"]) for call in turns.pop(turn)]
        elif not turns and (agent, attempt) in self.final_text:
            content = [TextBlock(text=self.final_text.pop((agent, attempt)))]
        else:
            pytest.fail(f"No recorded model turn {turn} for {agent} attempt {attempt}", pytrace=False)
        usage = self.usage.pop((agent, attempt, turn), {})
        return ChatResponse(
            content=content,
            is_last=True,
            usage=ChatUsage(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0), time=0.0),
        )

    def tool_output(self, agent: str, tool: str, arguments: dict) -> str:
        results = self.tool_results.get((agent, tool, canonical(arguments)))
        if not results:
            # AgentScope's parallel tool runner drops BaseException results, so the next model turn re-raises this.
            self.divergences.append(f"Unrecorded tool call: {agent} {tool}({canonical(arguments)})")
            pytest.fail(self.divergences[-1], pytrace=False)
        result = results.pop(0)
        if result["state"] != "success":
            raise RuntimeError(result["text"])
        return result["text"]

    def structured_response(self, agent: str, attempt: int, request_text: str, schema: dict):
        request = self.requests.pop((agent, attempt), None)
        if request is None or (agent, attempt) not in self.structured:
            pytest.fail(f"No recorded structured attempt {attempt} for {agent}", pytrace=False)
        if request["text"] != request_text or request["schema"] != schema:
            pytest.fail(f"Structured request for {agent} attempt {attempt} differs from the capture", pytrace=False)
        return SimpleNamespace(content=self.structured.pop((agent, attempt))["content"])

    def assert_exhausted(self) -> None:
        # Error results may never reach the tool: AgentScope rejects invalid arguments first.
        leftovers = {
            "divergences": self.divergences,
            "model_turns": [key for key, turns in self.turns.items() if turns],
            "final_text": list(self.final_text),
            "tool_results": [key for key, results in self.tool_results.items() if any(r["state"] == "success" for r in results)],
            "structured": list(self.structured),
        }
        if any(leftovers.values()):
            pytest.fail(f"Capture records were not replayed: {leftovers}", pytrace=False)


class _Parameters(BaseModel):
    pass


class ReplayChatModel(ChatModelBase):
    """Serve recorded model turns for one agent in place of the provider."""

    Parameters = _Parameters

    def __init__(self, capture: ReplayCapture, agent: str, attempt: int = 0) -> None:
        super().__init__(credential=DashScopeCredential(api_key="replay-only"), model="replay", parameters=_Parameters(), stream=False)
        self.formatter = DashScopeChatFormatter()
        self._capture = capture
        self._agent = agent
        self._attempt = attempt
        self._turn = 0

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs) -> ChatResponse:
        response = self._capture.model_response(self._agent, self._attempt, self._turn)
        self._turn += 1
        return response


def replay_tool(func, capture: ReplayCapture, agent: str):
    """Keep the production signature and docstring so AgentScope builds the real schema."""

    @functools.wraps(func)
    def replay(**kwargs):
        return capture.tool_output(agent, func.__name__, kwargs)

    return replay


class ReplayFormatter:
    def __init__(self, capture: ReplayCapture, agent: str) -> None:
        self._capture = capture
        self._agent = agent
        self._attempt = 0

    async def generate_structured_output(self, messages, structured_model, **kwargs):
        self._attempt += 1
        return self._capture.structured_response(self._agent, self._attempt, messages[0].get_text_content(), structured_model.model_json_schema())

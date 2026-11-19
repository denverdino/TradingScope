from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from agentscope.event import (
    ModelCallEndEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultDataDeltaEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)

logger = logging.getLogger(__name__)
AUDIT_VERSION = "1.0"
REDACTED = "<REDACTED>"
_SENSITIVE = re.compile(r"api[_-]?key|access[_-]?key|authorization|password|passwd|secret|credential|(?:^|[_-])token(?:$|[_-])", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>\"']+")
_INLINE = re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?key|token|password|secret|authorization)\b[\"']?\s*[:=]\s*[\"']?)([^\s,;\"'}]+)")


def _redact_url(match: re.Match) -> str:
    original = match.group(0)
    try:
        parts = urlsplit(original)
        query = parse_qsl(parts.query, keep_blank_values=True)
        sensitive_query = any(_SENSITIVE.search(key) or key.lower() == "key" for key, _ in query)
        if "@" not in parts.netloc and not sensitive_query:
            return original
        netloc = parts.netloc.rsplit("@", 1)[-1]
        masked = [(key, REDACTED if _SENSITIVE.search(key) or key.lower() == "key" else value) for key, value in query]
        return urlunsplit((parts.scheme, netloc, parts.path, urlencode(masked), parts.fragment))
    except ValueError:
        return REDACTED


def redact(value):
    if isinstance(value, dict):
        return {key: REDACTED if _SENSITIVE.search(str(key)) or str(key).lower() == "key" else redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value
    for name, secret in os.environ.items():
        if _SENSITIVE.search(name) and len(secret) >= 8:
            text = text.replace(secret, REDACTED)
            escaped = json.dumps(secret, ensure_ascii=False)[1:-1]
            text = text.replace(escaped, REDACTED)
    text = _URL.sub(_redact_url, text)
    text = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer " + REDACTED, text)
    return _INLINE.sub(lambda match: match.group(1) + REDACTED, text)


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _run_metadata() -> dict:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    sources = {}
    for name in ("agent_utils.py", "local_audit.py", "structured_output.py", "tool_response_helper.py"):
        path = Path(__file__).parent / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "agentscope_version": version("agentscope"),
        "revision": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_hashes": sources,
    }


class LocalAuditRecorder:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        missing = []
        current = directory
        while not current.exists():
            missing.append(current)
            current = current.parent
        for path in reversed(missing):
            path.mkdir(mode=0o700, exist_ok=True)
        self.run_id = uuid4().hex
        self.path = directory / f"audit-{self.run_id}.jsonl"
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        self.calls: dict[tuple, dict] = {}
        self.turns: dict[tuple, int] = {}
        self.record("run", **_run_metadata())

    def record(self, kind: str, **payload) -> None:
        try:
            entry = {"audit_version": AUDIT_VERSION, "run_id": self.run_id, "kind": kind, "recorded_at": datetime.now(timezone.utc).isoformat(), **payload}
            line = json.dumps(redact(entry), ensure_ascii=False) + "\n"
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except Exception:
            logger.warning("Local audit record could not be written")

    async def observe(self, agent, event, attempt: int) -> None:
        turn_key = (id(agent), attempt)
        if isinstance(event, ModelCallEndEvent):
            turn = self.turns.get(turn_key, 0)
            self.record("model_usage", agent=agent.name, attempt=attempt, model_turn=turn, **event.model_dump(mode="json"))
            self.turns[turn_key] = turn + 1
            return
        if not isinstance(
            event,
            (
                ToolCallStartEvent,
                ToolCallDeltaEvent,
                ToolCallEndEvent,
                ToolResultStartEvent,
                ToolResultTextDeltaEvent,
                ToolResultDataDeltaEvent,
                ToolResultEndEvent,
            ),
        ):
            return
        key = (id(agent), event.reply_id, event.tool_call_id)
        if isinstance(event, ToolCallStartEvent):
            self.calls[key] = {
                "agent": agent.name,
                "reply_id": event.reply_id,
                "call_id": event.tool_call_id,
                "tool": event.tool_call_name,
                "attempt": attempt,
                "model_turn": self.turns.get(turn_key, 0),
                "arguments_raw": "",
                "text": "",
                "media": [],
                "started_at": event.created_at,
                "clock": monotonic(),
            }
        elif key in self.calls:
            call = self.calls[key]
            if isinstance(event, ToolCallDeltaEvent):
                call["arguments_raw"] += event.delta
            elif isinstance(event, ToolCallEndEvent):
                try:
                    arguments = json.loads(call["arguments_raw"])
                except json.JSONDecodeError:
                    arguments = None
                schema = None
                toolkit = getattr(agent, "toolkit", None)
                if toolkit is not None:
                    tool = await toolkit.get_tool(call["tool"])
                    if tool is not None:
                        schema = tool.input_schema
                self.record(
                    "tool_call",
                    **self._identity(call),
                    arguments=arguments,
                    arguments_raw=call["arguments_raw"],
                    schema=schema,
                    schema_hash=_fingerprint(schema),
                )
            elif isinstance(event, ToolResultStartEvent):
                call["clock"] = monotonic()
                call["started_at"] = event.created_at
            elif isinstance(event, ToolResultTextDeltaEvent):
                call["text"] += event.delta
            elif isinstance(event, ToolResultDataDeltaEvent):
                call["media"].append({"block_id": event.block_id, "media_type": event.media_type, "url": event.url, "replay_supported": False})
            elif isinstance(event, ToolResultEndEvent):
                self.record(
                    "tool_result",
                    **self._identity(call),
                    text=call["text"],
                    media=call["media"],
                    state=event.state,
                    metadata=event.metadata,
                    started_at=call["started_at"],
                    completed_at=event.created_at,
                    elapsed_seconds=monotonic() - call["clock"],
                    replay_supported=not call["media"],
                )
                del self.calls[key]

    @staticmethod
    def _identity(call: dict) -> dict:
        return {name: call[name] for name in ("agent", "reply_id", "call_id", "tool", "attempt", "model_turn")}

    def finish(self, agent, attempt: int, error: bool = False) -> None:
        for key in [key for key in self.calls if key[0] == id(agent)]:
            call = self.calls.pop(key)
            self.record("tool_incomplete", **self._identity(call), arguments_raw=call["arguments_raw"], text=call["text"], replay_supported=False)
        self.turns.pop((id(agent), attempt), None)
        if error:
            self.record("agent_error", agent=agent.name, attempt=attempt)


@lru_cache(maxsize=8)
def _recorder(directory: str) -> LocalAuditRecorder | None:
    try:
        return LocalAuditRecorder(directory)
    except Exception:
        logger.warning("Local audit could not be initialized")
        return None


def get_audit_recorder() -> LocalAuditRecorder | None:
    directory = os.getenv("TOOL_AUDIT_DIR")
    return _recorder(directory) if directory else None

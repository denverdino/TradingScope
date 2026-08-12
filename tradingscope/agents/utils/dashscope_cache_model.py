"""DashScope model compatibility for explicit-cache usage metadata."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from agentscope.model import ChatResponse, DashScopeChatModel


def _cache_creation_input_tokens(usage: Any) -> int:
    """Read DashScope's provider-specific cache creation token count."""
    details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(details, dict):
        return int(details.get("cache_creation_input_tokens") or 0)
    return int(getattr(details, "cache_creation_input_tokens", 0) or 0)


class _CacheCreationTrackingStream:
    """Proxy an OpenAI async stream while retaining its latest usage data."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.stream: Any = None
        self.cache_creation_input_tokens = 0

    async def __aenter__(self):
        self.stream = await self.response.__aenter__()
        return self

    async def __aexit__(self, *args: Any):
        return await self.response.__aexit__(*args)

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self.stream.__anext__()
        if chunk.usage:
            self.cache_creation_input_tokens = _cache_creation_input_tokens(chunk.usage)
        return chunk


class CacheAwareDashScopeChatModel(DashScopeChatModel):
    """Preserve cache-creation usage omitted by AgentScope 2.0.6."""

    def _parse_completion_response(
        self,
        start_datetime: datetime,
        response: Any,
    ) -> ChatResponse:
        parsed = super()._parse_completion_response(start_datetime, response)
        if parsed.usage is not None and response.usage is not None:
            parsed.usage.cache_creation_input_tokens = _cache_creation_input_tokens(response.usage)
        return parsed

    async def _parse_stream_response(
        self,
        start_datetime: datetime,
        response: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        tracked = _CacheCreationTrackingStream(response)
        async for parsed in super()._parse_stream_response(start_datetime, tracked):
            if parsed.usage is not None:
                parsed.usage.cache_creation_input_tokens = tracked.cache_creation_input_tokens
            yield parsed

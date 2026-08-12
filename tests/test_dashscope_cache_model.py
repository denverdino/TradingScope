from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from agentscope.model import ChatResponse, ChatUsage, DashScopeChatModel

from tradingscope.agents.utils.dashscope_cache_model import CacheAwareDashScopeChatModel


def _raw_usage(*, created: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(
            cache_creation_input_tokens=created,
        ),
    )


def test_cache_aware_model_preserves_creation_usage_for_completion() -> None:
    parsed = ChatResponse(
        content=[],
        is_last=True,
        usage=ChatUsage(input_tokens=1_500, output_tokens=100, time=0.2),
    )
    raw = SimpleNamespace(usage=_raw_usage(created=1_200))
    model = CacheAwareDashScopeChatModel.__new__(CacheAwareDashScopeChatModel)

    with patch.object(
        DashScopeChatModel,
        "_parse_completion_response",
        return_value=parsed,
    ):
        result = model._parse_completion_response(datetime.now(), raw)

    assert result.usage.cache_creation_input_tokens == 1_200


def test_cache_aware_model_preserves_creation_usage_for_stream() -> None:
    class RawStream:
        def __init__(self) -> None:
            self._chunks = iter([SimpleNamespace(usage=_raw_usage(created=1_300))])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    async def parent_parser(self, start_datetime, response):
        async with response as stream:
            async for _ in stream:
                yield ChatResponse(
                    content=[],
                    is_last=True,
                    usage=ChatUsage(input_tokens=1_600, output_tokens=100, time=0.2),
                )

    model = CacheAwareDashScopeChatModel.__new__(CacheAwareDashScopeChatModel)

    async def parse() -> list[ChatResponse]:
        with patch.object(DashScopeChatModel, "_parse_stream_response", parent_parser):
            return [
                response
                async for response in model._parse_stream_response(
                    datetime.now(),
                    RawStream(),
                )
            ]

    responses = asyncio.run(parse())

    assert responses[-1].usage.cache_creation_input_tokens == 1_300

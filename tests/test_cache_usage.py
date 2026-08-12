from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from agentscope.model import ChatResponse, ChatUsage, FinishedReason

from tradingscope.agents.utils.cache_usage import CacheUsageCollector, CacheUsageMiddleware


def test_cache_usage_middleware_records_non_streaming_response() -> None:
    collector = CacheUsageCollector()
    middleware = CacheUsageMiddleware(collector)
    formatter = SimpleNamespace(confirm_cache_prefixes=lambda messages, *, confirmed_at: None)
    response = ChatResponse(
        content=[],
        is_last=True,
        usage=ChatUsage(
            input_tokens=1_500,
            output_tokens=200,
            cache_input_tokens=1_200,
            cache_creation_input_tokens=1_200,
            time=0.5,
        ),
    )

    async def next_handler(**kwargs):
        return response

    with (
        patch("agentscope.logger.info") as log_info,
        patch.object(formatter, "confirm_cache_prefixes", wraps=formatter.confirm_cache_prefixes) as confirm,
        patch("tradingscope.agents.utils.cache_usage.monotonic", return_value=100.0),
    ):
        result = asyncio.run(
            middleware.on_model_call(
                agent=SimpleNamespace(name="BullResearcher"),
                input_kwargs={
                    "current_model": SimpleNamespace(model="qwen3.8-max", formatter=formatter),
                    "messages": ["cacheable prompt"],
                },
                next_handler=next_handler,
            ),
        )

    assert result is response
    confirm.assert_called_once_with(["cacheable prompt"], confirmed_at=100.0)
    assert collector.snapshot() == {
        "calls": 1,
        "input_tokens": 1_500,
        "output_tokens": 200,
        "cache_input_tokens": 1_200,
    }
    log_args = log_info.call_args.args
    assert "cache_read=%d" in log_args[0]
    assert log_args[1:] == ("BullResearcher", "qwen3.8-max", 1_200, 80.0)


def test_cache_usage_middleware_does_not_confirm_ineligible_prefix() -> None:
    collector = CacheUsageCollector()
    middleware = CacheUsageMiddleware(collector)
    formatter = SimpleNamespace(confirm_cache_prefixes=lambda messages, *, confirmed_at: None)
    response = ChatResponse(
        content=[],
        is_last=True,
        usage=ChatUsage(
            input_tokens=800,
            output_tokens=100,
            cache_input_tokens=0,
            cache_creation_input_tokens=0,
            time=0.2,
        ),
    )

    async def next_handler(**kwargs):
        return response

    with (
        patch.object(
            formatter,
            "confirm_cache_prefixes",
            wraps=formatter.confirm_cache_prefixes,
        ) as confirm,
        patch("tradingscope.agents.utils.cache_usage.monotonic", return_value=200.0),
    ):
        asyncio.run(
            middleware.on_model_call(
                agent=SimpleNamespace(name="BullResearcher"),
                input_kwargs={
                    "current_model": SimpleNamespace(model="qwen3.8-max", formatter=formatter),
                    "messages": ["short prompt"],
                },
                next_handler=next_handler,
            ),
        )

    confirm.assert_not_called()


def test_cache_usage_middleware_does_not_confirm_interrupted_response() -> None:
    collector = CacheUsageCollector()
    middleware = CacheUsageMiddleware(collector)
    formatter = SimpleNamespace(confirm_cache_prefixes=lambda messages, *, confirmed_at: None)
    response = ChatResponse(
        content=[],
        is_last=True,
        finished_reason=FinishedReason.INTERRUPTED,
        usage=ChatUsage(
            input_tokens=1_500,
            output_tokens=50,
            cache_input_tokens=0,
            cache_creation_input_tokens=1_200,
            time=0.2,
        ),
    )

    async def next_handler(**kwargs):
        return response

    with patch.object(
        formatter,
        "confirm_cache_prefixes",
        wraps=formatter.confirm_cache_prefixes,
    ) as confirm:
        asyncio.run(
            middleware.on_model_call(
                agent=SimpleNamespace(name="BullResearcher"),
                input_kwargs={
                    "current_model": SimpleNamespace(model="qwen3.8-max", formatter=formatter),
                    "messages": ["cacheable prompt"],
                },
                next_handler=next_handler,
            ),
        )

    confirm.assert_not_called()


def test_cache_usage_middleware_records_final_streaming_response() -> None:
    collector = CacheUsageCollector()
    middleware = CacheUsageMiddleware(collector)
    formatter = SimpleNamespace(confirm_cache_prefixes=lambda messages, *, confirmed_at: None)

    async def response_stream():
        yield ChatResponse(content=[], is_last=False)
        yield ChatResponse(
            content=[],
            is_last=True,
            usage=ChatUsage(
                input_tokens=2_000,
                output_tokens=300,
                cache_input_tokens=1_500,
                cache_creation_input_tokens=1_000,
                time=0.8,
            ),
        )

    async def next_handler(**kwargs):
        return response_stream()

    with (
        patch.object(
            formatter,
            "confirm_cache_prefixes",
            wraps=formatter.confirm_cache_prefixes,
        ) as confirm,
        patch("tradingscope.agents.utils.cache_usage.monotonic", return_value=200.0),
    ):

        async def consume() -> list[ChatResponse]:
            stream = await middleware.on_model_call(
                agent=SimpleNamespace(name="BearResearcher"),
                input_kwargs={
                    "current_model": SimpleNamespace(model="qwen3.8-max", formatter=formatter),
                    "messages": ["cacheable prompt"],
                },
                next_handler=next_handler,
            )
            return [chunk async for chunk in stream]

        chunks = asyncio.run(consume())

    assert len(chunks) == 2
    confirm.assert_called_once_with(["cacheable prompt"], confirmed_at=200.0)
    assert collector.snapshot() == {
        "calls": 1,
        "input_tokens": 2_000,
        "output_tokens": 300,
        "cache_input_tokens": 1_500,
    }


def test_cache_usage_middleware_does_not_confirm_interrupted_stream() -> None:
    collector = CacheUsageCollector()
    middleware = CacheUsageMiddleware(collector)
    formatter = SimpleNamespace(confirm_cache_prefixes=lambda messages, *, confirmed_at: None)

    async def response_stream():
        yield ChatResponse(content=[], is_last=False)
        yield ChatResponse(
            content=[],
            is_last=True,
            finished_reason=FinishedReason.INTERRUPTED,
            usage=ChatUsage(
                input_tokens=2_000,
                output_tokens=100,
                cache_input_tokens=0,
                cache_creation_input_tokens=1_500,
                time=0.5,
            ),
        )

    async def next_handler(**kwargs):
        return response_stream()

    with patch.object(
        formatter,
        "confirm_cache_prefixes",
        wraps=formatter.confirm_cache_prefixes,
    ) as confirm:

        async def consume() -> None:
            stream = await middleware.on_model_call(
                agent=SimpleNamespace(name="BearResearcher"),
                input_kwargs={
                    "current_model": SimpleNamespace(model="qwen3.8-max", formatter=formatter),
                    "messages": ["cacheable prompt"],
                },
                next_handler=next_handler,
            )
            _ = [chunk async for chunk in stream]

        asyncio.run(consume())

    confirm.assert_not_called()


def test_cache_usage_collector_logs_workflow_summary() -> None:
    collector = CacheUsageCollector(
        calls=2,
        input_tokens=4_000,
        output_tokens=500,
        cache_input_tokens=3_000,
    )

    with patch("agentscope.logger.info") as log_info:
        collector.log_summary()

    log_args = log_info.call_args.args
    assert "workflow cache summary" in log_args[0]
    assert log_args[1:] == (2, 4_000, 500, 3_000, 75.0)


def test_cache_usage_uses_agentscope_logger() -> None:
    collector = CacheUsageCollector(
        calls=1,
        input_tokens=1_000,
        output_tokens=100,
        cache_input_tokens=800,
    )

    with patch("agentscope.logger.info") as log_info:
        collector.log_summary()

    log_info.assert_called_once()

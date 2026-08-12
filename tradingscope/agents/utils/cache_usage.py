"""Collect cache-related token usage from AgentScope model calls."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from time import monotonic

from agentscope import logger
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatResponse, FinishedReason


@dataclass
class CacheUsageCollector:
    """Workflow-level aggregate of model token usage."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_input_tokens: int = 0

    def record(self, response: ChatResponse) -> None:
        """Add usage from a completed model response when available."""
        if response.usage is None:
            return
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cache_input_tokens += response.usage.cache_input_tokens

    def snapshot(self) -> dict[str, int]:
        """Return a serializable aggregate snapshot."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_input_tokens": self.cache_input_tokens,
        }

    def log_summary(self) -> None:
        """Log aggregate usage and cache hit ratio for the workflow."""
        hit_ratio = self.cache_input_tokens / self.input_tokens if self.input_tokens else 0.0
        logger.info(
            "workflow cache summary: calls=%d input=%d output=%d cache_read=%d cache_hit_ratio=%.1f%%",
            self.calls,
            self.input_tokens,
            self.output_tokens,
            self.cache_input_tokens,
            hit_ratio * 100,
        )


class CacheUsageMiddleware(MiddlewareBase):
    """Capture usage returned by completed AgentScope model calls."""

    def __init__(self, collector: CacheUsageCollector) -> None:
        self.collector = collector

    async def on_model_call(self, agent, input_kwargs, next_handler):
        request_started_at = monotonic()
        response = await next_handler(**input_kwargs)
        agent_name = getattr(agent, "name", "unknown")
        current_model = input_kwargs.get("current_model")
        model_name = getattr(current_model, "model", "unknown")
        messages = input_kwargs.get("messages", [])
        if isinstance(response, ChatResponse):
            self._record(
                agent_name,
                model_name,
                response,
                current_model,
                messages,
                request_started_at,
            )
            return response
        if inspect.isasyncgen(response):
            return self._record_stream(
                agent_name,
                model_name,
                response,
                current_model,
                messages,
                request_started_at,
            )
        return response

    def _record(
        self,
        agent_name: str,
        model_name: str,
        response: ChatResponse,
        current_model,
        messages,
        request_started_at: float,
    ) -> None:
        formatter = getattr(current_model, "formatter", None)
        confirm = getattr(formatter, "confirm_cache_prefixes", None)
        usage = response.usage
        cache_available = usage is not None and (usage.cache_creation_input_tokens > 0 or usage.cache_input_tokens > 0)
        if callable(confirm) and response.is_last and response.finished_reason == FinishedReason.COMPLETED and cache_available:
            confirm(messages, confirmed_at=request_started_at)
        self.collector.record(response)
        if usage is None:
            return
        hit_ratio = usage.cache_input_tokens / usage.input_tokens if usage.input_tokens else 0.0
        logger.info(
            "[%s] model=%s cache_read=%d cache_hit_ratio=%.1f%%",
            agent_name,
            model_name,
            usage.cache_input_tokens,
            hit_ratio * 100,
        )

    async def _record_stream(
        self,
        agent_name: str,
        model_name: str,
        stream,
        current_model,
        messages,
        request_started_at: float,
    ) -> AsyncGenerator[ChatResponse, None]:
        async for response in stream:
            if response.is_last:
                self._record(
                    agent_name,
                    model_name,
                    response,
                    current_model,
                    messages,
                    request_started_at,
                )
            yield response

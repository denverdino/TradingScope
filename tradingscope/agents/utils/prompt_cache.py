"""Prompt helpers for DashScope explicit context caching."""

from __future__ import annotations

from time import monotonic
from typing import Any

from agentscope.formatter import DashScopeChatFormatter
from pydantic import PrivateAttr

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT

CACHE_PREFIX_BREAK = "\n<!-- tradingscope:cache-prefix-end -->\n"
CACHE_TTL_SECONDS = 300.0


def build_cacheable_system_prompt(*, shared_context: str, role_instructions: str) -> str:
    """Place an identical stage context before role-specific instructions."""
    shared_prefix = f"{COMPLIANCE_PROMPT}\n\n# 共享分析上下文\n\n{shared_context}"
    return f"{shared_prefix}{CACHE_PREFIX_BREAK}{role_instructions}"


class CachedDashScopeChatFormatter(DashScopeChatFormatter):
    """Translate the internal cache boundary into DashScope cache_control."""

    _confirmed_prefixes: dict[str, float] = PrivateAttr(default_factory=dict)

    @staticmethod
    def _cache_prefixes(msgs) -> set[str]:
        prefixes: set[str] = set()
        for msg in msgs:
            for block in getattr(msg, "content", []):
                text = getattr(block, "text", None)
                if isinstance(text, str) and CACHE_PREFIX_BREAK in text:
                    prefixes.add(text.split(CACHE_PREFIX_BREAK, 1)[0])
        return prefixes

    def confirm_cache_prefixes(self, msgs, *, confirmed_at: float) -> None:
        """Record prefixes whose cache-creation response completed."""
        for prefix in self._cache_prefixes(msgs):
            self._confirmed_prefixes[prefix] = confirmed_at

    def _is_confirmed(self, prefix: str) -> bool:
        confirmed_at = self._confirmed_prefixes.get(prefix)
        if confirmed_at is None:
            return False
        if monotonic() - confirmed_at < CACHE_TTL_SECONDS:
            return True
        self._confirmed_prefixes.pop(prefix, None)
        return False

    async def format(self, msgs) -> list[dict[str, Any]]:
        formatted_messages = await super().format(msgs)
        cache_aware_messages: list[dict[str, Any]] = []
        for message in formatted_messages:
            content = message.get("content")
            if not isinstance(content, list):
                cache_aware_messages.append(message)
                continue
            for block in content:
                text = block.get("text") if isinstance(block, dict) else None
                if not isinstance(text, str) or CACHE_PREFIX_BREAK not in text:
                    continue
                shared_prefix, role_instructions = text.split(CACHE_PREFIX_BREAK, 1)
                message_base = {key: value for key, value in message.items() if key != "content"}
                if self._is_confirmed(shared_prefix):
                    cache_aware_messages.extend(
                        [
                            {**message_base, "content": shared_prefix},
                            {
                                **message_base,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": role_instructions,
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                ],
                            },
                        ],
                    )
                else:
                    cache_aware_messages.extend(
                        [
                            {
                                **message_base,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": shared_prefix,
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                ],
                            },
                            {
                                **message_base,
                                "content": [{"type": "text", "text": role_instructions}],
                            },
                        ],
                    )
                break
            else:
                cache_aware_messages.append(message)
        return cache_aware_messages

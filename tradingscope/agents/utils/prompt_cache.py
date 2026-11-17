"""Prompt helpers for stable shared analysis context prefixes."""

from __future__ import annotations

from typing import Any

from agentscope.formatter import OpenAIResponseFormatter
from agentscope.message import Msg

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT

CACHE_PREFIX_BREAK = "\n<!-- tradingscope:cache-prefix-end -->\n"


def build_cacheable_system_prompt(*, shared_context: str, role_instructions: str) -> str:
    """Place an identical stage context before role-specific instructions."""
    shared_prefix = f"{COMPLIANCE_PROMPT}\n\n# 共享分析上下文\n\n{shared_context}"
    return f"{shared_prefix}{CACHE_PREFIX_BREAK}{role_instructions}"


class DashScopeResponseFormatter(OpenAIResponseFormatter):
    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        items = await super().format(msgs)
        for item in items:
            if item.get("role") != "system":
                continue
            content = []
            for block in item["content"]:
                if block["type"] != "input_text" or CACHE_PREFIX_BREAK not in block["text"]:
                    content.append(block)
                    continue
                first, *rest = block["text"].split(CACHE_PREFIX_BREAK)
                segments = [first, *(CACHE_PREFIX_BREAK + segment for segment in rest)]
                content.extend({"type": "input_text", "text": segment, "prompt_cache_breakpoint": {"mode": "explicit"}} for segment in segments[:-1])
                content.append({"type": "input_text", "text": segments[-1]})
            item["content"] = content
        return items

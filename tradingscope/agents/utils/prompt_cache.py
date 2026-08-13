"""Prompt helpers for stable shared analysis context prefixes."""

from __future__ import annotations

from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT

CACHE_PREFIX_BREAK = "\n<!-- tradingscope:cache-prefix-end -->\n"


def build_cacheable_system_prompt(*, shared_context: str, role_instructions: str) -> str:
    """Place an identical stage context before role-specific instructions."""
    shared_prefix = f"{COMPLIANCE_PROMPT}\n\n# 共享分析上下文\n\n{shared_context}"
    return f"{shared_prefix}{CACHE_PREFIX_BREAK}{role_instructions}"

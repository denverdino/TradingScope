"""LLM-based summarization for stock analysis content.

This module provides a function to summarize long stock analysis text
into concise content that fits within the Model Studio Memory API's
512-character custom_content limit.
"""

from __future__ import annotations

import os

from agentscope import logger
from openai import AsyncOpenAI

from tradingscope.default_config import DEFAULT_CONFIG

# DashScope OpenAI-compatible endpoint
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Use a fast, cheap model for summarization
_DEFAULT_MODEL = DEFAULT_CONFIG["quick_think_llm"]

_SUMMARIZE_PROMPT = (
    "你是一位专业的股票分析摘要专家。请将以下股票分析内容压缩为一段简洁的摘要。\n"
    "要求：\n"
    "1. 摘要必须严格控制在{max_chars}个字符以内\n"
    "2. 保留最关键的分析结论、买卖信号、风险提示和核心数据指标\n"
    "3. 优先保留：当前日期、股票代码/名称、投资建议、关键判断依据与风险\n"
    "4. 去除冗余的修饰语和背景描述\n"
    "5. 输出纯文本，禁止使用任何Markdown格式（不要使用 ** | # 等标记符号）\n"
    "6. 直接输出摘要内容，不要添加任何前缀或说明\n\n"
    "原始内容：\n{content}"
)


async def summarize_for_memory(
    content: str,
    max_chars: int = 500,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Summarize stock analysis content using an LLM.

    Calls a lightweight LLM to compress long analytical text into a concise
    summary that fits within the Model Studio Memory API character limit,
    while preserving key stock analysis insights.

    Args:
        content: The stock analysis text to summarize.
        max_chars: Maximum character count for the summary. Defaults to 500.
        model: Model name to use. Defaults to DEFAULT_CONFIG["quick_think_llm"].
        api_key: DashScope API key. Defaults to DASHSCOPE_API_KEY env var.

    Returns:
        Summarized text within max_chars, or the original content
        (truncated) if the LLM call fails.
    """
    if len(content) <= max_chars:
        return content

    resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not resolved_key:
        logger.warning("[summarize] No DASHSCOPE_API_KEY found, falling back to truncation")
        return content[:max_chars]

    resolved_model = model or _DEFAULT_MODEL

    client = AsyncOpenAI(
        api_key=resolved_key,
        base_url=_DASHSCOPE_BASE_URL,
    )

    prompt = _SUMMARIZE_PROMPT.format(max_chars=max_chars, content=content)

    try:
        response = await client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        summary = response.choices[0].message.content or ""
        summary = summary.strip()

        # Enforce hard limit in case the model overshoots
        if len(summary) > max_chars:
            summary = summary[:max_chars]

        if not summary:
            logger.warning("[summarize] LLM returned empty summary, falling back to truncation")
            return content[:max_chars]

        logger.debug(f"[summarize] Compressed {len(content)} -> {len(summary)} chars")
        return summary

    except Exception as e:
        logger.warning(f"[summarize] LLM call failed: {e}, falling back to truncation")
        return content[:max_chars]
    finally:
        await client.close()

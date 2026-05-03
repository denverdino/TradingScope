"""Data models for the evaluation module.

AnalysisRecord stores the key decision basis from each workflow run,
used later by the evaluation process to score against actual market data.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from tradingscope.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_SUMMARY_PROMPT = (
    "你是一位专业的股票分析师。请从以下交易决策文本中提炼关键影响因素摘要。\n"
    "要求：\n"
    "1. 提炼影响交易决策的关键因素（技术面、基本面、市场情绪、风险因素等）\n"
    "2. 保留具体数据指标（价格、ATR、支撑位/阻力位、盈亏比等）\n"
    "3. 总结各方观点的核心分歧\n"
    "4. 摘要控制在500字符以内\n"
    "5. 输出纯文本，不使用Markdown格式\n"
    "6. 直接输出摘要，不要前缀或说明\n\n"
    "决策文本：\n{content}"
)


@dataclass
class AnalysisRecord:
    """Record of key decision basis from a workflow run.

    Stored as local JSON files. Not subject to the 512-char Memory API limit.
    """

    ticker: str
    trade_date: str  # YYYY-MM-DD
    direction: str  # bullish/bearish/neutral
    action: str  # buy/sell/hold
    confidence: float  # 0-1
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""  # core reasoning (~100 chars)
    final_decision_summary: str = ""  # LLM-summarized key decision factors
    status: str = "pending"  # pending/evaluated
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "direction": self.direction,
            "action": self.action,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "reasoning": self.reasoning,
            "final_decision_summary": self.final_decision_summary,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRecord":
        """Deserialize from dictionary."""
        return cls(
            ticker=data["ticker"],
            trade_date=data["trade_date"],
            direction=data["direction"],
            action=data["action"],
            confidence=data["confidence"],
            entry_price=data.get("entry_price"),
            target_price=data.get("target_price"),
            stop_loss=data.get("stop_loss"),
            reasoning=data.get("reasoning", ""),
            final_decision_summary=data.get("final_decision_summary", ""),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
        )

    @classmethod
    async def create_from_context(cls, context: Any) -> "AnalysisRecord":
        """Create an AnalysisRecord from an AgentContext instance.

        Uses LLM to summarize the final decision text, extracting key
        influencing factors instead of raw truncation.

        Args:
            context: AgentContext with final trade decision populated

        Returns:
            New AnalysisRecord instance
        """
        pred_data = context.extract_prediction_data()

        decision_text = context.final_trade_decision or context.trader_investment_plan or ""
        summary = await _summarize_decision(decision_text)

        return cls(
            ticker=context.company_of_interest,
            trade_date=context.trade_date,
            direction=pred_data["direction"],
            action=pred_data["action"],
            confidence=pred_data["confidence"],
            entry_price=pred_data.get("entry_price"),
            target_price=pred_data.get("target_price"),
            stop_loss=pred_data.get("stop_loss"),
            reasoning=pred_data.get("reasoning", ""),
            final_decision_summary=summary,
        )


async def _summarize_decision(content: str, max_chars: int = 500) -> str:
    """Summarize decision text via LLM to extract key influencing factors.

    Falls back to truncation if LLM is unavailable.
    """
    if not content:
        return ""
    if len(content) <= max_chars:
        return content.strip()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.warning("[summary] No DASHSCOPE_API_KEY, falling back to truncation")
        return content[:max_chars].strip()

    model = DEFAULT_CONFIG.get("quick_think_llm", "qwen-plus")
    client = AsyncOpenAI(api_key=api_key, base_url=_DASHSCOPE_BASE_URL)
    prompt = _SUMMARY_PROMPT.format(content=content)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        summary = (response.choices[0].message.content or "").strip()
        if not summary:
            logger.warning("[summary] LLM returned empty, falling back to truncation")
            return content[:max_chars].strip()
        if len(summary) > max_chars:
            summary = summary[:max_chars]
        logger.debug("[summary] Compressed %d -> %d chars", len(content), len(summary))
        return summary
    except Exception as e:
        logger.warning("[summary] LLM call failed: %s, falling back to truncation", e)
        return content[:max_chars].strip()
    finally:
        await client.close()

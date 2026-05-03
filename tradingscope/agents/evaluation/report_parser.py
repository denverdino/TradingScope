"""Parse prediction data from portfolio manager markdown reports.

Extracts structured fields (direction, confidence, prices, reasoning)
from the markdown text of portfolio_manager.md reports.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .models import AnalysisRecord


def parse_prediction_from_report(text: str) -> Dict[str, Any]:
    """Extract structured prediction data from a portfolio manager report.

    Parses markdown text to extract:
    - direction: bullish/bearish/neutral
    - action: buy/sell/hold
    - confidence: 0-1
    - entry_price: suggested entry price
    - target_price: price target
    - stop_loss: stop loss price
    - reasoning: core reasoning (up to 100 chars)

    Args:
        text: Markdown content of the portfolio_manager.md report

    Returns:
        Dictionary with extracted prediction fields.
    """
    result: Dict[str, Any] = {
        "direction": "neutral",
        "action": "hold",
        "confidence": 0.5,
        "entry_price": None,
        "target_price": None,
        "stop_loss": None,
        "reasoning": "",
    }

    if not text:
        return result

    text_lower = text.lower()

    # Extract action (buy/sell/hold)
    if any(kw in text_lower for kw in ["买入", "buy", "做多", "增持"]):
        result["action"] = "buy"
        result["direction"] = "bullish"
    elif any(kw in text_lower for kw in ["卖出", "sell", "做空", "减持", "清仓"]):
        result["action"] = "sell"
        result["direction"] = "bearish"
    else:
        result["action"] = "hold"
        result["direction"] = "neutral"

    # Extract confidence (置信度)
    confidence_patterns = [
        r"置信度[：:]\s*(\d+\.?\d*)",
        r"信心[：:]\s*(\d+\.?\d*)",
        r"confidence[：:]\s*(\d+\.?\d*)",
    ]
    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            conf = float(match.group(1))
            if conf > 1:
                conf = conf / 100
            result["confidence"] = min(1.0, max(0.0, conf))
            break

    # Extract entry price
    entry_patterns = [
        r"入场价[：:位]\s*\$?(\d+\.?\d*)",
        r"建议入场[：:价]\s*\$?(\d+\.?\d*)",
        r"entry[：:\s]+\$?(\d+\.?\d*)",
    ]
    for pattern in entry_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["entry_price"] = float(match.group(1))
            break

    # Extract target price
    target_patterns = [
        r"目标价[：:位]\s*\$?(\d+\.?\d*)",
        r"目标[：:\s]+\$?(\d+\.?\d*)",
        r"target[：:\s]+\$?(\d+\.?\d*)",
    ]
    for pattern in target_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["target_price"] = float(match.group(1))
            break

    # Extract stop loss
    stop_patterns = [
        r"止损[价位]*[：:\s]+\$?(\d+\.?\d*)",
        r"stop.?loss[：:\s]+\$?(\d+\.?\d*)",
    ]
    for pattern in stop_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["stop_loss"] = float(match.group(1))
            break

    # Extract reasoning
    reasoning_parts = []
    reason_patterns = [
        r"决策理由[：:](.*?)(?:\n\n|\n#|$)",
        r"核心逻辑[：:](.*?)(?:\n\n|\n#|$)",
        r"主要原因[：:](.*?)(?:\n\n|\n#|$)",
        r"#+\s*决策理由\s*\n(.*?)(?:\n\n|\n#|$)",
        r"#+\s*核心逻辑\s*\n(.*?)(?:\n\n|\n#|$)",
    ]
    for pattern in reason_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            reasoning_parts.append(match.group(1).strip()[:100])
            break

    if not reasoning_parts:
        sentences = re.split(r"[。\.\n]", text)
        action_keywords = ["买入", "卖出", "持有", "buy", "sell", "hold"]
        for sent in sentences:
            if any(kw in sent.lower() for kw in action_keywords) and len(sent) > 20:
                reasoning_parts.append(sent.strip()[:100])
                break

    result["reasoning"] = "".join(reasoning_parts)[:100] if reasoning_parts else ""

    return result


def build_analysis_record(ticker: str, trade_date: str, report_text: str) -> AnalysisRecord:
    """Build an AnalysisRecord from a portfolio manager report.

    Args:
        ticker: Stock symbol (e.g. "AAPL")
        trade_date: Trade date (YYYY-MM-DD)
        report_text: Full markdown content of portfolio_manager.md

    Returns:
        AnalysisRecord populated from the parsed report.
    """
    pred = parse_prediction_from_report(report_text)

    return AnalysisRecord(
        ticker=ticker,
        trade_date=trade_date,
        direction=pred["direction"],
        action=pred["action"],
        confidence=pred["confidence"],
        entry_price=pred.get("entry_price"),
        target_price=pred.get("target_price"),
        stop_loss=pred.get("stop_loss"),
        reasoning=pred.get("reasoning", ""),
        final_decision_summary=report_text[:500],
        status="pending",
    )

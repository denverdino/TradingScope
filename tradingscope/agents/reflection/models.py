"""Data models for the Reflection Loop system.

This module defines all data structures used in the reflection system:
- PredictionRecord: Stores T-day prediction data
- EvaluationResult: Stores T+N day evaluation metrics
- ReflectionLesson: Stores generated lessons from reflection
- BiasType/BiasResult: Stores bias detection results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional


class BiasType(Enum):
    """Types of cognitive biases detected in prediction patterns."""

    CONFIRMATION = "confirmation"  # 确认偏差：只记住支持自己观点的案例
    DIRECTIONAL = "directional"  # 方向偏差：系统性倾向看涨或看跌
    OVERCONFIDENCE = "overconfidence"  # 过度自信：置信度与准确率不匹配
    RECENCY = "recency"  # 近因偏差：过度重视近期事件
    ANCHORING = "anchoring"  # 锚定偏差：过度依赖初始信息
    LOSS_AVERSION = "loss_aversion"  # 损失厌恶：亏损后过度保守


@dataclass
class PredictionRecord:
    """Record of a prediction made at T day.

    Stored in Model Studio Memory API with compact format to fit 512 char limit.
    Format:
        [P]AAPL|2026-02-15|2026-02-20|bullish|buy|0.85
        入:145.5|目:155|损:140
        RSI超卖+MACD金叉，财报预期正面
        状态:pending
    """

    symbol: str  # 股票代码
    prediction_date: str  # T日 (YYYY-MM-DD)
    evaluation_date: str  # T+N日 (YYYY-MM-DD)
    direction: str  # bullish/bearish/neutral
    action: str  # buy/sell/hold
    confidence: float  # 0-1 置信度
    entry_price: Optional[float] = None  # 建议入场价
    target_price: Optional[float] = None  # 目标价
    stop_loss: Optional[float] = None  # 止损价
    reasoning: str = ""  # 核心理由（压缩到100字以内）
    status: str = "pending"  # pending/evaluated

    @property
    def prediction_id(self) -> str:
        """Generate unique ID from symbol and prediction date."""
        return f"{self.symbol}_{self.prediction_date}"

    def to_compact_format(self) -> str:
        """Serialize to compact format for Memory API storage (< 500 chars).

        Format:
            [P]AAPL|2026-02-15|2026-02-20|bullish|buy|0.85
            入:145.5|目:155|损:140
            RSI超卖+MACD金叉，财报预期正面
            状态:pending
        """
        # Line 1: Core prediction info
        line1 = f"[P]{self.symbol}|{self.prediction_date}|{self.evaluation_date}|{self.direction}|{self.action}|{self.confidence:.2f}"

        # Line 2: Price targets (use - for None)
        entry = f"{self.entry_price:.2f}" if self.entry_price else "-"
        target = f"{self.target_price:.2f}" if self.target_price else "-"
        stop = f"{self.stop_loss:.2f}" if self.stop_loss else "-"
        line2 = f"入:{entry}|目:{target}|损:{stop}"

        # Line 3: Reasoning (truncate to 100 chars)
        reasoning_truncated = self.reasoning[:100] if self.reasoning else "-"
        line3 = reasoning_truncated

        # Line 4: Status
        line4 = f"状态:{self.status}"

        return f"{line1}\n{line2}\n{line3}\n{line4}"

    @classmethod
    def from_compact_format(cls, content: str) -> Optional["PredictionRecord"]:
        """Deserialize from compact format.

        Returns None if parsing fails.
        """
        try:
            lines = content.strip().split("\n")
            if len(lines) < 4 or not lines[0].startswith("[P]"):
                return None

            # Parse line 1
            line1 = lines[0][3:]  # Remove [P] prefix
            parts = line1.split("|")
            if len(parts) < 6:
                return None

            symbol = parts[0]
            prediction_date = parts[1]
            evaluation_date = parts[2]
            direction = parts[3]
            action = parts[4]
            confidence = float(parts[5])

            # Parse line 2: prices
            line2_parts = lines[1].split("|")
            entry_price = None
            target_price = None
            stop_loss = None

            for part in line2_parts:
                if part.startswith("入:") and part[2:] != "-":
                    entry_price = float(part[2:])
                elif part.startswith("目:") and part[2:] != "-":
                    target_price = float(part[2:])
                elif part.startswith("损:") and part[2:] != "-":
                    stop_loss = float(part[2:])

            # Parse line 3: reasoning
            reasoning = lines[2] if lines[2] != "-" else ""

            # Parse line 4: status
            status = "pending"
            if lines[3].startswith("状态:"):
                status = lines[3][3:]

            return cls(
                symbol=symbol,
                prediction_date=prediction_date,
                evaluation_date=evaluation_date,
                direction=direction,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning,
                status=status,
            )
        except (ValueError, IndexError):
            return None

    @classmethod
    def create(
        cls,
        symbol: str,
        prediction_date: str,
        direction: str,
        action: str,
        confidence: float,
        reasoning: str,
        evaluation_delay_days: int = 5,
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> "PredictionRecord":
        """Create a new prediction record with auto-calculated evaluation date."""
        pred_date = datetime.strptime(prediction_date, "%Y-%m-%d")
        eval_date = pred_date + timedelta(days=evaluation_delay_days)
        evaluation_date = eval_date.strftime("%Y-%m-%d")

        return cls(
            symbol=symbol,
            prediction_date=prediction_date,
            evaluation_date=evaluation_date,
            direction=direction,
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=reasoning[:100] if reasoning else "",  # Truncate reasoning
            status="pending",
        )


@dataclass
class EvaluationResult:
    """Result of evaluating a prediction against actual stock prices."""

    prediction_id: str  # 关联的预测ID
    actual_price_t: float  # T日收盘价
    actual_price_tn: float  # T+N日收盘价
    actual_return: float  # 实际收益率 (T+N - T) / T
    predicted_return: float  # 预测收益率 (target - entry) / entry
    direction_correct: bool  # 方向是否正确
    target_reached: bool  # 是否达到目标价
    stop_loss_triggered: bool  # 是否触发止损
    accuracy_score: float  # 综合准确度 0-1
    market_return: float  # 大盘同期收益（用于相对评估）
    evaluation_date: str  # 评估日期

    @property
    def lesson_type(self) -> str:
        """Determine lesson type based on evaluation result."""
        if self.direction_correct and self.target_reached:
            return "success"
        elif not self.direction_correct or self.stop_loss_triggered:
            return "failure"
        else:
            return "partial"


@dataclass
class ReflectionLesson:
    """Structured lesson generated from reflection analysis."""

    lesson_id: str  # UUID
    prediction_id: str  # 关联的预测ID
    symbol: str  # 股票代码
    lesson_type: str  # success/failure/partial
    severity: str  # critical/major/minor
    root_cause: str  # 根因分析
    lesson_learned: str  # 核心教训（存入长期记忆）
    improvement_actions: List[str]  # 改进行动列表
    bias_findings: List[BiasType] = field(default_factory=list)  # 检测到的偏差
    weight: float = 1.0  # 记忆权重
    created_at: str = ""  # 创建时间

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_memory_format(self) -> str:
        """Format lesson for long-term memory storage.

        Format:
            [元数据]
            类型: 反思教训
            权重: 0.9
            股票: AAPL
            日期: 2026-02-15
            结果: 错误预测

            [教训内容]
            市场情况：...
            错误决策：...
            实际结果：...
            经验教训：...
            改进建议：...
        """
        bias_str = ",".join([b.value for b in self.bias_findings]) if self.bias_findings else "无"
        actions_str = "; ".join(self.improvement_actions[:3]) if self.improvement_actions else "无"

        result_label = {"success": "正确预测", "failure": "错误预测", "partial": "部分正确"}.get(self.lesson_type, "未知")

        content = f"""[元数据]
类型: 反思教训
权重: {self.weight:.1f}
股票: {self.symbol}
日期: {self.created_at[:10]}
结果: {result_label}
偏差: {bias_str}

[教训内容]
{self.root_cause}
经验教训：{self.lesson_learned}
改进建议：{actions_str}"""

        return content[:500]  # Ensure within 512 char limit


@dataclass
class BiasResult:
    """Result of bias detection analysis."""

    bias_type: BiasType
    detected: bool
    severity: float  # 0-1, higher means more severe
    evidence: str  # 检测到偏差的证据描述
    recommendation: str  # 改进建议

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bias_type": self.bias_type.value,
            "detected": self.detected,
            "severity": self.severity,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }

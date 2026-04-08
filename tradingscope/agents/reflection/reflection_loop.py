"""Reflection Loop for evaluating predictions and generating lessons.

This module provides the core reflection loop functionality:
- Schedules and executes reflection tasks
- Uses LLM to analyze prediction errors
- Generates structured lessons for long-term memory
- Reduces self-reinforcing bias through error-weighted learning
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import List, Optional

from agentscope import logger

from .models import (
    BiasResult,
    BiasType,
    EvaluationResult,
    PredictionRecord,
    ReflectionLesson,
)
from .prediction_evaluator import PredictionEvaluator
from .prediction_store import PredictionStore

try:
    from tradingscope.agents.utils.memory import ModelStudioLongTermMemory
    from tradingscope.agents.utils.memory_manager import FinancialMemoryManager

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


# Reflection Agent system prompt
REFLECTION_AGENT_PROMPT = """你是一位客观的投资分析反思专家。你的任务是分析过去的预测与实际结果之间的差异，识别决策过程中的问题，并生成可操作的改进建议。

## 分析原则

1. **关注依据而非结果**：不要仅仅因为结果错误就否定预测。分析"预测依据是否正确"比"预测结果是否正确"更重要。

2. **识别认知偏差**：
   - 确认偏差：是否只关注了支持自己观点的信息？
   - 锚定偏差：是否过度依赖初始信息？
   - 近因偏差：是否过度重视最近的市场事件？
   - 过度自信：置信度是否与实际准确率匹配？

3. **根因分析**：找出决策链中的关键失误点，而非表面原因。

4. **可操作建议**：生成具体、可执行的改进措施。

5. **客观分析**：避免自我辩护，承认错误是学习的机会。

## 输出要求

请用中文输出，格式如下：

### 根因分析
[分析预测错误的根本原因，识别决策链中的关键问题]

### 遗漏信号
[列出预测时忽略但事后证明重要的信号]

### 偏差识别
[识别存在的认知偏差类型]

### 经验教训
[总结核心教训，应简洁明了，便于记忆]

### 改进建议
[提供1-3条具体可执行的改进措施]
"""


def calculate_lesson_weight(
    lesson_type: str,
    accuracy_score: float,
    confidence: float,
    error_weight_multiplier: float = 2.0,
) -> float:
    """Calculate memory weight for a lesson.

    Error cases get higher weight to counter self-reinforcing bias.
    Confidence calibration also affects weight.

    Args:
        lesson_type: success/failure/partial
        accuracy_score: 0-1 prediction accuracy
        confidence: 0-1 original prediction confidence
        error_weight_multiplier: multiplier for error cases

    Returns:
        Weight between 0 and 1
    """
    base_weight = 0.5

    # Error cases get higher weight (core anti-bias mechanism)
    if lesson_type == "failure":
        base_weight = 0.8 * error_weight_multiplier
    elif lesson_type == "partial":
        base_weight = 0.6 * error_weight_multiplier
    else:  # success
        base_weight = 0.4

    # Confidence calibration adjustment
    # High confidence + low accuracy = higher weight (overconfidence penalty)
    if confidence > 0.7 and accuracy_score < 0.4:
        base_weight *= 1.2

    # Low confidence + high accuracy = slightly higher weight (missed opportunity)
    if confidence < 0.4 and accuracy_score > 0.7:
        base_weight *= 1.1

    # Normalize to 0-1 range
    return min(1.0, max(0.1, base_weight))


def apply_time_decay(base_weight: float, days_old: int, lambda_: float = 0.01) -> float:
    """Apply time decay to memory weight.

    Uses exponential decay: weight * e^(-lambda * days)
    With lambda=0.01, half-life is about 70 days.

    Args:
        base_weight: Original weight
        days_old: Days since lesson was created
        lambda_: Decay coefficient

    Returns:
        Decayed weight
    """
    decay_factor = math.exp(-lambda_ * days_old)
    return base_weight * max(0.3, decay_factor)  # Minimum 30% of original


class ReflectionLoop:
    """Main reflection loop orchestrator.

    Manages the complete reflection process:
    1. Find pending predictions ready for evaluation
    2. Fetch actual stock prices
    3. Evaluate prediction accuracy
    4. Generate reflection lessons using LLM
    5. Store lessons in long-term memory with appropriate weights

    Usage:
        loop = ReflectionLoop()
        lessons = await loop.run_batch_reflection()
    """

    def __init__(
        self,
        memory_manager: Optional[FinancialMemoryManager] = None,
        evaluation_delay_days: int = 5,
        error_weight_multiplier: float = 2.0,
    ):
        """Initialize ReflectionLoop.

        Args:
            memory_manager: FinancialMemoryManager instance. If None, creates new one.
            evaluation_delay_days: Days to wait before evaluating predictions
            error_weight_multiplier: Weight multiplier for error cases
        """
        self._memory_manager = memory_manager
        self._owns_memory_manager = memory_manager is None
        self._evaluation_delay_days = evaluation_delay_days
        self._error_weight_multiplier = error_weight_multiplier
        self._evaluator = PredictionEvaluator()
        self._prediction_store: Optional[PredictionStore] = None

    async def _ensure_initialized(self) -> bool:
        """Ensure memory manager is initialized."""
        if not MEMORY_AVAILABLE:
            logger.warning("Memory system not available")
            return False

        if self._memory_manager is None:
            self._memory_manager = FinancialMemoryManager()

        if self._prediction_store is None:
            self._prediction_store = PredictionStore(memory=self._memory_manager.prediction_store_memory)

        return True

    async def close(self) -> None:
        """Close resources."""
        if self._owns_memory_manager and self._memory_manager:
            await self._memory_manager.close()

    async def run_batch_reflection(
        self,
        symbol: Optional[str] = None,
    ) -> List[ReflectionLesson]:
        """Run reflection on all pending predictions.

        Args:
            symbol: Optional stock symbol to filter by

        Returns:
            List of generated ReflectionLesson objects
        """
        if not await self._ensure_initialized():
            return []

        lessons = []

        try:
            # Get pending predictions
            pending = await self._prediction_store.get_pending_predictions(
                symbol=symbol,
                before_date=datetime.now().strftime("%Y-%m-%d"),
            )

            logger.info(f"[ReflectionLoop] Found {len(pending)} pending predictions")

            for prediction in pending:
                try:
                    lesson = await self.run_single_reflection(prediction)
                    if lesson:
                        lessons.append(lesson)
                except Exception as e:
                    logger.warning(f"[ReflectionLoop] Error reflecting on {prediction.prediction_id}: {e}")

            logger.info(f"[ReflectionLoop] Generated {len(lessons)} lessons")

        except Exception as e:
            logger.warning(f"[ReflectionLoop] Error in batch reflection: {e}")

        return lessons

    async def run_single_reflection(
        self,
        prediction: PredictionRecord,
    ) -> Optional[ReflectionLesson]:
        """Run reflection on a single prediction.

        Args:
            prediction: PredictionRecord to reflect on

        Returns:
            ReflectionLesson if successful, None otherwise
        """
        if not await self._ensure_initialized():
            return None

        logger.info(f"[ReflectionLoop] Reflecting on {prediction.prediction_id}")

        # Step 1: Evaluate prediction
        evaluation = await self._evaluator.evaluate(prediction)
        if evaluation is None:
            logger.warning(f"[ReflectionLoop] Could not evaluate {prediction.prediction_id}")
            return None

        # Step 2: Generate reflection lesson
        lesson = await self._generate_lesson(prediction, evaluation)
        if lesson is None:
            return None

        # Step 3: Store lesson in appropriate agent memories
        await self._store_lesson(lesson, prediction)

        # Step 4: Mark prediction as evaluated
        await self._prediction_store.mark_as_evaluated(prediction)

        return lesson

    async def _generate_lesson(
        self,
        prediction: PredictionRecord,
        evaluation: EvaluationResult,
    ) -> Optional[ReflectionLesson]:
        """Generate reflection lesson using LLM analysis.

        Args:
            prediction: Original prediction
            evaluation: Evaluation result

        Returns:
            ReflectionLesson if successful
        """
        try:
            # Calculate lesson weight
            weight = calculate_lesson_weight(
                lesson_type=evaluation.lesson_type,
                accuracy_score=evaluation.accuracy_score,
                confidence=prediction.confidence,
                error_weight_multiplier=self._error_weight_multiplier,
            )

            # Determine severity
            if evaluation.lesson_type == "failure" and evaluation.stop_loss_triggered:
                severity = "critical"
            elif evaluation.lesson_type == "failure":
                severity = "major"
            else:
                severity = "minor"

            # Generate root cause and lesson content
            # In production, this would use LLM. For now, use template.
            root_cause = self._generate_root_cause(prediction, evaluation)
            lesson_learned = self._generate_lesson_text(prediction, evaluation)
            improvement_actions = self._generate_improvements(prediction, evaluation)

            # Detect biases
            bias_findings = self._detect_biases(prediction, evaluation)

            lesson = ReflectionLesson(
                lesson_id=str(uuid.uuid4())[:8],
                prediction_id=prediction.prediction_id,
                symbol=prediction.symbol,
                lesson_type=evaluation.lesson_type,
                severity=severity,
                root_cause=root_cause,
                lesson_learned=lesson_learned,
                improvement_actions=improvement_actions,
                bias_findings=bias_findings,
                weight=weight,
            )

            return lesson

        except Exception as e:
            logger.warning(f"[ReflectionLoop] Error generating lesson: {e}")
            return None

    def _generate_root_cause(
        self,
        prediction: PredictionRecord,
        evaluation: EvaluationResult,
    ) -> str:
        """Generate root cause analysis text."""
        if evaluation.lesson_type == "success":
            return f"预测{prediction.symbol}方向正确，实际收益{evaluation.actual_return:.1%}，分析依据有效。"

        direction_text = "看涨" if prediction.direction == "bullish" else "看跌"
        actual_direction = "上涨" if evaluation.actual_return > 0 else "下跌"

        if not evaluation.direction_correct:
            return (
                f"预测{prediction.symbol}{direction_text}，"
                f"但实际{actual_direction}{abs(evaluation.actual_return):.1%}。"
                f"原因分析：{prediction.reasoning[:50]}... "
                f"可能忽略了市场反向信号。"
            )

        if evaluation.stop_loss_triggered:
            return "预测方向正确但触发止损。实际波动超出预期，止损设置可能过紧。"

        return "预测部分正确，方向对但未达目标价。目标价设置可能过于乐观。"

    def _generate_lesson_text(
        self,
        prediction: PredictionRecord,
        evaluation: EvaluationResult,
    ) -> str:
        """Generate core lesson text."""
        if evaluation.lesson_type == "success":
            return f"{prediction.symbol}分析方法有效，继续使用类似分析框架。"

        if not evaluation.direction_correct:
            actual_vs_expected = f"预期{prediction.direction}，实际{'上涨' if evaluation.actual_return > 0 else '下跌'}"
            return f"方向判断错误：{actual_vs_expected}。需要增加反向信号的关注度。"

        if evaluation.stop_loss_triggered:
            return "止损被触发，考虑使用更宽松的止损或分批建仓策略。"

        return "目标价未达到，考虑设置更保守的目标或分批止盈。"

    def _generate_improvements(
        self,
        prediction: PredictionRecord,
        evaluation: EvaluationResult,
    ) -> List[str]:
        """Generate improvement actions."""
        improvements = []

        if not evaluation.direction_correct:
            improvements.append("增加对反向观点的分析权重")
            improvements.append("检查是否存在被忽略的重要信号")

        if evaluation.stop_loss_triggered:
            improvements.append("评估止损位置设置是否合理")

        if prediction.confidence > 0.8 and evaluation.accuracy_score < 0.5:
            improvements.append("降低高置信度预测的比例，提高校准度")

        if not improvements:
            improvements.append("继续保持当前分析方法")

        return improvements[:3]

    def _detect_biases(
        self,
        prediction: PredictionRecord,
        evaluation: EvaluationResult,
    ) -> List[BiasType]:
        """Detect potential biases in the prediction."""
        biases = []

        # Overconfidence detection
        if prediction.confidence > 0.8 and evaluation.accuracy_score < 0.4:
            biases.append(BiasType.OVERCONFIDENCE)

        # Direction bias hint (would need historical data for proper detection)
        if prediction.direction == "bullish" and not evaluation.direction_correct:
            # Possible bullish bias if consistently wrong on bullish calls
            pass  # Need historical data

        return biases

    async def _store_lesson(
        self,
        lesson: ReflectionLesson,
        prediction: PredictionRecord,
    ) -> None:
        """Store lesson in appropriate agent memories.

        Stores the lesson in:
        - trader memory (always)
        - research_manager memory (if direction was wrong)
        - portfolio_manager memory (if stop loss triggered)
        """
        if not self._memory_manager:
            return

        content = lesson.to_memory_format()
        thinking = f"存储{prediction.symbol}反思教训，类型：{lesson.lesson_type}，权重：{lesson.weight:.2f}"

        # Store in trader memory
        trader_memory = self._memory_manager.trader_memory
        if trader_memory:
            await trader_memory.record_to_memory(thinking, content)
            logger.info("[ReflectionLoop] Stored lesson in trader memory")

        # Store in research_manager if direction was wrong
        if lesson.lesson_type == "failure":
            research_memory = self._memory_manager.research_manager_memory
            if research_memory:
                await research_memory.record_to_memory(thinking, content)

        # Store in portfolio_manager if critical
        if lesson.severity == "critical":
            risk_memory = self._memory_manager.portfolio_manager_memory
            if risk_memory:
                await risk_memory.record_to_memory(thinking, content)


async def run_reflection(symbol: Optional[str] = None) -> List[ReflectionLesson]:
    """Convenience function to run reflection loop.

    Args:
        symbol: Optional stock symbol to filter by

    Returns:
        List of generated lessons
    """
    loop = ReflectionLoop()
    try:
        return await loop.run_batch_reflection(symbol=symbol)
    finally:
        await loop.close()

"""Bias Detector for the Reflection Loop system.

This module analyzes prediction patterns to detect cognitive biases:
- Confirmation bias: Favoring information that confirms existing beliefs
- Directional bias: Systematic tendency toward bullish or bearish predictions
- Overconfidence bias: Confidence levels not matching actual accuracy
- Recency bias: Over-weighting recent events
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .models import BiasResult, BiasType, EvaluationResult, PredictionRecord


class BiasDetector:
    """Detects cognitive biases in prediction patterns.

    Analyzes historical predictions to identify systematic biases
    that could lead to self-reinforcing errors.

    Usage:
        detector = BiasDetector()
        results = detector.analyze(predictions, evaluations)
    """

    def __init__(
        self,
        min_samples: int = 10,
        confirmation_threshold: float = 0.3,
        directional_threshold: float = 0.2,
        overconfidence_threshold: float = 0.15,
    ):
        """Initialize BiasDetector.

        Args:
            min_samples: Minimum predictions needed for bias detection
            confirmation_threshold: Threshold for confirmation bias detection
            directional_threshold: Threshold for directional bias detection
            overconfidence_threshold: Threshold for overconfidence detection
        """
        self.min_samples = min_samples
        self.confirmation_threshold = confirmation_threshold
        self.directional_threshold = directional_threshold
        self.overconfidence_threshold = overconfidence_threshold

    def analyze(
        self,
        predictions: List[PredictionRecord],
        evaluations: Optional[Dict[str, EvaluationResult]] = None,
    ) -> List[BiasResult]:
        """Analyze predictions for cognitive biases.

        Args:
            predictions: List of historical predictions
            evaluations: Optional dict mapping prediction_id to EvaluationResult

        Returns:
            List of detected biases with severity and evidence
        """
        results = []

        if len(predictions) < self.min_samples:
            return results

        # Run each bias detection
        confirmation_result = self.detect_confirmation_bias(predictions, evaluations)
        if confirmation_result.detected:
            results.append(confirmation_result)

        directional_result = self.detect_directional_bias(predictions, evaluations)
        if directional_result.detected:
            results.append(directional_result)

        if evaluations:
            overconfidence_result = self.detect_overconfidence_bias(
                predictions, evaluations
            )
            if overconfidence_result.detected:
                results.append(overconfidence_result)

            recency_result = self.detect_recency_bias(predictions, evaluations)
            if recency_result.detected:
                results.append(recency_result)

        return results

    def detect_confirmation_bias(
        self,
        predictions: List[PredictionRecord],
        evaluations: Optional[Dict[str, EvaluationResult]] = None,
    ) -> BiasResult:
        """Detect confirmation bias in prediction patterns.

        Confirmation bias is detected when:
        - Bull/bear predictions are adopted at significantly different rates
        - The favored direction has lower accuracy than the disfavored one

        Args:
            predictions: List of predictions
            evaluations: Optional evaluation results

        Returns:
            BiasResult indicating if confirmation bias is detected
        """
        if len(predictions) < self.min_samples:
            return BiasResult(
                bias_type=BiasType.CONFIRMATION,
                detected=False,
                severity=0.0,
                evidence="样本数量不足",
                recommendation="需要更多预测数据",
            )

        # Count by direction
        bull_count = sum(1 for p in predictions if p.direction == "bullish")
        bear_count = sum(1 for p in predictions if p.direction == "bearish")
        total = len(predictions)

        bull_rate = bull_count / total if total > 0 else 0
        bear_rate = bear_count / total if total > 0 else 0

        # Check if there's a significant imbalance
        imbalance = abs(bull_rate - bear_rate)

        if imbalance < self.confirmation_threshold:
            return BiasResult(
                bias_type=BiasType.CONFIRMATION,
                detected=False,
                severity=imbalance,
                evidence=f"看涨率{bull_rate:.1%}，看跌率{bear_rate:.1%}，分布相对均衡",
                recommendation="继续保持平衡分析",
            )

        # Check accuracy if evaluations available
        accuracy_mismatch = False
        bull_accuracy = 0.5
        bear_accuracy = 0.5

        if evaluations:
            bull_correct = sum(
                1
                for p in predictions
                if p.direction == "bullish"
                and p.prediction_id in evaluations
                and evaluations[p.prediction_id].direction_correct
            )
            bear_correct = sum(
                1
                for p in predictions
                if p.direction == "bearish"
                and p.prediction_id in evaluations
                and evaluations[p.prediction_id].direction_correct
            )

            bull_accuracy = bull_correct / bull_count if bull_count > 0 else 0
            bear_accuracy = bear_correct / bear_count if bear_count > 0 else 0

            # Confirmation bias: favoring one direction but being less accurate
            if bull_rate > bear_rate and bull_accuracy < bear_accuracy:
                accuracy_mismatch = True
            elif bear_rate > bull_rate and bear_accuracy < bull_accuracy:
                accuracy_mismatch = True

        detected = imbalance > self.confirmation_threshold and accuracy_mismatch
        severity = imbalance * (1.5 if accuracy_mismatch else 1.0)

        favored = "看涨" if bull_rate > bear_rate else "看跌"
        evidence = (
            f"预测倾向{favored}（{max(bull_rate, bear_rate):.1%}），"
            f"但该方向准确率较低（看涨{bull_accuracy:.1%}，看跌{bear_accuracy:.1%}）"
        )

        recommendation = f"增加对{'看跌' if favored == '看涨' else '看涨'}观点的关注度"

        return BiasResult(
            bias_type=BiasType.CONFIRMATION,
            detected=detected,
            severity=min(1.0, severity),
            evidence=evidence,
            recommendation=recommendation,
        )

    def detect_directional_bias(
        self,
        predictions: List[PredictionRecord],
        evaluations: Optional[Dict[str, EvaluationResult]] = None,
    ) -> BiasResult:
        """Detect directional bias (systematic bullish/bearish tendency).

        Args:
            predictions: List of predictions
            evaluations: Optional evaluation results

        Returns:
            BiasResult indicating if directional bias is detected
        """
        if len(predictions) < self.min_samples:
            return BiasResult(
                bias_type=BiasType.DIRECTIONAL,
                detected=False,
                severity=0.0,
                evidence="样本数量不足",
                recommendation="需要更多预测数据",
            )

        # Count actions
        action_counts = defaultdict(int)
        for p in predictions:
            action_counts[p.action] += 1

        total = len(predictions)
        buy_rate = action_counts["buy"] / total
        sell_rate = action_counts["sell"] / total
        hold_rate = action_counts["hold"] / total

        # Calculate optimal distribution if evaluations available
        optimal_buy = 0.33
        optimal_sell = 0.33
        optimal_hold = 0.34

        if evaluations:
            # Determine optimal action based on actual returns
            optimal_counts = {"buy": 0, "sell": 0, "hold": 0}
            for p in predictions:
                if p.prediction_id in evaluations:
                    actual_return = evaluations[p.prediction_id].actual_return
                    if actual_return > 0.05:  # Should have bought
                        optimal_counts["buy"] += 1
                    elif actual_return < -0.05:  # Should have sold
                        optimal_counts["sell"] += 1
                    else:  # Hold was appropriate
                        optimal_counts["hold"] += 1

            opt_total = sum(optimal_counts.values())
            if opt_total > 0:
                optimal_buy = optimal_counts["buy"] / opt_total
                optimal_sell = optimal_counts["sell"] / opt_total
                optimal_hold = optimal_counts["hold"] / opt_total

        # Calculate deviation from optimal
        buy_deviation = abs(buy_rate - optimal_buy)
        sell_deviation = abs(sell_rate - optimal_sell)
        total_deviation = buy_deviation + sell_deviation

        detected = total_deviation > self.directional_threshold

        # Determine bias direction
        if buy_rate > optimal_buy + self.directional_threshold:
            direction = "看涨偏差"
        elif sell_rate > optimal_sell + self.directional_threshold:
            direction = "看跌偏差"
        else:
            direction = "无明显方向偏差"

        evidence = (
            f"操作分布：买入{buy_rate:.1%}，卖出{sell_rate:.1%}，持有{hold_rate:.1%}。"
            f"最优分布：买入{optimal_buy:.1%}，卖出{optimal_sell:.1%}"
        )

        if detected:
            recommendation = f"存在{direction}，建议更客观评估反向信号"
        else:
            recommendation = "方向分布合理，继续保持"

        return BiasResult(
            bias_type=BiasType.DIRECTIONAL,
            detected=detected,
            severity=min(1.0, total_deviation),
            evidence=evidence,
            recommendation=recommendation,
        )

    def detect_overconfidence_bias(
        self,
        predictions: List[PredictionRecord],
        evaluations: Dict[str, EvaluationResult],
    ) -> BiasResult:
        """Detect overconfidence bias (confidence not matching accuracy).

        Args:
            predictions: List of predictions
            evaluations: Evaluation results

        Returns:
            BiasResult indicating if overconfidence bias is detected
        """
        if len(predictions) < self.min_samples:
            return BiasResult(
                bias_type=BiasType.OVERCONFIDENCE,
                detected=False,
                severity=0.0,
                evidence="样本数量不足",
                recommendation="需要更多预测数据",
            )

        # Group by confidence buckets
        buckets: Dict[str, Tuple[int, int]] = {
            "high": (0, 0),  # (correct, total) for confidence >= 0.8
            "medium": (0, 0),  # 0.5 <= confidence < 0.8
            "low": (0, 0),  # confidence < 0.5
        }

        for p in predictions:
            if p.prediction_id not in evaluations:
                continue

            is_correct = evaluations[p.prediction_id].direction_correct

            if p.confidence >= 0.8:
                correct, total = buckets["high"]
                buckets["high"] = (correct + (1 if is_correct else 0), total + 1)
            elif p.confidence >= 0.5:
                correct, total = buckets["medium"]
                buckets["medium"] = (correct + (1 if is_correct else 0), total + 1)
            else:
                correct, total = buckets["low"]
                buckets["low"] = (correct + (1 if is_correct else 0), total + 1)

        # Calculate calibration error
        calibration_errors = []

        if buckets["high"][1] > 0:
            high_accuracy = buckets["high"][0] / buckets["high"][1]
            expected_high = 0.85  # Expected accuracy for high confidence
            calibration_errors.append(abs(high_accuracy - expected_high))

        if buckets["medium"][1] > 0:
            medium_accuracy = buckets["medium"][0] / buckets["medium"][1]
            expected_medium = 0.65
            calibration_errors.append(abs(medium_accuracy - expected_medium))

        if not calibration_errors:
            return BiasResult(
                bias_type=BiasType.OVERCONFIDENCE,
                detected=False,
                severity=0.0,
                evidence="无有效的校准数据",
                recommendation="需要更多评估数据",
            )

        avg_calibration_error = sum(calibration_errors) / len(calibration_errors)
        detected = avg_calibration_error > self.overconfidence_threshold

        # Determine if overconfident or underconfident
        high_accuracy = (
            buckets["high"][0] / buckets["high"][1] if buckets["high"][1] > 0 else 0
        )
        is_overconfident = high_accuracy < 0.7  # High confidence but low accuracy

        evidence = (
            f"校准误差：{avg_calibration_error:.1%}。"
            f"高置信度预测准确率：{high_accuracy:.1%}"
        )

        if is_overconfident and detected:
            recommendation = "降低高置信度预测的比例，或提高决策门槛"
        elif detected:
            recommendation = "置信度评估偏保守，可适当提高置信度"
        else:
            recommendation = "置信度校准良好，继续保持"

        return BiasResult(
            bias_type=BiasType.OVERCONFIDENCE,
            detected=detected,
            severity=min(1.0, avg_calibration_error * 2),
            evidence=evidence,
            recommendation=recommendation,
        )

    def detect_recency_bias(
        self,
        predictions: List[PredictionRecord],
        evaluations: Dict[str, EvaluationResult],
    ) -> BiasResult:
        """Detect recency bias (over-weighting recent events).

        This is detected by checking if recent predictions follow
        recent market trends rather than independent analysis.

        Args:
            predictions: List of predictions (sorted by date)
            evaluations: Evaluation results

        Returns:
            BiasResult indicating if recency bias is detected
        """
        if len(predictions) < self.min_samples:
            return BiasResult(
                bias_type=BiasType.RECENCY,
                detected=False,
                severity=0.0,
                evidence="样本数量不足",
                recommendation="需要更多预测数据",
            )

        # Sort by date
        sorted_preds = sorted(predictions, key=lambda p: p.prediction_date)

        # Check if predictions follow previous market direction
        follow_previous = 0
        total_pairs = 0

        for i in range(1, len(sorted_preds)):
            prev_pred = sorted_preds[i - 1]
            curr_pred = sorted_preds[i]

            if prev_pred.prediction_id not in evaluations:
                continue

            prev_result = evaluations[prev_pred.prediction_id]
            actual_direction = "bullish" if prev_result.actual_return > 0 else "bearish"

            # Check if current prediction follows previous actual result
            if curr_pred.direction == actual_direction:
                follow_previous += 1
            total_pairs += 1

        if total_pairs < 5:
            return BiasResult(
                bias_type=BiasType.RECENCY,
                detected=False,
                severity=0.0,
                evidence="配对样本不足",
                recommendation="需要更多连续预测数据",
            )

        follow_rate = follow_previous / total_pairs
        # High follow rate suggests recency bias
        detected = follow_rate > 0.75

        evidence = f"预测跟随前期市场方向的比例：{follow_rate:.1%}"

        if detected:
            recommendation = "避免过度依赖近期市场走势，增加独立分析权重"
        else:
            recommendation = "预测独立性良好"

        return BiasResult(
            bias_type=BiasType.RECENCY,
            detected=detected,
            severity=min(1.0, (follow_rate - 0.5) * 2) if follow_rate > 0.5 else 0,
            evidence=evidence,
            recommendation=recommendation,
        )

    def generate_report(self, results: List[BiasResult]) -> str:
        """Generate a human-readable bias detection report.

        Args:
            results: List of BiasResult from analyze()

        Returns:
            Formatted report string
        """
        if not results:
            return "未检测到明显的认知偏差。"

        lines = ["# 认知偏差检测报告\n"]

        detected = [r for r in results if r.detected]
        if detected:
            lines.append(f"## 检测到 {len(detected)} 个偏差\n")

            for result in detected:
                bias_name = {
                    BiasType.CONFIRMATION: "确认偏差",
                    BiasType.DIRECTIONAL: "方向偏差",
                    BiasType.OVERCONFIDENCE: "过度自信",
                    BiasType.RECENCY: "近因偏差",
                }.get(result.bias_type, result.bias_type.value)

                severity_label = (
                    "严重" if result.severity > 0.7 else ("中等" if result.severity > 0.4 else "轻微")
                )

                lines.append(f"### {bias_name} ({severity_label})")
                lines.append(f"- 证据：{result.evidence}")
                lines.append(f"- 建议：{result.recommendation}\n")

        else:
            lines.append("未检测到需要关注的偏差。\n")

        return "\n".join(lines)

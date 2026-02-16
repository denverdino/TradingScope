"""Prediction Store for the Reflection Loop system.

This module provides a storage layer for prediction records using
the Model Studio Memory API, enabling the reflection loop to:
- Store predictions at T day
- Retrieve pending predictions for evaluation at T+N day
- Update prediction status after evaluation
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from agentscope import logger

from .models import PredictionRecord

try:
    from tradingscope.agents.utils.memory import ModelStudioLongTermMemory

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logger.warning("ModelStudioLongTermMemory not available")


class PredictionStore:
    """Store and retrieve predictions using Model Studio Memory API.

    This class wraps ModelStudioLongTermMemory to provide specialized
    operations for prediction storage and retrieval.

    The predictions are stored in compact format to fit within the
    512 character limit of the Memory API.

    Usage:
        store = PredictionStore(memory)

        # Save a prediction
        await store.save(prediction)

        # Get pending predictions for a symbol
        pending = await store.get_pending_predictions("AAPL")

        # Update prediction status
        await store.mark_as_evaluated(prediction_id)
    """

    def __init__(
        self,
        memory: Optional["ModelStudioLongTermMemory"] = None,
        user_name: str = "tradingscope",
    ):
        """Initialize PredictionStore.

        Args:
            memory: ModelStudioLongTermMemory instance for prediction_store role.
                   If None, will create a new instance.
            user_name: User name prefix for memory namespace.
        """
        if memory is not None:
            self._memory = memory
        elif MEMORY_AVAILABLE:
            self._memory = ModelStudioLongTermMemory(
                agent_name="prediction_store",
                user_name=user_name,
                top_k=20,  # Higher top_k for prediction retrieval
            )
        else:
            self._memory = None
            logger.warning("PredictionStore initialized without memory backend")

    async def save(self, prediction: PredictionRecord) -> bool:
        """Save a prediction record to memory.

        Args:
            prediction: PredictionRecord to save

        Returns:
            True if saved successfully, False otherwise
        """
        if self._memory is None:
            logger.warning("Memory not available, prediction not saved")
            return False

        try:
            content = prediction.to_compact_format()
            thinking = f"存储{prediction.symbol}在{prediction.prediction_date}的预测记录，用于T+N日反思评估"

            result = await self._memory.record_to_memory(thinking, content)
            if "Successfully" in result:
                logger.info(
                    f"[PredictionStore] Saved prediction: {prediction.prediction_id}"
                )
                return True
            else:
                logger.warning(f"[PredictionStore] Failed to save: {result}")
                return False

        except Exception as e:
            logger.warning(f"[PredictionStore] Error saving prediction: {e}")
            return False

    async def get_pending_predictions(
        self,
        symbol: Optional[str] = None,
        before_date: Optional[str] = None,
    ) -> List[PredictionRecord]:
        """Retrieve pending predictions that need evaluation.

        Args:
            symbol: Optional stock symbol to filter by
            before_date: Only return predictions with evaluation_date <= this date
                        Defaults to today if not specified

        Returns:
            List of pending PredictionRecord objects
        """
        if self._memory is None:
            return []

        # Build search query
        query_parts = ["[P]", "pending"]  # Prediction marker and status
        if symbol:
            query_parts.insert(1, symbol)

        query = " ".join(query_parts)

        try:
            result = await self._memory.retrieve_from_memory(query)

            if result in ["Memory system unavailable", "No relevant memories found"]:
                return []

            # Parse returned memories
            predictions = self._parse_memory_results(result)

            # Filter by date if specified
            if before_date is None:
                before_date = datetime.now().strftime("%Y-%m-%d")

            filtered = []
            for pred in predictions:
                if pred.status == "pending" and pred.evaluation_date <= before_date:
                    if symbol is None or pred.symbol == symbol:
                        filtered.append(pred)

            return filtered

        except Exception as e:
            logger.warning(f"[PredictionStore] Error retrieving predictions: {e}")
            return []

    async def get_all_predictions(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[PredictionRecord]:
        """Retrieve all predictions for a symbol (for bias detection).

        Args:
            symbol: Optional stock symbol to filter by
            limit: Maximum number of predictions to return

        Returns:
            List of PredictionRecord objects (both pending and evaluated)
        """
        if self._memory is None:
            return []

        query = f"[P] {symbol}" if symbol else "[P]"

        try:
            result = await self._memory.retrieve_from_memory(query)

            if result in ["Memory system unavailable", "No relevant memories found"]:
                return []

            predictions = self._parse_memory_results(result)

            # Filter by symbol if specified
            if symbol:
                predictions = [p for p in predictions if p.symbol == symbol]

            return predictions[:limit]

        except Exception as e:
            logger.warning(f"[PredictionStore] Error retrieving all predictions: {e}")
            return []

    async def mark_as_evaluated(self, prediction: PredictionRecord) -> bool:
        """Mark a prediction as evaluated by storing updated record.

        Note: Model Studio Memory API doesn't support updates, so we store
        a new record with evaluated status. The old record remains but
        will be filtered out by status check.

        Args:
            prediction: PredictionRecord to mark as evaluated

        Returns:
            True if updated successfully, False otherwise
        """
        if self._memory is None:
            return False

        try:
            # Create updated prediction with evaluated status
            updated = PredictionRecord(
                symbol=prediction.symbol,
                prediction_date=prediction.prediction_date,
                evaluation_date=prediction.evaluation_date,
                direction=prediction.direction,
                action=prediction.action,
                confidence=prediction.confidence,
                entry_price=prediction.entry_price,
                target_price=prediction.target_price,
                stop_loss=prediction.stop_loss,
                reasoning=prediction.reasoning,
                status="evaluated",
            )

            content = updated.to_compact_format()
            thinking = f"更新{prediction.symbol}预测状态为已评估，反思循环完成"

            result = await self._memory.record_to_memory(thinking, content)
            if "Successfully" in result:
                logger.info(
                    f"[PredictionStore] Marked as evaluated: {prediction.prediction_id}"
                )
                return True
            return False

        except Exception as e:
            logger.warning(f"[PredictionStore] Error marking as evaluated: {e}")
            return False

    def _parse_memory_results(self, result: str) -> List[PredictionRecord]:
        """Parse memory retrieval results into PredictionRecord objects.

        The memory results contain formatted text with multiple memory nodes.
        We need to extract and parse each prediction record.

        Args:
            result: Raw memory retrieval result string

        Returns:
            List of parsed PredictionRecord objects
        """
        predictions = []

        # Split by numbered list items (1. 2. 3. etc)
        # Memory format: "1. [P]...\n\n2. [P]...\n\n"
        lines = result.split("\n")

        current_record = []
        in_record = False

        for line in lines:
            # Check if this is a prediction marker
            if "[P]" in line:
                # Save previous record if exists
                if current_record:
                    record_text = "\n".join(current_record)
                    pred = PredictionRecord.from_compact_format(record_text)
                    if pred:
                        predictions.append(pred)
                    current_record = []

                # Start new record - extract from [P] marker
                start_idx = line.find("[P]")
                current_record = [line[start_idx:]]
                in_record = True

            elif in_record and line.strip():
                # Continue building current record
                # Stop if we hit section headers or empty content
                if line.startswith("#") or line.startswith("请在分析"):
                    in_record = False
                else:
                    current_record.append(line)

        # Don't forget the last record
        if current_record:
            record_text = "\n".join(current_record)
            pred = PredictionRecord.from_compact_format(record_text)
            if pred:
                predictions.append(pred)

        return predictions

    async def close(self) -> None:
        """Close the memory connection."""
        if self._memory:
            await self._memory.close()

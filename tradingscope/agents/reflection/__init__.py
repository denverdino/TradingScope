"""Reflection Loop module for reducing self-reinforcing bias.

This module provides components for:
- Recording predictions at T day
- Evaluating predictions against actual stock prices at T+N day
- Detecting cognitive biases in prediction patterns
- Generating structured lessons and storing them in long-term memory
"""

from .models import (
    BiasResult,
    BiasType,
    EvaluationResult,
    PredictionRecord,
    ReflectionLesson,
)

# Lazy imports for modules that depend on agentscope
# These will fail gracefully if agentscope is not installed


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "BiasDetector":
        from .bias_detector import BiasDetector

        return BiasDetector
    elif name == "PredictionEvaluator":
        from .prediction_evaluator import PredictionEvaluator

        return PredictionEvaluator
    elif name == "PredictionStore":
        from .prediction_store import PredictionStore

        return PredictionStore
    elif name == "ReflectionLoop":
        from .reflection_loop import ReflectionLoop

        return ReflectionLoop
    elif name == "run_reflection":
        from .reflection_loop import run_reflection

        return run_reflection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PredictionRecord",
    "EvaluationResult",
    "ReflectionLesson",
    "BiasType",
    "BiasResult",
    "PredictionStore",
    "PredictionEvaluator",
    "BiasDetector",
    "ReflectionLoop",
    "run_reflection",
]

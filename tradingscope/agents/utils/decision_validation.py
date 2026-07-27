"""Workflow-only policy checks for newly generated agent outputs."""

from __future__ import annotations

from collections.abc import Sequence

from tradingscope.agents.output import (
    Action,
    AgentOutputBase,
    Direction,
    PortfolioManagerOutput,
    PositionAdvice,
    TradeIntent,
    TraderOutput,
)

PolicyError = dict[str, object]


class GeneratedOutputPolicyError(ValueError):
    """Raised when a new output is schema-valid but not executable."""

    def __init__(self, errors: list[PolicyError]) -> None:
        self.errors = errors
        details = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in errors)
        super().__init__(details)


def _error(location: tuple[str, ...], message: str) -> PolicyError:
    return {"loc": location, "msg": message}


ACTION_INTENTS = {
    Action.BUY: {TradeIntent.OPEN_LONG, TradeIntent.COVER_SHORT},
    Action.SELL: {TradeIntent.REDUCE_LONG, TradeIntent.CLOSE_LONG, TradeIntent.OPEN_SHORT},
    Action.HOLD: {TradeIntent.HOLD},
}

_ENTRY_FIELDS = ("entry_price", "entry_price_low", "entry_price_high")


def _require_fields(output: TraderOutput | PortfolioManagerOutput, field_names: tuple[str, ...]) -> list[PolicyError]:
    errors: list[PolicyError] = []
    for field_name in field_names:
        if getattr(output.price_plan, field_name) is None:
            errors.append(_error(("price_plan", field_name), f"{output.trade_intent.value} requires {field_name}"))
    return errors


def _validate_long_prices(output: TraderOutput | PortfolioManagerOutput) -> list[PolicyError]:
    plan = output.price_plan
    if any(getattr(plan, field_name) is None for field_name in (*_ENTRY_FIELDS, "target_price", "stop_loss")):
        return []

    errors: list[PolicyError] = []
    if plan.stop_loss >= plan.entry_price_low:
        errors.append(_error(("price_plan", "stop_loss"), "long stop_loss must be below the entry range"))
    if plan.target_price <= plan.entry_price_high:
        errors.append(_error(("price_plan", "target_price"), "long target_price must be above the entry range"))
    return errors


def _validate_short_prices(output: TraderOutput | PortfolioManagerOutput) -> list[PolicyError]:
    plan = output.price_plan
    if any(getattr(plan, field_name) is None for field_name in (*_ENTRY_FIELDS, "target_price", "stop_loss")):
        return []

    errors: list[PolicyError] = []
    if plan.target_price >= plan.entry_price_low:
        errors.append(_error(("price_plan", "target_price"), "short target_price must be below the entry range"))
    if plan.stop_loss <= plan.entry_price_high:
        errors.append(_error(("price_plan", "stop_loss"), "short stop_loss must be above the entry range"))
    return errors


def _validate_execution_plan(output: TraderOutput | PortfolioManagerOutput) -> list[PolicyError]:
    errors: list[PolicyError] = []
    action = output.decision.action
    intent = output.trade_intent

    if intent is None:
        return [_error(("trade_intent",), "new execution output requires trade_intent")]
    if intent not in ACTION_INTENTS[action]:
        return [_error(("trade_intent",), f"{action.value} action is inconsistent with {intent.value} trade_intent")]

    if action in (Action.BUY, Action.SELL):
        expected_direction = Direction.BULLISH if action is Action.BUY else Direction.BEARISH
        if output.decision.direction is not expected_direction:
            errors.append(
                _error(
                    ("decision", "direction"),
                    f"{action.value} action requires {expected_direction.value} direction",
                ),
            )

    if intent in (TradeIntent.OPEN_LONG, TradeIntent.OPEN_SHORT):
        errors.extend(_require_fields(output, (*_ENTRY_FIELDS, "target_price", "stop_loss")))
        if output.position_advice is PositionAdvice.NONE:
            errors.append(_error(("position_advice",), f"{intent.value} requires a residual position"))
        if output.time_stop_days is None:
            errors.append(_error(("time_stop_days",), f"{intent.value} requires time_stop_days"))
        price_validator = _validate_long_prices if intent is TradeIntent.OPEN_LONG else _validate_short_prices
        errors.extend(price_validator(output))
    elif intent is TradeIntent.REDUCE_LONG:
        errors.extend(_require_fields(output, _ENTRY_FIELDS))
        if output.position_advice is not PositionAdvice.NONE:
            errors.extend(_require_fields(output, ("target_price", "stop_loss")))
            if output.time_stop_days is None:
                errors.append(_error(("time_stop_days",), "reduce_long with a residual position requires time_stop_days"))
            errors.extend(_validate_long_prices(output))
        elif output.time_stop_days is not None:
            errors.append(_error(("time_stop_days",), "reduce_long without a residual position must omit time_stop_days"))
    elif intent in (TradeIntent.CLOSE_LONG, TradeIntent.COVER_SHORT):
        errors.extend(_require_fields(output, _ENTRY_FIELDS))
        if output.position_advice is not PositionAdvice.NONE:
            errors.append(_error(("position_advice",), f"{intent.value} requires no residual position"))
        if output.time_stop_days is not None:
            errors.append(_error(("time_stop_days",), f"{intent.value} must omit time_stop_days"))
        for field_name in ("target_price", "stop_loss"):
            if getattr(output.price_plan, field_name) is not None:
                errors.append(_error(("price_plan", field_name), f"{intent.value} must not fabricate {field_name}"))
    elif output.position_advice is not PositionAdvice.NONE:
        errors.extend(_require_fields(output, ("target_price", "stop_loss")))
        if output.time_stop_days is None:
            errors.append(_error(("time_stop_days",), "positioned hold requires time_stop_days"))
    elif output.time_stop_days is not None:
        errors.append(_error(("time_stop_days",), "hold without a position must omit time_stop_days"))
    return errors


def validate_generated_output(
    output: AgentOutputBase,
    references: Sequence[AgentOutputBase] = (),
) -> None:
    """Validate policies that apply to new workflow output, not stored JSON."""

    errors: list[PolicyError] = []
    if isinstance(output, (TraderOutput, PortfolioManagerOutput)):
        errors.extend(_validate_execution_plan(output))

    if errors:
        raise GeneratedOutputPolicyError(errors)

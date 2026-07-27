"""Tests for policies applied only to newly generated workflow outputs."""

from __future__ import annotations

import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents import output as models


def _with_decision(
    output: models.AgentOutputBase,
    *,
    direction: str,
    action: str,
    confidence: float,
    price_plan: dict | None = None,
    position_advice: str | None = None,
    trade_intent: str | None = None,
    time_stop_days: int | None = 3,
) -> models.AgentOutputBase:
    data = output.model_dump(mode="json")
    data["decision"] = {
        "direction": direction,
        "action": action,
        "confidence": confidence,
        "summary": "测试决策",
        "reasoning": ["测试依据"],
    }
    if price_plan is not None:
        data["price_plan"] = {
            "currency": "USD",
            "invalidation_conditions": ["条件失效"],
            **price_plan,
        }
    if position_advice is not None:
        data["position_advice"] = position_advice
    if isinstance(output, (models.TraderOutput, models.PortfolioManagerOutput)):
        data["trade_intent"] = (
            trade_intent
            or {
                "buy": "open_long",
                "sell": "open_short",
                "hold": "hold",
            }[action]
        )
        data["time_stop_days"] = time_stop_days
    return type(output).model_validate(data)


@pytest.mark.parametrize(
    ("direction", "action", "plan"),
    [
        (
            "bullish",
            "buy",
            {
                "entry_price": 100.0,
                "entry_price_low": 99.0,
                "entry_price_high": 101.0,
                "target_price": 110.0,
                "stop_loss": 100.0,
            },
        ),
        (
            "bearish",
            "sell",
            {
                "entry_price": 100.0,
                "entry_price_low": 99.0,
                "entry_price_high": 101.0,
                "target_price": 90.0,
                "stop_loss": 100.0,
            },
        ),
    ],
)
def test_generated_directional_plan_requires_stop_outside_entry_range(
    direction: str,
    action: str,
    plan: dict,
) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    trader = _with_decision(
        _all_outputs()[5],
        direction=direction,
        action=action,
        confidence=0.6,
        price_plan=plan,
        position_advice="light",
    )

    with pytest.raises(GeneratedOutputPolicyError, match="price_plan.stop_loss"):
        validate_generated_output(trader)


def test_generated_sell_plan_requires_entry_target_and_stop() -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    trader = _with_decision(
        _all_outputs()[5],
        direction="bearish",
        action="sell",
        confidence=0.6,
        price_plan={
            "entry_price": 100.0,
            "target_price": None,
            "stop_loss": 105.0,
        },
    )

    with pytest.raises(GeneratedOutputPolicyError, match="price_plan.target_price"):
        validate_generated_output(trader)


def test_generated_directional_plan_requires_entry_range() -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    trader = _with_decision(
        _all_outputs()[5],
        direction="bearish",
        action="sell",
        confidence=0.6,
        price_plan={
            "entry_price": 100.0,
            "target_price": 90.0,
            "stop_loss": 105.0,
        },
    )

    with pytest.raises(GeneratedOutputPolicyError, match="price_plan.entry_price_low"):
        validate_generated_output(trader)


@pytest.mark.parametrize(
    ("action", "plan"),
    [
        (
            "buy",
            {
                "entry_price": 100.0,
                "entry_price_low": 99.0,
                "entry_price_high": 101.0,
                "target_price": 110.0,
                "stop_loss": 95.0,
            },
        ),
        (
            "sell",
            {
                "entry_price": 100.0,
                "entry_price_low": 99.0,
                "entry_price_high": 101.0,
                "target_price": 90.0,
                "stop_loss": 105.0,
            },
        ),
    ],
)
def test_generated_directional_action_rejects_neutral_direction(action: str, plan: dict) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    trader = _with_decision(
        _all_outputs()[5],
        direction="neutral",
        action=action,
        confidence=0.99,
        price_plan=plan,
    )

    with pytest.raises(GeneratedOutputPolicyError, match="decision.direction"):
        validate_generated_output(trader)


def test_research_manager_is_not_validated_as_an_execution_order() -> None:
    from tradingscope.agents.utils.decision_validation import validate_generated_output

    research = _with_decision(
        _all_outputs()[4],
        direction="neutral",
        action="buy",
        confidence=0.6,
        price_plan={
            "entry_price": 100.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
        },
    )

    validate_generated_output(research)


def test_generated_positioned_hold_requires_target_and_stop() -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    portfolio = _with_decision(
        _all_outputs()[6],
        direction="neutral",
        action="hold",
        confidence=0.6,
        price_plan={
            "entry_price": None,
            "target_price": 110.0,
            "stop_loss": None,
        },
        position_advice="medium",
    )

    with pytest.raises(GeneratedOutputPolicyError, match="price_plan.stop_loss"):
        validate_generated_output(portfolio)


@pytest.mark.parametrize(
    ("action", "intent"),
    [
        ("buy", "open_short"),
        ("sell", "open_long"),
        ("hold", "close_long"),
    ],
)
def test_generated_output_rejects_action_intent_conflicts(action: str, intent: str) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    direction = {"buy": "bullish", "sell": "bearish", "hold": "neutral"}[action]
    output = _with_decision(
        _all_outputs()[5],
        direction=direction,
        action=action,
        confidence=0.6,
        price_plan=(
            {
                "entry_price": 100.0,
                "entry_price_low": 99.0,
                "entry_price_high": 101.0,
                "target_price": 110.0 if action == "buy" else 90.0,
                "stop_loss": 95.0 if action == "buy" else 105.0,
            }
            if action != "hold"
            else None
        ),
        trade_intent=intent,
    )

    with pytest.raises(GeneratedOutputPolicyError, match="trade_intent"):
        validate_generated_output(output)


@pytest.mark.parametrize("output_index", [5, 6])
def test_generated_execution_output_requires_trade_intent(output_index: int) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    output = _with_decision(
        _all_outputs()[output_index],
        direction="neutral",
        action="hold",
        confidence=0.8,
    ).model_copy(update={"trade_intent": None})

    with pytest.raises(GeneratedOutputPolicyError, match="trade_intent"):
        validate_generated_output(output)


def test_high_confidence_direction_change_is_not_rejected_by_numeric_threshold() -> None:
    from tradingscope.agents.utils.decision_validation import validate_generated_output

    outputs = _all_outputs()
    research = _with_decision(
        outputs[4],
        direction="bullish",
        action="buy",
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
        },
    )
    trader = _with_decision(
        outputs[5],
        direction="bearish",
        action="sell",
        confidence=0.99,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": 90.0,
            "stop_loss": 105.0,
        },
        position_advice="light",
        trade_intent="open_short",
    )

    validate_generated_output(trader, [research])


def test_large_portfolio_entry_shift_is_not_rejected_by_numeric_threshold() -> None:
    from tradingscope.agents.utils.decision_validation import validate_generated_output

    outputs = _all_outputs()
    trader = _with_decision(
        outputs[5],
        direction="bearish",
        action="sell",
        confidence=0.6,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": 90.0,
            "stop_loss": 110.0,
        },
        trade_intent="open_short",
    )
    portfolio = _with_decision(
        outputs[6],
        direction="bearish",
        action="sell",
        confidence=0.7,
        price_plan={
            "entry_price": 106.0,
            "entry_price_low": 105.0,
            "entry_price_high": 107.0,
            "target_price": 90.0,
            "stop_loss": 116.0,
        },
        position_advice="light",
        trade_intent="open_short",
    )

    validate_generated_output(portfolio, [trader])


@pytest.mark.parametrize("intent", ["open_long", "open_short"])
def test_open_intent_requires_time_stop(intent: str) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    action, direction, target, stop = ("buy", "bullish", 110.0, 95.0) if intent == "open_long" else ("sell", "bearish", 90.0, 105.0)
    trader = _with_decision(
        _all_outputs()[5],
        direction=direction,
        action=action,
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": target,
            "stop_loss": stop,
        },
        position_advice="light",
        trade_intent=intent,
        time_stop_days=None,
    )

    with pytest.raises(GeneratedOutputPolicyError, match="time_stop_days"):
        validate_generated_output(trader)


@pytest.mark.parametrize("output_index", [5, 6])
@pytest.mark.parametrize("intent", ["open_long", "open_short"])
def test_open_intent_requires_a_residual_position(output_index: int, intent: str) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    action, direction, target, stop = ("buy", "bullish", 110.0, 95.0) if intent == "open_long" else ("sell", "bearish", 90.0, 105.0)
    output = _with_decision(
        _all_outputs()[output_index],
        direction=direction,
        action=action,
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": target,
            "stop_loss": stop,
        },
        position_advice="none",
        trade_intent=intent,
    )

    with pytest.raises(GeneratedOutputPolicyError, match="position_advice"):
        validate_generated_output(output)


def test_generated_open_long_rejects_incomplete_plan_at_runtime() -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    trader = _with_decision(
        _all_outputs()[5],
        direction="bullish",
        action="buy",
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": None,
            "stop_loss": 95.0,
        },
        position_advice="light",
        trade_intent="open_long",
    )

    with pytest.raises(GeneratedOutputPolicyError, match="price_plan.target_price"):
        validate_generated_output(trader)


def test_cover_short_accepts_an_execution_range_without_exit_prices() -> None:
    from tradingscope.agents.utils.decision_validation import validate_generated_output

    trader = _with_decision(
        _all_outputs()[5],
        direction="bullish",
        action="buy",
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": None,
            "stop_loss": None,
        },
        position_advice="none",
        trade_intent="cover_short",
        time_stop_days=None,
    )

    validate_generated_output(trader)


@pytest.mark.parametrize(
    ("intent", "action", "direction"),
    [("close_long", "sell", "bearish"), ("cover_short", "buy", "bullish")],
)
def test_closing_intent_rejects_time_stop_without_a_residual_position(
    intent: str,
    action: str,
    direction: str,
) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    trader = _with_decision(
        _all_outputs()[5],
        direction=direction,
        action=action,
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": None,
            "stop_loss": None,
        },
        position_advice="none",
        trade_intent=intent,
        time_stop_days=3,
    )

    with pytest.raises(GeneratedOutputPolicyError, match="time_stop_days"):
        validate_generated_output(trader)


@pytest.mark.parametrize(
    ("intent", "action", "direction", "plan"),
    [
        (
            "reduce_long",
            "sell",
            "bearish",
            {
                "entry_price": 100.0,
                "entry_price_low": 99.0,
                "entry_price_high": 101.0,
            },
        ),
        ("hold", "hold", "neutral", None),
    ],
)
def test_flat_reduce_or_hold_rejects_time_stop(
    intent: str,
    action: str,
    direction: str,
    plan: dict | None,
) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    portfolio = _with_decision(
        _all_outputs()[6],
        direction=direction,
        action=action,
        confidence=0.8,
        price_plan=plan,
        position_advice="none",
        trade_intent=intent,
        time_stop_days=3,
    )

    with pytest.raises(GeneratedOutputPolicyError, match="time_stop_days"):
        validate_generated_output(portfolio)


def test_reduce_long_uses_long_position_price_semantics() -> None:
    from tradingscope.agents.utils.decision_validation import validate_generated_output

    trader = _with_decision(
        _all_outputs()[5],
        direction="bearish",
        action="sell",
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
        },
        position_advice="light",
        trade_intent="reduce_long",
    )

    validate_generated_output(trader)


@pytest.mark.parametrize("intent,action,direction", [("close_long", "sell", "bearish"), ("cover_short", "buy", "bullish")])
def test_closing_intent_requires_no_residual_position_or_fabricated_exit_prices(
    intent: str,
    action: str,
    direction: str,
) -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    trader = _with_decision(
        _all_outputs()[5],
        direction=direction,
        action=action,
        confidence=0.8,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
        },
        position_advice="light",
        trade_intent=intent,
    )

    with pytest.raises(GeneratedOutputPolicyError) as exc_info:
        validate_generated_output(trader)

    message = str(exc_info.value)
    assert "position_advice" in message
    assert "target_price" in message
    assert "stop_loss" in message


def test_hold_without_position_needs_no_execution_prices_or_time_stop() -> None:
    from tradingscope.agents.utils.decision_validation import validate_generated_output

    trader = _with_decision(
        _all_outputs()[5],
        direction="neutral",
        action="hold",
        confidence=0.8,
        position_advice="none",
        trade_intent="hold",
        time_stop_days=None,
    )

    validate_generated_output(trader)


def test_hold_with_position_requires_time_stop() -> None:
    from tradingscope.agents.utils.decision_validation import (
        GeneratedOutputPolicyError,
        validate_generated_output,
    )

    portfolio = _with_decision(
        _all_outputs()[6],
        direction="neutral",
        action="hold",
        confidence=0.8,
        price_plan={"target_price": 110.0, "stop_loss": 95.0},
        position_advice="medium",
        trade_intent="hold",
        time_stop_days=None,
    )

    with pytest.raises(GeneratedOutputPolicyError, match="time_stop_days"):
        validate_generated_output(portfolio)


def test_valid_generated_plan_passes_policy_validation() -> None:
    from tradingscope.agents.utils.decision_validation import validate_generated_output

    trader = _with_decision(
        _all_outputs()[5],
        direction="bearish",
        action="sell",
        confidence=0.6,
        price_plan={
            "entry_price": 100.0,
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": 90.0,
            "stop_loss": 106.0,
        },
        position_advice="light",
    )

    validate_generated_output(trader)

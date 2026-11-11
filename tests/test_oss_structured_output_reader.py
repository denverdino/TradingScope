"""Tests for manifest-gated current reads and historical schema-v2 migration."""

from unittest.mock import patch

import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents.output import PortfolioManagerOutput
from tradingscope.utils.oss_structured_output_reader import (
    async_fetch_completed_output,
    fetch_completed_output,
)


def _manifest() -> dict[str, str]:
    return {
        "schema_version": "3.0",
        "status": "complete",
        "ticker": "AAPL",
        "analysis_date": "2026-07-14",
    }


def test_reader_requires_complete_manifest() -> None:
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        return_value=None,
    ) as fetch:
        assert fetch_completed_output("2026-07-14", "AAPL") is None

    fetch.assert_called_once_with("2026-07-14", "AAPL", "manifest")


def test_v3_reader_fetches_output_after_valid_manifest() -> None:
    output = _all_outputs()[-1].model_dump(mode="json")
    output["price_plan"].update(
        execution_price=101.0,
        execution_price_low=100.0,
        execution_price_high=102.0,
    )

    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=[_manifest(), output],
    ) as fetch:
        result = fetch_completed_output("2026-07-14", "AAPL")

    assert result == output
    portfolio = PortfolioManagerOutput.model_validate(result)
    assert portfolio.schema_version == "3.0"
    assert portfolio.analysis_date.isoformat() == "2026-07-14"
    assert portfolio.last_completed_session.isoformat() == "2026-07-13"
    assert portfolio.price_plan.execution_price == output["price_plan"]["execution_price"]
    assert fetch.call_args_list[1].args == ("2026-07-14", "AAPL", "portfolio_manager")


def test_v2_manifest_and_artifact_migrate_to_valid_v3_portfolio() -> None:
    # Persisted historical fixtures intentionally retain schema-v2 terminology.
    manifest = {
        "schema_version": "2.0",
        "status": "complete",
        "ticker": "AAPL",
        "trade_date": "2026-07-14",
    }
    output = {
        "schema_version": "2.0",
        "agent_name": "portfolio_manager",
        "ticker": "AAPL",
        "trade_date": "2026-07-14",
        "latest_trading_date": "2026-07-13",
        "decision": {
            "direction": "bullish",
            "action": "buy",
            "confidence": 0.8,
            "summary": "盈利预期改善",
            "reasoning": ["现金流稳健"],
        },
        "evidence": [],
        "limitations": [],
        "aggressive_viewpoints": ["增长强劲"],
        "conservative_viewpoints": ["关注回撤"],
        "neutral_viewpoints": ["保持纪律"],
        "adopted_reasoning": ["控制风险"],
        "position_advice": "light",
        "risk_score": 0.3,
        "trade_intent": "open_long",
        "time_stop_days": 3,
        "risk_control_measures": ["严格止损"],
        "price_plan": {
            "entry_price": 101.0,
            "entry_price_low": 100.0,
            "entry_price_high": 102.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
            "currency": "USD",
            "invalidation_conditions": ["跌破支撑"],
        },
    }

    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=[manifest, output],
    ):
        result = fetch_completed_output("2026-07-14", "AAPL")

    portfolio = PortfolioManagerOutput.model_validate(result)
    assert portfolio.schema_version == "3.0"
    assert portfolio.analysis_date.isoformat() == "2026-07-14"
    assert portfolio.last_completed_session.isoformat() == "2026-07-13"
    assert portfolio.price_plan.execution_price == 101.0
    assert portfolio.price_plan.execution_price_low == 100.0
    assert portfolio.price_plan.execution_price_high == 102.0
    assert portfolio.trade_intent == "open_long"
    assert portfolio.time_stop_days == 3
    assert "trade_date" not in result
    assert "latest_trading_date" not in result
    assert "entry_price" not in result["price_plan"]
    # Migration must not mutate the fetched historical payload.
    assert manifest["schema_version"] == "2.0"
    assert output["schema_version"] == "2.0"
    assert output["price_plan"]["entry_price"] == 101.0


@pytest.mark.parametrize(
    "updates",
    [
        {"ticker": "MSFT"},
        {"analysis_date": "2026-07-13"},
        {"analysis_date": "not-a-date"},
        {"status": "incomplete"},
        {"schema_version": "1.0"},
        {"schema_version": "4.0"},
        {"unexpected": True},
    ],
)
def test_reader_rejects_invalid_manifest_before_fetching_artifact(updates) -> None:
    manifest = {**_manifest(), **updates}

    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        return_value=manifest,
    ) as fetch:
        assert fetch_completed_output("2026-07-14", "AAPL") is None

    fetch.assert_called_once_with("2026-07-14", "AAPL", "manifest")


@pytest.mark.parametrize("manifest", [None, {}, [], "invalid"])
def test_reader_rejects_missing_or_malformed_manifest(manifest) -> None:
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        return_value=manifest,
    ) as fetch:
        assert fetch_completed_output("2026-07-14", "AAPL") is None

    assert fetch.call_count == 1


@pytest.mark.parametrize(
    "output",
    [None, {}, [], "invalid", {"schema_version": "1.0"}, {"schema_version": "4.0"}],
)
def test_reader_rejects_missing_or_unsupported_artifact(output) -> None:
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=[_manifest(), output],
    ):
        assert fetch_completed_output("2026-07-14", "AAPL") is None


@pytest.mark.asyncio
async def test_async_reader_uses_manifest_gate_and_current_schema() -> None:
    output = _all_outputs()[-1].model_dump(mode="json")

    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=[_manifest(), output],
    ):
        result = await async_fetch_completed_output("2026-07-14", "AAPL")

    assert PortfolioManagerOutput.model_validate(result).schema_version == "3.0"


@pytest.mark.parametrize(
    "dates",
    [
        {"trade_date": "2026-07-13", "analysis_date": "2026-07-14"},
        {"trade_date": "2026-07-14", "analysis_date": "2026-07-13"},
        {"analysis_date": "2026-07-14"},
    ],
)
def test_v2_reader_rejects_manifest_with_v3_date_fields(dates) -> None:
    manifest = {
        "schema_version": "2.0",
        "status": "complete",
        "ticker": "AAPL",
        **dates,
    }
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        return_value=manifest,
    ) as fetch:
        assert fetch_completed_output("2026-07-14", "AAPL") is None

    fetch.assert_called_once_with("2026-07-14", "AAPL", "manifest")


@pytest.mark.parametrize(
    "price_plan",
    [
        {"entry_price": 101.0, "execution_price": 102.0},
        {"execution_price": 101.0},
    ],
)
def test_v2_reader_rejects_artifact_with_v3_price_fields(price_plan) -> None:
    output = {"schema_version": "2.0", "price_plan": price_plan}
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=[_manifest(), output],
    ):
        assert fetch_completed_output("2026-07-14", "AAPL") is None

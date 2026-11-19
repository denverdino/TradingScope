"""Tests for current-schema, manifest-gated completed bundle reads."""

from unittest.mock import patch

import pytest

from tests.support.evaluation import completed_run
from tests.test_main import _analysis_result
from tradingscope.agents.output import PortfolioManagerOutput
from tradingscope.utils.oss_structured_output_reader import async_fetch_completed_output, fetch_completed_manifest, fetch_completed_output


def _manifest():
    return completed_run().manifest.model_dump(mode="json")


def _artifacts():
    return {"manifest": _manifest(), "runs/run-1/full_report": _analysis_result().model_dump(mode="json")}


def test_reader_requires_complete_manifest():
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", return_value=None) as fetch:
        assert fetch_completed_output("2026-07-14", "AAPL") is None
    fetch.assert_called_once_with("2026-07-14", "AAPL", "manifest")


def test_reader_fetches_current_stage_from_immutable_bundle():
    data = _artifacts()
    portfolio = data["runs/run-1/full_report"]["portfolio_manager"]
    portfolio["price_plan"].update(execution_price=101.0, execution_price_low=100.0, execution_price_high=102.0)
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, name: data.get(name)) as fetch:
        result = fetch_completed_output("2026-07-14", "AAPL")
    assert result == portfolio
    assert PortfolioManagerOutput.model_validate(result).price_plan.execution_price == 101.0
    assert all(call.args[2] != "portfolio_manager" for call in fetch.call_args_list)


@pytest.mark.parametrize(
    "manifest",
    [
        {"schema_version": "3.0", "status": "complete", "ticker": "AAPL", "analysis_date": "2026-07-14"},
        {"schema_version": "2.0", "status": "complete", "ticker": "AAPL", "trade_date": "2026-07-14"},
    ],
)
def test_old_manifest_is_not_eligible_for_new_evaluation(manifest):
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", return_value=manifest) as fetch:
        assert fetch_completed_manifest("2026-07-14", "AAPL") is None
        assert fetch_completed_output("2026-07-14", "AAPL") is None
    assert all(call.args[2] == "manifest" for call in fetch.call_args_list)


@pytest.mark.parametrize(
    "updates",
    [
        {"ticker": "MSFT"},
        {"analysis_date": "2026-07-13"},
        {"analysis_date": "not-a-date"},
        {"status": "incomplete"},
        {"schema_version": "3.0"},
        {"schema_version": "2.0"},
        {"schema_version": "4.0"},
        {"run_id": None},
        {"forecast_started_at": None},
        {"forecast_completed_at": None},
        {"last_completed_session": None},
        {"unexpected": True},
    ],
)
def test_invalid_manifest_is_rejected_before_reading_bundle(updates):
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", return_value={**_manifest(), **updates}) as fetch:
        assert fetch_completed_output("2026-07-14", "AAPL") is None
    fetch.assert_called_once_with("2026-07-14", "AAPL", "manifest")


@pytest.mark.parametrize("manifest", [None, {}, [], "invalid"])
def test_missing_or_malformed_manifest_is_rejected(manifest):
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", return_value=manifest) as fetch:
        assert fetch_completed_output("2026-07-14", "AAPL") is None
    assert fetch.call_count == 1


@pytest.mark.parametrize("bundle", [None, {}, [], "invalid", {"schema_version": "2.0"}, {"schema_version": "4.0"}])
def test_missing_or_unsupported_bundle_is_rejected(bundle):
    data = {"manifest": _manifest(), "runs/run-1/full_report": bundle}
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, name: data.get(name)):
        assert fetch_completed_output("2026-07-14", "AAPL") is None


@pytest.mark.parametrize("stage", ["trader", "portfolio_manager"])
def test_missing_trade_intent_is_rejected_not_inferred(stage):
    data = _artifacts()
    data["runs/run-1/full_report"][stage].pop("trade_intent")
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, name: data.get(name)):
        assert fetch_completed_output("2026-07-14", "AAPL") is None


def test_old_price_field_is_rejected_not_migrated():
    data = _artifacts()
    plan = data["runs/run-1/full_report"]["portfolio_manager"]["price_plan"]
    plan["entry_price"] = plan.pop("execution_price")
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, name: data.get(name)):
        assert fetch_completed_output("2026-07-14", "AAPL") is None


def test_mutable_stage_file_cannot_replace_a_missing_bundle():
    data = {"manifest": _manifest(), "portfolio_manager": _analysis_result().portfolio_manager.model_dump(mode="json")}
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, name: data.get(name)):
        assert fetch_completed_output("2026-07-14", "AAPL") is None


@pytest.mark.asyncio
async def test_async_reader_uses_current_completed_bundle():
    data = _artifacts()
    with patch("tradingscope.utils.oss_structured_output_reader.fetch_structured_output", side_effect=lambda _date, _ticker, name: data.get(name)):
        result = await async_fetch_completed_output("2026-07-14", "AAPL")
    assert PortfolioManagerOutput.model_validate(result).trade_intent == "hold"

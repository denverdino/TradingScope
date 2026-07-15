"""Tests for manifest-gated schema-v2 reads."""

from unittest.mock import patch

from tradingscope.utils.oss_structured_output_reader import fetch_completed_v2_output


def test_v2_reader_requires_complete_manifest() -> None:
    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        return_value=None,
    ) as fetch:
        assert fetch_completed_v2_output("2026-07-14", "AAPL") is None

    fetch.assert_called_once_with("2026-07-14", "AAPL", "manifest")


def test_v2_reader_fetches_output_after_valid_manifest() -> None:
    manifest = {
        "schema_version": "2.0",
        "status": "complete",
        "ticker": "AAPL",
        "trade_date": "2026-07-14",
    }
    output = {"schema_version": "2.0", "agent_name": "portfolio_manager"}

    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        side_effect=[manifest, output],
    ):
        assert fetch_completed_v2_output("2026-07-14", "AAPL") == output


def test_v2_reader_rejects_mismatched_manifest() -> None:
    manifest = {
        "schema_version": "2.0",
        "status": "complete",
        "ticker": "MSFT",
        "trade_date": "2026-07-14",
    }

    with patch(
        "tradingscope.utils.oss_structured_output_reader.fetch_structured_output",
        return_value=manifest,
    ) as fetch:
        assert fetch_completed_v2_output("2026-07-14", "AAPL") is None

    assert fetch.call_count == 1

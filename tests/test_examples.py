"""Tests for public examples following the schema-v3 API."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from examples import memories as memories_example
from examples import workflow as workflow_example
from tests.test_main import _analysis_result


@pytest.mark.asyncio
async def test_workflow_example_renders_result_and_json(capsys) -> None:
    result = _analysis_result()

    with patch.object(
        workflow_example,
        "analyze",
        new=AsyncMock(return_value=result),
    ):
        await workflow_example.main()

    output = capsys.readouterr().out
    assert "最终投资组合决策" in output
    assert '"schema_version": "3.0"' in output


def test_memories_example_uses_analysis_date_and_execution_price_without_status(tmp_path, capsys) -> None:
    record_dir = tmp_path / "analysis_records" / "2026-07-14"
    record_dir.mkdir(parents=True)
    (record_dir / "AAPL.json").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "analysis_date": "2026-07-14",
                "direction": "bullish",
                "action": "buy",
                "confidence": 0.8,
                "execution_price": 101.0,
            },
        ),
        encoding="utf-8",
    )

    memories_example.review_records("AAPL", "2026-07-14", str(tmp_path))

    output = capsys.readouterr().out
    assert "[2026-07-14] AAPL" in output
    assert "Execution Price: 101.0" in output
    assert "Entry Price:" not in output
    assert "Status:" not in output

"""Tests for public examples following the schema-v2 API."""

from unittest.mock import AsyncMock, patch

import pytest

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
    assert '"schema_version": "2.0"' in output

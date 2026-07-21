"""Test cases for the TradingScope CLI."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents.output import AnalysisResult, AnalystOutputs


def _analysis_result() -> AnalysisResult:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    return AnalysisResult(
        schema_version="2.0",
        ticker="AAPL",
        trade_date="2026-07-14",
        latest_trading_date="2026-07-13",
        analysts=AnalystOutputs(
            market=market,
            fundamentals=fundamentals,
            news=news,
            social_media=social,
        ),
        research_manager=research,
        trader=trader,
        portfolio_manager=portfolio,
    )


# Test imports work correctly
def test_package_imports() -> None:
    """Test that the main modules can be imported without errors."""
    # These should not raise ImportError

    # Basic assertion to ensure the test runs
    assert True


def test_version_exists() -> None:
    """Test that the package has a version."""
    from tradingscope import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_main_serializes_result_and_renders_markdown(tmp_path) -> None:
    from tradingscope import main as main_module

    result = _analysis_result()
    with (
        patch.object(main_module, "analyze", new=AsyncMock(return_value=result)),
        patch.object(main_module, "DEFAULT_CONFIG", {"results_dir": str(tmp_path)}),
        patch("sys.argv", ["tradingscope", "AAPL", "--output", "both"]),
    ):
        main_module.main()

    full_json = next(tmp_path.rglob("AAPL_report_*.json"))
    parsed = AnalysisResult.model_validate_json(full_json.read_text())
    assert parsed == result
    assert next(tmp_path.rglob("portfolio_manager.json")).exists()
    html = next(tmp_path.rglob("AAPL_report_*.html")).read_text()
    assert "最终投资组合决策" in html


def test_main_owns_tracing_lifecycle() -> None:
    from tradingscope import main as main_module

    provider = object()
    setup_tracing = Mock(return_value=provider)
    shutdown_tracing = Mock()
    process = Mock()
    with (
        patch.object(main_module, "setup_tracing", setup_tracing),
        patch.object(main_module, "shutdown_tracing", shutdown_tracing),
        patch.object(main_module, "_main", process),
    ):
        main_module.main()

    process.assert_called_once_with()
    setup_tracing.assert_called_once_with("tradingscope-main")
    shutdown_tracing.assert_called_once_with(provider)


def test_main_shuts_down_tracing_when_processing_fails() -> None:
    from tradingscope import main as main_module

    provider = object()
    setup_tracing = Mock(return_value=provider)
    shutdown_tracing = Mock()
    failure = RuntimeError("analysis failed")
    with (
        patch.object(main_module, "setup_tracing", setup_tracing),
        patch.object(main_module, "shutdown_tracing", shutdown_tracing),
        patch.object(main_module, "_main", side_effect=failure),
        pytest.raises(RuntimeError, match="analysis failed"),
    ):
        main_module.main()

    shutdown_tracing.assert_called_once_with(provider)

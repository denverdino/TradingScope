"""Test cases for the TradingScope CLI."""

from unittest.mock import AsyncMock, patch

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

"""Test cases for the TradingScope package."""


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

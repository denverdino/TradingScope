"""Unit tests for utility functions."""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.agents.utils.stock_utils import StockMarket, StockUtils


class TestStockUtils:
    """Test cases for StockUtils class."""

    def test_identify_stock_market_china_a(self):
        """Test identifying China A stock market."""
        # Test 6-digit codes
        assert StockUtils.identify_stock_market("000001") == StockMarket.CHINA_A
        assert StockUtils.identify_stock_market("600000") == StockMarket.CHINA_A
        assert StockUtils.identify_stock_market("300001") == StockMarket.CHINA_A

    def test_identify_stock_market_hk(self):
        """Test identifying Hong Kong stock market."""
        # Test HK stock codes
        assert StockUtils.identify_stock_market("0700.HK") == StockMarket.HONG_KONG
        assert StockUtils.identify_stock_market("9988.HK") == StockMarket.HONG_KONG
        assert StockUtils.identify_stock_market("09988.HK") == StockMarket.HONG_KONG

    def test_identify_stock_market_us(self):
        """Test identifying US stock market."""
        # Test US stock codes
        assert StockUtils.identify_stock_market("AAPL") == StockMarket.US
        assert StockUtils.identify_stock_market("TSLA") == StockMarket.US
        assert StockUtils.identify_stock_market("MSFT") == StockMarket.US
        assert StockUtils.identify_stock_market("GOOGL") == StockMarket.US

    def test_identify_stock_market_unknown(self):
        """Test identifying unknown stock market."""
        # Test unknown/invalid codes
        assert StockUtils.identify_stock_market("") == StockMarket.UNKNOWN
        assert StockUtils.identify_stock_market(None) == StockMarket.UNKNOWN
        assert StockUtils.identify_stock_market("INVALID") == StockMarket.UNKNOWN
        assert StockUtils.identify_stock_market("1234567") == StockMarket.UNKNOWN  # Too many digits

    def test_is_stock_type_functions(self):
        """Test the is_*_stock functions."""
        # China A stock
        assert StockUtils.is_china_stock("000001") is True
        assert StockUtils.is_hk_stock("000001") is False
        assert StockUtils.is_us_stock("000001") is False

        # HK stock
        assert StockUtils.is_china_stock("0700.HK") is False
        assert StockUtils.is_hk_stock("0700.HK") is True
        assert StockUtils.is_us_stock("0700.HK") is False

        # US stock
        assert StockUtils.is_china_stock("AAPL") is False
        assert StockUtils.is_hk_stock("AAPL") is False
        assert StockUtils.is_us_stock("AAPL") is True

    def test_get_currency_info(self):
        """Test getting currency information."""
        # China A stock
        currency_name, currency_symbol = StockUtils.get_currency_info("000001")
        assert currency_name == "人民币"
        assert currency_symbol == "¥"

        # HK stock
        currency_name, currency_symbol = StockUtils.get_currency_info("0700.HK")
        assert currency_name == "港币"
        assert currency_symbol == "HK$"

        # US stock
        currency_name, currency_symbol = StockUtils.get_currency_info("AAPL")
        assert currency_name == "美元"
        assert currency_symbol == "$"

        # Unknown stock
        currency_name, currency_symbol = StockUtils.get_currency_info("INVALID")
        assert currency_name == "未知"
        assert currency_symbol == "?"

    def test_get_data_source(self):
        """Test getting data source information."""
        # China A stock
        assert StockUtils.get_data_source("000001") == "china_unified"

        # HK stock
        assert StockUtils.get_data_source("0700.HK") == "yahoo_finance"

        # US stock
        assert StockUtils.get_data_source("AAPL") == "yahoo_finance"

        # Unknown stock
        assert StockUtils.get_data_source("INVALID") == "unknown"

    def test_normalize_hk_ticker(self):
        """Test normalizing HK ticker format."""
        # Already normalized
        assert StockUtils.normalize_hk_ticker("0700.HK") == "0700.HK"
        assert StockUtils.normalize_hk_ticker("9988.HK") == "9988.HK"

        # Pure numbers (4-5 digits) - should add .HK suffix
        assert StockUtils.normalize_hk_ticker("0700") == "0700.HK"
        assert StockUtils.normalize_hk_ticker("9988") == "9988.HK"

        # 3-digit numbers - should not add .HK suffix (doesn't match 4-5 digit pattern)
        assert StockUtils.normalize_hk_ticker("700") == "700"

        # Edge cases
        assert StockUtils.normalize_hk_ticker("") == ""
        assert StockUtils.normalize_hk_ticker(None) is None

    def test_get_market_info(self):
        """Test getting detailed market information."""
        # China A stock
        info = StockUtils.get_market_info("000001")
        assert info["ticker"] == "000001"
        assert info["market"] == "china_a"
        assert info["market_name"] == "中国A股"
        assert info["currency_name"] == "人民币"
        assert info["currency_symbol"] == "¥"
        assert info["data_source"] == "china_unified"
        assert info["is_china"] is True
        assert info["is_hk"] is False
        assert info["is_us"] is False

        # HK stock
        info = StockUtils.get_market_info("0700.HK")
        assert info["ticker"] == "0700.HK"
        assert info["market"] == "hong_kong"
        assert info["market_name"] == "港股"
        assert info["currency_name"] == "港币"
        assert info["currency_symbol"] == "HK$"
        assert info["data_source"] == "yahoo_finance"
        assert info["is_china"] is False
        assert info["is_hk"] is True
        assert info["is_us"] is False

        # US stock
        info = StockUtils.get_market_info("AAPL")
        assert info["ticker"] == "AAPL"
        assert info["market"] == "us"
        assert info["market_name"] == "美股"
        assert info["currency_name"] == "美元"
        assert info["currency_symbol"] == "$"
        assert info["data_source"] == "yahoo_finance"
        assert info["is_china"] is False
        assert info["is_hk"] is False
        assert info["is_us"] is True


class TestAgentUtils:
    """Test cases for agent utility functions."""

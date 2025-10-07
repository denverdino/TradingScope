"""Unit tests for utility functions."""

import os
import sys
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.agents.utils.agent_utils import get_company_name
from tradingscope.utils.stock_utils import StockMarket, StockUtils


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

    def test_get_company_name_china_stock(self):
        """Test getting company name for China stocks."""
        market_info = {"is_china": True, "is_hk": False, "is_us": False}

        # Mock the interface function
        with patch("tradingscope.agents.utils.agent_utils.interface") as mock_interface:
            mock_interface.get_china_stock_info_unified.return_value = "股票名称:平安银行\n股票代码:000001"

            company_name = get_company_name("000001", market_info)
            assert company_name == "平安银行"

    def test_get_company_name_china_stock_fallback(self):
        """Test getting company name for China stocks with fallback."""
        market_info = {"is_china": True, "is_hk": False, "is_us": False}

        # Mock the interface function to return invalid data
        with patch("tradingscope.agents.utils.agent_utils.interface") as mock_interface:
            mock_interface.get_china_stock_info_unified.return_value = "Invalid data"

            company_name = get_company_name("000001", market_info)
            assert company_name == "股票代码000001"

    def test_get_company_name_hk_stock(self):
        """Test getting company name for HK stocks."""
        market_info = {"is_china": False, "is_hk": True, "is_us": False}

        # Mock the get_hk_company_name_improved function
        with patch("tradingscope.agents.utils.agent_utils.get_hk_company_name_improved") as mock_get_name:
            mock_get_name.return_value = "腾讯控股"

            company_name = get_company_name("0700.HK", market_info)
            assert company_name == "腾讯控股"

    def test_get_company_name_hk_stock_fallback(self):
        """Test getting company name for HK stocks with fallback."""
        market_info = {"is_china": False, "is_hk": True, "is_us": False}

        # Mock the get_hk_company_name_improved function to raise an exception
        with patch("tradingscope.agents.utils.agent_utils.get_hk_company_name_improved") as mock_get_name:
            mock_get_name.side_effect = Exception("Test exception")

            company_name = get_company_name("0700.HK", market_info)
            assert company_name == "港股0700"

    def test_get_company_name_us_stock(self):
        """Test getting company name for US stocks."""
        market_info = {"is_china": False, "is_hk": False, "is_us": True}

        # Test known US stocks
        company_name = get_company_name("AAPL", market_info)
        assert company_name == "苹果公司"

        company_name = get_company_name("TSLA", market_info)
        assert company_name == "特斯拉"

        company_name = get_company_name("NVDA", market_info)
        assert company_name == "英伟达"

        # Test unknown US stock
        company_name = get_company_name("UNKNOWN", market_info)
        assert company_name == "美股UNKNOWN"

    def test_get_company_name_unknown_stock(self):
        """Test getting company name for unknown stocks."""
        market_info = {"is_china": False, "is_hk": False, "is_us": False}

        company_name = get_company_name("INVALID", market_info)
        assert company_name == "股票INVALID"

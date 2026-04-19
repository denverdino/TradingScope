from typing import Annotated

from tradingscope.dataflows.interface import route_to_vendor

from .tool_response_helper import agentscope_tool


@agentscope_tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)


@agentscope_tool
def get_stock_info(
    symbol: Annotated[str, "ticker symbol of the company"],
) -> str:
    """
    Retrieve latest stock info for a given ticker symbol.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
    Returns:
        str: A formatted json string containing the latest stock info for the specified ticker symbol, including currentPrice, preMarketPrice, preMarketChange, postMarketPrice, postMarketChange, volume, averageVolume, etc.
    """
    return route_to_vendor("get_stock_info", symbol)


@agentscope_tool
def get_sector_performance(
    symbol: Annotated[str, "ticker symbol of the company"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve sector and industry performance data relative to the stock.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted string containing the stock's sector, industry, and performance comparison with sector ETF.
    """
    return route_to_vendor("get_sector_performance", symbol, look_back_days)


@agentscope_tool
def get_market_indices(
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve major market indices performance (S&P 500, NASDAQ, Dow Jones, VIX, Russell 2000).
    Args:
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted string containing performance data for major market indices.
    """
    return route_to_vendor("get_market_indices", look_back_days)


@agentscope_tool
def get_options_analysis(
    symbol: Annotated[str, "ticker symbol of the company"],
) -> str:
    """
    Retrieve options chain analysis for a given ticker symbol, including Put/Call ratio,
    support/resistance levels from open interest, and max pain price for the nearest
    expiration date. Primarily available for US-listed stocks.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
    Returns:
        str: A formatted string containing options chain analysis including Put/Call ratio, support/resistance levels, and max pain price.
    """
    return route_to_vendor("get_options_analysis", symbol)


@agentscope_tool
def get_volume_analysis(
    symbol: Annotated[str, "ticker symbol of the company"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve comprehensive volume analysis for a given ticker symbol, including
    daily volume metrics, volume trend, OBV analysis, volume-price divergence
    detection, and up/down day distribution statistics.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted string containing comprehensive volume analysis.
    """
    return route_to_vendor("get_volume_analysis", symbol, look_back_days)

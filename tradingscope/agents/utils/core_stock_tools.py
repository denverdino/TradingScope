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
        str: A formatted json string containing the latest stock info for the specified ticker symbol, including currentPrice, preMarketPrice, etc.
    """
    return route_to_vendor("get_stock_info", symbol)

import logging
import os
from datetime import datetime
from typing import Annotated

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry

logger = logging.getLogger(__name__)


def get_YFin_stock_info(symbol: Annotated[str, "ticker symbol of the company"]):
    info = yf_retry(lambda: yf.Ticker(symbol.upper()).info)
    info.pop("longBusinessSummary", None)
    info.pop("companyOfficers", None)
    return info


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_date))

    # Check if data is empty
    if data.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}")

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)

        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []

        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")

            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"

            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"

    except Exception as e:
        logger.warning("Error getting bulk stockstats data: %s", e)
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(symbol, indicator, curr_date_dt.strftime("%Y-%m-%d"))
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"],
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    from .config import get_config

    config = get_config()
    online = config["data_vendors"]["technical_indicators"] != "local"

    if not online:
        # Local data path
        try:
            data = pd.read_csv(
                os.path.join(
                    config.get("data_cache_dir", "data"),
                    f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                ),
                on_bad_lines="skip",
            )
        except FileNotFoundError:
            raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
    else:
        # Online data fetching with caching
        today_date = pd.Timestamp.today()
        curr_date_dt = pd.to_datetime(curr_date)

        end_date = today_date
        start_date = today_date - pd.DateOffset(years=15)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        os.makedirs(config["data_cache_dir"], exist_ok=True)

        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
        )

        if os.path.exists(data_file):
            data = pd.read_csv(data_file, on_bad_lines="skip")
        else:
            data = yf_retry(
                lambda: yf.download(
                    symbol,
                    start=start_date_str,
                    end=end_date_str,
                    multi_level_index=False,
                    progress=False,
                    auto_adjust=True,
                )
            )
            data = data.reset_index()
            data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator

    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]

        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)

    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        logger.warning(
            "Error getting stockstats indicator data for indicator %s on %s: %s",
            indicator,
            curr_date,
            e,
        )
        return ""

    return str(indicator_value)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None,
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet)
        else:
            data = yf_retry(lambda: ticker_obj.balance_sheet)

        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None,
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_cashflow)
        else:
            data = yf_retry(lambda: ticker_obj.cashflow)

        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None,
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_income_stmt)
        else:
            data = yf_retry(lambda: ticker_obj.income_stmt)

        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(ticker: Annotated[str, "ticker symbol of the company"]):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.insider_transactions)

        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"


def get_sector_performance(
    ticker: Annotated[str, "ticker symbol of the company"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
):
    """Get sector and industry performance data relative to the stock.

    Returns the stock's sector ETF performance compared to the stock itself,
    helping analyze whether the stock is outperforming or underperforming its sector.
    """
    import pandas as pd

    # Sector ETF mappings for US stocks
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financial Services": "XLF",
        "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP",
        "Industrials": "XLI",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Basic Materials": "XLB",
        "Communication Services": "XLC",
    }

    # Tickers whose yfinance sector info is inaccurate and should be ignored.
    # e.g. BABA is classified as "Consumer Cyclical" by yfinance, but as a
    # Chinese ADR it should not be compared against the US sector ETF (XLY).
    IGNORE_SECTOR_TICKERS = {"BABA"}

    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)

        ignore_sector = ticker.upper() in IGNORE_SECTOR_TICKERS
        sector = "Unknown" if ignore_sector else info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        # Get stock price performance
        end_date = datetime.now()
        start_date = end_date - relativedelta(days=look_back_days + 10)  # Extra buffer for trading days

        stock_hist = yf_retry(lambda: ticker_obj.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")))

        if stock_hist.empty:
            return f"No historical data found for symbol '{ticker}'"

        result = f"# Sector and Industry Analysis for {ticker.upper()}\n"
        result += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        result += "## Stock Information\n"
        result += f"- Sector: {sector}\n"
        result += f"- Industry: {industry}\n\n"

        # Calculate stock performance
        stock_close = stock_hist["Close"]
        if len(stock_close) >= 2:
            stock_return_1d = ((stock_close.iloc[-1] / stock_close.iloc[-2]) - 1) * 100
            result += f"- Stock 1-Day Return: {stock_return_1d:.2f}%\n"

        if len(stock_close) >= 5:
            stock_return_5d = ((stock_close.iloc[-1] / stock_close.iloc[-5]) - 1) * 100
            result += f"- Stock 5-Day Return: {stock_return_5d:.2f}%\n"

        if len(stock_close) >= look_back_days:
            stock_return_period = ((stock_close.iloc[-1] / stock_close.iloc[-look_back_days]) - 1) * 100
            result += f"- Stock {look_back_days}-Day Return: {stock_return_period:.2f}%\n"

        # Get sector ETF performance if available
        sector_etf = SECTOR_ETFS.get(sector)
        if sector_etf:
            try:
                etf_obj = yf.Ticker(sector_etf)
                etf_hist = yf_retry(lambda: etf_obj.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")))

                if not etf_hist.empty:
                    etf_close = etf_hist["Close"]
                    result += f"\n## Sector ETF Performance ({sector_etf} - {sector})\n"

                    if len(etf_close) >= 2:
                        etf_return_1d = ((etf_close.iloc[-1] / etf_close.iloc[-2]) - 1) * 100
                        result += f"- Sector 1-Day Return: {etf_return_1d:.2f}%\n"
                        if len(stock_close) >= 2:
                            result += f"- Stock vs Sector (1D): {stock_return_1d - etf_return_1d:+.2f}%\n"

                    if len(etf_close) >= 5:
                        etf_return_5d = ((etf_close.iloc[-1] / etf_close.iloc[-5]) - 1) * 100
                        result += f"- Sector 5-Day Return: {etf_return_5d:.2f}%\n"
                        if len(stock_close) >= 5:
                            result += f"- Stock vs Sector (5D): {stock_return_5d - etf_return_5d:+.2f}%\n"

                    if len(etf_close) >= look_back_days:
                        etf_return_period = ((etf_close.iloc[-1] / etf_close.iloc[-look_back_days]) - 1) * 100
                        result += f"- Sector {look_back_days}-Day Return: {etf_return_period:.2f}%\n"
                        if len(stock_close) >= look_back_days:
                            result += f"- Stock vs Sector ({look_back_days}D): {stock_return_period - etf_return_period:+.2f}%\n"
            except Exception as e:
                result += f"\n## Sector ETF ({sector_etf}) data unavailable: {str(e)}\n"
        else:
            result += f"\n## Note: No sector ETF mapping available for sector '{sector}'\n"

        return result

    except Exception as e:
        return f"Error retrieving sector performance for {ticker}: {str(e)}"


def get_fundamentals(ticker: Annotated[str, "ticker symbol of the company"], curr_date: Annotated[str, "current date (not used for yfinance)"] = None):
    """Get company fundamentals overview from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)
        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("Price to Sales (TTM)", info.get("priceToSalesTrailing12Months")),
            ("Enterprise Value", info.get("enterpriseValue")),
            ("EV/Revenue", info.get("enterpriseToRevenue")),
            ("EV/EBITDA", info.get("enterpriseToEbitda")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit (TTM)", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income (TTM)", info.get("netIncomeToCommon")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Assets (TTM)", info.get("returnOnAssets")),
            ("Return on Equity (TTM)", info.get("returnOnEquity")),
            ("Revenue Growth", info.get("revenueGrowth")),
            ("Earnings Growth", info.get("earningsGrowth")),
            ("Total Cash", info.get("totalCash")),
            ("Total Debt", info.get("totalDebt")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Dividend Rate", info.get("dividendRate")),
            ("Dividend Yield", info.get("dividendYield") / 100),
            ("Payout Ratio", info.get("payoutRatio")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Shares Outstanding", info.get("sharesOutstanding")),
            ("Float Shares", info.get("floatShares")),
            ("Short Ratio", info.get("shortRatio")),
            ("Short % of Float", info.get("shortPercentOfFloat")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                # Format percentages
                if label in [
                    "Profit Margin",
                    "Operating Margin",
                    "ROA",
                    "ROE",
                    "Revenue Growth",
                    "Earnings Growth",
                    "Dividend Yield",
                    "Payout Ratio",
                    "Short % of Float",
                ]:
                    if isinstance(value, (int, float)):
                        lines.append(f"{label}: {value:.2%}")
                    else:
                        lines.append(f"{label}: {value}")
                # Format large numbers
                elif label in [
                    "Market Cap",
                    "Enterprise Value",
                    "Revenue (TTM)",
                    "Gross Profit (TTM)",
                    "EBITDA",
                    "Net Income (TTM)",
                    "Total Cash",
                    "Total Debt",
                    "Free Cash Flow",
                    "Shares Outstanding",
                    "Float Shares",
                ]:
                    if isinstance(value, (int, float)) and value >= 1_000_000:
                        if value >= 1_000_000_000_000:
                            lines.append(f"{label}: ${value / 1_000_000_000_000:.2f}T")
                        elif value >= 1_000_000_000:
                            lines.append(f"{label}: ${value / 1_000_000_000:.2f}B")
                        elif value >= 1_000_000:
                            lines.append(f"{label}: ${value / 1_000_000:.2f}M")
                        else:
                            lines.append(f"{label}: {value:,.0f}")
                    else:
                        lines.append(f"{label}: {value}")
                else:
                    lines.append(f"{label}: {value}")

        header = f"# Company Fundamentals for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_market_indices(
    look_back_days: Annotated[int, "how many days to look back"] = 30,
):
    """Get major market indices performance.

    Returns performance data for major market indices (S&P 500, Nasdaq, Dow Jones)
    and China-related indices (for Chinese ADRs like BABA, PDD, JD, etc.)
    to provide market context for stock analysis.
    """
    # Major market indices (US + China-related)
    INDICES = {
        # US Market
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ Composite",
        "^DJI": "Dow Jones Industrial Average",
        "^VIX": "VIX Volatility Index",
        "^RUT": "Russell 2000 (Small Cap)",
        # China Market (important for Chinese ADRs like BABA, PDD, JD, NTES, etc.)
        "^HXC": "Nasdaq Golden Dragon China Index (中概股)",
        "MCHI": "iShares MSCI China ETF (中国ETF)",
        "^HSI": "Hang Seng Index (恒生指数)",
    }

    try:
        end_date = datetime.now()
        start_date = end_date - relativedelta(days=look_back_days + 10)

        result = "# Major Market Indices Performance\n"
        result += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"# Look back period: {look_back_days} days\n\n"

        for symbol, name in INDICES.items():
            try:
                ticker_obj = yf.Ticker(symbol)
                hist = yf_retry(lambda t=ticker_obj: t.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")))

                if hist.empty:
                    result += f"## {name} ({symbol}): No data available\n\n"
                    continue

                close = hist["Close"]
                current_price = close.iloc[-1]

                result += f"## {name} ({symbol})\n"
                result += f"- Current Level: {current_price:,.2f}\n"

                # Calculate returns for different periods
                if len(close) >= 2:
                    return_1d = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100
                    result += f"- 1-Day Change: {return_1d:+.2f}%\n"

                if len(close) >= 5:
                    return_5d = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100
                    result += f"- 5-Day Change: {return_5d:+.2f}%\n"

                if len(close) >= 10:
                    return_10d = ((close.iloc[-1] / close.iloc[-10]) - 1) * 100
                    result += f"- 10-Day Change: {return_10d:+.2f}%\n"

                if len(close) >= look_back_days:
                    return_period = ((close.iloc[-1] / close.iloc[-look_back_days]) - 1) * 100
                    result += f"- {look_back_days}-Day Change: {return_period:+.2f}%\n"

                # For VIX, add detailed interpretation and trend analysis
                if symbol == "^VIX":
                    if current_price < 15:
                        result += "- VIX Interpretation: Low volatility (complacency / 极度乐观)\n"
                    elif current_price < 20:
                        result += "- VIX Interpretation: Normal volatility (正常波动)\n"
                    elif current_price < 25:
                        result += "- VIX Interpretation: Elevated volatility (波动偏高，需谨慎)\n"
                    elif current_price < 30:
                        result += "- VIX Interpretation: High volatility (高度恐慌，市场承压)\n"
                    else:
                        result += "- VIX Interpretation: Extreme fear (极端恐慌，市场可能超卖)\n"

                    # VIX trend analysis
                    if len(close) >= 5:
                        vix_5d_high = close.iloc[-5:].max()
                        vix_5d_low = close.iloc[-5:].min()
                        vix_5d_avg = close.iloc[-5:].mean()
                        result += f"- VIX 5-Day Range: {vix_5d_low:.2f} - {vix_5d_high:.2f} (Avg: {vix_5d_avg:.2f})\n"

                        # Spike detection: VIX surged more than 20% in 5 days
                        vix_5d_change = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100
                        if vix_5d_change > 20:
                            result += f"- ⚠ VIX Spike Detected: 5-day surge {vix_5d_change:+.2f}%, 市场恐慌情绪急剧上升\n"
                        elif vix_5d_change < -20:
                            result += f"- VIX Rapid Decline: 5-day drop {vix_5d_change:+.2f}%, 恐慌情绪快速消退\n"

                    if len(close) >= 20:
                        vix_20d_avg = close.iloc[-20:].mean()
                        result += f"- VIX 20-Day Average: {vix_20d_avg:.2f}\n"
                        if current_price > vix_20d_avg * 1.1:
                            result += "- VIX above 20-day MA (市场波动高于近期均值，风险偏高)\n"
                        elif current_price < vix_20d_avg * 0.9:
                            result += "- VIX below 20-day MA (市场波动低于近期均值，情绪偏乐观)\n"
                        else:
                            result += "- VIX near 20-day MA (波动处于近期正常范围)\n"

                # For NASDAQ, add trend and momentum analysis
                if symbol == "^IXIC":
                    if len(close) >= 20:
                        nasdaq_20d_avg = close.iloc[-20:].mean()
                        result += f"- NASDAQ 20-Day MA: {nasdaq_20d_avg:,.2f}\n"
                        if current_price > nasdaq_20d_avg:
                            pct_above = ((current_price / nasdaq_20d_avg) - 1) * 100
                            result += f"- Price above 20-Day MA by {pct_above:.2f}% (中短期趋势偏多)\n"
                        else:
                            pct_below = ((nasdaq_20d_avg / current_price) - 1) * 100
                            result += f"- Price below 20-Day MA by {pct_below:.2f}% (中短期趋势偏空)\n"

                    if len(close) >= 5:
                        nasdaq_5d_high = close.iloc[-5:].max()
                        nasdaq_5d_low = close.iloc[-5:].min()
                        result += f"- NASDAQ 5-Day Range: {nasdaq_5d_low:,.2f} - {nasdaq_5d_high:,.2f}\n"

                        # Momentum: consecutive up/down days
                        up_days = sum(1 for i in range(-5, 0) if close.iloc[i] > close.iloc[i - 1])
                        result += f"- NASDAQ Recent Momentum: {up_days}/5 trading days positive (近5日上涨天数)\n"

                    if len(close) >= 2:
                        # Daily range as volatility indicator
                        if "High" in hist.columns and "Low" in hist.columns:
                            daily_range_pct = ((hist["High"].iloc[-1] - hist["Low"].iloc[-1]) / close.iloc[-2]) * 100
                            result += f"- NASDAQ Daily Range: {daily_range_pct:.2f}% (当日振幅)\n"

                result += "\n"

            except Exception as e:
                result += f"## {name} ({symbol}): Error - {str(e)}\n\n"

        return result

    except Exception as e:
        return f"Error retrieving market indices: {str(e)}"


def get_options_analysis(ticker: Annotated[str, "ticker symbol of the company"]):
    """Get options chain analysis for the nearest expiration date.

    Returns Put/Call ratio, support/resistance levels from open interest,
    and max pain price. Primarily available for US-listed stocks.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        # Get available expiration dates
        expirations = yf_retry(lambda: ticker_obj.options)
        if not expirations:
            return f"No options data available for symbol '{ticker}'. Options data is typically only available for US-listed stocks."

        nearest_exp = expirations[0]

        # Fetch options chain
        opt = yf_retry(lambda: ticker_obj.option_chain(nearest_exp))
        calls_df = opt.calls
        puts_df = opt.puts

        if calls_df.empty and puts_df.empty:
            return f"Options chain is empty for symbol '{ticker}' (expiration: {nearest_exp})."

        # Get current stock price for context
        info = yf_retry(lambda: ticker_obj.info)
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        # Calculate days to expiration
        exp_date = datetime.strptime(nearest_exp, "%Y-%m-%d")
        today = datetime.now()
        days_to_exp = (exp_date - today).days + 1  # +1 because expiration is end of day

        # --- Put/Call Ratio ---
        total_call_oi = calls_df["openInterest"].fillna(0).sum()
        total_put_oi = puts_df["openInterest"].fillna(0).sum()
        total_call_vol = calls_df["volume"].fillna(0).sum()
        total_put_vol = puts_df["volume"].fillna(0).sum()

        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else float("inf")
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else float("inf")

        # PCR sentiment interpretation
        if pcr_oi == float("inf"):
            pcr_oi_str = "N/A (Call OI = 0)"
            sentiment = "无法判断（Call 未平仓量为零）"
        elif pcr_oi < 0.7:
            pcr_oi_str = f"{pcr_oi:.2f}"
            sentiment = "偏多情绪（看涨占主导，市场乐观）"
        elif pcr_oi <= 1.0:
            pcr_oi_str = f"{pcr_oi:.2f}"
            sentiment = "中性偏多（多空力量相对均衡，略偏看涨）"
        elif pcr_oi <= 1.5:
            pcr_oi_str = f"{pcr_oi:.2f}"
            sentiment = "中性偏空（看跌占优，市场谨慎）"
        else:
            pcr_oi_str = f"{pcr_oi:.2f}"
            sentiment = "偏空/防御情绪（大量看跌期权，市场恐慌或机构对冲）"

        pcr_vol_str = f"{pcr_vol:.2f}" if pcr_vol != float("inf") else "N/A (Call Volume = 0)"

        # --- Support Levels (Top Put OI) ---
        puts_sorted = puts_df[puts_df["openInterest"].fillna(0) > 0].sort_values("openInterest", ascending=False).head(5)

        # --- Resistance Levels (Top Call OI) ---
        calls_sorted = calls_df[calls_df["openInterest"].fillna(0) > 0].sort_values("openInterest", ascending=False).head(5)

        # --- Max Pain Calculation ---
        all_strikes = sorted(set(calls_df["strike"].tolist() + puts_df["strike"].tolist()))
        call_oi_map = dict(zip(calls_df["strike"], calls_df["openInterest"].fillna(0)))
        put_oi_map = dict(zip(puts_df["strike"], puts_df["openInterest"].fillna(0)))

        max_pain_price = None
        min_total_pain = float("inf")
        for test_price in all_strikes:
            total_pain = 0
            # Pain for call holders: if test_price > strike, call is ITM, holder profits, writer loses
            for strike, oi in call_oi_map.items():
                if test_price > strike:
                    total_pain += (test_price - strike) * oi
            # Pain for put holders: if test_price < strike, put is ITM, holder profits, writer loses
            for strike, oi in put_oi_map.items():
                if test_price < strike:
                    total_pain += (strike - test_price) * oi
            if total_pain < min_total_pain:
                min_total_pain = total_pain
                max_pain_price = test_price

        # --- Build Result String ---
        result = f"# Options Chain Analysis for {ticker.upper()}\n"
        result += f"# Expiration Date: {nearest_exp}\n"
        result += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        result += "## Current Price Context\n"
        if current_price:
            result += f"- Current Price: ${current_price:.2f}\n"
        result += f"- Nearest Expiration: {nearest_exp} ({days_to_exp} calendar days away)\n"
        result += f"- Total Call Open Interest: {int(total_call_oi):,}\n"
        result += f"- Total Put Open Interest: {int(total_put_oi):,}\n\n"

        result += "## Put/Call Ratio\n"
        result += f"- PCR (Open Interest): {pcr_oi_str}\n"
        result += f"- PCR (Volume): {pcr_vol_str}\n"
        result += f"- Sentiment: {sentiment}\n\n"

        result += "## Support Levels (Top Put Open Interest)\n"
        if not puts_sorted.empty:
            result += "| Rank | Strike | Open Interest | Implied Volatility |\n"
            result += "|------|--------|---------------|--------------------|\n"
            for rank, (_, row) in enumerate(puts_sorted.iterrows(), 1):
                iv = f"{row['impliedVolatility']:.1%}" if pd.notna(row.get("impliedVolatility")) else "N/A"
                result += f"| {rank} | ${row['strike']:.2f} | {int(row['openInterest']):,} | {iv} |\n"
        else:
            result += "- No significant put open interest found.\n"
        result += "\n"

        result += "## Resistance Levels (Top Call Open Interest)\n"
        if not calls_sorted.empty:
            result += "| Rank | Strike | Open Interest | Implied Volatility |\n"
            result += "|------|--------|---------------|--------------------|\n"
            for rank, (_, row) in enumerate(calls_sorted.iterrows(), 1):
                iv = f"{row['impliedVolatility']:.1%}" if pd.notna(row.get("impliedVolatility")) else "N/A"
                result += f"| {rank} | ${row['strike']:.2f} | {int(row['openInterest']):,} | {iv} |\n"
        else:
            result += "- No significant call open interest found.\n"
        result += "\n"

        if max_pain_price is not None:
            result += f"## Max Pain Price: ${max_pain_price:.2f}\n"
            if current_price:
                diff_pct = ((max_pain_price - current_price) / current_price) * 100
                result += f"- Distance from current price: {diff_pct:+.2f}%\n"
            result += "- Max Pain is the price at which the most options expire worthless, often acting as a gravitational anchor near expiration.\n"

        return result

    except Exception as e:
        return f"Error retrieving options analysis for {ticker}: {str(e)}"


def get_volume_analysis(
    ticker: Annotated[str, "ticker symbol of the company"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Get comprehensive volume analysis for a given ticker symbol.

    Returns volume metrics, trend analysis, OBV trend, volume-price divergence,
    and up/down day distribution statistics.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        end_date = datetime.now()
        # Extra buffer for moving average warm-up and non-trading day gaps
        start_date = end_date - relativedelta(days=look_back_days + 40)

        hist = yf_retry(
            lambda: ticker_obj.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
        )

        if hist.empty:
            return f"No historical data found for symbol '{ticker}'"

        # Remove timezone info from index
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        close = hist["Close"]
        volume = hist["Volume"]
        n = len(hist)

        result = f"# Volume Analysis for {ticker.upper()}\n"
        result += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"# Look back period: {look_back_days} days\n"
        result += f"# Total trading days available: {n}\n\n"

        # --- Section 1: Daily Volume Metrics ---
        result += "## Daily Volume Metrics\n\n"

        latest_volume = int(volume.iloc[-1])
        avg_volume = int(volume.iloc[-look_back_days:].mean()) if n >= look_back_days else int(volume.mean())
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0

        if volume_ratio > 1.5:
            vol_classification = "显著放量"
        elif volume_ratio > 1.2:
            vol_classification = "温和放量"
        elif volume_ratio >= 0.8:
            vol_classification = "正常水平"
        elif volume_ratio >= 0.5:
            vol_classification = "缩量"
        else:
            vol_classification = "显著缩量"

        result += f"- Latest Volume: {latest_volume:,}\n"
        result += f"- {look_back_days}-Day Average Volume: {avg_volume:,}\n"
        result += f"- Volume Ratio: {volume_ratio:.2f}x ({vol_classification})\n\n"

        # Recent 5-day volume table
        recent_days = min(5, n)
        result += "| Date | Volume | vs Avg |\n"
        result += "|------|--------|--------|\n"
        for i in range(-recent_days, 0):
            date_str = hist.index[i].strftime("%Y-%m-%d")
            day_vol = int(volume.iloc[i])
            day_ratio = day_vol / avg_volume if avg_volume > 0 else 0
            result += f"| {date_str} | {day_vol:,} | {day_ratio:.2f}x |\n"
        result += "\n"

        # --- Section 2: Volume Moving Average Comparison ---
        result += "## Volume Moving Average Comparison\n\n"

        vol_5d_avg = int(volume.iloc[-5:].mean()) if n >= 5 else None
        vol_10d_avg = int(volume.iloc[-10:].mean()) if n >= 10 else None
        vol_20d_avg = int(volume.iloc[-20:].mean()) if n >= 20 else None

        if vol_5d_avg is not None:
            result += f"- 5-Day Avg Volume: {vol_5d_avg:,}\n"
        if vol_10d_avg is not None:
            result += f"- 10-Day Avg Volume: {vol_10d_avg:,}\n"
        if vol_20d_avg is not None:
            result += f"- 20-Day Avg Volume: {vol_20d_avg:,}\n"

        if vol_5d_avg is not None and vol_10d_avg is not None and vol_20d_avg is not None:
            if vol_5d_avg > vol_10d_avg > vol_20d_avg:
                vol_trend = "成交量递增趋势（放量）"
            elif vol_5d_avg < vol_10d_avg < vol_20d_avg:
                vol_trend = "成交量递减趋势（缩量）"
            else:
                vol_trend = "成交量震荡"
            result += f"- Volume Trend: {vol_trend}\n"
        result += "\n"

        # --- Section 3: Volume Expansion/Contraction Phase ---
        result += "## Volume Expansion/Contraction Phase\n\n"

        if vol_20d_avg is not None and n >= 5:
            expansion_days = 0
            contraction_days = 0
            result += "| Date | Volume | vs 20D Avg | Status |\n"
            result += "|------|--------|------------|--------|\n"
            for i in range(-min(5, n), 0):
                date_str = hist.index[i].strftime("%Y-%m-%d")
                day_vol = int(volume.iloc[i])
                ratio = day_vol / vol_20d_avg if vol_20d_avg > 0 else 0
                if day_vol > vol_20d_avg:
                    expansion_days += 1
                    status = "放量"
                else:
                    contraction_days += 1
                    status = "缩量"
                result += f"| {date_str} | {day_vol:,} | {ratio:.2f}x | {status} |\n"
            result += "\n"

            if expansion_days >= 4:
                phase = "放量阶段"
            elif contraction_days >= 4:
                phase = "缩量阶段"
            else:
                phase = "量能过渡阶段"
            result += f"- Recent 5-Day: {expansion_days} expansion / {contraction_days} contraction days\n"
            result += f"- Phase: {phase}\n"
        else:
            result += "- Insufficient data for phase analysis (need >= 20 trading days)\n"
        result += "\n"

        # --- Section 4: OBV Trend Analysis ---
        result += "## OBV Trend Analysis\n\n"

        if n >= 2:
            obv = [0] * n
            for i in range(1, n):
                if close.iloc[i] > close.iloc[i - 1]:
                    obv[i] = obv[i - 1] + int(volume.iloc[i])
                elif close.iloc[i] < close.iloc[i - 1]:
                    obv[i] = obv[i - 1] - int(volume.iloc[i])
                else:
                    obv[i] = obv[i - 1]

            obv_latest = obv[-1]

            # OBV 5-day trend
            if n >= 6:
                obv_5d_ago = obv[-6]
                obv_5d_change = obv_latest - obv_5d_ago
                obv_direction = "上升" if obv_5d_change > 0 else ("下降" if obv_5d_change < 0 else "持平")
                result += f"- OBV Current: {obv_latest:,}\n"
                result += f"- OBV 5-Day Change: {obv_5d_change:+,}\n"
                result += f"- OBV 5-Day Direction: {obv_direction}\n"

            # Price direction for cross-validation
            if n >= 6:
                price_5d_change = close.iloc[-1] - close.iloc[-6]
                price_direction = "上涨" if price_5d_change > 0 else ("下跌" if price_5d_change < 0 else "持平")

                result += f"- Price 5-Day Direction: {price_direction} ({price_5d_change:+.2f})\n"

                # Cross-validation
                if price_5d_change > 0 and obv_5d_change > 0:
                    confirmation = "价量齐升，上涨趋势确认"
                elif price_5d_change < 0 and obv_5d_change < 0:
                    confirmation = "价量齐跌，下跌趋势确认"
                elif price_5d_change > 0 and obv_5d_change <= 0:
                    confirmation = "价升量缩，上涨动能不足，警惕回调"
                elif price_5d_change < 0 and obv_5d_change >= 0:
                    confirmation = "价跌量增，可能有资金吸筹"
                else:
                    confirmation = "量价关系中性"
                result += f"- OBV Confirmation: {confirmation}\n"
        else:
            result += "- Insufficient data for OBV analysis\n"
        result += "\n"

        # --- Section 5: Volume-Price Divergence ---
        result += "## Volume-Price Divergence\n\n"

        if n >= 6 and vol_20d_avg is not None and vol_5d_avg is not None:
            price_5d_pct = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100
            vol_ratio_change = vol_5d_avg / vol_20d_avg if vol_20d_avg > 0 else 1

            result += f"- 5-Day Price Change: {price_5d_pct:+.2f}%\n"
            result += f"- 5D Avg Vol / 20D Avg Vol: {vol_ratio_change:.2f}x\n"

            if price_5d_pct < -1 and vol_ratio_change > 1.1:
                divergence = "看涨背离（底部放量）- 价格下跌但成交量增加，可能有资金进场"
            elif price_5d_pct > 1 and vol_ratio_change < 0.9:
                divergence = "看跌背离（顶部缩量）- 价格上涨但成交量萎缩，上涨动能可能衰竭"
            else:
                divergence = "无明显背离 - 量价方向一致或变化幅度不显著"
            result += f"- Divergence: {divergence}\n"
        else:
            result += "- Insufficient data for divergence analysis\n"
        result += "\n"

        # --- Section 6: Volume Distribution (Up/Down Days) ---
        result += "## Volume Distribution (Up/Down Days)\n\n"

        period = min(look_back_days, n - 1)
        if period >= 2:
            up_volume = 0
            down_volume = 0
            up_days = 0
            down_days = 0

            for i in range(-period, 0):
                if close.iloc[i] > close.iloc[i - 1]:
                    up_volume += int(volume.iloc[i])
                    up_days += 1
                elif close.iloc[i] < close.iloc[i - 1]:
                    down_volume += int(volume.iloc[i])
                    down_days += 1

            result += f"- Up Days: {up_days} (Total Volume: {up_volume:,})\n"
            result += f"- Down Days: {down_days} (Total Volume: {down_volume:,})\n"

            if up_days > 0:
                result += f"- Avg Volume on Up Days: {up_volume // up_days:,}\n"
            if down_days > 0:
                result += f"- Avg Volume on Down Days: {down_volume // down_days:,}\n"

            if down_volume > 0:
                up_down_ratio = up_volume / down_volume
                if up_down_ratio > 1.5:
                    dist_interpretation = "多头主导（上涨日成交量远大于下跌日）"
                elif up_down_ratio >= 1.0:
                    dist_interpretation = "略偏多头"
                elif up_down_ratio >= 0.67:
                    dist_interpretation = "略偏空头"
                else:
                    dist_interpretation = "空头主导（下跌日成交量远大于上涨日）"
                result += f"- Up/Down Volume Ratio: {up_down_ratio:.2f} ({dist_interpretation})\n"
            elif up_volume > 0:
                result += "- Up/Down Volume Ratio: N/A (no down-day volume)\n"
            else:
                result += "- Up/Down Volume Ratio: N/A (insufficient data)\n"
        else:
            result += "- Insufficient data for distribution analysis\n"
        result += "\n"

        # --- Summary ---
        result += "## Volume Analysis Summary\n\n"
        result += f"- 整体成交量状态: {vol_classification}（量比 {volume_ratio:.2f}x）\n"
        if vol_5d_avg is not None and vol_10d_avg is not None and vol_20d_avg is not None:
            result += f"- 均量趋势: {vol_trend}\n"
        if n >= 6:
            result += f"- OBV 趋势确认: {confirmation}\n"
        if n >= 6 and vol_20d_avg is not None and vol_5d_avg is not None:
            result += f"- 量价关系: {divergence}\n"

        return result

    except Exception as e:
        return f"Error retrieving volume analysis for {ticker}: {str(e)}"

import logging
import os
from datetime import date, datetime, timedelta, timezone
from math import isclose
from typing import Annotated
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from .stockstats_utils import StockstatsUtils, _clean_dataframe, completed_daily_cutoff, load_yfinance_data, market_timezone, yf_retry

logger = logging.getLogger(__name__)


def get_YFin_stock_info(symbol: Annotated[str, "ticker symbol of the company"]):
    info = yf_retry(lambda: yf.Ticker(symbol.upper()).info)
    # 移除分析师长期目标价与推荐评级，短期交易场景不应依赖这些数据
    for key in (
        "longBusinessSummary",
        "companyOfficers",
        "targetHighPrice",
        "targetLowPrice",
        "targetMeanPrice",
        "targetMedianPrice",
        "recommendationMean",
        "recommendationKey",
        "numberOfAnalystOpinions",
        "averageAnalystRating",
    ):
        info.pop(key, None)
    return info


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    exclusive_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = yf_retry(lambda: ticker.history(start=start_date, end=exclusive_end))
    if not data.empty:
        today = datetime.now(timezone.utc)
        if data.index.tz is not None:
            today_date = today.astimezone(data.index.tz).date()
        else:
            today_date = today.date() - timedelta(days=1)
        if max(data.index.date) >= today_date:
            try:
                metadata = ticker.history_metadata
            except Exception:
                logger.warning("Market schedule unavailable for %s", symbol)
                metadata = {}
            market_tz = market_timezone(metadata, data.index.tz)
            cutoff = completed_daily_cutoff(metadata, data.index.tz)
            if data.index.tz is not None:
                data = data.tz_convert(market_tz)
            data = data[data.index.date <= cutoff]
        data = data[(data.index.date >= datetime.strptime(start_date, "%Y-%m-%d").date()) & (data.index.date <= datetime.strptime(end_date, "%Y-%m-%d").date())]

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
                indicator_value = "N/A: No price data available for this date"

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
        data = load_yfinance_data(symbol, 15, config["data_cache_dir"])

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


def _format_session_return(close: pd.Series, sessions: int, label: str) -> str:
    if len(close) <= sessions:
        return f"- {label}: N/A (need {sessions + 1} completed closes, got {len(close)})\n"
    value = (close.iloc[-1] / close.iloc[-sessions - 1] - 1) * 100
    start = close.index[-sessions - 1].strftime("%Y-%m-%d")
    end = close.index[-1].strftime("%Y-%m-%d")
    return f"- {label}: {value:+.2f}% ({start} to {end}; {sessions} completed-session intervals)\n"


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

# yfinance classifies BABA as "Consumer Cyclical", but as a Chinese ADR it
# should not be compared against the US sector ETF (XLY).
IGNORE_SECTOR_TICKERS = {"BABA"}
MARKET_BENCHMARK = "SPY"


def get_sector_benchmark_symbol(ticker: str) -> str:
    """Return the ticker's sector ETF, or the broad-market fund when no sector mapping applies."""
    if ticker.upper() in IGNORE_SECTOR_TICKERS:
        return MARKET_BENCHMARK
    info = yf_retry(lambda: yf.Ticker(ticker.upper()).info)
    return SECTOR_ETFS.get(info.get("sector"), MARKET_BENCHMARK)


def _earnings_dates_frame(ticker: str, limit: int) -> pd.DataFrame | None:
    frame = yf_retry(lambda: yf.Ticker(ticker.upper()).get_earnings_dates(limit=limit))
    if frame is None or frame.empty:
        return None
    return frame


def _earnings_release_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = frame.index
    return index.tz_convert(ZoneInfo("America/New_York")) if index.tz is not None else index


def fetch_earnings_dates(ticker: str, start: str, end: str) -> list[datetime] | None:
    """Earnings release timestamps (New York time) in [start, end], or None when unavailable."""
    start_date = date.fromisoformat(start)
    quarters_back = max(0, (date.today() - start_date).days // 90) + 6
    try:
        frame = _earnings_dates_frame(ticker, limit=quarters_back)
    except Exception as exc:
        logger.warning("Earnings dates unavailable for %s: %s", ticker, exc)
        return None
    if frame is None:
        return None
    index = _earnings_release_index(frame)
    end_date = date.fromisoformat(end)
    if min(index.date) > start_date:
        return None
    return sorted({stamp.to_pydatetime() for stamp in index if start_date <= stamp.date() <= end_date})


def get_earnings_calendar(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """Report the next scheduled earnings release and the most recent ones."""
    current = date.fromisoformat(curr_date)
    frame = _earnings_dates_frame(ticker, limit=8)
    if frame is None:
        return f"No earnings calendar available for {ticker.upper()}"
    index = _earnings_release_index(frame)
    stamps = sorted(index)
    upcoming = [stamp for stamp in stamps if stamp.date() >= current]
    past = [stamp for stamp in stamps if stamp.date() < current][-4:]
    lines = [f"# Earnings calendar for {ticker.upper()} (as of {curr_date}, New York time)"]
    if upcoming:
        lines.append(f"- Next earnings release: {upcoming[0].strftime('%Y-%m-%d %H:%M %Z')}")
    else:
        lines.append("- Next earnings release: not announced")
    lines.extend(f"- Past release: {stamp.strftime('%Y-%m-%d %H:%M %Z')}" for stamp in reversed(past))
    return "\n".join(lines) + "\n"


def get_sector_performance(
    ticker: Annotated[str, "ticker symbol of the company"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
):
    """Get sector and industry performance data relative to the stock.

    Returns the stock's sector ETF performance compared to the stock itself,
    helping analyze whether the stock is outperforming or underperforming its sector.
    """
    import pandas as pd

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

        stock_close = stock_hist["Close"]
        periods = tuple(dict.fromkeys((1, 5, look_back_days)))
        for sessions in periods:
            result += _format_session_return(stock_close, sessions, f"Stock {sessions}-Day Return")

        # Get sector ETF performance if available
        sector_etf = SECTOR_ETFS.get(sector)
        if sector_etf:
            try:
                etf_obj = yf.Ticker(sector_etf)
                etf_hist = yf_retry(lambda: etf_obj.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")))

                if not etf_hist.empty:
                    etf_close = etf_hist["Close"]
                    result += f"\n## Sector ETF Performance ({sector_etf} - {sector})\n"

                    for sessions in periods:
                        result += _format_session_return(etf_close, sessions, f"Sector {sessions}-Day Return")
                        if len(stock_close) <= sessions or len(etf_close) <= sessions:
                            result += f"- Stock vs Sector ({sessions}D): N/A (insufficient completed closes)\n"
                            continue
                        if stock_close.index[-sessions - 1 :].date.tolist() != etf_close.index[-sessions - 1 :].date.tolist():
                            result += f"- Stock vs Sector ({sessions}D): N/A (session windows differ)\n"
                            continue
                        stock_return = (stock_close.iloc[-1] / stock_close.iloc[-sessions - 1] - 1) * 100
                        etf_return = (etf_close.iloc[-1] / etf_close.iloc[-sessions - 1] - 1) * 100
                        start = stock_close.index[-sessions - 1].strftime("%Y-%m-%d")
                        end = stock_close.index[-1].strftime("%Y-%m-%d")
                        result += f"- Stock vs Sector ({sessions}D): {stock_return - etf_return:+.2f}pp ({start} to {end})\n"
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

                for sessions in dict.fromkeys((1, 5, 10, look_back_days)):
                    result += _format_session_return(close, sessions, f"{sessions}-Day Change")

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

                    if len(close) >= 6:
                        vix_5d_change = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100
                        if vix_5d_change > 20:
                            result += f"- VIX Spike Detected: 5-day surge {vix_5d_change:+.2f}%, 市场恐慌情绪急剧上升\n"
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

                    if len(close) >= 6:
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
    """Analyze a future US options snapshot without imputing missing open interest."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        # Get available expiration dates
        expirations = yf_retry(lambda: ticker_obj.options)
        if not expirations:
            return f"No options data available for symbol '{ticker}'. Options data is typically only available for US-listed stocks."

        retrieved_at = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
        market_today = retrieved_at.date()
        future_expirations = sorted({exp for exp in expirations if datetime.strptime(exp, "%Y-%m-%d").date() > market_today})
        if not future_expirations:
            return f"No future options expiration available for symbol '{ticker}' after {market_today} (America/New_York)."

        def numeric_column(frame, column):
            if frame.empty or column not in frame:
                return None
            values = pd.to_numeric(frame[column], errors="coerce")
            if not values.notna().all() or not values.between(0, float("inf"), inclusive="left").all():
                return None
            if column in ("openInterest", "volume") and not (values % 1 == 0).all():
                return None
            return values

        skipped = [f"{exp}: expired or same-day expiry excluded by forward-window policy" for exp in sorted(set(expirations) - set(future_expirations))]
        selected = None
        fallback = None
        for expiration in future_expirations[:3]:
            opt = yf_retry(lambda exp=expiration: ticker_obj.option_chain(exp))
            calls, puts = opt.calls, opt.puts
            if calls.empty and puts.empty:
                skipped.append(f"{expiration}: empty chain")
                continue
            if fallback is None:
                fallback = (expiration, calls, puts)
            call_oi = numeric_column(calls, "openInterest")
            put_oi = numeric_column(puts, "openInterest")
            if call_oi is not None and put_oi is not None and call_oi.sum() + put_oi.sum() > 0:
                selected = (expiration, calls, puts)
                break
            reason = "combined OI = 0" if call_oi is not None and put_oi is not None else "incomplete OI"
            skipped.append(f"{expiration}: {reason}")

        selected = selected or fallback
        if selected is None:
            return f"Options chains are empty for symbol '{ticker}'.\n" + "\n".join(skipped)
        nearest_exp, calls_df, puts_df = selected

        # Get current stock price for context
        info = yf_retry(lambda: ticker_obj.info)
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        # Calculate days to expiration
        exp_date = datetime.strptime(nearest_exp, "%Y-%m-%d").date()
        days_to_exp = (exp_date - market_today).days

        call_oi = numeric_column(calls_df, "openInterest")
        put_oi = numeric_column(puts_df, "openInterest")
        call_volume = numeric_column(calls_df, "volume")
        put_volume = numeric_column(puts_df, "volume")
        total_call_oi = call_oi.sum() if call_oi is not None else None
        total_put_oi = put_oi.sum() if put_oi is not None else None
        oi_complete = total_call_oi is not None and total_put_oi is not None
        positive_oi = oi_complete and total_call_oi + total_put_oi > 0
        pcr_oi = total_put_oi / total_call_oi if oi_complete and total_call_oi > 0 else None

        if pcr_oi is None:
            pcr_oi_str = "N/A (Call OI = 0)" if oi_complete else "N/A (incomplete OI)"
            sentiment = "N/A (OI ratio unavailable)"
        else:
            pcr_oi_str = f"{pcr_oi:.2f}"
            if pcr_oi < 0.7:
                sentiment = "偏多情绪（看涨占主导，市场乐观）"
            elif pcr_oi <= 1.0:
                sentiment = "中性偏多（多空力量相对均衡，略偏看涨）"
            elif pcr_oi <= 1.5:
                sentiment = "中性偏空（看跌占优，市场偏谨慎，但也可能包含机构对冲需求）"
            else:
                sentiment = "偏空/防御情绪（大量看跌期权，可能包含机构对冲需求，需结合Put分布行权价判断是保护性建仓还是方向性看跌）"

        if call_volume is None or put_volume is None:
            pcr_vol_str = "N/A (incomplete volume)"
        elif call_volume.sum() == 0:
            pcr_vol_str = "N/A (Call Volume = 0)"
        else:
            pcr_vol_str = f"{put_volume.sum() / call_volume.sum():.2f}"

        call_strikes = numeric_column(calls_df, "strike")
        put_strikes = numeric_column(puts_df, "strike")
        max_pain_price = None
        max_pain_interval = None
        if positive_oi and call_strikes is not None and put_strikes is not None:
            all_strikes = sorted({0.0, *call_strikes[call_oi > 0].tolist(), *put_strikes[put_oi > 0].tolist()})
            payouts = [((price - call_strikes).clip(lower=0) * call_oi).sum() + ((put_strikes - price).clip(lower=0) * put_oi).sum() for price in all_strikes]
            minimum = min(payouts)
            minima = [price for price, payout in zip(all_strikes, payouts, strict=True) if isclose(payout, minimum, rel_tol=1e-12, abs_tol=1e-8)]
            if total_call_oi == 0:
                max_pain_interval = (minima[0], None)
            elif len(minima) > 1:
                max_pain_interval = (minima[0], minima[-1])
            else:
                max_pain_price = minima[0]

        # --- Build Result String ---
        result = f"# Options Chain Analysis for {ticker.upper()}\n"
        result += f"# Expiration Date: {nearest_exp}\n"
        result += f"# Data queried at: {retrieved_at.isoformat()} (America/New_York)\n\n"
        result += "- Forward-window reference: today's expiry is excluded; at most three future chains are checked.\n"
        result += "- OI update time: unknown (Yahoo supplies no verified OI cutoff here). Missing/zero OI does not establish absence of market positions.\n"
        result += "- Volume is a supplier snapshot, not verified complete current-session volume; OI is not real-time volume.\n"
        result += "- Completeness checks cover returned rows only; full contract-universe coverage is unverified.\n"
        result += "- This options snapshot and historical daily prices are not necessarily from the same instant.\n\n"
        if not positive_oi:
            result += "- Limited report: no complete positive OI chain found within the candidate limit; using the earliest nonempty future chain.\n\n"
        if skipped:
            result += "## Skipped / OI-rejected Expirations\n" + "\n".join(f"- {reason}" for reason in skipped) + "\n\n"

        result += "## Current Price Context\n"
        if current_price:
            result += f"- Current Price: ${current_price:.2f}\n"
        result += f"- Selected Expiration: {nearest_exp} ({days_to_exp} calendar days away)\n"
        call_oi_text = f"{int(total_call_oi):,}" if total_call_oi is not None else "N/A (missing or invalid OI)"
        put_oi_text = f"{int(total_put_oi):,}" if total_put_oi is not None else "N/A (missing or invalid OI)"
        result += f"- Total Call Open Interest: {call_oi_text}\n"
        result += f"- Total Put Open Interest: {put_oi_text}\n"
        result += f"- OI completeness: Call {'complete' if call_oi is not None else 'incomplete'}, Put {'complete' if put_oi is not None else 'incomplete'}\n\n"

        result += "## Put/Call Ratio\n"
        result += f"- PCR (Open Interest): {pcr_oi_str}\n"
        result += f"- PCR (Volume): {pcr_vol_str}\n"
        result += f"- Sentiment: {sentiment}\n\n"

        for title, frame, oi, strikes in (
            ("Support Levels (Top Put Open Interest)", puts_df, put_oi, put_strikes),
            ("Resistance Levels (Top Call Open Interest)", calls_df, call_oi, call_strikes),
        ):
            result += f"## {title}\n"
            if oi is None or strikes is None:
                result += "- N/A (incomplete OI or strikes).\n\n"
                continue
            if oi.sum() == 0:
                result += "- N/A (supplier reports zero OI).\n\n"
                continue
            levels = frame.assign(openInterest=oi, strike=strikes).loc[oi > 0].sort_values("openInterest", ascending=False).head(5)
            result += "| Rank | Strike | Open Interest | Implied Volatility |\n"
            result += "|------|--------|---------------|--------------------|\n"
            for rank, (_, row) in enumerate(levels.iterrows(), 1):
                iv = f"{row['impliedVolatility']:.1%}" if pd.notna(row.get("impliedVolatility")) else "N/A"
                result += f"| {rank} | ${row['strike']:.2f} | {int(row['openInterest']):,} | {iv} |\n"
            result += "- OI concentrations are reference levels, not guaranteed support/resistance.\n\n"

        if max_pain_price is not None:
            result += f"## Max Pain Price: ${max_pain_price:.2f}\n"
            if current_price:
                diff_pct = ((max_pain_price - current_price) / current_price) * 100
                result += f"- Distance from current price: {diff_pct:+.2f}%\n"
            result += "- Max Pain minimizes aggregate intrinsic option payout weighted by reported OI. It serves as a reference anchor for near-term options activity — prices may show increased stickiness near this level approaching expiration, but it is not a high-probability price predictor.\n"
        elif max_pain_interval is not None:
            low, high = max_pain_interval
            upper = f"${high:.2f}" if high is not None else "unbounded"
            result += "## Max Pain Price: N/A (non-unique minimum)\n"
            result += f"- Max Pain Minimum Interval: ${low:.2f}–{upper}\n"
            result += "- OI-weighted intrinsic payout has a flat minimum throughout this interval; no unique price or directional target is implied. It is not a high-probability price predictor.\n"
        else:
            result += "## Max Pain Price: N/A (incomplete OI/strikes or combined OI = 0)\n"

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
                result += f"- OBV 5-Day Change: {obv_5d_change:+,} ({hist.index[-6]:%Y-%m-%d} to {hist.index[-1]:%Y-%m-%d}; 5 completed-session intervals)\n"
                result += f"- OBV 5-Day Direction: {obv_direction}\n"

            # Price direction for cross-validation
            if n >= 6:
                price_5d_change = close.iloc[-1] - close.iloc[-6]
                price_direction = "上涨" if price_5d_change > 0 else ("下跌" if price_5d_change < 0 else "持平")

                result += f"- Price 5-Day Direction: {price_direction} ({price_5d_change:+.2f}; {hist.index[-6]:%Y-%m-%d} to {hist.index[-1]:%Y-%m-%d})\n"

                # Cross-validation
                if price_5d_change > 0 and obv_5d_change > 0:
                    confirmation = "价量齐升，上涨趋势得到量能确认"
                elif price_5d_change < 0 and obv_5d_change < 0:
                    confirmation = "价量齐跌，下跌趋势得到量能确认"
                elif price_5d_change > 0 and obv_5d_change <= 0:
                    confirmation = "价格微涨但OBV走弱，轻度顶背离，量价配合一般，短线增量买盘持续性不强"
                elif price_5d_change < 0 and obv_5d_change >= 0:
                    confirmation = "价格走弱但OBV未同步下降，轻度底背离，可能有逢低买盘介入"
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

            result += _format_session_return(close, 5, "5-Day Price Change")
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

        # --- Section 7: Previous Trading Day Volume Impact ---
        result += "## Previous Trading Day Volume Impact Analysis\n\n"

        if n >= 3 and avg_volume > 0:
            prev_volume = int(volume.iloc[-2])
            prev_close = close.iloc[-2]
            prev_open = hist["Open"].iloc[-2]
            prev_high = hist["High"].iloc[-2]
            prev_low = hist["Low"].iloc[-2]
            prev_vol_ratio = prev_volume / avg_volume

            # Previous day price change
            prev_prev_close = close.iloc[-3]
            prev_day_return = ((prev_close / prev_prev_close) - 1) * 100
            prev_day_direction = "上涨" if prev_day_return > 0 else ("下跌" if prev_day_return < 0 else "持平")

            # Previous day volume classification
            if prev_vol_ratio > 1.5:
                prev_vol_class = "显著放量"
            elif prev_vol_ratio > 1.2:
                prev_vol_class = "温和放量"
            elif prev_vol_ratio >= 0.8:
                prev_vol_class = "正常水平"
            elif prev_vol_ratio >= 0.5:
                prev_vol_class = "缩量"
            else:
                prev_vol_class = "显著缩量"

            # Current day (latest) metrics
            curr_close = close.iloc[-1]
            curr_day_return = ((curr_close / prev_close) - 1) * 100
            curr_day_direction = "上涨" if curr_day_return > 0 else ("下跌" if curr_day_return < 0 else "持平")

            result += f"- Previous Trading Day Date: {hist.index[-2].strftime('%Y-%m-%d')}\n"
            result += f"- Previous Day Volume: {prev_volume:,}\n"
            result += f"- Previous Day Volume Ratio: {prev_vol_ratio:.2f}x ({prev_vol_class})\n"
            result += f"- Previous Day Price Change: {prev_day_return:+.2f}% ({prev_day_direction})\n"
            result += f"- Previous Day OHLC: O={prev_open:.2f} / H={prev_high:.2f} / L={prev_low:.2f} / C={prev_close:.2f}\n"
            result += f"- Current Day Price Change: {curr_day_return:+.2f}% ({curr_day_direction})\n\n"

            # Volume-to-price impact pattern
            result += "### Previous Day Volume-Price Impact Pattern\n\n"

            if prev_vol_ratio > 1.5 and prev_day_return > 1:
                pattern = "前日放量大涨"
                impact = "短期存在惯性上冲动能，但也可能因获利盘涌出而出现冲高回落；需关注当日能否继续放量维持涨势"
            elif prev_vol_ratio > 1.5 and prev_day_return < -1:
                pattern = "前日放量大跌"
                impact = "短期抛压较重，可能延续调整；若当日缩量企稳则恐慌情绪消退，反弹概率增加"
            elif prev_vol_ratio > 1.5 and abs(prev_day_return) <= 1:
                pattern = "前日放量滞涨/滞跌"
                impact = "大量换手但价格变动不大，多空博弈激烈；通常预示短期方向选择临近，需警惕突破方向"
            elif prev_vol_ratio > 1.2 and prev_day_return > 0:
                pattern = "前日温和放量上涨"
                impact = "量价配合健康，短期上行动能较为扎实；若当日能维持量能则趋势延续性较好"
            elif prev_vol_ratio > 1.2 and prev_day_return < 0:
                pattern = "前日温和放量下跌"
                impact = "下行有一定量能支撑，短期偏空；关注是否在关键支撑位附近出现缩量企稳信号"
            elif prev_vol_ratio < 0.8 and prev_day_return > 0:
                pattern = "前日缩量上涨"
                impact = "上涨缺乏量能确认，持续性存疑；若当日继续缩量则警惕回调，放量则可能确认突破"
            elif prev_vol_ratio < 0.8 and prev_day_return < 0:
                pattern = "前日缩量下跌"
                impact = "下跌动能不足，恐慌性抛售有限；若当日放量反弹则可能是短期底部信号"
            else:
                pattern = "前日量价平稳"
                impact = "无明显量能异动，市场观望情绪为主；短期走势更依赖其他催化因素"

            result += f"- Pattern: {pattern}\n"
            result += f"- Impact Assessment: {impact}\n\n"

            # Volume continuity check: previous vs current day
            curr_volume = int(volume.iloc[-1])
            vol_continuity = curr_volume / prev_volume if prev_volume > 0 else 0

            result += "### Volume Continuity (Previous → Current Day)\n\n"
            result += f"- Previous Day Volume: {prev_volume:,}\n"
            result += f"- Current Day Volume: {curr_volume:,}\n"
            result += f"- Continuity Ratio: {vol_continuity:.2f}x\n"

            if vol_continuity > 1.2:
                continuity_signal = "量能放大延续，趋势动能增强"
            elif vol_continuity >= 0.8:
                continuity_signal = "量能基本持平，市场参与度稳定"
            elif vol_continuity >= 0.5:
                continuity_signal = "量能萎缩，参与度下降，趋势持续性减弱"
            else:
                continuity_signal = "量能大幅萎缩，市场观望情绪明显加重"

            result += f"- Continuity Signal: {continuity_signal}\n"

            # Direction continuity
            if prev_day_return > 0 and curr_day_return > 0 and vol_continuity > 1:
                result += "- Direction Signal: 连续上涨且量能放大，多头动能强劲\n"
            elif prev_day_return > 0 and curr_day_return > 0 and vol_continuity < 0.8:
                result += "- Direction Signal: 连续上涨但量能萎缩，上涨持续性存疑\n"
            elif prev_day_return < 0 and curr_day_return < 0 and vol_continuity > 1:
                result += "- Direction Signal: 连续下跌且量能放大，空头压力加剧\n"
            elif prev_day_return < 0 and curr_day_return < 0 and vol_continuity < 0.8:
                result += "- Direction Signal: 连续下跌但量能萎缩，抛压或在减弱\n"
            elif prev_day_return * curr_day_return < 0:
                result += f"- Direction Signal: 价格方向反转（前日{prev_day_direction} → 当日{curr_day_direction}），关注反转量能是否充分\n"
            else:
                result += "- Direction Signal: 价格变动不大，方向不明朗\n"
        else:
            result += "- Insufficient data for previous day volume impact analysis (need >= 3 trading days)\n"
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
        if n >= 3 and avg_volume > 0:
            result += f"- 前一交易日量能影响: {pattern}，{continuity_signal}\n"

        return result

    except Exception as e:
        return f"Error retrieving volume analysis for {ticker}: {str(e)}"

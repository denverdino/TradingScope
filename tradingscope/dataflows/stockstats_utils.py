import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yfinance as yf
from stockstats import wrap
from yfinance.exceptions import YFRateLimitError

from .config import DATA_DIR, get_config

logger = logging.getLogger(__name__)

_REFRESHED_PRICE_FILES: set[str] = set()
_US_MARKET_TZ = ZoneInfo("America/New_York")


def market_timezone(metadata: dict | None = None, index_timezone=None, *, us_market: bool = False) -> ZoneInfo:
    for name in ((metadata or {}).get("exchangeTimezoneName"), index_timezone):
        if name is not None:
            try:
                return ZoneInfo(str(name))
            except (ValueError, ZoneInfoNotFoundError):
                logger.warning("Invalid exchange timezone: %s", name)
    if us_market:
        return _US_MARKET_TZ
    raise RuntimeError("Cannot determine exchange timezone for daily prices")


def completed_daily_cutoff(metadata: dict | None = None, index_timezone=None, *, us_market: bool = False) -> date:
    market_tz = market_timezone(metadata, index_timezone, us_market=us_market)
    now = datetime.now(timezone.utc).astimezone(market_tz)
    regular_end = (metadata or {}).get("currentTradingPeriod", {}).get("regular", {}).get("end")
    if regular_end is not None:
        try:
            close = datetime.fromtimestamp(regular_end, market_tz)
            if close.date() == now.date():
                return now.date() if now >= close else now.date() - timedelta(days=1)
        except (TypeError, ValueError, OverflowError, OSError):
            logger.warning("Invalid regular session end: %s", regular_end)
    if market_tz == _US_MARKET_TZ:
        logger.warning("Today's market schedule unavailable; using conservative 16:00 New York close (early closes unknown)")
        close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return now.date() if now >= close else now.date() - timedelta(days=1)
    logger.warning("Cannot confirm today's close for %s; excluding today's daily bar", market_tz)
    return now.date() - timedelta(days=1)


@lru_cache(maxsize=32)
def _market_metadata(symbol: str, market_date: date) -> dict:
    try:
        return dict(yf.Ticker(symbol).history_metadata or {})
    except Exception:
        logger.warning("Market schedule unavailable for %s", symbol)
        return {}


def load_yfinance_data(symbol: str, years: int, cache_dir: str) -> pd.DataFrame:
    metadata = _market_metadata(symbol, datetime.now(timezone.utc).date())
    data = None
    if not metadata.get("exchangeTimezoneName"):
        # A superset download supplies the exchange timezone when metadata is unavailable.
        end = pd.Timestamp(datetime.now(timezone.utc).date()) + pd.Timedelta(days=2)
        start = end - pd.DateOffset(years=years)
        data = yf_retry(
            lambda: yf.download(
                symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), multi_level_index=False, progress=False, auto_adjust=True
            )
        )
        if data.empty:
            raise RuntimeError(f"No price data returned for {symbol}")
        metadata["exchangeTimezoneName"] = str(market_timezone(metadata, data.index.tz))
    cutoff = completed_daily_cutoff(metadata)
    exclusive_end = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
    start_date = (exclusive_end - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    end_date = exclusive_end.strftime("%Y-%m-%d")
    data_file = os.path.abspath(os.path.join(cache_dir, f"{symbol}-YFin-data-{start_date}-{end_date}.csv"))
    os.makedirs(cache_dir, exist_ok=True)
    if data_file in _REFRESHED_PRICE_FILES and os.path.exists(data_file):
        return pd.read_csv(data_file, on_bad_lines="skip")

    if data is None:
        data = yf_retry(lambda: yf.download(symbol, start=start_date, end=end_date, multi_level_index=False, progress=False, auto_adjust=True))
    if data.empty:
        raise RuntimeError(f"No price data returned for {symbol}")
    if data.index.tz is not None:
        data = data.tz_convert(market_timezone(metadata)).tz_localize(None)
    data = _clean_dataframe(data.reset_index())
    data = data[(data["Date"].dt.date >= pd.Timestamp(start_date).date()) & (data["Date"].dt.date <= cutoff)]
    if data.empty:
        raise RuntimeError(f"No price data returned for {symbol}")
    data.to_csv(data_file, index=False)
    _REFRESHED_PRICE_FILES.add(data_file)
    return data


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning("Yahoo Finance rate limited, retrying in %.0fs (attempt %d/%d)", delay, attempt + 1, max_retries)
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[str, "quantitative indicators based off of the stock data for the company"],
        curr_date: Annotated[str, "curr date for retrieving stock price data, YYYY-mm-dd"],
    ):
        # Get config and set up data directory path
        config = get_config()
        online = config["data_vendors"]["technical_indicators"] != "local"
        logger.debug("Data cache dir: %s", config.get("data_cache_dir"))
        logger.debug("Online mode: %s", online)
        df = None
        data = None

        if not online:
            try:
                data = pd.read_csv(
                    os.path.join(
                        DATA_DIR,
                        f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                    ),
                    on_bad_lines="skip",
                )
                data = _clean_dataframe(data)
                df = wrap(data)
            except FileNotFoundError:
                raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
        else:
            curr_date = pd.to_datetime(curr_date)
            data = load_yfinance_data(symbol, 15, config["data_cache_dir"])

            data = _clean_dataframe(data)
            df = wrap(data)
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            curr_date = curr_date.strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: No price data available for this date"

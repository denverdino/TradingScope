import logging
import os
from typing import Annotated

import pandas as pd
import yfinance as yf
from stockstats import wrap

from tradingscope.dataflows.interface import route_to_vendor
from tradingscope.dataflows.y_finance import _clean_dataframe, yf_retry

from .tool_response_helper import agentscope_tool

logger = logging.getLogger(__name__)

MARKET_ANALYST_INDICATORS = (
    "close_10_ema",
    "rsi",
    "macd",
    "macds",
    "macdh",
    "atr",
    "boll",
    "boll_ub",
    "boll_lb",
)


@agentscope_tool
def get_weekly_bollinger_signal(
    symbol: Annotated[str, "ticker symbol of the company, e.g. AAPL, TSM"],
    curr_date: Annotated[str, "The current trading date, YYYY-mm-dd"],
) -> str:
    """Check weekly Bollinger Band buy/sell signals for a given stock.

    Resamples daily OHLCV data to weekly, computes 20-week Bollinger Bands,
    and checks if the current price is near the weekly upper or lower band
    (within 0.5% deviation).

    Buy signal:  price near weekly lower band (deviation < 0.5%)
    Sell signal: price near weekly upper band (deviation < 0.5%)

    Args:
        symbol: Ticker symbol of the company, e.g. AAPL, TSM
        curr_date: The current trading date, YYYY-mm-dd

    Returns:
        str: Weekly Bollinger Band signal analysis.
    """
    from tradingscope.dataflows.config import get_config

    config = get_config()

    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=2)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = today_date.strftime("%Y-%m-%d")

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

    curr_date_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_date_dt]

    if len(data) < 100:
        return f"Insufficient data for weekly Bollinger Band analysis (need >= 100 daily bars, got {len(data)})"

    data = data.set_index("Date")
    weekly = (
        data.resample("W")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Close"])
    )

    weekly = weekly.reset_index()
    wdf = wrap(weekly.rename(columns={"Date": "date"}))

    wdf["boll"]
    wdf["boll_ub"]
    wdf["boll_lb"]

    latest = wdf.iloc[-1]
    prev = wdf.iloc[-2] if len(wdf) >= 2 else None

    close = latest["close"]
    boll_mid = latest["boll"]
    boll_ub = latest["boll_ub"]
    boll_lb = latest["boll_lb"]

    if pd.isna(boll_mid) or pd.isna(boll_ub) or pd.isna(boll_lb):
        return "Weekly Bollinger Band data contains N/A values (insufficient weekly bars for 20-week calculation)"

    dev_upper = (close - boll_ub) / boll_ub * 100
    dev_lower = (close - boll_lb) / boll_lb * 100
    dev_mid = (close - boll_mid) / boll_mid * 100

    bandwidth = (boll_ub - boll_lb) / boll_mid * 100

    signal = "无信号"
    if abs(dev_lower) < 0.5:
        signal = "买入信号：现价接近周线布林下轨（偏差 < 0.5%）"
    elif abs(dev_upper) < 0.5:
        signal = "卖出信号：现价接近周线布林上轨（偏差 < 0.5%）"
    elif close < boll_lb:
        signal = "买入参考：现价已跌破周线布林下轨"
    elif close > boll_ub:
        signal = "卖出参考：现价已突破周线布林上轨"

    result = f"# 周线布林带信号分析 - {symbol.upper()}\n\n"
    result += "## 周线布林带数值\n"
    result += f"- 现价（周收盘）: {close:.2f}\n"
    result += f"- 周线布林上轨: {boll_ub:.2f}\n"
    result += f"- 周线布林中轨: {boll_mid:.2f}\n"
    result += f"- 周线布林下轨: {boll_lb:.2f}\n"
    result += f"- 带宽: {bandwidth:.2f}%\n\n"

    result += "## 偏差分析\n"
    result += f"- 现价 vs 上轨偏差: {dev_upper:+.2f}%\n"
    result += f"- 现价 vs 中轨偏差: {dev_mid:+.2f}%\n"
    result += f"- 现价 vs 下轨偏差: {dev_lower:+.2f}%\n\n"

    result += "## 信号判定\n"
    result += f"- **{signal}**\n"
    result += "- 判定标准: 现价距周线上/下轨偏差 < 0.5% 时触发信号\n"

    if prev is not None and not pd.isna(prev["boll_ub"]):
        prev_bw = (prev["boll_ub"] - prev["boll_lb"]) / prev["boll"] * 100
        if bandwidth < prev_bw:
            result += "- 带宽变化: 收口中（波动收缩，可能酝酿突破）\n"
        else:
            result += "- 带宽变化: 扩口中（波动扩大）\n"

    return result


@agentscope_tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Retrieve the short-term technical indicators used by MarketAnalyst.

    Uses the configured technical_indicators vendor.

    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30

    Returns:
        str: Formatted data for every short-term indicator used by MarketAnalyst.
    """
    results = []
    success_count = 0

    for indicator in MARKET_ANALYST_INDICATORS:
        try:
            result = route_to_vendor("get_indicators", symbol, indicator, curr_date, look_back_days)
        except Exception as exc:
            logger.warning("Failed to retrieve %s for %s: %s", indicator, symbol, exc)
            results.append(f"## {indicator} unavailable\n\nError: {exc}")
        else:
            results.append(result)
            success_count += 1

    if success_count == 0:
        raise RuntimeError("Failed to retrieve all market indicators")

    return "\n\n".join(results)

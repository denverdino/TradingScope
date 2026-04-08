import json
import logging
from datetime import datetime, timedelta

from .alpha_vantage_common import _make_api_request, format_datetime_for_api

logger = logging.getLogger(__name__)


def _format_news_with_citations(response_text: str, ticker: str, start_date: str, end_date: str) -> str:
    """Format Alpha Vantage news response with inline citation links.

    Args:
        response_text: Raw API response text (JSON)
        ticker: Stock ticker symbol
        start_date: Start date of the query
        end_date: End date of the query

    Returns:
        Formatted news string with inline links
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse news response as JSON")
        return response_text

    feed = data.get("feed", [])
    if not feed:
        return f"## {ticker} 新闻 ({start_date} 至 {end_date}):\n\n未找到相关新闻。"

    news_str = f"## {ticker} 新闻 ({start_date} 至 {end_date}):\n\n"

    for article in feed:
        title = article.get("title", "无标题")
        url = article.get("url", "")
        source = article.get("source", "未知来源")
        summary = article.get("summary", "")
        time_published = article.get("time_published", "")
        sentiment_label = article.get("overall_sentiment_label", "")
        sentiment_score = article.get("overall_sentiment_score", "")

        # Format publication time
        if time_published:
            try:
                # Alpha Vantage format: 20240101T120000
                dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                formatted_time = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                formatted_time = time_published
        else:
            formatted_time = "未知时间"

        # Build article entry with inline link
        if url:
            news_str += f"### [{title}]({url})\n"
        else:
            news_str += f"### {title}\n"

        news_str += f"**来源:** {source} | **时间:** {formatted_time}"
        if sentiment_label:
            news_str += f" | **情绪:** {sentiment_label} ({sentiment_score})"
        news_str += "\n\n"

        if summary:
            news_str += f"{summary}\n\n"

        news_str += "---\n\n"

    return news_str


def get_news(ticker, start_date, end_date) -> str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Formatted news string with inline citation links.
    """

    params = {
        "tickers": ticker,
        "sort": "LATEST",
        "limit": "50",
    }

    today = datetime.now().strftime("%Y-%m-%d")

    if start_date == today:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        params["time_from"] = format_datetime_for_api(yesterday)
    else:
        params["time_from"] = format_datetime_for_api(start_date)

    if end_date != today:
        params["time_to"] = format_datetime_for_api(end_date)

    response_text = _make_api_request("NEWS_SENTIMENT", params)
    return _format_news_with_citations(response_text, ticker, start_date, end_date)


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)

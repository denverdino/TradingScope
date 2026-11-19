"""Finnhub company news with original sources and publication times."""

import os
from datetime import date, datetime, timezone
from html import escape
from urllib.parse import urlsplit

import requests


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    if not ticker.strip():
        raise ValueError("Ticker must not be empty")
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start.isoformat() != start_date or end.isoformat() != end_date:
        raise ValueError("Dates must use yyyy-mm-dd format")
    if start > end:
        raise ValueError("start_date must not be after end_date")
    api_key = os.getenv("FINHUB_API_KEY")
    if not api_key:
        raise ValueError("FINHUB_API_KEY environment variable is not set")

    response = requests.get(
        "https://api.finnhub.io/api/v1/company-news",
        params={"symbol": ticker, "from": start_date, "to": end_date},
        headers={"X-Finnhub-Token": api_key},
        timeout=30,
    )
    response.raise_for_status()
    articles = response.json()
    if not isinstance(articles, list):
        raise RuntimeError("Finnhub company news returned a non-news response")
    for article in articles:
        if not isinstance(article, dict) or any(not isinstance(article.get(field), str) for field in ("headline", "source", "url", "summary")):
            raise RuntimeError("Finnhub company news returned an invalid article")
        timestamp = article.get("datetime")
        if type(timestamp) is not int or timestamp <= 0 or not article["headline"].strip():
            raise RuntimeError("Finnhub company news returned an invalid article")
        try:
            datetime.fromtimestamp(timestamp, timezone.utc)
            url = urlsplit(article["url"])
        except (OverflowError, OSError, ValueError) as exc:
            raise RuntimeError("Finnhub company news returned an invalid article") from exc
        if url.scheme not in ("http", "https") or not url.netloc:
            raise RuntimeError("Finnhub company news returned an invalid article URL")
    articles = [article for article in articles if start <= datetime.fromtimestamp(article["datetime"], timezone.utc).date() <= end]
    articles = sorted(articles, key=lambda article: article["datetime"], reverse=True)[:50]
    sections = [f"## {ticker} 新闻 ({start_date} 至 {end_date})\n\n来源：Finnhub。公司新闻接口不提供情绪评分。"]
    if not articles:
        sections.append("未找到相关新闻。")
    for article in articles:
        published_at = datetime.fromtimestamp(article["datetime"], timezone.utc)
        sections.append(
            f"### [{escape(article['headline'])}]({escape(article['url'])})\n"
            f"**来源:** {escape(article['source'])} | **时间:** {published_at:%Y-%m-%d %H:%M:%S} UTC\n\n"
            f"{escape(article['summary'])}"
        )
    return "\n\n".join(sections)

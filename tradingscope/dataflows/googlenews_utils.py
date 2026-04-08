import logging
import random
import re
import time
from datetime import datetime
from urllib.parse import quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Google News RSS base URL
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


def is_rate_limited(response):
    """Check if the response indicates rate limiting (status code 429)"""
    return response.status_code == 429


@retry(
    retry=(retry_if_result(is_rate_limited)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_request(url, headers):
    """Make a request with retry logic for rate limiting"""
    # Random delay before each request to avoid detection
    time.sleep(random.uniform(1, 3))
    response = requests.get(url, headers=headers)
    return response


def _extract_original_url(google_news_url: str) -> str:
    """Extract original article URL from Google News redirect URL.

    Google News URLs are in format:
    https://news.google.com/rss/articles/xxx?oc=5&hl=en-US&gl=US&ceid=US:en

    We need to follow the redirect to get the actual article URL.
    """
    if not google_news_url:
        return google_news_url

    # If it's already a direct URL (not a Google News redirect), return as-is
    if "news.google.com" not in google_news_url:
        return google_news_url

    try:
        headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")}
        # Follow redirects to get the final URL
        response = requests.head(google_news_url, headers=headers, allow_redirects=True, timeout=10)
        return response.url
    except Exception as e:
        logger.warning("Failed to extract original URL from %s: %s", google_news_url, e)
        return google_news_url


def _parse_date_filter(start_date: str, end_date: str) -> str:
    """Build date filter query parameter for Google News.

    Google News supports 'when:Xd' for last X days or 'after:YYYY-MM-DD before:YYYY-MM-DD' format.
    """
    # Calculate days difference for the 'when' parameter
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_diff = (end_dt - start_dt).days + 1

        # Use 'when:Xd' format for recent news (up to 30 days)
        if days_diff <= 30:
            return f"when:{days_diff}d"
        else:
            # For longer periods, use after/before format
            return f"after:{start_date} before:{end_date}"
    except ValueError:
        return ""


def getNewsData(query, start_date, end_date):
    """
    Fetch Google News search results using RSS feed from news.google.com.

    Args:
        query: str - search query
        start_date: str - start date in the format yyyy-mm-dd or mm/dd/yyyy
        end_date: str - end date in the format yyyy-mm-dd or mm/dd/yyyy

    Returns:
        list: List of news articles with title, link, snippet, date, source
    """
    # Normalize date format to yyyy-mm-dd
    if "/" in start_date:
        start_date = datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    if "/" in end_date:
        end_date = datetime.strptime(end_date, "%m/%d/%Y").strftime("%Y-%m-%d")

    # Build the RSS URL with query and date filter
    date_filter = _parse_date_filter(start_date, end_date)
    search_query = f"{query} {date_filter}".strip()
    encoded_query = quote_plus(search_query)

    rss_url = f"{GOOGLE_NEWS_RSS_URL}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    logger.debug("Fetching Google News RSS: %s", rss_url)

    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")}

    news_results = []

    try:
        response = make_request(rss_url, headers)

        if response.status_code != 200:
            logger.error("Failed to fetch Google News RSS: %s", response.status_code)
            return news_results

        # Parse RSS feed
        feed = feedparser.parse(response.content)

        for entry in feed.entries:
            try:
                title = entry.get("title", "")
                google_link = entry.get("link", "")
                published = entry.get("published", "")
                summary = entry.get("summary", "")

                # Extract source from title (format: "Title - Source")
                source = ""
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    if len(parts) == 2:
                        title = parts[0]
                        source = parts[1]

                # Clean HTML from summary
                if summary:
                    soup = BeautifulSoup(summary, "html.parser")
                    summary = soup.get_text(separator=" ").strip()

                # Extract original article URL (follow Google News redirect)
                original_link = _extract_original_url(google_link)

                # Parse publication date
                date_str = ""
                if published:
                    try:
                        # feedparser provides time.struct_time in published_parsed
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            dt = datetime(*entry.published_parsed[:6])
                            date_str = dt.strftime("%Y-%m-%d %H:%M")
                        else:
                            date_str = published
                    except Exception:
                        date_str = published

                news_results.append(
                    {
                        "link": original_link,
                        "title": title,
                        "snippet": summary,
                        "date": date_str,
                        "source": source,
                    }
                )

            except Exception as e:
                logger.warning("Error processing RSS entry: %s", e)
                continue

        logger.info("Fetched %d news articles from Google News", len(news_results))

    except Exception as e:
        logger.error("Failed to fetch Google News: %s", e)

    return news_results

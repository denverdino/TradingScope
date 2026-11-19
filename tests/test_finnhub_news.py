"""Tests for Finnhub company news through the provider and routed tool."""

from datetime import datetime
from unittest.mock import Mock

import pytest
import requests


def _article(headline="Company update", published_at="2026-11-17T18:00:00+00:00"):
    return {
        "headline": headline,
        "datetime": int(datetime.fromisoformat(published_at).timestamp()),
        "source": "Example News",
        "url": "https://example.org/news/company-update",
        "summary": "The company announced a new product.",
        "id": 123,
        "related": "AAPL",
        "category": "company",
        "image": "",
    }


@pytest.fixture
def news_request(monkeypatch):
    monkeypatch.setenv("FINHUB_API_KEY", "test-finnhub-key")
    response = Mock()
    response.json.return_value = [_article()]
    request = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", request)
    return request


def test_company_news_preserves_sources_dates_links_and_uses_header_auth(news_request):
    from tradingscope.dataflows import finnhub_news

    text = finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")

    news_request.assert_called_once_with(
        "https://api.finnhub.io/api/v1/company-news",
        params={"symbol": "AAPL", "from": "2026-11-10", "to": "2026-11-18"},
        headers={"X-Finnhub-Token": "test-finnhub-key"},
        timeout=30,
    )
    news_request.return_value.raise_for_status.assert_called_once_with()
    assert "AAPL" in text
    assert "2026-11-10" in text and "2026-11-18" in text
    assert "[Company update](https://example.org/news/company-update)" in text
    assert "Example News" in text
    assert "2026-11-17 18:00:00 UTC" in text
    assert "The company announced a new product." in text
    assert "Finnhub" in text
    assert "不提供情绪评分" in text


def test_company_news_filters_inclusive_utc_dates_and_limits_latest_articles(news_request):
    from tradingscope.dataflows import finnhub_news

    news_request.return_value.json.return_value = [
        _article("Outside before", "2026-11-09T23:59:59+00:00"),
        _article("Start boundary", "2026-11-10T00:00:00+00:00"),
        _article("End boundary", "2026-11-18T23:59:59+00:00"),
        _article("Outside after", "2026-11-19T00:00:00+00:00"),
    ]

    text = finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")

    assert "Outside" not in text
    assert text.index("End boundary") < text.index("Start boundary")

    articles = [_article(f"Article {index:02d}", f"2026-11-17T18:00:{index:02d}+00:00") for index in range(55)]
    news_request.return_value.json.return_value = articles
    text = finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")

    assert text.count("### [") == 50
    assert "Article 54" in text and "Article 05" in text
    assert "Article 04" not in text
    assert text.index("Article 54") < text.index("Article 05")
    assert articles[0]["headline"] == "Article 00"


@pytest.mark.parametrize("articles", [[], [_article(published_at="2026-11-09T23:59:59+00:00")]])
def test_company_news_reports_valid_empty_results(news_request, articles):
    from tradingscope.dataflows import finnhub_news

    news_request.return_value.json.return_value = articles

    text = finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")

    assert "未找到相关新闻" in text
    assert "Finnhub" in text


def test_company_news_requires_the_configured_key_before_network_access(news_request, monkeypatch):
    from tradingscope.dataflows import finnhub_news

    monkeypatch.delenv("FINHUB_API_KEY")

    with pytest.raises(ValueError, match="FINHUB_API_KEY"):
        finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")

    news_request.assert_not_called()


@pytest.mark.parametrize(
    ("ticker", "start_date", "end_date"),
    [
        ("", "2026-11-10", "2026-11-18"),
        (" ", "2026-11-10", "2026-11-18"),
        ("AAPL", "invalid", "2026-11-18"),
        ("AAPL", "2026-11-10", "2026-11-31"),
        ("AAPL", "20261110", "2026-11-18"),
        ("AAPL", "2026-11-10", "20261118"),
        ("AAPL", "2026-11-19", "2026-11-18"),
    ],
)
def test_company_news_rejects_invalid_queries_before_network_access(news_request, ticker, start_date, end_date):
    from tradingscope.dataflows import finnhub_news

    with pytest.raises(ValueError):
        finnhub_news.get_news(ticker, start_date, end_date)

    news_request.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "Permission denied"},
        {"Information": "Premium endpoint"},
        None,
        "not a news array",
        ["not an article"],
        [{}],
        [{**_article(), "datetime": "invalid"}],
        [{**_article(), "datetime": 0}],
        [{**_article(), "datetime": True}],
        [{**_article(), "headline": None}],
        [{**_article(), "headline": ""}],
        [{**_article(), "summary": {"text": "invalid"}}],
        [{**_article(), "url": "javascript:alert(1)"}],
    ],
)
def test_company_news_rejects_error_and_malformed_payloads(news_request, payload):
    from tradingscope.dataflows import finnhub_news

    news_request.return_value.json.return_value = payload

    with pytest.raises(RuntimeError, match="Finnhub"):
        finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
def test_company_news_propagates_http_failures(news_request, status_code):
    from tradingscope.dataflows import finnhub_news

    error = requests.HTTPError(f"HTTP {status_code}")
    news_request.return_value.raise_for_status.side_effect = error

    with pytest.raises(requests.HTTPError) as raised:
        finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")

    assert raised.value is error
    news_request.return_value.json.assert_not_called()


def test_company_news_propagates_timeout_without_retry(news_request):
    from tradingscope.dataflows import finnhub_news

    news_request.side_effect = requests.Timeout("Request timed out")

    with pytest.raises(requests.Timeout):
        finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")

    assert news_request.call_count == 1


def test_company_news_rejects_invalid_json(news_request):
    from tradingscope.dataflows import finnhub_news

    news_request.return_value.json.side_effect = ValueError("Invalid JSON")

    with pytest.raises(ValueError, match="Invalid JSON"):
        finnhub_news.get_news("AAPL", "2026-11-10", "2026-11-18")


@pytest.mark.parametrize(("ticker", "payload"), [("AAPL", [_article()]), ("MSFT", [])])
def test_routed_news_tool_selects_finnhub_without_calling_other_vendors(news_request, monkeypatch, ticker, payload):
    from tradingscope.agents.utils.news_data_tools import get_news
    from tradingscope.dataflows import interface
    from tradingscope.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["tool_vendors"]["get_news"] == "finnhub"
    assert interface.get_vendor("news_data", "get_news") == "finnhub"
    vendors = interface.VENDOR_METHODS["get_news"]
    for name in vendors:
        if name != "finnhub":
            monkeypatch.setitem(vendors, name, Mock(__name__=name, side_effect=AssertionError("Unexpected vendor call")))
    news_request.return_value.json.return_value = payload

    result = get_news(ticker, "2026-11-12", "2026-11-18")

    assert result.state.value == "success"
    text = "".join(block.text for block in result.content)
    assert "Finnhub" in text
    assert ("未找到相关新闻" in text) == (not payload)
    assert news_request.call_count == 1
    for name, vendor in vendors.items():
        if name != "finnhub":
            vendor.assert_not_called()


def test_routed_news_tool_falls_back_on_finnhub_permission_error(news_request, monkeypatch):
    from tradingscope.agents.utils.news_data_tools import get_news
    from tradingscope.dataflows import interface

    vendors = interface.VENDOR_METHODS["get_news"]
    assert "finnhub" in vendors
    google = Mock(__name__="google", return_value="Google fallback news")
    for name in vendors:
        if name != "finnhub":
            monkeypatch.setitem(vendors, name, google if name == "google" else Mock(__name__=name, side_effect=AssertionError("Unexpected vendor call")))
    news_request.return_value.json.return_value = {"error": "Permission denied"}

    result = get_news("GOOG", "2026-11-12", "2026-11-18")

    assert "".join(block.text for block in result.content) == "Google fallback news"
    news_request.assert_called_once()
    google.assert_called_once_with("GOOG", "2026-11-12", "2026-11-18")
    for name, vendor in vendors.items():
        if name not in ("finnhub", "google"):
            vendor.assert_not_called()


def test_routed_news_tool_raises_when_all_vendors_fail(news_request, monkeypatch):
    from tradingscope.agents.utils.news_data_tools import get_news
    from tradingscope.dataflows import interface

    vendors = interface.VENDOR_METHODS["get_news"]
    assert "finnhub" in vendors
    for name in vendors:
        if name != "finnhub":
            monkeypatch.setitem(vendors, name, Mock(__name__=name, side_effect=RuntimeError("Vendor unavailable")))
    news_request.side_effect = requests.Timeout("Request timed out")

    with pytest.raises(RuntimeError, match="All vendor implementations failed"):
        get_news("NVDA", "2026-11-12", "2026-11-18")

    news_request.assert_called_once()
    for name, vendor in vendors.items():
        if name != "finnhub":
            vendor.assert_called_once()

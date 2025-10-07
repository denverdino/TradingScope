from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

# 你项目中的依赖（保持不变）
from agentscope.message import TextBlock

# AgentScope
from agentscope.tool import ToolResponse

import tradingscope.dataflows.interface as interface
from tradingscope.dataflows.akshare_utils import get_stock_news_em
from tradingscope.dataflows.improved_hk_utils import get_hk_company_name_improved
from tradingscope.dataflows.optimized_china_data import OptimizedChinaDataProvider
from tradingscope.dataflows.optimized_us_data import get_us_stock_data_cached
from tradingscope.dataflows.realtime_news_utils import get_realtime_stock_news
from tradingscope.dataflows.tdx_utils import get_china_market_overview
from tradingscope.dataflows.tushare_adapter import get_tushare_adapter
from tradingscope.dataflows.unified_news_tool import UnifiedNewsAnalyzer
from tradingscope.utils.logging_manager import get_logger
from tradingscope.utils.stock_utils import StockUtils

logger = get_logger("agents")


def _tb(text: str) -> ToolResponse:
    """把纯文本封装成 AgentScope 的 ToolResponse"""
    return ToolResponse(content=[TextBlock(type="text", text=str(text))])


# ----------------- 新闻 / 舆情 -----------------


def get_reddit_news(current_date: str) -> ToolResponse:
    """Retrieve global news from Reddit within a specified time frame.

    Args:
        current_date (str):
            current date in 'yyyy-mm-dd' format
    Returns:
            str: A formatted dataframe containing the latest global news from Reddit in the specified time frame.
    """

    try:
        res = interface.get_reddit_global_news(current_date, 7, 5)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_reddit_news 失败：{e}")


def get_finnhub_news(ticker: str, start_date: str, end_date: str) -> ToolResponse:
    """Retrieve the latest news about a given stock from Finnhub within a date range

    Args:
        ticker (str):
            股票代码（如 'AAPL'、'0700.HK'）
        start_date (str):
            Start date in yyyy-mm-dd format
        end_date (str):
            End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing news about the company within the date range from start_date to end_date
    """
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        look_back_days = (end_dt - start_dt).days
        res = interface.get_finnhub_news(ticker, end_date, look_back_days)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_finnhub_news 失败：{e}")


def get_reddit_stock_info(ticker: str, current_date: str) -> ToolResponse:
    """Retrieve latest Reddit news for a stock.

    Args:
        ticker (str): Ticker of a company. e.g. AAPL, TSM
        current_date (str): current date in yyyy-mm-dd format to get news for
    Returns:
        str: A formatted dataframe containing the latest news about the company on the given date
    """
    try:
        res = interface.get_reddit_company_news(ticker, current_date, 7, 5)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_reddit_stock_info 失败：{e}")


def get_chinese_social_sentiment(ticker: str, current_date: str) -> ToolResponse:
    """抓取中国社交媒体/财经平台对指定股票的情绪与热度；失败时回退到 Reddit 摘要

    Args:
        ticker (str):
            股票代码（支持A股符号）
        current_date (str):
            当前日期，格式为 'yyyy-mm-dd'
    """
    try:
        res = interface.get_chinese_social_sentiment(ticker, current_date)
        return _tb(res)
    except Exception:
        # 回退到 Reddit
        try:
            res_fb = interface.get_reddit_company_news(ticker, current_date, 7, 5)
            return _tb(res_fb)
        except Exception as e:
            logger.exception(e)
            return _tb(f"❌ get_chinese_social_sentiment 失败：{e}")


def get_google_news(query: str, current_date: str) -> ToolResponse:
    """
    Retrieve the latest news from Google News based on a query and date range.
    Args:
        query (str): Query to search with
        current_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): How many days to look back
    Returns:
        str: A formatted string containing the latest news from Google News based on the query and date range.
    """

    try:
        res = interface.get_google_news(query, current_date, 7)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_google_news 失败：{e}")


def get_realtime_stock_news(ticker: str, current_date: str) -> ToolResponse:
    """多源聚合实时股票新闻（15-30min 延迟级）"""
    try:

        res = get_realtime_stock_news(ticker, current_date, hours_back=6)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_realtime_stock_news 失败：{e}")


def get_stock_news_openai(ticker: str, current_date: str) -> ToolResponse:
    """
    Retrieve the latest news about a given stock by using OpenAI's news API.
    Args:
        ticker (str): Ticker of a company. e.g. AAPL, TSM
        current_date (str): Current date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing the latest news about the company on the given date.
    """

    try:
        res = interface.get_stock_news_openai(ticker, current_date)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_stock_news_openai 失败：{e}")


def get_global_news_openai(current_date: str) -> ToolResponse:
    """
    Retrieve the latest macroeconomics news on a given date using OpenAI's macroeconomics news API.
    Args:
        current_date (str): Current date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing the latest macroeconomic news on the given date.
    """

    try:
        res = interface.get_global_news_openai(current_date)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_global_news_openai 失败：{e}")


# ----------------- 市场概览 / 行情 / 技术指标 -----------------


def get_china_market_overview_tool(current_date: str) -> ToolResponse:
    """获取中国主要指数市场概览

    Args:
        current_date (str):
            当前日期，格式为 'yyyy-mm-dd'
    """
    try:

        adapter = get_tushare_adapter()
        if not adapter.provider or not adapter.provider.connected:

            return _tb(get_china_market_overview())

        return _tb(
            f"""# 中国股市概览 - {current_date}

## 📊 主要指数
- 上证指数: 数据获取中...
- 深证成指: 数据获取中...
- 创业板指: 数据获取中...
- 科创50: 数据获取中...

说明：市场概览正从 Tushare 对接完善中
"""
        )
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_china_market_overview 失败：{e}")


def get_YFin_data(symbol: str, start_date: str, end_date: str) -> ToolResponse:
    """Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    try:
        res = interface.get_YFin_data(symbol, start_date, end_date)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_YFin_data 失败：{e}")


def get_YFin_data_online(symbol: str, start_date: str, end_date: str) -> ToolResponse:
    """Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    try:
        res = interface.get_YFin_data_online(symbol, start_date, end_date)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_YFin_data_online 失败：{e}")


def get_stockstats_indicators_report(symbol: str, indicator: str, current_date: str, look_back_days: int = 30) -> ToolResponse:
    """
    Retrieve stock stats indicators for a given ticker symbol and indicator.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        current_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
    """

    try:
        res = interface.get_stock_stats_indicators_window(symbol, indicator, current_date, look_back_days, False)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_stockstats_indicators_report 失败：{e}")


def get_stockstats_indicators_report_online(symbol: str, indicator: str, current_date: str, look_back_days: int = 30) -> ToolResponse:
    """
    Retrieve stock stats indicators for a given ticker symbol and indicator.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        current_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
    """
    try:
        res = interface.get_stock_stats_indicators_window(symbol, indicator, current_date, look_back_days, True)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_stockstats_indicators_report_online 失败：{e}")


# ----------------- 财务报表 / 内幕交易 -----------------


def get_finnhub_company_insider_sentiment(ticker: str, current_date: str) -> ToolResponse:
    """
    Retrieve insider sentiment information about a company (retrieved from public SEC information) for the past 30 days
    Args:
        ticker (str): ticker symbol of the company
        current_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 30 days starting at current_date
    """

    try:
        res = interface.get_finnhub_company_insider_sentiment(ticker, current_date, 30)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_finnhub_company_insider_sentiment 失败：{e}")


def get_finnhub_company_insider_transactions(ticker: str, current_date: str) -> ToolResponse:
    """
    Retrieve insider transaction information about a company (retrieved from public SEC information) for the past 30 days
    Args:
        ticker (str): ticker symbol of the company
        current_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transactions/trading information in the past 30 days
    """

    try:
        res = interface.get_finnhub_company_insider_transactions(ticker, current_date, 30)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_finnhub_company_insider_transactions 失败：{e}")


def get_simfin_balance_sheet(ticker: str, freq: str, current_date: str) -> ToolResponse:
    """
    Retrieve the most recent balance sheet of a company
    Args:
        ticker (str): ticker symbol of the company
        freq (str): reporting frequency of the company's financial history: annual / quarterly
        current_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's most recent balance sheet
    """

    try:
        res = interface.get_simfin_balance_sheet(ticker, freq, current_date)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_simfin_balance_sheet 失败：{e}")


def get_simfin_cashflow(ticker: str, freq: str, current_date: str) -> ToolResponse:
    """
    Retrieve the most recent cash flow statement of a company
    Args:
        ticker (str): ticker symbol of the company
        freq (str): reporting frequency of the company's financial history: annual / quarterly
        current_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
            str: a report of the company's most recent cash flow statement
    """

    try:
        res = interface.get_simfin_cashflow(ticker, freq, current_date)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_simfin_cashflow 失败：{e}")


def get_simfin_income_stmt(ticker: str, freq: str, current_date: str) -> ToolResponse:
    """
    Retrieve the most recent income statement of a company
    Args:
        ticker (str): ticker symbol of the company
        freq (str): reporting frequency of the company's financial history: annual / quarterly
        current_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
            str: a report of the company's most recent income statement
    """

    try:
        res = interface.get_simfin_income_statements(ticker, freq, current_date)
        return _tb(res)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_simfin_income_stmt 失败：{e}")


# ----------------- 统一：基本面 / 行情 / 新闻 / 情绪 -----------------


def get_stock_fundamentals_unified(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    current_date: str | None = None,
) -> ToolResponse:
    """统一基本面分析（A股/港股/美股自动路由）
    Args:
        ticker (str):
            要分析的股票代码
        start_date (str, optional):
            分析的起始日期，格式为 'yyyy-mm-dd'
        end_date (str, optional):
            分析的结束日期，格式为 'yyyy-mm-dd'
        current_date (str, optional):
            当前日期，格式为 'yyyy-mm-dd'
    """

    try:
        logger.info(f"📊 统一基本面工具: {ticker}")
        market_info = StockUtils.get_market_info(ticker)
        is_china, is_hk = market_info["is_china"], market_info["is_hk"]

        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = current_date

        parts: List[str] = []
        if is_china:
            # A股
            try:

                stock_data = interface.get_china_stock_data_unified(ticker, start_date, end_date)
                parts.append(f"## A股价格数据\n{stock_data}")
            except Exception as e:
                parts.append(f"## A股价格数据\n获取失败: {e}")

            try:

                analyzer = OptimizedChinaDataProvider()
                fundamentals = analyzer._generate_fundamentals_report(ticker, locals().get("stock_data", ""))
                parts.append(f"## A股基本面数据\n{fundamentals}")
            except Exception as e:
                parts.append(f"## A股基本面数据\n获取失败: {e}")

        elif is_hk:
            # 港股
            try:

                hk_data = interface.get_hk_stock_data_unified(ticker, start_date, end_date)
                parts.append(f"## 港股数据\n{hk_data}")
            except Exception as e:
                parts.append(f"## 港股数据\n获取失败: {e}")
        else:
            # 美股
            try:

                us_data = interface.get_fundamentals_openai(ticker, current_date)
                parts.append(f"## 美股基本面数据\n{us_data}")
            except Exception as e:
                parts.append(f"## 美股基本面数据\n获取失败: {e}")

        text = f"""# {ticker} 基本面分析数据

**股票类型**: {market_info['market_name']}
**货币**: {market_info['currency_name']} ({market_info['currency_symbol']})
**分析日期**: {current_date}

{chr(10).join(parts)}

---
*数据来源: 自动根据股票类型选择*
"""
        return _tb(text)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_stock_fundamentals_unified 失败：{e}")


def get_stock_market_data_unified(ticker: str, start_date: str, end_date: str) -> ToolResponse:
    """统一行情/技术分析（A股/港股/美股自动路由）

    Args:
        ticker (str):
            股票代码
        start_date (str):
            起始日期，格式为 'yyyy-mm-dd'
        end_date (str):
            结束日期，格式为 'yyyy-mm-dd'
    """
    try:
        logger.info(f"📈 统一市场工具: {ticker}")
        market_info = StockUtils.get_market_info(ticker)
        is_china, is_hk = market_info["is_china"], market_info["is_hk"]

        parts: List[str] = []
        if is_china:
            try:
                stock_data = interface.get_china_stock_data_unified(ticker, start_date, end_date)
                parts.append(f"## A股市场数据\n{stock_data}")
            except Exception as e:
                parts.append(f"## A股市场数据\n获取失败: {e}")
        elif is_hk:
            try:
                hk_data = interface.get_hk_stock_data_unified(ticker, start_date, end_date)
                parts.append(f"## 港股市场数据\n{hk_data}")
            except Exception as e:
                parts.append(f"## 港股市场数据\n获取失败: {e}")
        else:
            try:
                us_data = get_us_stock_data_cached(ticker, start_date, end_date)
                parts.append(f"## 美股市场数据\n{us_data}")
            except Exception as e:
                parts.append(f"## 美股市场数据\n获取失败: {e}")

        text = f"""# {ticker} 市场数据分析

**股票类型**: {market_info['market_name']}
**货币**: {market_info['currency_name']} ({market_info['currency_symbol']})
**分析期间**: {start_date} 至 {end_date}

{chr(10).join(parts)}

---
*数据来源: 自动根据股票类型选择*
"""
        return _tb(text)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_stock_market_data_unified 失败：{e}")


def get_stock_news_unified(ticker: str, current_date: str) -> ToolResponse:
    """统一新闻抓取（A股/港股中文源 + 美股 Finnhub）

    Args:
        ticker (str):
            股票代码
        current_date (str):
            当前日期，格式为 'yyyy-mm-dd'
    """
    try:
        logger.info(f"📰 统一新闻工具: {ticker}")
        market_info = StockUtils.get_market_info(ticker)
        is_china, is_hk = market_info["is_china"], market_info["is_hk"]

        end_dt = datetime.strptime(current_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=7)
        start_date_str = start_dt.strftime("%Y-%m-%d")

        parts: List[str] = []
        if is_china or is_hk:
            # 东方财富
            try:
                clean_ticker = ticker.replace(".SH", "").replace(".SZ", "").replace(".SS", "").replace(".HK", "").replace(".XSHE", "").replace(".XSHG", "")

                news_df = get_stock_news_em(clean_ticker)
                if news_df is not None and not news_df.empty:
                    items = []
                    for _, row in news_df.iterrows():
                        t = row.get("标题", "")
                        tm = row.get("时间", "")
                        url = row.get("链接", "")
                        items.append(f"- **{t}** [{tm}]({url})")
                    if items:
                        parts.append("## 东方财富新闻\n" + "\n".join(items))
            except Exception as e:
                parts.append(f"## 东方财富新闻\n获取失败: {e}")
            # Google 中文新闻补充
            try:
                if is_china:
                    clean_ticker = ticker.replace(".SH", "").replace(".SZ", "").replace(".SS", "").replace(".XSHE", "").replace(".XSHG", "")
                    q = f"{clean_ticker} 股票 公司 财报 新闻"
                else:
                    q = f"{ticker} 港股"
                data = interface.get_google_news(q, current_date)
                parts.append(f"## Google新闻\n{data}")
            except Exception as e:
                parts.append(f"## Google新闻\n获取失败: {e}")
        else:
            # 美股：Finnhub
            try:
                data = interface.get_finnhub_news(ticker, start_date_str, current_date)
                parts.append(f"## 美股新闻\n{data}")
            except Exception as e:
                parts.append(f"## 美股新闻\n获取失败: {e}")

        text = f"""# {ticker} 新闻分析

**股票类型**: {market_info['market_name']}
**分析日期**: {current_date}
**新闻时间范围**: {start_date_str} 至 {current_date}

{chr(10).join(parts)}

---
*数据来源: 自动根据股票类型选择*
"""
        return _tb(text)
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_stock_news_unified 失败：{e}")


def get_stock_sentiment_unified(ticker: str, current_date: str) -> ToolResponse:
    """统一情绪分析（A股/港股：中文社媒；美股：Reddit）

    Args:
        ticker (str):
            股票代码
        current_date (str):
            当前日期，格式为 'yyyy-mm-dd'
    """
    try:
        logger.info(f"😊 统一情绪工具: {ticker}")
        market_info = StockUtils.get_market_info(ticker)
        is_china, is_hk = market_info["is_china"], market_info["is_hk"]

        if is_china or is_hk:
            text = f"""
# {ticker} 情绪分析

**股票类型**: {market_info['market_name']}
**分析日期**: {current_date}

## 中文市场情绪（基础版）
- 关注雪球、东方财富、同花顺等平台的讨论热度
- 港股请同时参考香港本地财经媒体
（完整中文情绪源集成开发中）
"""
            return _tb(text)
        else:
            try:
                data = interface.get_reddit_company_news(ticker, current_date)
                text = f"""# {ticker} 情绪分析

**股票类型**: {market_info['market_name']}
**分析日期**: {current_date}

## 美股 Reddit 情绪
{data}
"""
                return _tb(text)
            except Exception as e:
                return _tb(f"❌ 美股 Reddit 情绪获取失败：{e}")
    except Exception as e:
        logger.exception(e)
        return _tb(f"❌ get_stock_sentiment_unified 失败：{e}")


def get_stock_news_unified(stock_code: str, max_news: int = 100, model_info: str = "") -> ToolResponse:
    """
    统一新闻获取工具 - 根据股票代码自动获取相应市场的新闻

    功能:
    - 自动识别股票类型（A股/港股/美股）
    - 根据股票类型选择最佳新闻源
    - A股: 优先东方财富 -> Google中文 -> OpenAI
    - 港股: 优先Google -> OpenAI -> 实时新闻
    - 美股: 优先OpenAI -> Google英文 -> FinnHub
    - 返回格式化的新闻内容

    Args:
        stock_code (str): 股票代码 (支持A股如000001、港股如0700.HK、美股如AAPL)
        max_news (int): 最大新闻数量，默认100
        model_info (str): 当前使用的模型信息，用于特殊处理

    Returns:
        str: 格式化的新闻内容
    """
    analyzer = UnifiedNewsAnalyzer()

    if not stock_code:
        return _tb("❌ 错误: 未提供股票代码")

    return _tb(analyzer.get_stock_news_unified(stock_code, max_news, model_info))


def get_company_name(ticker: str, market_info: dict) -> str:
    try:
        if market_info.get("is_china"):

            stock_info = interface.get_china_stock_info_unified(ticker)
            if "股票名称:" in stock_info:
                company_name = stock_info.split("股票名称:")[1].split("\n")[0].strip()
                logger.debug(f"[分析师] 统一接口: {ticker} -> {company_name}")
                return company_name
            logger.warning(f"[分析师] 统一接口未解析到名称: {ticker}")
            return f"股票代码{ticker}"

        if market_info.get("is_hk"):
            try:
                company_name = get_hk_company_name_improved(ticker)
                logger.debug(f"[分析师] 港股名称: {ticker} -> {company_name}")
                return company_name
            except Exception as e:
                logger.debug(f"[分析师] 港股名称降级: {e}")
                clean = ticker.replace(".HK", "").replace(".hk", "")
                return f"港股{clean}"

        if market_info.get("is_us"):
            us_map = {
                "AAPL": "苹果公司",
                "TSLA": "特斯拉",
                "NVDA": "英伟达",
                "MSFT": "微软",
                "GOOGL": "谷歌",
                "AMZN": "亚马逊",
                "META": "Meta",
                "NFLX": "奈飞",
            }
            return us_map.get(ticker.upper(), f"美股{ticker}")

        return f"股票{ticker}"
    except Exception as e:
        logger.error(f"[分析师] 获取公司名称失败: {e}")
        return f"股票{ticker}"

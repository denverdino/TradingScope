#!/usr/bin/env python3
"""
AKShare数据源工具
提供AKShare数据获取的统一接口
"""

import hashlib
import json
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

# 导入日志模块
from tradingscope.utils.logging_manager import get_logger

logger = get_logger("agents")
warnings.filterwarnings("ignore")

# 缓存存储字典
_akshare_cache = {}
# 默认缓存时间（30分钟）
DEFAULT_CACHE_TIMEOUT = timedelta(minutes=30)


class AKShareProvider:
    """AKShare数据提供器"""

    def __init__(self):
        """初始化AKShare提供器"""
        try:
            import akshare as ak

            self.ak = ak
            self.connected = True

            # 设置更长的超时时间
            self._configure_timeout()

            logger.debug("✅ AKShare初始化成功")
        except ImportError:
            self.ak = None
            self.connected = False
            logger.error("❌ AKShare未安装")

    def _configure_timeout(self):
        """配置AKShare的超时设置"""
        try:
            import socket

            import requests

            # 设置更长的超时时间
            socket.setdefaulttimeout(60)  # 60秒超时

            # 如果AKShare使用requests，设置默认超时
            if hasattr(requests, "adapters"):
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry

                # 创建重试策略
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )

                # 设置适配器
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session = requests.Session()
                session.mount("http://", adapter)
                session.mount("https://", adapter)

                logger.debug("🔧 AKShare超时配置完成: 60秒超时，3次重试")

        except Exception as e:
            logger.error(f"⚠️ AKShare超时配置失败: {e}")
            logger.info("🔧 使用默认超时设置")

    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """获取股票历史数据"""
        # 生成缓存键
        cache_key = _generate_cache_key("get_stock_data", symbol, start_date, end_date)

        # 尝试从缓存获取
        cached_data = _get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        if not self.connected:
            return None

        try:
            # 转换股票代码格式
            if len(symbol) == 6:
                symbol = symbol
            else:
                symbol = symbol.replace(".SZ", "").replace(".SS", "")

            # 获取数据
            data = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", "") if start_date else "20240101",
                end_date=end_date.replace("-", "") if end_date else "20241231",
                adjust="",
            )

            # 存入缓存
            _set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"❌ AKShare获取股票数据失败: {e}")
            return None

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票基本信息"""
        # 生成缓存键
        cache_key = _generate_cache_key("get_stock_info", symbol)

        # 尝试从缓存获取
        cached_data = _get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        if not self.connected:
            return {}

        try:
            # 获取股票基本信息
            stock_list = self.ak.stock_info_a_code_name()
            stock_info = stock_list[stock_list["code"] == symbol]

            if not stock_info.empty:
                result = {"symbol": symbol, "name": stock_info.iloc[0]["name"], "source": "akshare"}
            else:
                result = {"symbol": symbol, "name": f"股票{symbol}", "source": "akshare"}

            # 存入缓存
            _set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"❌ AKShare获取股票信息失败: {e}")
            result = {"symbol": symbol, "name": f"股票{symbol}", "source": "akshare"}
            # 存入缓存
            _set_cache(cache_key, result)
            return result

    def get_hk_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取港股历史数据

        Args:
            symbol: 港股代码 (如: 00700 或 0700.HK)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame: 港股历史数据
        """
        # 生成缓存键
        cache_key = _generate_cache_key("get_hk_stock_data", symbol, start_date, end_date)

        # 尝试从缓存获取
        cached_data = _get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        if not self.connected:
            logger.error("❌ AKShare未连接")
            return None

        try:
            # 标准化港股代码 - AKShare使用5位数字格式
            hk_symbol = self._normalize_hk_symbol_for_akshare(symbol)

            logger.info(f"🇭🇰 AKShare获取港股数据: {hk_symbol} ({start_date} 到 {end_date})")

            # 格式化日期为AKShare需要的格式
            start_date_formatted = start_date.replace("-", "") if start_date else "20240101"
            end_date_formatted = end_date.replace("-", "") if end_date else "20241231"

            # 使用AKShare获取港股历史数据（带超时保护）
            import threading

            result = [None]
            exception = [None]

            def fetch_hist_data():
                try:
                    result[0] = self.ak.stock_hk_hist(symbol=hk_symbol, period="daily", start_date=start_date_formatted, end_date=end_date_formatted, adjust="")
                except Exception as e:
                    exception[0] = e

            # 启动线程
            thread = threading.Thread(target=fetch_hist_data)
            thread.daemon = True
            thread.start()

            # 等待60秒
            thread.join(timeout=60)

            if thread.is_alive():
                # 超时了
                logger.warning(f"⚠️ AKShare港股历史数据获取超时（60秒）: {symbol}")
                raise Exception(f"AKShare港股历史数据获取超时（60秒）: {symbol}")
            elif exception[0]:
                # 有异常
                raise exception[0]
            else:
                # 成功
                data = result[0]

            if not data.empty:
                # 数据预处理
                data = data.reset_index()
                data["Symbol"] = symbol  # 保持原始格式

                # 重命名列以保持一致性
                column_mapping = {"日期": "Date", "开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Amount"}

                for old_col, new_col in column_mapping.items():
                    if old_col in data.columns:
                        data = data.rename(columns={old_col: new_col})

                logger.info(f"✅ AKShare港股数据获取成功: {symbol}, {len(data)}条记录")
                # 存入缓存
                _set_cache(cache_key, data)
                return data
            else:
                logger.warning(f"⚠️ AKShare港股数据为空: {symbol}")
                # 存入缓存
                _set_cache(cache_key, None)
                return None

        except Exception as e:
            logger.error(f"❌ AKShare获取港股数据失败: {e}")
            # 存入缓存
            _set_cache(cache_key, None)
            return None

    def get_hk_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取港股基本信息

        Args:
            symbol: 港股代码

        Returns:
            Dict: 港股基本信息
        """
        # 生成缓存键
        cache_key = _generate_cache_key("get_hk_stock_info", symbol)

        # 尝试从缓存获取
        cached_data = _get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        if not self.connected:
            result = {"symbol": symbol, "name": f"港股{symbol}", "currency": "HKD", "exchange": "HKG", "source": "akshare_unavailable"}
            # 存入缓存
            _set_cache(cache_key, result)
            return result

        try:
            hk_symbol = self._normalize_hk_symbol_for_akshare(symbol)

            logger.info(f"🇭🇰 AKShare获取港股信息: {hk_symbol}")

            # 尝试获取港股实时行情数据来获取基本信息
            # 使用线程超时包装（兼容Windows）
            import threading

            result = [None]
            exception = [None]

            def fetch_data():
                try:
                    result[0] = self.ak.stock_hk_spot_em()
                except Exception as e:
                    exception[0] = e

            # 启动线程
            thread = threading.Thread(target=fetch_data)
            thread.daemon = True
            thread.start()

            # 等待60秒
            thread.join(timeout=60)

            if thread.is_alive():
                # 超时了
                logger.warning("⚠️ AKShare港股信息获取超时（60秒），使用备用方案")
                raise Exception("AKShare港股信息获取超时（60秒）")
            elif exception[0]:
                # 有异常
                raise exception[0]
            else:
                # 成功
                spot_data = result[0]

            # 查找对应的股票信息
            if not spot_data.empty:
                # 查找匹配的股票
                matching_stocks = spot_data[spot_data["代码"].str.contains(hk_symbol[:5], na=False)]

                if not matching_stocks.empty:
                    stock_info = matching_stocks.iloc[0]
                    result = {
                        "symbol": symbol,
                        "name": stock_info.get("名称", f"港股{symbol}"),
                        "currency": "HKD",
                        "exchange": "HKG",
                        "latest_price": stock_info.get("最新价", None),
                        "source": "akshare",
                    }
                else:
                    # 如果没有找到，返回基本信息
                    result = {"symbol": symbol, "name": f"港股{symbol}", "currency": "HKD", "exchange": "HKG", "source": "akshare"}
            else:
                # 如果没有找到，返回基本信息
                result = {"symbol": symbol, "name": f"港股{symbol}", "currency": "HKD", "exchange": "HKG", "source": "akshare"}

            # 存入缓存
            _set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"❌ AKShare获取港股信息失败: {e}")
            result = {"symbol": symbol, "name": f"港股{symbol}", "currency": "HKD", "exchange": "HKG", "source": "akshare_error", "error": str(e)}
            # 存入缓存
            _set_cache(cache_key, result)
            return result

    def _normalize_hk_symbol_for_akshare(self, symbol: str) -> str:
        """
        标准化港股代码为AKShare格式

        Args:
            symbol: 原始港股代码 (如: 0700.HK 或 700)

        Returns:
            str: AKShare格式的港股代码 (如: 00700)
        """
        if not symbol:
            return symbol

        # 移除.HK后缀
        clean_symbol = symbol.replace(".HK", "").replace(".hk", "")

        # 确保是5位数字格式
        if clean_symbol.isdigit():
            return clean_symbol.zfill(5)

        return clean_symbol

    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取股票财务数据

        Args:
            symbol: 股票代码 (6位数字)

        Returns:
            Dict: 包含主要财务指标的财务数据
        """
        # 生成缓存键
        cache_key = _generate_cache_key("get_financial_data", symbol)

        # 尝试从缓存获取
        cached_data = _get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        if not self.connected:
            logger.error(f"❌ AKShare未连接，无法获取{symbol}财务数据")
            result = {}
            # 存入缓存
            _set_cache(cache_key, result)
            return result

        try:
            logger.info(f"🔍 开始获取{symbol}的AKShare财务数据")

            financial_data = {}

            # 1. 优先获取主要财务指标
            try:
                logger.debug(f"📊 尝试获取{symbol}主要财务指标...")
                main_indicators = self.ak.stock_financial_abstract(symbol=symbol)
                if main_indicators is not None and not main_indicators.empty:
                    financial_data["main_indicators"] = main_indicators
                    logger.info(f"✅ 成功获取{symbol}主要财务指标: {len(main_indicators)}条记录")
                    logger.debug(f"主要财务指标列名: {list(main_indicators.columns)}")
                else:
                    logger.warning(f"⚠️ {symbol}主要财务指标为空")
            except Exception as e:
                logger.warning(f"❌ 获取{symbol}主要财务指标失败: {e}")

            # 2. 尝试获取资产负债表（可能失败，降级为debug日志）
            try:
                logger.debug(f"📊 尝试获取{symbol}资产负债表...")
                balance_sheet = self.ak.stock_balance_sheet_by_report_em(symbol=symbol)
                if balance_sheet is not None and not balance_sheet.empty:
                    financial_data["balance_sheet"] = balance_sheet
                    logger.debug(f"✅ 成功获取{symbol}资产负债表: {len(balance_sheet)}条记录")
                else:
                    logger.debug(f"⚠️ {symbol}资产负债表为空")
            except Exception as e:
                logger.debug(f"❌ 获取{symbol}资产负债表失败: {e}")

            # 3. 尝试获取利润表（可能失败，降级为debug日志）
            try:
                logger.debug(f"📊 尝试获取{symbol}利润表...")
                income_statement = self.ak.stock_profit_sheet_by_report_em(symbol=symbol)
                if income_statement is not None and not income_statement.empty:
                    financial_data["income_statement"] = income_statement
                    logger.debug(f"✅ 成功获取{symbol}利润表: {len(income_statement)}条记录")
                else:
                    logger.debug(f"⚠️ {symbol}利润表为空")
            except Exception as e:
                logger.debug(f"❌ 获取{symbol}利润表失败: {e}")

            # 4. 尝试获取现金流量表（可能失败，降级为debug日志）
            try:
                logger.debug(f"📊 尝试获取{symbol}现金流量表...")
                cash_flow = self.ak.stock_cash_flow_sheet_by_report_em(symbol=symbol)
                if cash_flow is not None and not cash_flow.empty:
                    financial_data["cash_flow"] = cash_flow
                    logger.debug(f"✅ 成功获取{symbol}现金流量表: {len(cash_flow)}条记录")
                else:
                    logger.debug(f"⚠️ {symbol}现金流量表为空")
            except Exception as e:
                logger.debug(f"❌ 获取{symbol}现金流量表失败: {e}")

            # 记录最终结果
            if financial_data:
                logger.info(f"✅ AKShare财务数据获取完成: {symbol}, 包含{len(financial_data)}个数据集")
                for key, value in financial_data.items():
                    if hasattr(value, "__len__"):
                        logger.info(f"  - {key}: {len(value)}条记录")
            else:
                logger.warning(f"⚠️ 未能获取{symbol}的任何AKShare财务数据")

            # 存入缓存
            _set_cache(cache_key, financial_data)
            return financial_data

        except Exception as e:
            logger.error(f"❌ AKShare获取{symbol}财务数据失败: {e}")
            result = {}
            # 存入缓存
            _set_cache(cache_key, result)
            return result


def _generate_cache_key(func_name: str, *args, **kwargs) -> str:
    """
    生成缓存键

    Args:
        func_name: 函数名
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        str: 缓存键
    """
    # 将参数转换为可哈希的字符串
    key_data = {"function": func_name, "args": args, "kwargs": kwargs}

    # 对于包含不可序列化对象的参数，进行特殊处理
    try:
        key_str = json.dumps(key_data, sort_keys=True, default=str)
    except Exception:
        # 如果JSON序列化失败，使用更简单的表示方法
        key_str = f"{func_name}_{hash(str(args) + str(sorted(kwargs.items())))}"

    # 生成MD5哈希作为缓存键
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def _get_from_cache(cache_key: str) -> Any:
    """
    从缓存中获取数据

    Args:
        cache_key: 缓存键

    Returns:
        Any: 缓存的数据，如果缓存不存在或已过期则返回None
    """
    if cache_key in _akshare_cache:
        cached_item = _akshare_cache[cache_key]
        # 检查是否过期
        if datetime.now() < cached_item["expires_at"]:
            logger.debug(f"✅ 缓存命中: {cache_key}")
            return cached_item["data"]
        else:
            # 缓存已过期，删除它
            logger.debug(f"⏰ 缓存过期: {cache_key}")
            del _akshare_cache[cache_key]
    return None


def _set_cache(cache_key: str, data: Any, timeout: timedelta = DEFAULT_CACHE_TIMEOUT) -> None:
    """
    将数据存入缓存

    Args:
        cache_key: 缓存键
        data: 要缓存的数据
        timeout: 缓存超时时间
    """
    # 不缓存None或空数据
    if data is None or (isinstance(data, (list, dict, pd.DataFrame)) and len(data) == 0):
        logger.debug(f"🚫 不缓存空数据: {cache_key}")
        return

    expires_at = datetime.now() + timeout
    _akshare_cache[cache_key] = {"data": data, "expires_at": expires_at}
    logger.debug(f"💾 数据已缓存: {cache_key}, 过期时间: {expires_at}")


def get_akshare_provider() -> AKShareProvider:
    """获取AKShare提供器实例"""
    return AKShareProvider()


# 便捷函数
def get_hk_stock_data_akshare(symbol: str, start_date: str = None, end_date: str = None) -> str:
    """
    使用AKShare获取港股数据的便捷函数

    Args:
        symbol: 港股代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        str: 格式化的港股数据
    """
    # 生成缓存键
    cache_key = _generate_cache_key("get_hk_stock_data_akshare", symbol, start_date, end_date)

    # 尝试从缓存获取
    cached_data = _get_from_cache(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        provider = get_akshare_provider()
        data = provider.get_hk_stock_data(symbol, start_date, end_date)

        if data is not None and not data.empty:
            result = format_hk_stock_data_akshare(symbol, data, start_date, end_date)
        else:
            result = f"❌ 无法获取港股 {symbol} 的AKShare数据"

        # 存入缓存
        _set_cache(cache_key, result)
        return result

    except Exception as e:
        result = f"❌ AKShare港股数据获取失败: {e}"
        # 存入缓存
        _set_cache(cache_key, result)
        return result


def get_hk_stock_info_akshare(symbol: str) -> Dict[str, Any]:
    """
    使用AKShare获取港股信息的便捷函数

    Args:
        symbol: 港股代码

    Returns:
        Dict: 港股信息
    """
    # 生成缓存键
    cache_key = _generate_cache_key("get_hk_stock_info_akshare", symbol)

    # 尝试从缓存获取
    cached_data = _get_from_cache(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        provider = get_akshare_provider()
        result = provider.get_hk_stock_info(symbol)
        # 存入缓存
        _set_cache(cache_key, result)
        return result
    except Exception as e:
        result = {"symbol": symbol, "name": f"港股{symbol}", "currency": "HKD", "exchange": "HKG", "source": "akshare_error", "error": str(e)}
        # 存入缓存
        _set_cache(cache_key, result)
        return result


def format_hk_stock_data_akshare(symbol: str, data: pd.DataFrame, start_date: str, end_date: str) -> str:
    """
    格式化AKShare港股数据为文本格式

    Args:
        symbol: 股票代码
        data: 股票数据DataFrame
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        str: 格式化的股票数据文本
    """
    if data is None or data.empty:
        return f"❌ 无法获取港股 {symbol} 的AKShare数据"

    try:
        # 获取股票基本信息（允许失败）
        stock_name = f"港股{symbol}"  # 默认名称
        try:
            provider = get_akshare_provider()
            stock_info = provider.get_hk_stock_info(symbol)
            stock_name = stock_info.get("name", f"港股{symbol}")
            logger.info(f"✅ 港股信息获取成功: {stock_name}")
        except Exception as info_error:
            logger.error(f"⚠️ 港股信息获取失败，使用默认信息: {info_error}")
            # 继续处理，使用默认信息

        # 计算统计信息
        latest_price = data["Close"].iloc[-1]
        price_change = data["Close"].iloc[-1] - data["Close"].iloc[0]
        price_change_pct = (price_change / data["Close"].iloc[0]) * 100

        avg_volume = data["Volume"].mean() if "Volume" in data.columns else 0
        max_price = data["High"].max()
        min_price = data["Low"].min()

        # 格式化输出
        formatted_text = f"""
🇭🇰 港股数据报告 (AKShare)
================

股票信息:
- 代码: {symbol}
- 名称: {stock_name}
- 货币: 港币 (HKD)
- 交易所: 香港交易所 (HKG)

价格信息:
- 最新价格: HK${latest_price:.2f}
- 期间涨跌: HK${price_change:+.2f} ({price_change_pct:+.2f}%)
- 期间最高: HK${max_price:.2f}
- 期间最低: HK${min_price:.2f}

交易信息:
- 数据期间: {start_date} 至 {end_date}
- 交易天数: {len(data)}天
- 平均成交量: {avg_volume:,.0f}股

最近5个交易日:
"""

        # 添加最近5天的数据
        recent_data = data.tail(5)
        for _, row in recent_data.iterrows():
            date = row["Date"].strftime("%Y-%m-%d") if "Date" in row else row.name.strftime("%Y-%m-%d")
            volume = row.get("Volume", 0)
            formatted_text += f"- {date}: 开盘HK${row['Open']:.2f}, 收盘HK${row['Close']:.2f}, 成交量{volume:,.0f}\n"

        formatted_text += "\n数据来源: AKShare (港股)\n"

        return formatted_text

    except Exception as e:
        logger.error(f"❌ 格式化AKShare港股数据失败: {e}")
        return f"❌ AKShare港股数据格式化失败: {symbol}"


def get_stock_news_em(symbol: str, max_news: int = 10) -> pd.DataFrame:
    """
    使用AKShare获取东方财富个股新闻

    Args:
        symbol: 股票代码，如 "600000" 或 "300059"
        max_news: 最大新闻数量，默认10条

    Returns:
        pd.DataFrame: 包含新闻标题、内容、日期和链接的DataFrame
    """
    # 生成缓存键
    cache_key = _generate_cache_key("get_stock_news_em", symbol, max_news)

    # 尝试从缓存获取
    cached_data = _get_from_cache(cache_key)
    if cached_data is not None:
        return cached_data

    start_time = datetime.now()
    logger.info(f"[东方财富新闻] 开始获取股票 {symbol} 的东方财富新闻数据")

    try:
        provider = get_akshare_provider()
        if not provider.connected:
            logger.error("[东方财富新闻] ❌ AKShare未连接，无法获取东方财富新闻")
            result = pd.DataFrame()
            # 存入缓存
            _set_cache(cache_key, result)
            return result

        logger.info(f"[东方财富新闻] 📰 准备调用AKShare API获取个股新闻: {symbol}")

        # 使用线程超时包装（兼容Windows）
        import threading
        import time

        result = [None]
        exception = [None]

        def fetch_news():
            try:
                logger.debug(f"[东方财富新闻] 线程开始执行 stock_news_em API调用: {symbol}")
                thread_start = time.time()
                result[0] = provider.ak.stock_news_em(symbol=symbol)
                thread_end = time.time()
                logger.debug(f"[东方财富新闻] 线程执行完成，耗时: {thread_end - thread_start:.2f}秒")
            except Exception as e:
                logger.error(f"[东方财富新闻] 线程执行异常: {e}")
                exception[0] = e

        # 启动线程
        thread = threading.Thread(target=fetch_news)
        thread.daemon = True
        logger.debug("[东方财富新闻] 启动线程获取新闻数据")
        thread.start()

        # 等待30秒
        logger.debug("[东方财富新闻] 等待线程完成，最长等待30秒")
        thread.join(timeout=30)

        if thread.is_alive():
            # 超时了
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.warning(f"[东方财富新闻] ⚠️ 获取超时（30秒）: {symbol}，总耗时: {elapsed_time:.2f}秒")
            raise Exception(f"东方财富个股新闻获取超时（30秒）: {symbol}")
        elif exception[0]:
            # 有异常
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[东方财富新闻] ❌ API调用异常: {exception[0]}，总耗时: {elapsed_time:.2f}秒")
            raise exception[0]
        else:
            # 成功
            news_df = result[0]

        if news_df is not None and not news_df.empty:
            # 限制新闻数量为最新的max_news条
            if len(news_df) > max_news:
                news_df = news_df.head(max_news)
                logger.info(f"[东方财富新闻] 📰 新闻数量限制: 从{len(news_df)}条限制为{max_news}条最新新闻")

            news_count = len(news_df)
            elapsed_time = (datetime.now() - start_time).total_seconds()

            # 记录一些新闻标题示例
            sample_titles = [row.get("标题", "无标题") for _, row in news_df.head(3).iterrows()]
            logger.info(f"[东方财富新闻] 新闻标题示例: {', '.join(sample_titles)}")

            logger.info(f"[东方财富新闻] ✅ 获取成功: {symbol}, 共{news_count}条记录，耗时: {elapsed_time:.2f}秒")
            # 存入缓存
            _set_cache(cache_key, news_df)
            return news_df
        else:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.warning(f"[东方财富新闻] ⚠️ 数据为空: {symbol}，API返回成功但无数据，耗时: {elapsed_time:.2f}秒")
            result = pd.DataFrame()
            # 存入缓存
            _set_cache(cache_key, result)
            return result

    except Exception as e:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[东方财富新闻] ❌ 获取失败: {symbol}, 错误: {e}, 耗时: {elapsed_time:.2f}秒")
        result = pd.DataFrame()
        # 存入缓存
        _set_cache(cache_key, result)
        return result

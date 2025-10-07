#!/usr/bin/env python3
"""
Tushare数据接口数据获取工具
支持A股、港股实时数据和历史数据
"""

import warnings
from datetime import datetime
from typing import Dict

import pandas as pd

# 导入日志模块
from tradingscope.utils.logging_manager import get_logger

from .cache_manager import get_cache

logger = get_logger("agents")
warnings.filterwarnings("ignore")

# 尝试导入文件缓存管理器

FILE_CACHE_AVAILABLE = True

try:
    # 中国股票数据Python接口
    import pytdx
    from pytdx.exhq import TdxExHq_API
    from pytdx.hq import TdxHq_API

    TDX_AVAILABLE = True
except ImportError:
    TDX_AVAILABLE = False
    logger.warning("⚠️ pytdx库未安装，无法使用Tushare数据接口")
    logger.info("💡 安装命令: pip install pytdx")


class TongDaXinDataProvider:
    """通达信数据提供器"""

    def __init__(self):
        logger.debug("🔍 [DEBUG] 初始化通达信数据提供器...")
        self.api = None
        self.exapi = None  # 扩展行情API
        self.connected = False

        logger.debug(f"🔍 [DEBUG] 检查pytdx库可用性: {TDX_AVAILABLE}")

    def connect(self) -> bool:
        """连接到通达信服务器"""
        if not TDX_AVAILABLE:
            logger.error("❌ pytdx库不可用，无法连接到通达信服务器")
            return False

        try:
            logger.debug("🔍 [DEBUG] 开始连接通达信服务器...")
            if self.api is None:
                self.api = TdxHq_API()
                logger.debug("🔍 [DEBUG] TdxHq_API创建成功")

            # 尝试连接主服务器
            connected = self.api.connect("120.76.152.87", 7709)  # 通达信主服务器
            logger.debug(f"🔍 [DEBUG] 主服务器连接结果: {connected}")

            if not connected:
                # 尝试备用服务器
                logger.debug("🔍 [DEBUG] 尝试连接备用服务器...")
                connected = self.api.connect("120.76.152.87", 7709)  # 备用服务器
                logger.debug(f"🔍 [DEBUG] 备用服务器连接结果: {connected}")

            if connected:
                self.connected = True
                logger.info("✅ 成功连接到通达信服务器")
                return True
            else:
                self.connected = False
                logger.error("❌ 无法连接到通达信服务器")
                return False

        except Exception as e:
            logger.error(f"⚠️ 连接通达信服务器时发生错误: {e}")
            self.connected = False
            return False

    def _get_market_code(self, stock_code: str) -> int:
        """根据股票代码获取市场代码"""
        if stock_code.startswith(("600", "601", "603", "688", "689")):  # 上海股票
            return 1
        elif stock_code.startswith(("000", "001", "002", "003", "300", "301")):  # 深圳股票
            return 0
        else:
            # 默认返回深圳市场代码
            return 0

    def _get_stock_name(self, stock_code: str) -> str:
        """获取股票名称（独立方法，不依赖实时数据）"""
        # 简化版本，仅基于代码前缀返回市场名称
        if stock_code.startswith(("600", "601", "603", "688", "689")):
            return f"沪市股票{stock_code}"
        elif stock_code.startswith(("000", "001", "002", "003", "300", "301")):
            return f"深市股票{stock_code}"
        else:
            return f"股票{stock_code}"

    def get_real_time_data(self, stock_code: str) -> Dict:
        """
        获取股票实时数据
        Args:
            stock_code: 股票代码
        Returns:
            Dict: 实时数据
        """
        if not self.connected:
            if not self.connect():
                return {}

        try:
            market = self._get_market_code(stock_code)

            # 获取实时数据
            data = self.api.get_security_quotes([(market, stock_code)])

            if not data:
                return {}

            quote = data[0]

            # 安全获取字段，避免KeyError
            def safe_get(key, default=0):
                return quote.get(key, default)

            return {
                "code": stock_code,
                "name": self._get_stock_name(stock_code),  # 使用独立的股票名称获取方法
                "price": safe_get("price"),
                "last_close": safe_get("last_close"),
                "open": safe_get("open"),
                "high": safe_get("high"),
                "low": safe_get("low"),
                "volume": safe_get("vol"),
                "amount": safe_get("amount"),
                "bid1": safe_get("bid1"),
                "ask1": safe_get("ask1"),
                "change": safe_get("price") - safe_get("last_close"),
                "change_percent": ((safe_get("price") - safe_get("last_close")) / safe_get("last_close") * 100) if safe_get("last_close") != 0 else 0,
                "update_time": datetime.now().strftime("%H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"⚠️ 获取实时数据失败: {e}")
            return {}

    def get_stock_history_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票历史数据
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        Returns:
            pd.DataFrame: 历史数据
        """
        if not self.connected:
            if not self.connect():
                return pd.DataFrame()

        try:
            # 计算日期范围
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days_diff = (end_dt - start_dt).days

            # 获取K线数据（日线）
            market = self._get_market_code(stock_code)
            data = self.api.get_security_bars(9, market, stock_code, 0, min(days_diff + 10, 1000))  # 获取更多数据以确保覆盖所需范围

            if not data:
                return pd.DataFrame()

            # 转换为DataFrame
            df_data = []
            for bar in data:
                bar_date = datetime.strptime(str(bar["datetime"]), "%Y%m%d")
                if start_dt <= bar_date <= end_dt:
                    df_data.append(
                        {
                            "Date": bar_date,
                            "Open": bar["open"],
                            "High": bar["high"],
                            "Low": bar["low"],
                            "Close": bar["close"],
                            "Volume": bar["vol"],
                        }
                    )

            df = pd.DataFrame(df_data)
            if not df.empty:
                df = df.sort_values("Date").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"⚠️ 获取历史数据失败: {e}")
            return pd.DataFrame()

    def get_stock_technical_indicators(self, stock_code: str) -> Dict:
        """
        获取股票技术指标
        Args:
            stock_code: 股票代码
        Returns:
            Dict: 技术指标
        """
        try:
            # 简化版本，仅计算基本指标
            realtime_data = self.get_real_time_data(stock_code)
            if not realtime_data:
                return {}

            return {
                "MA5": realtime_data["price"],  # 简化处理
                "MA10": realtime_data["price"],
                "MA20": realtime_data["price"],
                "RSI": 50.0,  # 简化处理
                "MACD": 0.0,  # 简化处理
            }

        except Exception as e:
            logger.error(f"⚠️ 获取技术指标失败: {e}")
            return {}

    def get_market_overview(self) -> Dict:
        """获取市场概览"""
        if not self.connected:
            if not self.connect():
                return {}

        try:
            # 获取主要指数数据
            indices = [
                ("000001", 1),  # 上证指数
                ("399001", 0),  # 深证成指
                ("399006", 0),  # 创业板指
            ]

            result = {}
            for code, market in indices:
                data = self.api.get_security_quotes([(market, code)])
                if data:
                    quote = data[0]
                    last_close = quote.get("last_close", 0)
                    price = quote.get("price", 0)
                    change = price - last_close
                    change_percent = (change / last_close * 100) if last_close != 0 else 0

                    result[code] = {
                        "price": price,
                        "change": change,
                        "change_percent": change_percent,
                        "volume": quote.get("vol", 0),
                    }

            return result

        except Exception as e:
            logger.error(f"⚠️ 获取市场概览失败: {e}")
            return {}

    def __del__(self):
        """析构函数，确保连接被关闭"""
        try:
            if self.api:
                self.api.disconnect()
                logger.debug("🔍 [DEBUG] 通达信API连接已断开")
        except Exception as e:
            logger.error(f"⚠️ 断开连接时发生错误: {e}")


# 全局通达信提供器实例
_tdx_provider = None


def get_tdx_provider() -> TongDaXinDataProvider:
    """获取通达信数据提供器实例（单例模式）"""
    global _tdx_provider
    if _tdx_provider is None or not _tdx_provider.connected:
        logger.debug("🔍 [DEBUG] 重新创建通达信数据提供器实例...")
        _tdx_provider = TongDaXinDataProvider()
        # 尝试连接
        _tdx_provider.connect()
        logger.debug("🔍 [DEBUG] 通达信数据提供器重新创建完成")
    return _tdx_provider


def get_china_stock_data(stock_code: str, start_date: str, end_date: str) -> str:
    """
    获取中国股票数据的主要接口函数（支持缓存）
    Args:
        stock_code: 股票代码 (如 '000001')
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
    Returns:
        str: 格式化的股票数据
    """
    logger.info(f"📊 正在获取中国股票数据: {stock_code} ({start_date} 到 {end_date})")

    # 如果文件缓存可用，尝试从缓存加载数据
    if FILE_CACHE_AVAILABLE:
        cache = get_cache()
        cache_key = cache.find_cached_stock_data(
            symbol=stock_code, start_date=start_date, end_date=end_date, data_source="tdx", max_age_hours=6  # 6小时内的缓存有效
        )

        if cache_key:
            cached_data = cache.load_stock_data(cache_key)
            if cached_data:
                logger.info(f"💾 从文件缓存加载数据: {stock_code} -> {cache_key}")
                return cached_data

    logger.info(f"🌐 从Tushare数据接口获取数据: {stock_code}")

    try:
        provider = get_tdx_provider()

        # 获取历史数据
        df = provider.get_stock_history_data(stock_code, start_date, end_date)

        if df.empty:
            error_msg = f"❌ 未能获取股票 {stock_code} 的历史数据"
            print(error_msg)
            return error_msg

        # 获取实时数据
        realtime_data = provider.get_real_time_data(stock_code)

        # 获取技术指标
        indicators = provider.get_stock_technical_indicators(stock_code)

        # 格式化输出
        result = f"""
# {stock_code} 股票数据分析

## 📊 实时行情
- 股票名称: {realtime_data.get('name', 'N/A')}
- 当前价格: ¥{realtime_data.get('price', 0):.2f}
- 涨跌幅: {realtime_data.get('change_percent', 0):.2f}%
- 成交量: {realtime_data.get('volume', 0):,}手
- 更新时间: {realtime_data.get('update_time', 'N/A')}

## 📈 历史数据概览
- 数据期间: {start_date} 至 {end_date}
- 数据条数: {len(df)}条
- 期间最高: ¥{df['High'].max():.2f}
- 期间最低: ¥{df['Low'].min():.2f}
- 期间涨幅: {((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):.2f}%

## 🔍 技术指标
- MA5: ¥{indicators.get('MA5', 0):.2f}
- MA10: ¥{indicators.get('MA10', 0):.2f}
- MA20: ¥{indicators.get('MA20', 0):.2f}
- RSI: {indicators.get('RSI', 0):.2f}
- MACD: {indicators.get('MACD', 0):.4f}

## 📋 最近5日数据
{df.tail().to_string()}

数据来源: Tushare数据接口 (实时数据)
"""

        # 保存到文件缓存作为备份
        if FILE_CACHE_AVAILABLE:
            cache = get_cache()
            cache.save_stock_data(symbol=stock_code, data=result, start_date=start_date, end_date=end_date, data_source="tdx")

        return result

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error("❌ [DEBUG] Tushare数据接口调用失败:")
        logger.error(f"❌ [DEBUG] 错误类型: {type(e).__name__}")
        logger.error(f"❌ [DEBUG] 错误信息: {str(e)}")
        logger.error("❌ [DEBUG] 详细堆栈:")
        print(error_details)

        return f"""
❌ 中国股票数据获取失败 - {stock_code}
错误类型: {type(e).__name__}
错误信息: {str(e)}

🔍 调试信息:
{error_details}

💡 解决建议:
1. 检查pytdx库是否已安装: pip install pytdx
2. 确认股票代码格式正确 (如: 000001, 600519)
3. 检查网络连接是否正常
4. 尝试重新连接数据服务器

注: 数据接口需要网络连接到数据服务器
"""


def get_china_market_overview() -> str:
    """获取中国股市概览"""
    try:
        provider = get_tdx_provider()
        market_data = provider.get_market_overview()

        if not market_data:
            return "无法获取市场概览数据"

        result = "# 中国股市概览\n\n"

        for name, data in market_data.items():
            change_symbol = "📈" if data["change"] >= 0 else "📉"
            result += f"## {change_symbol} {name}\n"
            result += f"- 当前点位: {data['price']:.2f}\n"
            result += f"- 涨跌点数: {data['change']:+.2f}\n"
            result += f"- 涨跌幅: {data['change_percent']:+.2f}%\n"
            result += f"- 成交量: {data['volume']:,}\n\n"

        result += f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += "数据来源: Tushare数据接口\n"

        return result

    except Exception as e:
        return f"获取市场概览失败: {str(e)}"


# 在文件末尾添加以下函数


def get_china_stock_data_enhanced(stock_code: str, start_date: str, end_date: str) -> str:
    """
    增强版中国股票数据获取函数（完整降级机制）
    这是get_china_stock_data的增强版本

    Args:
        stock_code: 股票代码 (如 '000001')
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
    Returns:
        str: 格式化的股票数据
    """
    try:
        from .stock_data_service import get_stock_data_service

        service = get_stock_data_service()
        return service.get_stock_data_with_fallback(stock_code, start_date, end_date)
    except ImportError:
        # 如果新服务不可用，降级到原有函数
        logger.warning("⚠️ 增强服务不可用，使用原有函数")
        return get_china_stock_data(stock_code, start_date, end_date)
    except Exception as e:
        logger.warning(f"⚠️ 增强服务出错，降级到原有函数: {e}")
        return get_china_stock_data(stock_code, start_date, end_date)

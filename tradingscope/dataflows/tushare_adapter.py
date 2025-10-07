#!/usr/bin/env python3
"""
Tushare数据适配器
提供统一的中国股票数据接口，支持缓存和错误处理
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

# 导入统一日志系统
from tradingscope.utils.logging_init import get_logger

warnings.filterwarnings("ignore")

logger = get_logger("default")

# 导入Tushare工具

from .tushare_utils import get_tushare_provider

TUSHARE_AVAILABLE = True

CACHE_AVAILABLE = True


class TushareDataAdapter:
    """Tushare数据适配器"""

    def __init__(self, enable_cache: bool = True):
        """
        初始化Tushare数据适配器

        Args:
            enable_cache: 是否启用缓存
        """
        self.enable_cache = enable_cache and CACHE_AVAILABLE
        self.provider = None

        # 初始化缓存管理器
        self.cache_manager = None
        if self.enable_cache:
            try:
                from .cache_manager import get_cache

                self.cache_manager = get_cache()
            except Exception as e:
                logger.warning(f"⚠️ 缓存管理器初始化失败: {e}")
                self.enable_cache = False

        # 初始化Tushare提供器
        if TUSHARE_AVAILABLE:
            try:
                self.provider = get_tushare_provider()
                if self.provider.connected:
                    logger.info("📊 Tushare数据适配器初始化完成")
                else:
                    logger.warning("⚠️ Tushare连接失败，数据适配器功能受限")
            except Exception as e:
                logger.warning(f"⚠️ Tushare提供器初始化失败: {e}")
        else:
            logger.error("❌ Tushare不可用")

    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None, data_type: str = "daily") -> pd.DataFrame:
        """
        获取股票数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型 ("daily", "realtime")

        Returns:
            DataFrame: 股票数据
        """
        if not self.provider or not self.provider.connected:
            logger.error("❌ Tushare数据源不可用")
            return pd.DataFrame()

        try:
            logger.debug(f"🔄 获取{symbol}数据 (类型: {data_type})...")

            # 添加详细的股票代码追踪日志
            logger.info(f"🔍 [股票代码追踪] TushareAdapter.get_stock_data 接收到的股票代码: '{symbol}' (类型: {type(symbol)})")
            logger.info(f"🔍 [股票代码追踪] 股票代码长度: {len(str(symbol))}")
            logger.info(f"🔍 [股票代码追踪] 股票代码字符: {list(str(symbol))}")

            if data_type == "daily":
                logger.info(f"🔍 [股票代码追踪] 调用 _get_daily_data，传入参数: symbol='{symbol}'")
                return self._get_daily_data(symbol, start_date, end_date)
            elif data_type == "realtime":
                return self._get_realtime_data(symbol)
            else:
                logger.error(f"❌ 不支持的数据类型: {data_type}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ 获取{symbol}数据失败: {e}")
            return pd.DataFrame()

    def _get_daily_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取日线数据"""

        # 记录详细的调用信息
        logger.info("🔍 [TushareAdapter详细日志] _get_daily_data 开始执行")
        logger.info(f"🔍 [TushareAdapter详细日志] 输入参数: symbol='{symbol}', start_date='{start_date}', end_date='{end_date}'")
        logger.info(f"🔍 [TushareAdapter详细日志] 缓存启用状态: {self.enable_cache}")

        # 1. 尝试从缓存获取
        if self.enable_cache:
            try:
                logger.info("🔍 [TushareAdapter详细日志] 开始查找缓存数据...")
                cache_key = self.cache_manager.find_cached_stock_data(
                    symbol=symbol, start_date=start_date, end_date=end_date, max_age_hours=24  # 日线数据缓存24小时
                )

                if cache_key:
                    logger.info(f"🔍 [TushareAdapter详细日志] 找到缓存键: {cache_key}")
                    cached_data = self.cache_manager.load_stock_data(cache_key)
                    if cached_data is not None:
                        # 检查是否为DataFrame且不为空
                        if hasattr(cached_data, "empty") and not cached_data.empty:
                            logger.debug(f"📦 从缓存获取{symbol}数据: {len(cached_data)}条")
                            logger.info("🔍 [TushareAdapter详细日志] 缓存数据有效，确保标准化后返回")
                            # 确保缓存数据也经过标准化验证（修复KeyError: 'volume'问题）
                            return self._validate_and_standardize_data(cached_data)
                        elif isinstance(cached_data, str) and cached_data.strip():
                            logger.debug(f"📦 从缓存获取{symbol}数据: 字符串格式")
                            logger.info("🔍 [TushareAdapter详细日志] 缓存数据为字符串格式")
                            return cached_data
                        else:
                            logger.info(f"🔍 [TushareAdapter详细日志] 缓存数据无效: {type(cached_data)}")
                    else:
                        logger.info("🔍 [TushareAdapter详细日志] 缓存数据为None")
                else:
                    logger.info("🔍 [TushareAdapter详细日志] 未找到有效缓存")
            except Exception as e:
                logger.warning(f"⚠️ 缓存获取失败: {e}")
                logger.warning(f"⚠️ [TushareAdapter详细日志] 缓存异常类型: {type(e).__name__}")
        else:
            logger.info("🔍 [TushareAdapter详细日志] 缓存未启用，直接从API获取")

        # 2. 从Tushare获取数据
        logger.info(f"🔍 [股票代码追踪] _get_daily_data 调用 provider.get_stock_daily，传入参数: symbol='{symbol}'")
        logger.info("🔍 [TushareAdapter详细日志] 开始调用Tushare Provider...")

        import time

        provider_start_time = time.time()
        data = self.provider.get_stock_daily(symbol, start_date, end_date)
        provider_duration = time.time() - provider_start_time

        logger.info(f"🔍 [TushareAdapter详细日志] Provider调用完成，耗时: {provider_duration:.3f}秒")
        logger.info(f"🔍 [股票代码追踪] adapter.get_stock_data 返回数据形状: {data.shape if data is not None and hasattr(data, 'shape') else 'None'}")

        if data is not None and not data.empty:
            logger.debug(f"✅ 从Tushare获取{symbol}数据成功: {len(data)}条")
            logger.info(f"🔍 [股票代码追踪] provider.get_stock_daily 返回数据形状: {data.shape}")
            logger.info("🔍 [TushareAdapter详细日志] 数据获取成功，开始检查数据内容...")

            # 检查数据中的股票代码列
            if "ts_code" in data.columns:
                unique_codes = data["ts_code"].unique()
                logger.info(f"🔍 [股票代码追踪] 返回数据中的股票代码: {unique_codes}")
            if "symbol" in data.columns:
                unique_symbols = data["symbol"].unique()
                logger.info(f"🔍 [股票代码追踪] 返回数据中的symbol: {unique_symbols}")

            logger.info("🔍 [TushareAdapter详细日志] 开始标准化数据...")
            standardized_data = self._standardize_data(data)
            logger.info("🔍 [TushareAdapter详细日志] 数据标准化完成")
            return standardized_data
        else:
            logger.warning("⚠️ Tushare返回空数据")
            logger.warning(f"⚠️ [TushareAdapter详细日志] 空数据详情: data={data}, type={type(data)}")
            if data is not None:
                logger.warning(f"⚠️ [TushareAdapter详细日志] DataFrame为空: {data.empty}")
            return pd.DataFrame()

    def _get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """获取实时数据（使用最新日线数据）"""

        # Tushare免费版不支持实时数据，使用最新日线数据
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        data = self.provider.get_stock_daily(symbol, start_date, end_date)

        if data is not None and not data.empty:
            # 返回最新一条数据
            latest_data = data.tail(1)
            logger.debug(f"✅ 从Tushare获取{symbol}最新数据")
            return self._standardize_data(latest_data)
        else:
            logger.warning(f"⚠️ 无法获取{symbol}实时数据")
            return pd.DataFrame()

    def _validate_and_standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证并标准化数据格式，增强版本（修复KeyError: 'volume'问题）"""
        if data.empty:
            logger.info("🔍 [数据标准化] 输入数据为空，直接返回")
            return data

        try:
            logger.info(f"🔍 [数据标准化] 开始标准化数据，输入列名: {list(data.columns)}")

            # 复制数据避免修改原始数据
            standardized = data.copy()

            # 列名映射
            column_mapping = {
                "trade_date": "date",
                "ts_code": "code",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",  # 关键映射：vol -> volume
                "amount": "amount",
                "pct_chg": "pct_change",
                "change": "change",
            }

            # 记录映射过程
            mapped_columns = []

            # 重命名列
            for old_col, new_col in column_mapping.items():
                if old_col in standardized.columns:
                    standardized = standardized.rename(columns={old_col: new_col})
                    mapped_columns.append(f"{old_col}->{new_col}")
                    logger.debug(f"🔄 [数据标准化] 列映射: {old_col} -> {new_col}")

            logger.info(f"🔍 [数据标准化] 完成列映射: {mapped_columns}")

            # 验证关键列是否存在，添加备用处理
            required_columns = ["volume", "close", "high", "low"]
            missing_columns = [col for col in required_columns if col not in standardized.columns]
            if missing_columns:
                logger.warning(f"⚠️ [数据标准化] 缺少关键列: {missing_columns}")
                self._add_fallback_columns(standardized, missing_columns, data)

            # 确保日期列存在且格式正确
            if "date" in standardized.columns:
                standardized["date"] = pd.to_datetime(standardized["date"])
                standardized = standardized.sort_values("date")
                logger.debug("✅ [数据标准化] 日期列格式化完成")

            # 添加股票代码列（如果不存在）
            if "code" in standardized.columns and "股票代码" not in standardized.columns:
                standardized["股票代码"] = standardized["code"].str.replace(".SH", "").str.replace(".SZ", "").str.replace(".BJ", "")
                logger.debug("✅ [数据标准化] 股票代码列添加完成")

            # 添加涨跌幅列（如果不存在）
            if "pct_change" in standardized.columns and "涨跌幅" not in standardized.columns:
                standardized["涨跌幅"] = standardized["pct_change"]
                logger.debug("✅ [数据标准化] 涨跌幅列添加完成")

            logger.info("✅ [数据标准化] 数据标准化完成")
            return standardized

        except Exception as e:
            logger.error(f"❌ [数据标准化] 数据标准化失败: {e}", exc_info=True)
            logger.error(f"❌ [数据标准化] 原始数据列名: {list(data.columns) if not data.empty else '空数据'}")
            return data

    def _add_fallback_columns(self, standardized: pd.DataFrame, missing_columns: list, original_data: pd.DataFrame):
        """为缺失的关键列添加备用值"""
        try:
            import numpy as np

            for col in missing_columns:
                if col == "volume":
                    # 尝试寻找可能的成交量列名
                    volume_candidates = ["vol", "volume", "turnover", "trade_volume"]
                    for candidate in volume_candidates:
                        if candidate in original_data.columns:
                            standardized["volume"] = original_data[candidate]
                            logger.info(f"✅ [数据标准化] 使用备用列 {candidate} 作为 volume")
                            break
                    else:
                        # 如果找不到任何成交量列，设置为0
                        standardized["volume"] = 0
                        logger.warning("⚠️ [数据标准化] 未找到成交量数据，设置为0")

                elif col in ["close", "high", "low", "open"]:
                    # 对于价格列，如果缺失则设置为NaN
                    if col not in standardized.columns:
                        standardized[col] = np.nan
                        logger.warning(f"⚠️ [数据标准化] 缺失价格列 {col}，设置为NaN")

        except Exception as e:
            logger.error(f"❌ [数据标准化] 添加备用列失败: {e}")

    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据格式 - 保持向后兼容性，调用增强版本"""
        return self._validate_and_standardize_data(data)

    def get_stock_info(self, symbol: str) -> Dict:
        """
        获取股票基本信息

        Args:
            symbol: 股票代码

        Returns:
            Dict: 股票基本信息
        """
        if not self.provider or not self.provider.connected:
            return {"symbol": symbol, "name": f"股票{symbol}", "source": "unknown"}

        try:
            info = self.provider.get_stock_info(symbol)
            if info and info.get("name") and info.get("name") != f"股票{symbol}":
                logger.debug(f"✅ 从Tushare获取{symbol}基本信息成功")
                return info
            else:
                return {"symbol": symbol, "name": f"股票{symbol}", "source": "unknown"}

        except Exception as e:
            logger.error(f"❌ 获取{symbol}股票信息失败: {e}")
            return {"symbol": symbol, "name": f"股票{symbol}", "source": "unknown"}

    def search_stocks(self, keyword: str) -> pd.DataFrame:
        """
        搜索股票

        Args:
            keyword: 搜索关键词

        Returns:
            DataFrame: 搜索结果
        """
        if not self.provider or not self.provider.connected:
            logger.error("❌ Tushare数据源不可用")
            return pd.DataFrame()

        try:
            results = self.provider.search_stocks(keyword)

            if results is not None and not results.empty:
                logger.debug(f"✅ 搜索'{keyword}'成功: {len(results)}条结果")
                return results
            else:
                logger.warning(f"⚠️ 未找到匹配'{keyword}'的股票")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ 搜索股票失败: {e}")
            return pd.DataFrame()

    def get_fundamentals(self, symbol: str) -> str:
        """
        获取基本面数据

        Args:
            symbol: 股票代码

        Returns:
            str: 基本面分析报告
        """
        if not self.provider or not self.provider.connected:
            return f"❌ Tushare数据源不可用，无法获取{symbol}基本面数据"

        try:
            logger.debug(f"📊 获取{symbol}基本面数据...")

            # 获取股票基本信息
            stock_info = self.get_stock_info(symbol)

            # 获取财务数据
            financial_data = self.provider.get_financial_data(symbol)

            # 生成基本面分析报告
            report = self._generate_fundamentals_report(symbol, stock_info, financial_data)

            # 缓存基本面数据
            if self.enable_cache and self.cache_manager:
                try:
                    cache_key = self.cache_manager.save_fundamentals_data(symbol=symbol, fundamentals_data=report, data_source="tushare_analysis")
                    logger.debug(f"💼 A股基本面数据已缓存: {symbol} (tushare_analysis) -> {cache_key}")
                except Exception as e:
                    logger.warning(f"⚠️ 基本面数据缓存失败: {e}")

            return report

        except Exception as e:
            logger.error(f"❌ 获取{symbol}基本面数据失败: {e}")
            return f"❌ 获取{symbol}基本面数据失败: {e}"

    def _generate_fundamentals_report(self, symbol: str, stock_info: Dict, financial_data: Dict) -> str:
        """生成基本面分析报告"""

        report = f"📊 {symbol} 基本面分析报告 (Tushare数据源)\n"
        report += "=" * 50 + "\n\n"

        # 基本信息
        report += "📋 基本信息\n"
        report += f"股票代码: {symbol}\n"
        report += f"股票名称: {stock_info.get('name', '未知')}\n"
        report += f"所属地区: {stock_info.get('area', '未知')}\n"
        report += f"所属行业: {stock_info.get('industry', '未知')}\n"
        report += f"上市市场: {stock_info.get('market', '未知')}\n"
        report += f"上市日期: {stock_info.get('list_date', '未知')}\n\n"

        # 财务数据
        if financial_data:
            report += "💰 财务数据\n"

            # 资产负债表
            balance_sheet = financial_data.get("balance_sheet", [])
            if balance_sheet:
                latest_balance = balance_sheet[0] if balance_sheet else {}
                report += f"总资产: {latest_balance.get('total_assets', 'N/A')}\n"
                report += f"总负债: {latest_balance.get('total_liab', 'N/A')}\n"
                report += f"股东权益: {latest_balance.get('total_hldr_eqy_exc_min_int', 'N/A')}\n"

            # 利润表
            income_statement = financial_data.get("income_statement", [])
            if income_statement:
                latest_income = income_statement[0] if income_statement else {}
                report += f"营业收入: {latest_income.get('total_revenue', 'N/A')}\n"
                report += f"营业利润: {latest_income.get('operate_profit', 'N/A')}\n"
                report += f"净利润: {latest_income.get('n_income', 'N/A')}\n"

            # 现金流量表
            cash_flow = financial_data.get("cash_flow", [])
            if cash_flow:
                latest_cash = cash_flow[0] if cash_flow else {}
                report += f"经营活动现金流: {latest_cash.get('c_fr_sale_sg', 'N/A')}\n"
        else:
            report += "💰 财务数据: 暂无数据\n"

        report += f"\n📅 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "📊 数据来源: Tushare\n"

        return report


# 全局适配器实例
_tushare_adapter = None


def get_tushare_adapter() -> TushareDataAdapter:
    """获取全局Tushare数据适配器实例"""
    global _tushare_adapter
    if _tushare_adapter is None:
        _tushare_adapter = TushareDataAdapter()
    return _tushare_adapter


def get_china_stock_data_tushare_adapter(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取中国股票数据的便捷函数（Tushare适配器）

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame: 股票数据
    """
    adapter = get_tushare_adapter()
    return adapter.get_stock_data(symbol, start_date, end_date)


def get_china_stock_info_tushare_adapter(symbol: str) -> Dict:
    """
    获取中国股票信息的便捷函数（Tushare适配器）

    Args:
        symbol: 股票代码

    Returns:
        Dict: 股票信息
    """
    adapter = get_tushare_adapter()
    return adapter.get_stock_info(symbol)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的股票数据获取服务
实现Tushare数据接口的完整降级机制
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

# 导入日志模块
from tradingscope.utils.logging_manager import get_logger

logger = get_logger("agents")

try:
    from .tdx_utils import TongDaXinDataProvider, get_tdx_provider

    TDX_AVAILABLE = True
except ImportError:
    TDX_AVAILABLE = False

# 尝试导入增强的股票列表获取器
try:
    import sys
    from pathlib import Path

    # 添加工具目录到路径
    utils_path = Path(__file__).parent.parent / "utils"
    if str(utils_path) not in sys.path:
        sys.path.append(utils_path)
    from enhanced_stock_list_fetcher import enhanced_fetch_stock_list

    ENHANCED_FETCHER_AVAILABLE = True
except ImportError:
    ENHANCED_FETCHER_AVAILABLE = False

logger = logging.getLogger(__name__)


class StockDataService:
    """
    统一的股票数据获取服务
    实现完整的降级机制：Tushare数据接口 -> 缓存 -> 错误处理
    """

    def __init__(self):
        self.tdx_provider = None
        self._init_services()

    def _init_services(self):
        """初始化服务"""
        # 尝试初始化通达信提供器
        if TDX_AVAILABLE:
            try:
                self.tdx_provider = get_tdx_provider()
                logger.info("✅ Tushare数据接口初始化成功")
            except Exception as e:
                logger.error(f"⚠️ Tushare数据接口初始化失败: {e}")
                self.tdx_provider = None

    def get_stock_basic_info(self, stock_code: str = None) -> Optional[Dict[str, Any]]:
        """
        获取股票基础信息（单个股票或全部股票）

        Args:
            stock_code: 股票代码，如果为None则返回所有股票

        Returns:
            Dict: 股票基础信息
        """
        logger.info(f"📊 获取股票基础信息: {stock_code or '全部股票'}")

        # 直接从Tushare数据接口获取
        logger.info("🔄 从Tushare数据接口获取数据")
        if ENHANCED_FETCHER_AVAILABLE:
            try:
                result = self._get_from_tdx_api(stock_code)
                if result:
                    logger.info(f"✅ 从Tushare数据接口获取成功: {len(result) if isinstance(result, list) else 1}条记录")
                    return result
            except Exception as e:
                logger.error(f"⚠️ Tushare数据接口查询失败: {e}")

        # 最后的降级方案
        logger.error("❌ 所有数据源都不可用")
        return self._get_fallback_data(stock_code)

    def _get_from_tdx_api(self, stock_code: str = None) -> Optional[Dict[str, Any]]:
        """从Tushare数据接口获取数据"""
        try:
            if stock_code:
                # 获取单个股票信息
                if self.tdx_provider:
                    # 使用现有的股票名称获取方法
                    stock_name = self.tdx_provider._get_stock_name(stock_code)
                    return {
                        "code": stock_code,
                        "name": stock_name,
                        "market": self._get_market_name(stock_code),
                        "category": self._get_stock_category(stock_code),
                        "source": "tdx_api",
                        "updated_at": datetime.now().isoformat(),
                    }
            else:
                # 获取所有股票列表
                stock_df = enhanced_fetch_stock_list(type_="stock", enable_server_failover=True, max_retries=3)

                if stock_df is not None and not stock_df.empty:
                    # 转换为字典列表
                    results = []
                    for _, row in stock_df.iterrows():
                        results.append(
                            {
                                "code": row.get("code", ""),
                                "name": row.get("name", ""),
                                "market": row.get("market", ""),
                                "category": row.get("category", ""),
                                "source": "tdx_api",
                                "updated_at": datetime.now().isoformat(),
                            }
                        )
                    return results

        except Exception as e:
            logger.error(f"Tushare数据接口查询失败: {e}")
            return None

    def _get_fallback_data(self, stock_code: str = None) -> Dict[str, Any]:
        """最后的降级数据"""
        if stock_code:
            return {
                "code": stock_code,
                "name": f"股票{stock_code}",
                "market": self._get_market_name(stock_code),
                "category": "未知",
                "source": "fallback",
                "updated_at": datetime.now().isoformat(),
                "error": "所有数据源都不可用",
            }
        else:
            return {"error": "无法获取股票列表，请检查网络连接", "suggestion": "请确保网络连接正常以访问Tushare数据接口"}

    def _get_market_name(self, stock_code: str) -> str:
        """根据股票代码判断市场"""
        if stock_code.startswith(("60", "68", "90")):
            return "上海"
        elif stock_code.startswith(("00", "30", "20")):
            return "深圳"
        else:
            return "未知"

    def _get_stock_category(self, stock_code: str) -> str:
        """根据股票代码判断类别"""
        if stock_code.startswith("60"):
            return "沪市主板"
        elif stock_code.startswith("68"):
            return "科创板"
        elif stock_code.startswith("00"):
            return "深市主板"
        elif stock_code.startswith("30"):
            return "创业板"
        elif stock_code.startswith("20"):
            return "深市B股"
        else:
            return "其他"

    def get_stock_data_with_fallback(self, stock_code: str, start_date: str, end_date: str) -> str:
        """
        获取股票数据（带降级机制）
        这是对现有get_china_stock_data函数的增强
        """
        logger.info(f"📊 获取股票数据: {stock_code} ({start_date} 到 {end_date})")

        # 首先确保股票基础信息可用
        stock_info = self.get_stock_basic_info(stock_code)
        if stock_info and "error" in stock_info:
            return f"❌ 无法获取股票{stock_code}的基础信息: {stock_info.get('error', '未知错误')}"

        # 调用现有的get_china_stock_data函数
        try:
            from .tdx_utils import get_china_stock_data

            return get_china_stock_data(stock_code, start_date, end_date)
        except Exception as e:
            return f"❌ 获取股票数据失败: {str(e)}\n\n💡 建议：\n1. 检查网络连接\n2. 确认股票代码格式正确"


# 全局服务实例
_stock_data_service = None


def get_stock_data_service() -> StockDataService:
    """获取股票数据服务实例（单例模式）"""
    global _stock_data_service
    if _stock_data_service is None:
        _stock_data_service = StockDataService()
    return _stock_data_service

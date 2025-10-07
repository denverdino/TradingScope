#!/usr/bin/env python3
"""
改进的港股数据获取工具
解决API速率限制和数据获取问题
"""

import json
import os
import time
from typing import Any, Dict

# 导入统一日志系统
from tradingscope.utils.logging_init import get_logger

logger = get_logger("default")


class ImprovedHKStockProvider:
    """改进的港股数据提供器"""

    def __init__(self):
        self.cache_file = "hk_stock_cache.json"
        self.cache_ttl = 3600 * 24  # 24小时缓存
        self.rate_limit_wait = 5  # 速率限制等待时间
        self.last_request_time = 0

        # 内置港股名称映射（避免API调用）
        self.hk_stock_names = {
            # 腾讯系
            "0700.HK": "腾讯控股",
            "0700": "腾讯控股",
            "00700": "腾讯控股",
            # 电信运营商
            "0941.HK": "中国移动",
            "0941": "中国移动",
            "00941": "中国移动",
            "0762.HK": "中国联通",
            "0762": "中国联通",
            "00762": "中国联通",
            "0728.HK": "中国电信",
            "0728": "中国电信",
            "00728": "中国电信",
            # 银行
            "0939.HK": "建设银行",
            "0939": "建设银行",
            "00939": "建设银行",
            "1398.HK": "工商银行",
            "1398": "工商银行",
            "01398": "工商银行",
            "3988.HK": "中国银行",
            "3988": "中国银行",
            "03988": "中国银行",
            "0005.HK": "汇丰控股",
            "0005": "汇丰控股",
            "00005": "汇丰控股",
            # 保险
            "1299.HK": "友邦保险",
            "1299": "友邦保险",
            "01299": "友邦保险",
            "2318.HK": "中国平安",
            "2318": "中国平安",
            "02318": "中国平安",
            "2628.HK": "中国人寿",
            "2628": "中国人寿",
            "02628": "中国人寿",
            # 石油化工
            "0857.HK": "中国石油",
            "0857": "中国石油",
            "00857": "中国石油",
            "0386.HK": "中国石化",
            "0386": "中国石化",
            "00386": "中国石化",
            # 地产
            "1109.HK": "华润置地",
            "1109": "华润置地",
            "01109": "华润置地",
            "1997.HK": "九龙仓置业",
            "1997": "九龙仓置业",
            "01997": "九龙仓置业",
            # 科技
            "9988.HK": "阿里巴巴",
            "9988": "阿里巴巴",
            "09988": "阿里巴巴",
            "3690.HK": "美团",
            "3690": "美团",
            "03690": "美团",
            "1024.HK": "快手",
            "1024": "快手",
            "01024": "快手",
            "9618.HK": "京东集团",
            "9618": "京东集团",
            "09618": "京东集团",
            # 消费
            "1876.HK": "百威亚太",
            "1876": "百威亚太",
            "01876": "百威亚太",
            "0291.HK": "华润啤酒",
            "0291": "华润啤酒",
            "00291": "华润啤酒",
            # 医药
            "1093.HK": "石药集团",
            "1093": "石药集团",
            "01093": "石药集团",
            "0867.HK": "康师傅",
            "0867": "康师傅",
            "00867": "康师傅",
            # 汽车
            "2238.HK": "广汽集团",
            "2238": "广汽集团",
            "02238": "广汽集团",
            "1211.HK": "比亚迪",
            "1211": "比亚迪",
            "01211": "比亚迪",
            # 航空
            "0753.HK": "中国国航",
            "0753": "中国国航",
            "00753": "中国国航",
            "0670.HK": "中国东航",
            "0670": "中国东航",
            "00670": "中国东航",
            # 钢铁
            "0347.HK": "鞍钢股份",
            "0347": "鞍钢股份",
            "00347": "鞍钢股份",
            # 电力
            "0902.HK": "华能国际",
            "0902": "华能国际",
            "00902": "华能国际",
            "0991.HK": "大唐发电",
            "0991": "大唐发电",
            "00991": "大唐发电",
        }

        self._load_cache()

    def _load_cache(self):
        """加载缓存"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            logger.debug(f"📊 [港股缓存] 加载缓存失败: {e}")
            self.cache = {}

    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"📊 [港股缓存] 保存缓存失败: {e}")

    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self.cache:
            return False

        cache_time = self.cache[key].get("timestamp", 0)
        return (time.time() - cache_time) < self.cache_ttl

    def _normalize_hk_symbol(self, symbol: str) -> str:
        """标准化港股代码"""
        # 移除.HK后缀
        clean_symbol = symbol.replace(".HK", "").replace(".hk", "")

        # 补齐到5位数字
        if len(clean_symbol) == 4:
            clean_symbol = "0" + clean_symbol
        elif len(clean_symbol) == 3:
            clean_symbol = "00" + clean_symbol
        elif len(clean_symbol) == 2:
            clean_symbol = "000" + clean_symbol
        elif len(clean_symbol) == 1:
            clean_symbol = "0000" + clean_symbol

        return clean_symbol

    def get_company_name(self, symbol: str) -> str:
        """
        获取港股公司名称

        Args:
            symbol: 港股代码

        Returns:
            str: 公司名称
        """
        try:
            # 检查缓存
            cache_key = f"name_{symbol}"
            if self._is_cache_valid(cache_key):
                cached_name = self.cache[cache_key]["data"]
                logger.debug(f"📊 [港股缓存] 从缓存获取公司名称: {symbol} -> {cached_name}")
                return cached_name

            # 方案1：使用内置映射
            normalized_symbol = self._normalize_hk_symbol(symbol)

            # 尝试多种格式匹配
            for format_symbol in [symbol, normalized_symbol, f"{normalized_symbol}.HK"]:
                if format_symbol in self.hk_stock_names:
                    company_name = self.hk_stock_names[format_symbol]

                    # 缓存结果
                    self.cache[cache_key] = {"data": company_name, "timestamp": time.time(), "source": "builtin_mapping"}
                    self._save_cache()

                    logger.debug(f"📊 [港股映射] 获取公司名称: {symbol} -> {company_name}")
                    return company_name

            # 方案2：优先尝试AKShare API获取（有速率限制保护）
            try:
                # 速率限制保护
                current_time = time.time()
                if current_time - self.last_request_time < self.rate_limit_wait:
                    wait_time = self.rate_limit_wait - (current_time - self.last_request_time)
                    logger.debug(f"📊 [港股API] 速率限制保护，等待 {wait_time:.1f} 秒")
                    time.sleep(wait_time)

                self.last_request_time = time.time()

                # 优先尝试AKShare获取
                try:
                    from tradingscope.dataflows.akshare_utils import get_hk_stock_info_akshare

                    logger.debug(f"📊 [港股API] 优先使用AKShare获取: {symbol}")

                    akshare_info = get_hk_stock_info_akshare(symbol)
                    if akshare_info and isinstance(akshare_info, dict) and "name" in akshare_info:
                        akshare_name = akshare_info["name"]
                        if not akshare_name.startswith("港股"):
                            # 缓存AKShare结果
                            self.cache[cache_key] = {"data": akshare_name, "timestamp": time.time(), "source": "akshare_api"}
                            self._save_cache()

                            logger.debug(f"📊 [港股AKShare] 获取公司名称: {symbol} -> {akshare_name}")
                            return akshare_name
                except Exception as e:
                    logger.debug(f"📊 [港股AKShare] AKShare获取失败: {e}")

                # 备用：尝试从统一接口获取（包含Yahoo Finance）
                from tradingscope.dataflows.interface import get_hk_stock_info_unified

                hk_info = get_hk_stock_info_unified(symbol)

                if hk_info and isinstance(hk_info, dict) and "name" in hk_info:
                    api_name = hk_info["name"]
                    if not api_name.startswith("港股"):
                        # 缓存API结果
                        self.cache[cache_key] = {"data": api_name, "timestamp": time.time(), "source": "unified_api"}
                        self._save_cache()

                        logger.debug(f"📊 [港股统一API] 获取公司名称: {symbol} -> {api_name}")
                        return api_name

            except Exception as e:
                logger.debug(f"📊 [港股API] API获取失败: {e}")

            # 方案3：生成友好的默认名称
            clean_symbol = self._normalize_hk_symbol(symbol)
            default_name = f"港股{clean_symbol}"

            # 缓存默认结果（较短的TTL）
            self.cache[cache_key] = {"data": default_name, "timestamp": time.time() - self.cache_ttl + 3600, "source": "default"}  # 1小时后过期
            self._save_cache()

            logger.debug(f"📊 [港股默认] 使用默认名称: {symbol} -> {default_name}")
            return default_name

        except Exception as e:
            logger.error(f"❌ [港股] 获取公司名称失败: {e}")
            clean_symbol = self._normalize_hk_symbol(symbol)
            return f"港股{clean_symbol}"

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取港股基本信息

        Args:
            symbol: 港股代码

        Returns:
            Dict: 港股信息
        """
        try:
            company_name = self.get_company_name(symbol)

            return {"symbol": symbol, "name": company_name, "currency": "HKD", "exchange": "HKG", "market": "港股", "source": "improved_hk_provider"}

        except Exception as e:
            logger.error(f"❌ [港股] 获取股票信息失败: {e}")
            clean_symbol = self._normalize_hk_symbol(symbol)
            return {"symbol": symbol, "name": f"港股{clean_symbol}", "currency": "HKD", "exchange": "HKG", "market": "港股", "source": "error", "error": str(e)}


# 全局实例
_improved_hk_provider = None


def get_improved_hk_provider() -> ImprovedHKStockProvider:
    """获取改进的港股提供器实例"""
    global _improved_hk_provider
    if _improved_hk_provider is None:
        _improved_hk_provider = ImprovedHKStockProvider()
    return _improved_hk_provider


def get_hk_company_name_improved(symbol: str) -> str:
    """
    获取港股公司名称的改进版本

    Args:
        symbol: 港股代码

    Returns:
        str: 公司名称
    """
    provider = get_improved_hk_provider()
    return provider.get_company_name(symbol)


def get_hk_stock_info_improved(symbol: str) -> Dict[str, Any]:
    """
    获取港股信息的改进版本

    Args:
        symbol: 港股代码

    Returns:
        Dict: 港股信息
    """
    provider = get_improved_hk_provider()
    return provider.get_stock_info(symbol)

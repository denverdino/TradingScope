#!/usr/bin/env python3
"""
文件缓存系统
提供基于文件的缓存功能
"""

import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class FileCacheSystem:
    """文件缓存系统"""

    def __init__(self, cache_dir: str = "data/cache"):
        self.logger = logging.getLogger(__name__)

        # 设置缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("文件缓存系统初始化")

    def _get_cache_key(self, symbol: str, start_date: str = "", end_date: str = "", data_source: str = "default", data_type: str = "stock_data") -> str:
        """生成缓存键"""
        key_data = f"{symbol}_{start_date}_{end_date}_{data_source}_{data_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_ttl_seconds(self, symbol: str, data_type: str = "stock_data") -> int:
        """获取TTL秒数"""
        # 默认TTL为2小时
        return 7200

    def _is_cache_valid(self, cache_time: datetime, ttl_seconds: int) -> bool:
        """检查缓存是否有效"""
        if cache_time is None:
            return False

        expiry_time = cache_time + timedelta(seconds=ttl_seconds)
        return datetime.now() < expiry_time

    def _save_to_file(self, cache_key: str, data: Any, metadata: Dict) -> bool:
        """保存到文件缓存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_data = {"data": data, "metadata": metadata, "timestamp": datetime.now(), "backend": "file"}

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            self.logger.debug(f"文件缓存保存成功: {cache_key}")
            return True

        except Exception as e:
            self.logger.error(f"文件缓存保存失败: {e}")
            return False

    def _load_from_file(self, cache_key: str) -> Optional[Dict]:
        """从文件缓存加载"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if not cache_file.exists():
                return None

            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            self.logger.debug(f"文件缓存加载成功: {cache_key}")
            return cache_data

        except Exception as e:
            self.logger.error(f"文件缓存加载失败: {e}")
            return None

    def save_data(self, symbol: str, data: Any, start_date: str = "", end_date: str = "", data_source: str = "default", data_type: str = "stock_data") -> str:
        """保存数据到缓存"""
        # 生成缓存键
        cache_key = self._get_cache_key(symbol, start_date, end_date, data_source, data_type)

        # 准备元数据
        metadata = {"symbol": symbol, "start_date": start_date, "end_date": end_date, "data_source": data_source, "data_type": data_type}

        # 保存到文件缓存
        success = self._save_to_file(cache_key, data, metadata)

        if success:
            self.logger.info(f"数据缓存成功: {symbol} -> {cache_key}")
        else:
            self.logger.error(f"数据缓存失败: {symbol}")

        return cache_key

    def load_data(self, cache_key: str) -> Optional[Any]:
        """从缓存加载数据"""
        cache_data = self._load_from_file(cache_key)

        if not cache_data:
            return None

        # 检查缓存是否有效
        symbol = cache_data["metadata"].get("symbol", "")
        data_type = cache_data["metadata"].get("data_type", "stock_data")
        ttl_seconds = self._get_ttl_seconds(symbol, data_type)

        if not self._is_cache_valid(cache_data["timestamp"], ttl_seconds):
            # 删除过期的缓存文件
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            self.logger.debug(f"文件缓存已过期: {cache_key}")
            return None

        return cache_data["data"]

    def find_cached_data(
        self, symbol: str, start_date: str = "", end_date: str = "", data_source: str = "default", data_type: str = "stock_data"
    ) -> Optional[str]:
        """查找缓存的数据"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, data_source, data_type)

        # 检查缓存是否存在且有效
        if self.load_data(cache_key) is not None:
            return cache_key

        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "file_cache_directory": str(self.cache_dir),
            "file_cache_count": len(list(self.cache_dir.glob("*.pkl"))),
        }

        return stats

    def clear_expired_cache(self):
        """清理过期缓存"""
        self.logger.info("开始清理过期缓存...")

        # 清理文件缓存
        cleared_files = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                symbol = cache_data["metadata"].get("symbol", "")
                data_type = cache_data["metadata"].get("data_type", "stock_data")
                ttl_seconds = self._get_ttl_seconds(symbol, data_type)

                if not self._is_cache_valid(cache_data["timestamp"], ttl_seconds):
                    cache_file.unlink()
                    cleared_files += 1

            except Exception as e:
                self.logger.error(f"清理缓存文件失败 {cache_file}: {e}")

        self.logger.info(f"文件缓存清理完成，删除 {cleared_files} 个过期文件")


# 全局缓存系统实例
_cache_system = None


def get_cache_system() -> FileCacheSystem:
    """获取全局文件缓存系统实例"""
    global _cache_system
    if _cache_system is None:
        _cache_system = FileCacheSystem()
    return _cache_system

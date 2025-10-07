#!/usr/bin/env python3
"""
统一新闻分析工具
整合A股、港股、美股等不同市场的新闻获取逻辑到一个工具函数中
让大模型只需要调用一个工具就能获取所有类型股票的新闻数据
"""

import logging
import re
from datetime import datetime

from .interface import get_finnhub_news, get_global_news_openai, get_google_news
from .realtime_news_utils import get_realtime_stock_news

logger = logging.getLogger(__name__)


class UnifiedNewsAnalyzer:
    """统一新闻分析器，整合所有新闻获取逻辑"""

    def __init__(self):
        """初始化统一新闻分析器"""

    def get_stock_news_unified(self, stock_code: str, max_news: int = 10, model_info: str = "") -> str:
        """
        统一新闻获取接口
        根据股票代码自动识别股票类型并获取相应新闻

        Args:
            stock_code: 股票代码
            max_news: 最大新闻数量
            model_info: 当前使用的模型信息，用于特殊处理

        Returns:
            str: 格式化的新闻内容
        """
        logger.info(f"[统一新闻工具] 开始获取 {stock_code} 的新闻，模型: {model_info}")
        logger.info(f"[统一新闻工具] 🤖 当前模型信息: {model_info}")

        # 识别股票类型
        stock_type = self._identify_stock_type(stock_code)
        logger.info(f"[统一新闻工具] 股票类型: {stock_type}")

        # 根据股票类型调用相应的获取方法
        if stock_type == "A股":
            result = self._get_a_share_news(stock_code, max_news, model_info)
        elif stock_type == "港股":
            result = self._get_hk_share_news(stock_code, max_news, model_info)
        elif stock_type == "美股":
            result = self._get_us_share_news(stock_code, max_news, model_info)
        else:
            # 默认使用A股逻辑
            result = self._get_a_share_news(stock_code, max_news, model_info)

        # 🔍 添加详细的结果调试日志
        logger.info(f"[统一新闻工具] 📊 新闻获取完成，结果长度: {len(result)} 字符")
        logger.info(f"[统一新闻工具] 📋 返回结果预览 (前1000字符): {result[:1000]}")

        # 如果结果为空或过短，记录警告
        if not result or len(result.strip()) < 50:
            logger.warning("[统一新闻工具] ⚠️ 返回结果异常短或为空！")
            logger.warning(f"[统一新闻工具] 📝 完整结果内容: '{result}'")

        return result

    def _identify_stock_type(self, stock_code: str) -> str:
        """识别股票类型"""
        stock_code = stock_code.upper().strip()

        # A股判断
        if re.match(r"^(00|30|60|68)\d{4}$", stock_code):
            return "A股"
        elif re.match(r"^(SZ|SH)\d{6}$", stock_code):
            return "A股"

        # 港股判断
        elif re.match(r"^\d{4,5}\.HK$", stock_code):
            return "港股"
        elif re.match(r"^\d{4,5}$", stock_code) and len(stock_code) <= 5:
            return "港股"

        # 美股判断
        elif re.match(r"^[A-Z]{1,5}$", stock_code):
            return "美股"
        elif "." in stock_code and not stock_code.endswith(".HK"):
            return "美股"

        # 默认按A股处理
        else:
            return "A股"

    def _get_a_share_news(self, stock_code: str, max_news: int, model_info: str = "") -> str:
        """获取A股新闻"""
        logger.info(f"[统一新闻工具] 获取A股 {stock_code} 新闻")

        # 获取当前日期
        curr_date = datetime.now().strftime("%Y-%m-%d")

        # 优先级1: 东方财富实时新闻
        try:
            logger.info("[统一新闻工具] 尝试东方财富实时新闻...")
            result = get_realtime_stock_news(ticker=stock_code, current_date=curr_date)

            # 🔍 详细记录东方财富返回的内容
            logger.info(f"[统一新闻工具] 📊 东方财富返回内容长度: {len(result) if result else 0} 字符")
            logger.info(f"[统一新闻工具] 📋 东方财富返回内容预览 (前500字符): {result[:500] if result else 'None'}")

            if result and len(result.strip()) > 100:
                logger.info(f"[统一新闻工具] ✅ 东方财富新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "东方财富实时新闻", model_info)
            else:
                logger.warning("[统一新闻工具] ⚠️ 东方财富新闻内容过短或为空")
        except Exception as e:
            logger.warning(f"[统一新闻工具] 东方财富新闻获取失败: {e}")

        # 优先级2: Google新闻（中文搜索）
        try:
            logger.info("[统一新闻工具] 尝试Google新闻...")
            query = f"{stock_code} 股票 新闻 财报 业绩"
            result = get_google_news(query=query, curr_date=curr_date)
            if result and len(result.strip()) > 50:
                logger.info(f"[统一新闻工具] ✅ Google新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "Google新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] Google新闻获取失败: {e}")

        # 优先级3: OpenAI全球新闻
        try:
            logger.info("[统一新闻工具] 尝试OpenAI全球新闻...")
            result = get_global_news_openai(curr_date=curr_date)
            if result and len(result.strip()) > 50:
                logger.info(f"[统一新闻工具] ✅ OpenAI新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "OpenAI全球新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] OpenAI新闻获取失败: {e}")

        return "❌ 无法获取A股新闻数据，所有新闻源均不可用"

    def _get_hk_share_news(self, stock_code: str, max_news: int, model_info: str = "") -> str:
        """获取港股新闻"""
        logger.info(f"[统一新闻工具] 获取港股 {stock_code} 新闻")

        # 获取当前日期
        curr_date = datetime.now().strftime("%Y-%m-%d")

        # 优先级1: Google新闻（港股搜索）
        try:
            logger.info("[统一新闻工具] 尝试Google港股新闻...")
            query = f"{stock_code} 港股 香港股票 新闻"
            # 使用LangChain工具的正确调用方式：.invoke()方法和字典参数
            result = self.toolkit.get_google_news.invoke(query=query, curr_date=curr_date)
            if result and len(result.strip()) > 50:
                logger.info(f"[统一新闻工具] ✅ Google港股新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "Google港股新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] Google港股新闻获取失败: {e}")

        # 优先级2: OpenAI全球新闻
        try:
            logger.info("[统一新闻工具] 尝试OpenAI港股新闻...")
            result = get_global_news_openai(curr_date=curr_date)
            if result and len(result.strip()) > 50:
                logger.info(f"[统一新闻工具] ✅ OpenAI港股新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "OpenAI港股新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] OpenAI港股新闻获取失败: {e}")

        # 优先级3: 实时新闻（如果支持港股）
        try:
            logger.info("[统一新闻工具] 尝试实时港股新闻...")
            result = get_realtime_stock_news(ticker=stock_code, curr_date=curr_date)
            if result and len(result.strip()) > 100:
                logger.info(f"[统一新闻工具] ✅ 实时港股新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "实时港股新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] 实时港股新闻获取失败: {e}")

        return "❌ 无法获取港股新闻数据，所有新闻源均不可用"

    def _get_us_share_news(self, stock_code: str, max_news: int, model_info: str = "") -> str:
        """获取美股新闻"""
        logger.info(f"[统一新闻工具] 获取美股 {stock_code} 新闻")

        # 获取当前日期
        curr_date = datetime.now().strftime("%Y-%m-%d")

        # 优先级1: OpenAI全球新闻
        try:
            logger.info("[统一新闻工具] 尝试OpenAI美股新闻...")
            # 使用LangChain工具的正确调用方式：.invoke()方法和字典参数
            result = get_global_news_openai(curr_date=curr_date)
            if result and len(result.strip()) > 50:
                logger.info(f"[统一新闻工具] ✅ OpenAI美股新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "OpenAI美股新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] OpenAI美股新闻获取失败: {e}")

        # 优先级2: Google新闻（英文搜索）
        try:
            logger.info("[统一新闻工具] 尝试Google美股新闻...")
            query = f"{stock_code} stock news earnings financial"
            result = get_google_news(query=query, curr_date=curr_date)
            if result and len(result.strip()) > 50:
                logger.info(f"[统一新闻工具] ✅ Google美股新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "Google美股新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] Google美股新闻获取失败: {e}")

        # 优先级3: FinnHub新闻（如果可用）
        try:
            logger.info("[统一新闻工具] 尝试FinnHub美股新闻...")
            result = get_finnhub_news(symbol=stock_code, max_results=min(max_news, 50))
            if result and len(result.strip()) > 50:
                logger.info(f"[统一新闻工具] ✅ FinnHub美股新闻获取成功: {len(result)} 字符")
                return self._format_news_result(result, "FinnHub美股新闻", model_info)
        except Exception as e:
            logger.warning(f"[统一新闻工具] FinnHub美股新闻获取失败: {e}")

        return "❌ 无法获取美股新闻数据，所有新闻源均不可用"

    def _format_news_result(self, news_content: str, source: str) -> str:
        """格式化新闻结果"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 🔍 添加调试日志：打印原始新闻内容
        logger.info(f"[统一新闻工具] 📋 原始新闻内容预览 (前500字符): {news_content[:500]}")
        logger.info(f"[统一新闻工具] 📊 原始内容长度: {len(news_content)} 字符")

        formatted_result = f"""
=== 📰 新闻数据来源: {source} ===
获取时间: {timestamp}
数据长度: {len(news_content)} 字符

=== 📋 新闻内容 ===
{news_content}

=== ✅ 数据状态 ===
状态: 成功获取
来源: {source}
时间戳: {timestamp}
"""
        return formatted_result.strip()

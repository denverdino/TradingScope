#!/usr/bin/env python3
"""
优化的A股数据获取工具
集成缓存策略和Tushare数据接口，提高数据获取效率
"""

import random
import time
from datetime import datetime, timedelta
from typing import Optional

# 导入日志模块
from tradingscope.utils.logging_manager import get_logger

from .cache_manager import get_cache
from .config import get_config

logger = get_logger("agents")


class OptimizedChinaDataProvider:
    """优化的A股数据提供器 - 集成缓存和Tushare数据接口"""

    def __init__(self):
        self.cache = get_cache()
        self.config = get_config()
        self.last_api_call = 0
        self.min_api_interval = 0.5  # Tushare数据接口调用间隔较短

        logger.info("📊 优化A股数据提供器初始化完成")

    def _wait_for_rate_limit(self):
        """等待API限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.min_api_interval:
            wait_time = self.min_api_interval - time_since_last_call
            time.sleep(wait_time)

        self.last_api_call = time.time()

    def get_stock_data(self, symbol: str, start_date: str, end_date: str, force_refresh: bool = False) -> str:
        """
        获取A股数据 - 优先使用缓存

        Args:
            symbol: 股票代码（6位数字）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            force_refresh: 是否强制刷新缓存

        Returns:
            格式化的股票数据字符串
        """
        logger.info(f"📈 获取A股数据: {symbol} ({start_date} 到 {end_date})")

        # 检查缓存（除非强制刷新）
        if not force_refresh:
            cache_key = self.cache.find_cached_stock_data(symbol=symbol, start_date=start_date, end_date=end_date, data_source="tdx")

            if cache_key:
                cached_data = self.cache.load_stock_data(cache_key)
                if cached_data:
                    logger.info(f"⚡ 从缓存加载A股数据: {symbol}")
                    return cached_data

        # 缓存未命中，从Tushare数据接口获取
        logger.info(f"🌐 从Tushare数据接口获取数据: {symbol}")

        try:
            # API限制处理
            self._wait_for_rate_limit()

            # 调用统一数据源接口（默认Tushare，支持备用数据源）
            from .data_source_manager import get_china_stock_data_unified

            formatted_data = get_china_stock_data_unified(symbol=symbol, start_date=start_date, end_date=end_date)

            # 检查是否获取成功
            if "❌" in formatted_data or "错误" in formatted_data:
                logger.error(f"❌ 数据源API调用失败: {symbol}")
                # 尝试从旧缓存获取数据
                old_cache = self._try_get_old_cache(symbol, start_date, end_date)
                if old_cache:
                    logger.info(f"📁 使用过期缓存数据: {symbol}")
                    return old_cache

                # 生成备用数据
                return self._generate_fallback_data(symbol, start_date, end_date, "数据源API调用失败")

            # 保存到缓存
            self.cache.save_stock_data(
                symbol=symbol, data=formatted_data, start_date=start_date, end_date=end_date, data_source="unified"  # 使用统一数据源标识
            )

            logger.info(f"✅ A股数据获取成功: {symbol}")
            return formatted_data

        except Exception as e:
            error_msg = f"Tushare数据接口调用异常: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # 尝试从旧缓存获取数据
            old_cache = self._try_get_old_cache(symbol, start_date, end_date)
            if old_cache:
                logger.info(f"📁 使用过期缓存数据: {symbol}")
                return old_cache

            # 生成备用数据
            return self._generate_fallback_data(symbol, start_date, end_date, error_msg)

    def get_fundamentals_data(self, symbol: str, force_refresh: bool = False) -> str:
        """
        获取A股基本面数据 - 优先使用缓存

        Args:
            symbol: 股票代码
            force_refresh: 是否强制刷新缓存

        Returns:
            格式化的基本面数据字符串
        """
        logger.info(f"📊 获取A股基本面数据: {symbol}")

        # 检查缓存（除非强制刷新）
        if not force_refresh:
            # 查找基本面数据缓存
            for metadata_file in self.cache.metadata_dir.glob("*_meta.json"):
                try:
                    import json

                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    if metadata.get("symbol") == symbol and metadata.get("data_type") == "fundamentals" and metadata.get("market_type") == "china":

                        cache_key = metadata_file.stem.replace("_meta", "")
                        if self.cache.is_cache_valid(cache_key, symbol=symbol, data_type="fundamentals"):
                            cached_data = self.cache.load_stock_data(cache_key)
                            if cached_data:
                                logger.info(f"⚡ 从缓存加载A股基本面数据: {symbol}")
                                return cached_data
                except Exception:
                    continue

        # 缓存未命中，生成基本面分析
        logger.debug(f"🔍 生成A股基本面分析: {symbol}")

        try:
            # 先获取股票数据
            current_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            stock_data = self.get_stock_data(symbol, start_date, current_date)

            # 生成基本面分析报告
            fundamentals_data = self._generate_fundamentals_report(symbol, stock_data)

            # 保存到缓存
            self.cache.save_fundamentals_data(symbol=symbol, fundamentals_data=fundamentals_data, data_source="tdx_analysis")

            logger.info(f"✅ A股基本面数据生成成功: {symbol}")
            return fundamentals_data

        except Exception as e:
            error_msg = f"基本面数据生成失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return self._generate_fallback_fundamentals(symbol, error_msg)

    def _generate_fundamentals_report(self, symbol: str, stock_data: str) -> str:
        """基于股票数据生成真实的基本面分析报告"""

        # 添加详细的股票代码追踪日志
        logger.debug(f"🔍 [股票代码追踪] _generate_fundamentals_report 接收到的股票代码: '{symbol}' (类型: {type(symbol)})")
        logger.debug(f"🔍 [股票代码追踪] 股票代码长度: {len(str(symbol))}")
        logger.debug(f"🔍 [股票代码追踪] 股票代码字符: {list(str(symbol))}")
        logger.debug(f"🔍 [股票代码追踪] 接收到的股票数据前200字符: {stock_data[:200] if stock_data else 'None'}")

        # 从股票数据中提取信息
        company_name = "未知公司"
        current_price = "N/A"
        volume = "N/A"
        change_pct = "N/A"

        # 首先尝试从统一接口获取股票基本信息
        try:
            logger.debug(f"🔍 [股票代码追踪] 尝试获取{symbol}的基本信息...")
            from .interface import get_china_stock_info_unified

            stock_info = get_china_stock_info_unified(symbol)
            logger.debug(f"🔍 [股票代码追踪] 获取到的股票信息: {stock_info}")

            if "股票名称:" in stock_info:
                lines = stock_info.split("\n")
                for line in lines:
                    if "股票名称:" in line:
                        company_name = line.split(":")[1].strip()
                        logger.debug(f"🔍 [股票代码追踪] 从统一接口获取到股票名称: {company_name}")
                        break
        except Exception as e:
            logger.warning(f"⚠️ 获取股票基本信息失败: {e}")

        # 然后从股票数据中提取价格信息
        if "股票名称:" in stock_data:
            lines = stock_data.split("\n")
            for line in lines:
                if "股票名称:" in line and company_name == "未知公司":
                    company_name = line.split(":")[1].strip()
                elif "当前价格:" in line:
                    current_price = line.split(":")[1].strip()
                elif "涨跌幅:" in line:
                    change_pct = line.split(":")[1].strip()
                elif "成交量:" in line:
                    volume = line.split(":")[1].strip()

        # 尝试从股票数据表格中提取最新价格信息
        if current_price == "N/A" and stock_data:
            try:
                lines = stock_data.split("\n")
                for i, line in enumerate(lines):
                    if "最新数据:" in line and i + 1 < len(lines):
                        # 查找数据行
                        for j in range(i + 1, min(i + 5, len(lines))):
                            data_line = lines[j].strip()
                            if data_line and not data_line.startswith("日期") and not data_line.startswith("-"):
                                # 尝试解析数据行
                                parts = data_line.split()
                                if len(parts) >= 4:
                                    try:
                                        # 假设格式: 日期 股票代码 开盘 收盘 最高 最低 成交量 成交额...
                                        current_price = parts[3]  # 收盘价
                                        logger.debug(f"🔍 [股票代码追踪] 从数据表格提取到收盘价: {current_price}")
                                        break
                                    except (IndexError, ValueError):
                                        continue
                        break
            except Exception as e:
                logger.debug(f"🔍 [股票代码追踪] 解析股票数据表格失败: {e}")

        # 根据股票代码判断行业和基本信息
        logger.debug(f"🔍 [股票代码追踪] 调用 _get_industry_info，传入参数: '{symbol}'")
        industry_info = self._get_industry_info(symbol)
        logger.debug(f"🔍 [股票代码追踪] _get_industry_info 返回结果: {industry_info}")

        logger.debug(f"🔍 [股票代码追踪] 调用 _estimate_financial_metrics，传入参数: '{symbol}'")
        financial_estimates = self._estimate_financial_metrics(symbol, current_price)
        logger.debug(f"🔍 [股票代码追踪] _estimate_financial_metrics 返回结果: {financial_estimates}")

        logger.debug(f"🔍 [股票代码追踪] 开始生成报告，使用股票代码: '{symbol}'")

        # 检查数据来源并生成相应说明
        data_source_note = ""
        data_source = financial_estimates.get("data_source", "")

        if any("（估算值）" in str(v) for v in financial_estimates.values() if isinstance(v, str)):
            data_source_note = "\n⚠️ **数据说明**: 部分财务指标为估算值，建议结合最新财报数据进行分析"
        elif data_source == "AKShare":
            data_source_note = "\n✅ **数据说明**: 财务指标基于AKShare真实财务数据计算"
        elif data_source == "Tushare":
            data_source_note = "\n✅ **数据说明**: 财务指标基于Tushare真实财务数据计算"
        else:
            data_source_note = "\n✅ **数据说明**: 财务指标基于真实财务数据计算"

        report = f"""# 中国A股基本面分析报告 - {symbol}

## 📊 股票基本信息
- **股票代码**: {symbol}
- **股票名称**: {company_name}
- **所属行业**: {industry_info['industry']}
- **市场板块**: {industry_info['market']}
- **当前股价**: {current_price}
- **涨跌幅**: {change_pct}
- **成交量**: {volume}
- **分析日期**: {datetime.now().strftime('%Y年%m月%d日')}{data_source_note}

## 💰 财务数据分析

### 估值指标
- **市盈率(PE)**: {financial_estimates['pe']}
- **市净率(PB)**: {financial_estimates['pb']}
- **市销率(PS)**: {financial_estimates['ps']}
- **股息收益率**: {financial_estimates['dividend_yield']}

### 盈利能力指标
- **净资产收益率(ROE)**: {financial_estimates['roe']}
- **总资产收益率(ROA)**: {financial_estimates['roa']}
- **毛利率**: {financial_estimates['gross_margin']}
- **净利率**: {financial_estimates['net_margin']}

### 财务健康度
- **资产负债率**: {financial_estimates['debt_ratio']}
- **流动比率**: {financial_estimates['current_ratio']}
- **速动比率**: {financial_estimates['quick_ratio']}
- **现金比率**: {financial_estimates['cash_ratio']}

## 📈 行业分析

### 行业地位
{industry_info['analysis']}

### 竞争优势
- **市场份额**: {industry_info['market_share']}
- **品牌价值**: {industry_info['brand_value']}
- **技术优势**: {industry_info['tech_advantage']}

## 🎯 投资价值评估

### 估值水平分析
{self._analyze_valuation(financial_estimates)}

### 成长性分析
{self._analyze_growth_potential(symbol, industry_info)}

### 风险评估
{self._analyze_risks(symbol, financial_estimates, industry_info)}

## 💡 投资建议

### 综合评分
- **基本面评分**: {financial_estimates['fundamental_score']}/10
- **估值吸引力**: {financial_estimates['valuation_score']}/10
- **成长潜力**: {financial_estimates['growth_score']}/10
- **风险等级**: {financial_estimates['risk_level']}

### 操作建议
{self._generate_investment_advice(financial_estimates, industry_info)}

### 绝对估值
- **DCF估值**：基于现金流贴现的内在价值
- **资产价值**：净资产重估价值
- **分红收益率**：股息回报分析

## 风险分析
### 系统性风险
- **宏观经济风险**：经济周期对公司的影响
- **政策风险**：行业政策变化的影响
- **市场风险**：股市波动对估值的影响

### 非系统性风险
- **经营风险**：公司特有的经营风险
- **财务风险**：债务结构和偿债能力风险
- **管理风险**：管理层变动和决策风险

## 投资建议
### 综合评价
基于以上分析，该股票的投资价值评估：

**优势：**
- A股市场上市公司，监管相对完善
- 具备一定的市场地位和品牌价值
- 财务信息透明度较高

**风险：**
- 需要关注宏观经济环境变化
- 行业竞争加剧的影响
- 政策调整对业务的潜在影响

### 操作建议
- **投资策略**：建议采用价值投资策略，关注长期基本面
- **仓位建议**：根据风险承受能力合理配置仓位
- **关注指标**：重点关注ROE、PE、现金流等核心指标

---
**重要声明**: 本报告基于公开数据和模型估算生成，仅供参考，不构成投资建议。
实际投资决策请结合最新财报数据和专业分析师意见。

**数据来源**: {data_source if data_source else "多源数据"}数据接口 + 基本面分析模型
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return report

    def _get_industry_info(self, symbol: str) -> dict:
        """根据股票代码获取行业信息"""

        # 添加详细的股票代码追踪日志
        logger.debug(f"🔍 [股票代码追踪] _get_industry_info 接收到的股票代码: '{symbol}' (类型: {type(symbol)})")
        logger.debug(f"🔍 [股票代码追踪] 股票代码长度: {len(str(symbol))}")
        logger.debug(f"🔍 [股票代码追踪] 股票代码字符: {list(str(symbol))}")

        # 根据股票代码前缀判断行业（简化版）
        code_prefix = symbol[:3]
        logger.debug(f"🔍 [股票代码追踪] 提取的代码前缀: '{code_prefix}'")

        industry_map = {
            "000": {"industry": "深市主板", "market": "深圳证券交易所", "type": "综合"},
            "001": {"industry": "深市主板", "market": "深圳证券交易所", "type": "综合"},
            "002": {"industry": "中小板", "market": "深圳证券交易所", "type": "成长型"},
            "003": {"industry": "创业板", "market": "深圳证券交易所", "type": "创新型"},
            "300": {"industry": "创业板", "market": "深圳证券交易所", "type": "高科技"},
            "600": {"industry": "沪市主板", "market": "上海证券交易所", "type": "大盘蓝筹"},
            "601": {"industry": "沪市主板", "market": "上海证券交易所", "type": "大盘蓝筹"},
            "603": {"industry": "沪市主板", "market": "上海证券交易所", "type": "中小盘"},
            "688": {"industry": "科创板", "market": "上海证券交易所", "type": "科技创新"},
        }

        info = industry_map.get(code_prefix, {"industry": "其他", "market": "未知市场", "type": "综合"})

        # 特殊股票的详细信息
        special_stocks = {
            "000001": {
                "industry": "银行业",
                "analysis": "平安银行是中国领先的股份制商业银行，在零售银行业务方面具有显著优势。",
                "market_share": "股份制银行前列",
                "brand_value": "知名金融品牌",
                "tech_advantage": "金融科技创新领先",
            },
            "600036": {
                "industry": "银行业",
                "analysis": "招商银行是中国优质的股份制银行，零售银行业务和财富管理业务领先。",
                "market_share": "股份制银行龙头",
                "brand_value": "优质银行品牌",
                "tech_advantage": "数字化银行先锋",
            },
            "000002": {
                "industry": "房地产",
                "analysis": "万科A是中国房地产行业龙头企业，在住宅开发领域具有领先地位。",
                "market_share": "房地产行业前三",
                "brand_value": "知名地产品牌",
                "tech_advantage": "绿色建筑技术",
            },
        }

        if symbol in special_stocks:
            info.update(special_stocks[symbol])
        else:
            info.update(
                {
                    "analysis": f"该股票属于{info['industry']}，具体业务需要进一步分析。",
                    "market_share": "待分析",
                    "brand_value": "待评估",
                    "tech_advantage": "待分析",
                }
            )

        return info

    def _estimate_financial_metrics(self, symbol: str, current_price: str) -> dict:
        """获取真实财务指标（优先使用Tushare真实数据，失败时使用估算）"""

        # 提取价格数值
        try:
            price_value = float(current_price.replace("¥", "").replace(",", ""))
        except:
            price_value = 10.0  # 默认值

        # 尝试获取真实财务数据
        real_metrics = self._get_real_financial_metrics(symbol, price_value)
        if real_metrics:
            logger.debug(f"✅ 使用真实财务数据: {symbol}")
            return real_metrics

        # 如果无法获取真实数据，使用估算数据并标注
        logger.warning(f"⚠️ 无法获取真实财务数据，使用估算数据: {symbol}")
        estimated_metrics = self._get_estimated_financial_metrics(symbol, price_value)

        # 在所有指标后添加估算标注
        for key in estimated_metrics:
            if isinstance(estimated_metrics[key], str) and key not in ["fundamental_score", "valuation_score", "growth_score", "risk_level"]:
                if "（" not in estimated_metrics[key]:
                    estimated_metrics[key] += "（估算值）"

        return estimated_metrics

    def _get_real_financial_metrics(self, symbol: str, price_value: float) -> dict:
        """获取真实财务指标 - 优先使用AKShare"""
        try:
            # 优先尝试AKShare数据源
            logger.info(f"🔄 优先尝试AKShare获取{symbol}财务数据")
            from .akshare_utils import get_akshare_provider

            akshare_provider = get_akshare_provider()

            if akshare_provider.connected:
                financial_data = akshare_provider.get_financial_data(symbol)

                if financial_data and any(not v.empty if hasattr(v, "empty") else bool(v) for v in financial_data.values()):
                    logger.info(f"✅ AKShare财务数据获取成功: {symbol}")
                    # 获取股票基本信息
                    stock_info = akshare_provider.get_stock_info(symbol)

                    # 解析AKShare财务数据
                    logger.debug(f"🔧 调用AKShare解析函数，股价: {price_value}")
                    metrics = self._parse_akshare_financial_data(financial_data, stock_info, price_value)
                    logger.debug(f"🔧 AKShare解析结果: {metrics}")
                    if metrics:
                        logger.info("✅ AKShare解析成功，返回指标")
                        return metrics
                    else:
                        logger.warning("⚠️ AKShare解析失败，返回None")
                else:
                    logger.warning(f"⚠️ AKShare未获取到{symbol}财务数据，尝试Tushare")
            else:
                logger.warning("⚠️ AKShare未连接，尝试Tushare")

            # 备用方案：使用Tushare数据源
            logger.info(f"🔄 使用Tushare备用数据源获取{symbol}财务数据")
            from .tushare_utils import get_tushare_provider

            provider = get_tushare_provider()
            if not provider.connected:
                logger.debug(f"Tushare未连接，无法获取{symbol}真实财务数据")
                return None

            # 获取财务数据
            financial_data = provider.get_financial_data(symbol)
            if not financial_data:
                logger.debug(f"未获取到{symbol}的财务数据")
                return None

            # 获取股票基本信息
            stock_info = provider.get_stock_info(symbol)

            # 解析Tushare财务数据
            metrics = self._parse_financial_data(financial_data, stock_info, price_value)
            if metrics:
                return metrics

        except Exception as e:
            logger.debug(f"获取{symbol}真实财务数据失败: {e}")

        return None

    def _parse_akshare_financial_data(self, financial_data: dict, stock_info: dict, price_value: float) -> dict:
        """解析AKShare财务数据为指标"""
        try:
            # 获取最新的财务数据
            balance_sheet = financial_data.get("balance_sheet", [])
            income_statement = financial_data.get("income_statement", [])
            cash_flow = financial_data.get("cash_flow", [])
            main_indicators = financial_data.get("main_indicators")

            if main_indicators is None or main_indicators.empty:
                logger.warning("AKShare主要财务指标为空")
                return None

            # main_indicators是DataFrame，需要转换为字典格式便于查找
            # 获取最新数据列（第3列，索引为2）
            latest_col = main_indicators.columns[2] if len(main_indicators.columns) > 2 else None
            if not latest_col:
                logger.warning("AKShare主要财务指标缺少数据列")
                return None

            logger.info(f"📅 使用AKShare最新数据期间: {latest_col}")

            # 创建指标名称到值的映射
            indicators_dict = {}
            for _, row in main_indicators.iterrows():
                indicator_name = row["指标"]
                value = row[latest_col]
                indicators_dict[indicator_name] = value

            logger.debug(f"AKShare主要财务指标数量: {len(indicators_dict)}")

            # 计算财务指标
            metrics = {}

            # 获取ROE - 直接从指标中获取
            roe_value = indicators_dict.get("净资产收益率(ROE)")
            if roe_value is not None and str(roe_value) != "nan" and roe_value != "--":
                try:
                    roe_val = float(roe_value)
                    # ROE通常是百分比形式
                    metrics["roe"] = f"{roe_val:.1f}%"
                    logger.debug(f"✅ 获取ROE: {metrics['roe']}")
                except (ValueError, TypeError):
                    metrics["roe"] = "N/A"
            else:
                metrics["roe"] = "N/A"

            # 获取每股收益 - 用于计算PE
            eps_value = indicators_dict.get("基本每股收益")
            if eps_value is not None and str(eps_value) != "nan" and eps_value != "--":
                try:
                    eps_val = float(eps_value)
                    if eps_val > 0:
                        # 计算PE = 股价 / 每股收益
                        pe_val = price_value / eps_val
                        metrics["pe"] = f"{pe_val:.1f}倍"
                        logger.debug(f"✅ 计算PE: 股价{price_value} / EPS{eps_val} = {metrics['pe']}")
                    else:
                        metrics["pe"] = "N/A（亏损）"
                except (ValueError, TypeError):
                    metrics["pe"] = "N/A"
            else:
                metrics["pe"] = "N/A"

            # 获取每股净资产 - 用于计算PB
            bps_value = indicators_dict.get("每股净资产_最新股数")
            if bps_value is not None and str(bps_value) != "nan" and bps_value != "--":
                try:
                    bps_val = float(bps_value)
                    if bps_val > 0:
                        # 计算PB = 股价 / 每股净资产
                        pb_val = price_value / bps_val
                        metrics["pb"] = f"{pb_val:.2f}倍"
                        logger.debug(f"✅ 计算PB: 股价{price_value} / BPS{bps_val} = {metrics['pb']}")
                    else:
                        metrics["pb"] = "N/A"
                except (ValueError, TypeError):
                    metrics["pb"] = "N/A"
            else:
                metrics["pb"] = "N/A"

            # 尝试获取其他指标
            # 总资产收益率(ROA)
            roa_value = indicators_dict.get("总资产报酬率")
            if roa_value is not None and str(roa_value) != "nan" and roa_value != "--":
                try:
                    roa_val = float(roa_value)
                    metrics["roa"] = f"{roa_val:.1f}%"
                except (ValueError, TypeError):
                    metrics["roa"] = "N/A"
            else:
                metrics["roa"] = "N/A"

            # 毛利率
            gross_margin_value = indicators_dict.get("毛利率")
            if gross_margin_value is not None and str(gross_margin_value) != "nan" and gross_margin_value != "--":
                try:
                    gross_margin_val = float(gross_margin_value)
                    metrics["gross_margin"] = f"{gross_margin_val:.1f}%"
                except (ValueError, TypeError):
                    metrics["gross_margin"] = "N/A"
            else:
                metrics["gross_margin"] = "N/A"

            # 销售净利率
            net_margin_value = indicators_dict.get("销售净利率")
            if net_margin_value is not None and str(net_margin_value) != "nan" and net_margin_value != "--":
                try:
                    net_margin_val = float(net_margin_value)
                    metrics["net_margin"] = f"{net_margin_val:.1f}%"
                except (ValueError, TypeError):
                    metrics["net_margin"] = "N/A"
            else:
                metrics["net_margin"] = "N/A"

            # 资产负债率
            debt_ratio_value = indicators_dict.get("资产负债率")
            if debt_ratio_value is not None and str(debt_ratio_value) != "nan" and debt_ratio_value != "--":
                try:
                    debt_ratio_val = float(debt_ratio_value)
                    metrics["debt_ratio"] = f"{debt_ratio_val:.1f}%"
                except (ValueError, TypeError):
                    metrics["debt_ratio"] = "N/A"
            else:
                metrics["debt_ratio"] = "N/A"

            # 流动比率
            current_ratio_value = indicators_dict.get("流动比率")
            if current_ratio_value is not None and str(current_ratio_value) != "nan" and current_ratio_value != "--":
                try:
                    current_ratio_val = float(current_ratio_value)
                    metrics["current_ratio"] = f"{current_ratio_val:.2f}"
                except (ValueError, TypeError):
                    metrics["current_ratio"] = "N/A"
            else:
                metrics["current_ratio"] = "N/A"

            # 速动比率
            quick_ratio_value = indicators_dict.get("速动比率")
            if quick_ratio_value is not None and str(quick_ratio_value) != "nan" and quick_ratio_value != "--":
                try:
                    quick_ratio_val = float(quick_ratio_value)
                    metrics["quick_ratio"] = f"{quick_ratio_val:.2f}"
                except (ValueError, TypeError):
                    metrics["quick_ratio"] = "N/A"
            else:
                metrics["quick_ratio"] = "N/A"

            # 补充其他指标的默认值
            metrics.update({"ps": "待计算", "dividend_yield": "待查询", "cash_ratio": "待分析"})

            # 评分（基于AKShare数据的简化评分）
            fundamental_score = self._calculate_fundamental_score(metrics, stock_info)
            valuation_score = self._calculate_valuation_score(metrics)
            growth_score = self._calculate_growth_score(metrics, stock_info)
            risk_level = self._calculate_risk_level(metrics, stock_info)

            metrics.update(
                {
                    "fundamental_score": fundamental_score,
                    "valuation_score": valuation_score,
                    "growth_score": growth_score,
                    "risk_level": risk_level,
                    "data_source": "AKShare",
                }
            )

            logger.info(f"✅ AKShare财务数据解析成功: PE={metrics['pe']}, PB={metrics['pb']}, ROE={metrics['roe']}")
            return metrics

        except Exception as e:
            logger.error(f"❌ AKShare财务数据解析失败: {e}")
            return None

    def _parse_financial_data(self, financial_data: dict, stock_info: dict, price_value: float) -> dict:
        """解析财务数据为指标"""
        try:
            # 获取最新的财务数据
            balance_sheet = financial_data.get("balance_sheet", [])
            income_statement = financial_data.get("income_statement", [])
            cash_flow = financial_data.get("cash_flow", [])

            if not (balance_sheet or income_statement):
                return None

            latest_balance = balance_sheet[0] if balance_sheet else {}
            latest_income = income_statement[0] if income_statement else {}
            latest_cash = cash_flow[0] if cash_flow else {}

            # 计算财务指标
            metrics = {}

            # 基础数据
            total_assets = latest_balance.get("total_assets", 0) or 0
            total_liab = latest_balance.get("total_liab", 0) or 0
            total_equity = latest_balance.get("total_hldr_eqy_exc_min_int", 0) or 0
            total_revenue = latest_income.get("total_revenue", 0) or 0
            net_income = latest_income.get("n_income", 0) or 0
            operate_profit = latest_income.get("operate_profit", 0) or 0

            # 估算市值（简化计算）
            market_cap = price_value * 1000000000  # 假设10亿股本

            # 计算各项指标
            # PE比率
            if net_income > 0:
                pe_ratio = market_cap / (net_income * 10000)  # 转换单位
                metrics["pe"] = f"{pe_ratio:.1f}倍"
            else:
                metrics["pe"] = "N/A（亏损）"

            # PB比率
            if total_equity > 0:
                pb_ratio = market_cap / (total_equity * 10000)
                metrics["pb"] = f"{pb_ratio:.2f}倍"
            else:
                metrics["pb"] = "N/A"

            # PS比率
            if total_revenue > 0:
                ps_ratio = market_cap / (total_revenue * 10000)
                metrics["ps"] = f"{ps_ratio:.1f}倍"
            else:
                metrics["ps"] = "N/A"

            # ROE
            if total_equity > 0 and net_income > 0:
                roe = (net_income / total_equity) * 100
                metrics["roe"] = f"{roe:.1f}%"
            else:
                metrics["roe"] = "N/A"

            # ROA
            if total_assets > 0 and net_income > 0:
                roa = (net_income / total_assets) * 100
                metrics["roa"] = f"{roa:.1f}%"
            else:
                metrics["roa"] = "N/A"

            # 净利率
            if total_revenue > 0 and net_income > 0:
                net_margin = (net_income / total_revenue) * 100
                metrics["net_margin"] = f"{net_margin:.1f}%"
            else:
                metrics["net_margin"] = "N/A"

            # 资产负债率
            if total_assets > 0:
                debt_ratio = (total_liab / total_assets) * 100
                metrics["debt_ratio"] = f"{debt_ratio:.1f}%"
            else:
                metrics["debt_ratio"] = "N/A"

            # 其他指标设为默认值
            metrics.update({"dividend_yield": "待查询", "gross_margin": "待计算", "current_ratio": "待计算", "quick_ratio": "待计算", "cash_ratio": "待分析"})

            # 评分（基于真实数据的简化评分）
            fundamental_score = self._calculate_fundamental_score(metrics, stock_info)
            valuation_score = self._calculate_valuation_score(metrics)
            growth_score = self._calculate_growth_score(metrics, stock_info)
            risk_level = self._calculate_risk_level(metrics, stock_info)

            metrics.update({"fundamental_score": fundamental_score, "valuation_score": valuation_score, "growth_score": growth_score, "risk_level": risk_level})

            return metrics

        except Exception as e:
            logger.error(f"解析财务数据失败: {e}")
            return None

    def _calculate_fundamental_score(self, metrics: dict, stock_info: dict) -> float:
        """计算基本面评分"""
        score = 5.0  # 基础分

        # ROE评分
        roe_str = metrics.get("roe", "N/A")
        if roe_str != "N/A":
            try:
                roe = float(roe_str.replace("%", ""))
                if roe > 15:
                    score += 1.5
                elif roe > 10:
                    score += 1.0
                elif roe > 5:
                    score += 0.5
            except:
                pass

        # 净利率评分
        net_margin_str = metrics.get("net_margin", "N/A")
        if net_margin_str != "N/A":
            try:
                net_margin = float(net_margin_str.replace("%", ""))
                if net_margin > 20:
                    score += 1.0
                elif net_margin > 10:
                    score += 0.5
            except:
                pass

        return min(score, 10.0)

    def _calculate_valuation_score(self, metrics: dict) -> float:
        """计算估值评分"""
        score = 5.0  # 基础分

        # PE评分
        pe_str = metrics.get("pe", "N/A")
        if pe_str != "N/A" and "亏损" not in pe_str:
            try:
                pe = float(pe_str.replace("倍", ""))
                if pe < 15:
                    score += 2.0
                elif pe < 25:
                    score += 1.0
                elif pe > 50:
                    score -= 1.0
            except:
                pass

        # PB评分
        pb_str = metrics.get("pb", "N/A")
        if pb_str != "N/A":
            try:
                pb = float(pb_str.replace("倍", ""))
                if pb < 1.5:
                    score += 1.0
                elif pb < 3:
                    score += 0.5
                elif pb > 5:
                    score -= 0.5
            except:
                pass

        return min(max(score, 1.0), 10.0)

    def _calculate_growth_score(self, metrics: dict, stock_info: dict) -> float:
        """计算成长性评分"""
        score = 6.0  # 基础分

        # 根据行业调整
        industry = stock_info.get("industry", "")
        if "科技" in industry or "软件" in industry or "互联网" in industry:
            score += 1.0
        elif "银行" in industry or "保险" in industry:
            score -= 0.5

        return min(max(score, 1.0), 10.0)

    def _calculate_risk_level(self, metrics: dict, stock_info: dict) -> str:
        """计算风险等级"""
        # 资产负债率
        debt_ratio_str = metrics.get("debt_ratio", "N/A")
        if debt_ratio_str != "N/A":
            try:
                debt_ratio = float(debt_ratio_str.replace("%", ""))
                if debt_ratio > 70:
                    return "较高"
                elif debt_ratio > 50:
                    return "中等"
                else:
                    return "较低"
            except:
                pass

        # 根据行业判断
        industry = stock_info.get("industry", "")
        if "银行" in industry:
            return "中等"
        elif "科技" in industry or "创业板" in industry:
            return "较高"

        return "中等"

    def _get_estimated_financial_metrics(self, symbol: str, price_value: float) -> dict:
        """获取估算财务指标（原有的分类方法）"""
        # 根据股票代码和价格估算指标
        if symbol.startswith(("000001", "600036")):  # 银行股
            return {
                "pe": "5.2倍（银行业平均水平）",
                "pb": "0.65倍（破净状态，银行业常见）",
                "ps": "2.1倍",
                "dividend_yield": "4.2%（银行业分红较高）",
                "roe": "12.5%（银行业平均）",
                "roa": "0.95%",
                "gross_margin": "N/A（银行业无毛利率概念）",
                "net_margin": "28.5%",
                "debt_ratio": "92%（银行业负债率高属正常）",
                "current_ratio": "N/A（银行业特殊）",
                "quick_ratio": "N/A（银行业特殊）",
                "cash_ratio": "充足",
                "fundamental_score": 7.5,
                "valuation_score": 8.0,
                "growth_score": 6.5,
                "risk_level": "中等",
            }
        elif symbol.startswith("300"):  # 创业板
            return {
                "pe": "35.8倍（创业板平均）",
                "pb": "3.2倍",
                "ps": "5.8倍",
                "dividend_yield": "1.2%",
                "roe": "15.2%",
                "roa": "8.5%",
                "gross_margin": "42.5%",
                "net_margin": "18.2%",
                "debt_ratio": "35%",
                "current_ratio": "2.1倍",
                "quick_ratio": "1.8倍",
                "cash_ratio": "良好",
                "fundamental_score": 7.0,
                "valuation_score": 5.5,
                "growth_score": 8.5,
                "risk_level": "较高",
            }
        else:  # 其他股票
            return {
                "pe": "18.5倍（市场平均）",
                "pb": "1.8倍",
                "ps": "2.5倍",
                "dividend_yield": "2.5%",
                "roe": "12.8%",
                "roa": "6.2%",
                "gross_margin": "25.5%",
                "net_margin": "12.8%",
                "debt_ratio": "45%",
                "current_ratio": "1.5倍",
                "quick_ratio": "1.2倍",
                "cash_ratio": "一般",
                "fundamental_score": 6.5,
                "valuation_score": 6.0,
                "growth_score": 7.0,
                "risk_level": "中等",
            }

    def _analyze_valuation(self, financial_estimates: dict) -> str:
        """分析估值水平"""
        valuation_score = financial_estimates["valuation_score"]

        if valuation_score >= 8:
            return "当前估值水平较为合理，具有一定的投资价值。市盈率和市净率相对较低，安全边际较高。"
        elif valuation_score >= 6:
            return "估值水平适中，需要结合基本面和成长性综合判断投资价值。"
        else:
            return "当前估值偏高，投资需谨慎。建议等待更好的买入时机。"

    def _analyze_growth_potential(self, symbol: str, industry_info: dict) -> str:
        """分析成长潜力"""
        if symbol.startswith(("000001", "600036")):
            return "银行业整体增长稳定，受益于经济发展和金融深化。数字化转型和财富管理业务是主要增长点。"
        elif symbol.startswith("300"):
            return "创业板公司通常具有较高的成长潜力，但也伴随着较高的风险。需要关注技术创新和市场拓展能力。"
        else:
            return "成长潜力需要结合具体行业和公司基本面分析。建议关注行业发展趋势和公司竞争优势。"

    def _analyze_risks(self, symbol: str, financial_estimates: dict, industry_info: dict) -> str:
        """分析投资风险"""
        risk_level = financial_estimates["risk_level"]

        risk_analysis = f"**风险等级**: {risk_level}\n\n"

        if symbol.startswith(("000001", "600036")):
            risk_analysis += """**主要风险**:
- 利率环境变化对净息差的影响
- 信贷资产质量风险
- 监管政策变化风险
- 宏观经济下行对银行业的影响"""
        elif symbol.startswith("300"):
            risk_analysis += """**主要风险**:
- 技术更新换代风险
- 市场竞争加剧风险
- 估值波动较大
- 业绩不确定性较高"""
        else:
            risk_analysis += """**主要风险**:
- 行业周期性风险
- 宏观经济环境变化
- 市场竞争风险
- 政策调整风险"""

        return risk_analysis

    def _generate_investment_advice(self, financial_estimates: dict, industry_info: dict) -> str:
        """生成投资建议"""
        fundamental_score = financial_estimates["fundamental_score"]
        valuation_score = financial_estimates["valuation_score"]
        growth_score = financial_estimates["growth_score"]

        total_score = (fundamental_score + valuation_score + growth_score) / 3

        if total_score >= 7.5:
            return """**投资建议**: 🟢 **买入**
- 基本面良好，估值合理，具有较好的投资价值
- 建议分批建仓，长期持有
- 适合价值投资者和稳健型投资者"""
        elif total_score >= 6.0:
            return """**投资建议**: 🟡 **观望**
- 基本面一般，需要进一步观察
- 可以小仓位试探，等待更好时机
- 适合有经验的投资者"""
        else:
            return """**投资建议**: 🔴 **回避**
- 当前风险较高，不建议投资
- 建议等待基本面改善或估值回落
- 风险承受能力较低的投资者应避免"""

    def _try_get_old_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[str]:
        """尝试获取过期的缓存数据作为备用"""
        try:
            # 查找任何相关的缓存，不考虑TTL
            for metadata_file in self.cache.metadata_dir.glob("*_meta.json"):
                try:
                    import json

                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    if metadata.get("symbol") == symbol and metadata.get("data_type") == "stock_data" and metadata.get("market_type") == "china":

                        cache_key = metadata_file.stem.replace("_meta", "")
                        cached_data = self.cache.load_stock_data(cache_key)
                        if cached_data:
                            return cached_data + "\n\n⚠️ 注意: 使用的是过期缓存数据"
                except Exception:
                    continue
        except Exception:
            pass

        return None

    def _generate_fallback_data(self, symbol: str, start_date: str, end_date: str, error_msg: str) -> str:
        """生成备用数据"""
        return f"""# {symbol} A股数据获取失败

## ❌ 错误信息
{error_msg}

## 📊 模拟数据（仅供演示）
- 股票代码: {symbol}
- 股票名称: 模拟公司
- 数据期间: {start_date} 至 {end_date}
- 模拟价格: ¥{random.uniform(10, 50):.2f}
- 模拟涨跌: {random.uniform(-5, 5):+.2f}%

## ⚠️ 重要提示
由于数据接口限制或网络问题，无法获取实时数据。
建议稍后重试或检查网络连接。

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _generate_fallback_fundamentals(self, symbol: str, error_msg: str) -> str:
        """生成备用基本面数据"""
        return f"""# {symbol} A股基本面分析失败

## ❌ 错误信息
{error_msg}

## 📊 基本信息
- 股票代码: {symbol}
- 分析状态: 数据获取失败
- 建议: 稍后重试或检查网络连接

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""


# 全局实例
_china_data_provider = None


def get_optimized_china_data_provider() -> OptimizedChinaDataProvider:
    """获取全局A股数据提供器实例"""
    global _china_data_provider
    if _china_data_provider is None:
        _china_data_provider = OptimizedChinaDataProvider()
    return _china_data_provider


def get_china_stock_data_cached(symbol: str, start_date: str, end_date: str, force_refresh: bool = False) -> str:
    """
    获取A股数据的便捷函数

    Args:
        symbol: 股票代码（6位数字）
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        force_refresh: 是否强制刷新缓存

    Returns:
        格式化的股票数据字符串
    """
    provider = get_optimized_china_data_provider()
    return provider.get_stock_data(symbol, start_date, end_date, force_refresh)


def get_china_fundamentals_cached(symbol: str, force_refresh: bool = False) -> str:
    """
    获取A股基本面数据的便捷函数

    Args:
        symbol: 股票代码（6位数字）
        force_refresh: 是否强制刷新缓存

    Returns:
        格式化的基本面数据字符串
    """
    provider = get_optimized_china_data_provider()
    return provider.get_fundamentals_data(symbol, force_refresh)

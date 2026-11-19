"""Plain-text and HTML rendering of evaluation records and the track record."""

from __future__ import annotations

from collections import defaultdict
from html import escape
from typing import Sequence

from .models import EvaluationRecord, StageEvaluation, effective_return_basis
from .track_record import ALL_TICKERS, TrackRecord

STAGE_LABELS = {
    "market_analyst": "市场分析师",
    "fundamentals_analyst": "基本面分析师",
    "news_analyst": "新闻分析师",
    "social_media_analyst": "社交媒体分析师",
    "research_manager": "研究经理",
    "trader": "交易员",
    "portfolio_manager": "投资组合经理",
}


def _percent(value: float | None) -> str:
    return f"{value:+.2%}" if value is not None else "-"


def _rate(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "-"


def _trade_summary(stage: StageEvaluation) -> str:
    trade = stage.trade
    if trade is None:
        return "-"
    basis = effective_return_basis(trade)
    outcome = "收盘估值（未退出）" if trade.exit_reason == "horizon" else f"退出={trade.exit_reason or '-'}"
    summary = f"{trade.trade_intent or '-'} {trade.status} {outcome} {basis}={_percent(trade.strategy_return)}"
    if trade.residual_exit_reason is not None:
        residual = "收盘估值（未退出）" if trade.residual_exit_reason == "horizon" else trade.residual_exit_reason
        summary += f"；剩余仓位={residual}@{trade.residual_exit_date}，价格={trade.residual_exit_price}"
    if trade.limitations:
        summary += "；限制：" + "；".join(trade.limitations)
    return summary


def _shock_summary(record: EvaluationRecord) -> str:
    if not record.market.shock_sessions:
        return "无"
    return "；".join(
        f"{shock.session_date.isoformat()} {shock.classification.value} 超额{_percent(shock.excess_return)}" for shock in record.market.shock_sessions
    )


def record_headline(record: EvaluationRecord) -> str:
    portfolio = record.stage("portfolio_manager")
    market = record.market
    return (
        f"[{record.ticker}|{record.analysis_date.isoformat()}|{record.horizon}交易日|{market.sample_class.value}] "
        f"评估交易日：{record.horizon_end_session.isoformat()} 标的{_percent(market.underlying_return)} 基准{market.benchmark_symbol}{_percent(market.benchmark_return)} "
        f"超额{_percent(market.excess_return)} PM方向={portfolio.direction if portfolio else '-'}"
        f"/{portfolio.excess_direction_status if portfolio else '-'} 交易={_trade_summary(portfolio) if portfolio else '-'} 冲击={_shock_summary(record)}"
    )


def render_records_html(records: Sequence[EvaluationRecord]) -> str:
    grouped: dict[str, dict[str, list[EvaluationRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.ticker][record.analysis_date.isoformat()].append(record)

    parts = []
    for ticker, by_date in grouped.items():
        parts.append(f"<h2>{escape(ticker)}</h2>")
        for analysis_date, items in sorted(by_date.items()):
            parts.append(f"<h3>分析日期：{analysis_date}</h3>")
            for record in sorted(items, key=lambda item: item.horizon):
                market = record.market
                parts.append(
                    f"<h4>[{record.horizon}交易日|{market.sample_class.value}] 评估交易日：{record.horizon_end_session.isoformat()}，"
                    f"标的 {_percent(market.underlying_return)}，"
                    f"基准 {escape(market.benchmark_symbol)} {_percent(market.benchmark_return)}，超额 {_percent(market.excess_return)}，"
                    f"冲击：{escape(_shock_summary(record))}</h4>"
                )
                parts.append("<table><tr><th>角色</th><th>方向</th><th>置信度</th><th>原始判定</th><th>超额判定</th><th>交易模拟</th></tr>")
                for stage in record.stages:
                    parts.append(
                        f"<tr><td>{STAGE_LABELS.get(stage.stage, escape(stage.stage))}</td><td>{escape(stage.direction)}</td>"
                        f"<td>{stage.confidence:.0%}</td><td>{escape(stage.direction_status)}</td>"
                        f"<td>{escape(stage.excess_direction_status)}</td><td>{escape(_trade_summary(stage))}</td></tr>"
                    )
                parts.append("</table>")
    return "\n".join(parts)


def _return_summary(cell) -> str:
    if cell.risk is None:
        return "-"
    return (
        "；".join(
            f"{item.trade_intent}/{item.return_basis} n={item.n} 均值={_percent(item.mean_strategy_return)}"
            f" 对标的差值={_percent(item.mean_strategy_minus_underlying)}"
            for item in cell.risk.return_performance
        )
        or "-"
    )


def render_track_record_text(track_record: TrackRecord) -> str:
    lines = [f"Track Record（n < {track_record.min_samples} 时不显示比率）"]
    for cell in track_record.cells:
        lines.append(f"{cell.scope}/{cell.stage}/h{cell.horizon} n={cell.direction.n} 方向命中率={_rate(cell.direction.hit_rate)}")
        for bucket in cell.calibration:
            lines.append(
                f"置信度 [{bucket.lower:.0%}, {bucket.upper if bucket.upper is not None else '1 inclusive'}) "
                f"n={bucket.skill.n} 命中率={_rate(bucket.skill.hit_rate)}"
            )
        if cell.risk is not None:
            lines.append(f"成交率={_rate(cell.risk.fill_rate)} 剩余仓位止损率={_rate(cell.risk.residual_stop_rate)} {_return_summary(cell)}")
    return "\n".join(lines)


def render_track_record_html(track_record: TrackRecord) -> str:
    parts = [
        f"<h2>Track Record（样本少于 {track_record.min_samples} 时只显示 n）</h2>",
        "<table><tr><th>范围</th><th>角色</th><th>周期</th><th>干净样本 n</th><th>方向命中率</th><th>inconclusive 占比</th>"
        "<th>交易 n</th><th>成交率</th><th>止损率</th><th>剩余仓位止损率</th><th>分口径收益（非完整持仓收益）</th></tr>",
    ]
    for cell in track_record.cells:
        risk = cell.risk
        parts.append(
            f"<tr><td>{escape(cell.scope)}</td><td>{STAGE_LABELS.get(cell.stage, escape(cell.stage))}</td><td>{cell.horizon}交易日</td><td>{cell.direction.n}</td>"
            f"<td>{_rate(cell.direction.hit_rate)}</td><td>{_rate(cell.direction.inconclusive_share)}</td>"
            f"<td>{risk.n if risk else '-'}</td><td>{_rate(risk.fill_rate) if risk else '-'}</td>"
            f"<td>{_rate(risk.stop_rate) if risk else '-'}</td><td>{_rate(risk.residual_stop_rate) if risk else '-'}</td>"
            f"<td>{escape(_return_summary(cell))}</td></tr>"
        )
    parts.extend(["</table>", "<h3>置信度校准</h3><table><tr><th>范围/角色/周期</th><th>置信度区间</th><th>n</th><th>方向命中率</th></tr>"])
    for cell in track_record.cells:
        for bucket in cell.calibration:
            upper = f"{bucket.upper:.0%})" if bucket.upper is not None else "100%]"
            parts.append(
                f"<tr><td>{escape(cell.scope)}/{escape(cell.stage)}/{cell.horizon}交易日</td>"
                f"<td>[{bucket.lower:.0%}, {upper}</td><td>{bucket.skill.n}</td><td>{_rate(bucket.skill.hit_rate)}</td></tr>"
            )
    parts.extend(
        ["</table>", "<h3>PM 与分析师方向一致性（非因果归因）</h3><table><tr><th>范围/周期/分析师</th><th>一致 n/命中率</th><th>不一致 n/命中率</th></tr>"]
    )
    for item in track_record.attribution:
        parts.append(
            f"<tr><td>{escape(item.scope)}/{item.horizon}交易日/{escape(item.analyst_stage)}</td>"
            f"<td>{item.agreed.n}/{_rate(item.agreed.hit_rate)}</td><td>{item.disagreed.n}/{_rate(item.disagreed.hit_rate)}</td></tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)

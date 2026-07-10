"""Tests for structured output models (Pydantic) and AgentScope integration."""

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tradingscope.agents.output import (
    AnalysisResult,
    AnalystDecision,
    AnalystReports,
    AnalystStructuredOutput,
    FundamentalsAnalystStructuredOutput,
    MarketAnalystStructuredOutput,
    NewsAnalystStructuredOutput,
    PortfolioDecision,
    PortfolioStructuredOutput,
    PredictionData,
    ResearchDecision,
    ResearchManagerStructuredOutput,
    SocialMediaAnalystStructuredOutput,
    TraderDecision,
    TraderStructuredOutput,
)
from tradingscope.agents.utils.context import AgentContext

# --- PredictionData (Pydantic) tests ---


class TestPredictionData:
    def test_defaults(self):
        p = PredictionData()
        assert p.direction == "neutral"
        assert p.action == "hold"
        assert p.confidence == 0.5
        assert p.entry_price is None

    def test_model_dump_roundtrip(self):
        p = PredictionData(action="buy", direction="bullish", confidence=0.7, entry_price=150.0)
        d = p.model_dump()
        p2 = PredictionData.model_validate(d)
        assert p2.action == "buy"
        assert p2.direction == "bullish"
        assert p2.confidence == 0.7
        assert p2.entry_price == 150.0

    def test_literal_validation(self):
        PredictionData(action="buy", direction="bullish")
        with pytest.raises(ValidationError):
            PredictionData(action="invalid_action")


# --- TraderStructuredOutput (AgentScope target model) ---


class TestTraderStructuredOutput:
    def test_defaults(self):
        t = TraderStructuredOutput()
        assert t.direction == "neutral"
        assert t.action == "hold"
        assert t.confidence == 0.5
        assert t.position_advice == ""
        assert t.risk_score is None

    def test_model_dump(self):
        t = TraderStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.7,
            entry_price=150.0,
            target_price=160.0,
            stop_loss=145.0,
            position_advice="中等仓位",
            risk_score=0.4,
            time_stop_days=3,
        )
        d = t.model_dump()
        assert d["action"] == "buy"
        assert d["confidence"] == 0.7
        assert d["entry_price"] == 150.0
        assert d["position_advice"] == "中等仓位"

    def test_confidence_bounds(self):
        TraderStructuredOutput(confidence=0.5)
        with pytest.raises(ValidationError):
            TraderStructuredOutput(confidence=1.5)
        with pytest.raises(ValidationError):
            TraderStructuredOutput(confidence=-0.1)


# --- PortfolioStructuredOutput (AgentScope target model) ---


class TestPortfolioStructuredOutput:
    def test_defaults(self):
        p = PortfolioStructuredOutput()
        assert p.direction == "neutral"
        assert p.action == "hold"
        assert p.aggressive_viewpoint == ""

    def test_model_dump(self):
        p = PortfolioStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.75,
            aggressive_viewpoint="技术面强势",
            conservative_viewpoint="估值偏高",
            risk_control_measures=["严格止损", "分批建仓"],
        )
        d = p.model_dump()
        assert d["action"] == "buy"
        assert d["aggressive_viewpoint"] == "技术面强势"
        assert d["risk_control_measures"] == ["严格止损", "分批建仓"]


# --- TraderDecision (from_structured_output) ---


class TestTraderDecision:
    def test_from_structured_output(self):
        data = TraderStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.7,
            entry_price=150.0,
            target_price=160.0,
            stop_loss=145.0,
            position_advice="中等仓位",
            risk_score=0.4,
            time_stop_days=3,
        ).model_dump()
        td = TraderDecision.from_structured_output(data)
        assert td.prediction.action == "buy"
        assert td.prediction.entry_price == 150.0
        assert td.prediction.confidence == 0.7
        assert td.position_advice == "中等仓位"
        assert td.risk_score == 0.4
        assert td.time_stop_days == 3

    def test_from_structured_output_defaults(self):
        td = TraderDecision.from_structured_output({})
        assert td.prediction.action == "hold"
        assert td.prediction.confidence == 0.5

    def test_fallback_from_prediction_data(self):
        td = TraderDecision(prediction=PredictionData(action="buy", confidence=0.7))
        assert td.prediction.action == "buy"


# --- PortfolioDecision (from_structured_output) ---


class TestPortfolioDecision:
    def test_from_structured_output(self):
        data = PortfolioStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.75,
            entry_price=151.0,
            target_price=161.0,
            stop_loss=146.0,
            aggressive_viewpoint="技术面强势",
            conservative_viewpoint="估值偏高",
            neutral_viewpoint="观望",
            adopted_reasoning="趋势明确",
            position_advice="中等仓位",
            risk_score=0.3,
            risk_control_measures=["严格止损", "分批建仓"],
        ).model_dump()
        pd = PortfolioDecision.from_structured_output(data)
        assert pd.prediction.action == "buy"
        assert pd.prediction.confidence == 0.75
        assert pd.prediction.entry_price == 151.0
        assert pd.viewpoints_aggressive == "技术面强势"
        assert pd.viewpoints_conservative == "估值偏高"
        assert pd.adopted_reasoning == "趋势明确"
        assert pd.risk_control_measures == ["严格止损", "分批建仓"]

    def test_from_structured_output_defaults(self):
        pd = PortfolioDecision.from_structured_output({})
        assert pd.prediction.action == "hold"


# --- AnalysisResult ---


class TestAnalysisResult:
    def test_defaults(self):
        r = AnalysisResult()
        assert r.ticker == ""
        assert r.created_at != ""

    def test_model_dump_has_all_fields(self):
        r = AnalysisResult(ticker="AAPL", trade_date="2025-01-01")
        d = r.model_dump()
        assert "ticker" in d
        assert "analyst_reports" in d
        assert "trader_decision" in d
        assert "portfolio_decision" in d

    def test_to_json_is_valid(self):
        r = AnalysisResult(ticker="AAPL", trade_date="2025-01-01")
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["ticker"] == "AAPL"

    def test_from_context_with_structured_output(self, mock_context):
        ctx = mock_context
        ctx.company_of_interest = "AAPL"
        ctx.trade_date = "2025-01-01"
        ctx.latest_trading_date = "2024-12-31"
        ctx.market_report = "Market is up"
        ctx.fundamentals_report = "Revenue growing"
        ctx.news_report = "New product launch"
        ctx.sentiment_report = "Positive sentiment"
        ctx.researcher_investment_plan = "建议买入"
        ctx.trader_investment_plan = "操作建议：买入"
        ctx.final_trade_decision = "交易决策：买入"

        trader_data = TraderStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.7,
            entry_price=150.0,
            target_price=160.0,
            stop_loss=145.0,
            position_advice="中等仓位",
            risk_score=0.4,
            time_stop_days=3,
        ).model_dump()

        portfolio_data = PortfolioStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.75,
            entry_price=151.0,
            target_price=161.0,
            stop_loss=146.0,
            aggressive_viewpoint="技术面强势",
            conservative_viewpoint="估值偏高",
            adopted_reasoning="趋势明确",
            position_advice="中等仓位",
            risk_score=0.3,
            risk_control_measures=["严格止损", "分批建仓"],
        ).model_dump()

        result = AnalysisResult.from_context(
            ctx,
            trader_structured=trader_data,
            portfolio_structured=portfolio_data,
        )

        assert result.ticker == "AAPL"
        assert result.trader_decision.prediction.action == "buy"
        assert result.trader_decision.prediction.entry_price == 150.0
        assert result.trader_decision.prediction.confidence == 0.7
        assert result.trader_decision.position_advice == "中等仓位"
        assert result.portfolio_decision.prediction.action == "buy"
        assert result.portfolio_decision.prediction.confidence == 0.75
        assert result.portfolio_decision.viewpoints_aggressive == "技术面强势"

        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["ticker"] == "AAPL"
        assert parsed["trader_decision"]["prediction"]["action"] == "buy"

    def test_from_context_with_fallback(self, mock_context):
        """When no structured output is provided, falls back to regex extraction."""
        ctx = mock_context
        ctx.company_of_interest = "AAPL"
        ctx.trade_date = "2025-01-01"
        ctx.latest_trading_date = "2024-12-31"
        ctx.market_report = "Market report"
        ctx.trader_investment_plan = "建议买入"
        ctx.final_trade_decision = "交易决策：买入\n置信度：0.8"

        result = AnalysisResult.from_context(ctx)
        assert result.ticker == "AAPL"
        # Fallback uses regex-based extraction
        assert result.trader_decision.prediction.action == "buy"
        assert result.portfolio_decision.prediction.action == "buy"


# --- AgentContext extract_prediction_data with source parameter ---


class TestExtractPredictionDataWithSource:
    def test_explicit_source(self, mock_context):
        ctx = mock_context
        ctx.final_trade_decision = "交易决策：买入\n置信度：0.8"

        trader_data = ctx.extract_prediction_data(source="操作建议：卖出\n置信度：0.3")
        assert trader_data["action"] == "sell"
        assert trader_data["confidence"] == 0.3

        default_data = ctx.extract_prediction_data()
        assert default_data["action"] == "buy"

    def test_source_none_uses_default(self, mock_context):
        ctx = mock_context
        ctx.final_trade_decision = "交易决策：持有"
        data = ctx.extract_prediction_data(source=None)
        assert data["action"] == "hold"


# --- Analyst structured output model tests ---


class TestAnalystStructuredOutput:
    def test_defaults(self):
        a = AnalystStructuredOutput()
        assert a.direction == "neutral"
        assert a.action == "hold"
        assert a.confidence == 0.5
        assert a.reasoning == ""

    def test_model_dump(self):
        a = AnalystStructuredOutput(direction="bullish", action="buy", confidence=0.8, reasoning="趋势向上")
        d = a.model_dump()
        assert d["direction"] == "bullish"
        assert d["action"] == "buy"
        assert d["confidence"] == 0.8
        assert d["reasoning"] == "趋势向上"


class TestMarketAnalystStructuredOutput:
    def test_defaults(self):
        m = MarketAnalystStructuredOutput()
        assert m.direction == "neutral"
        assert m.signal_strength == ""
        assert m.invalidation_conditions == ""

    def test_model_dump(self):
        m = MarketAnalystStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.7,
            signal_strength="强",
            invalidation_conditions="收盘跌破10EMA",
        )
        d = m.model_dump()
        assert d["signal_strength"] == "强"
        assert d["invalidation_conditions"] == "收盘跌破10EMA"


class TestFundamentalsAnalystStructuredOutput:
    def test_defaults(self):
        f = FundamentalsAnalystStructuredOutput()
        assert f.valuation_assessment == ""
        assert f.key_catalysts == []
        assert f.key_risks == []

    def test_model_dump(self):
        f = FundamentalsAnalystStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.6,
            valuation_assessment="合理",
            key_catalysts=["财报超预期"],
            key_risks=["竞争加剧"],
        )
        d = f.model_dump()
        assert d["valuation_assessment"] == "合理"
        assert d["key_catalysts"] == ["财报超预期"]
        assert d["key_risks"] == ["竞争加剧"]


class TestNewsAnalystStructuredOutput:
    def test_defaults(self):
        n = NewsAnalystStructuredOutput()
        assert n.sentiment == "neutral"
        assert n.key_events == []

    def test_model_dump(self):
        n = NewsAnalystStructuredOutput(
            direction="bearish",
            action="sell",
            confidence=0.6,
            sentiment="negative",
            key_events=["监管调查", "高管离职"],
        )
        d = n.model_dump()
        assert d["sentiment"] == "negative"
        assert d["key_events"] == ["监管调查", "高管离职"]

    def test_sentiment_literal_validation(self):
        NewsAnalystStructuredOutput(sentiment="positive")
        with pytest.raises(ValidationError):
            NewsAnalystStructuredOutput(sentiment="invalid")


class TestSocialMediaAnalystStructuredOutput:
    def test_defaults(self):
        s = SocialMediaAnalystStructuredOutput()
        assert s.sentiment == "neutral"
        assert s.sentiment_score is None

    def test_model_dump(self):
        s = SocialMediaAnalystStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.7,
            sentiment="positive",
            sentiment_score=8.5,
        )
        d = s.model_dump()
        assert d["sentiment"] == "positive"
        assert d["sentiment_score"] == 8.5

    def test_sentiment_score_bounds(self):
        SocialMediaAnalystStructuredOutput(sentiment_score=5.0)
        with pytest.raises(ValidationError):
            SocialMediaAnalystStructuredOutput(sentiment_score=0.5)
        with pytest.raises(ValidationError):
            SocialMediaAnalystStructuredOutput(sentiment_score=11)


# --- ResearchManagerStructuredOutput tests ---


class TestResearchManagerStructuredOutput:
    def test_defaults(self):
        r = ResearchManagerStructuredOutput()
        assert r.direction == "neutral"
        assert r.action == "hold"
        assert r.confidence == 0.5
        assert r.bull_viewpoints == ""
        assert r.bear_viewpoints == ""

    def test_model_dump(self):
        r = ResearchManagerStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.75,
            bull_viewpoints="增长潜力大",
            bear_viewpoints="估值偏高",
            adopted_reasoning="增长前景更重要",
        )
        d = r.model_dump()
        assert d["bull_viewpoints"] == "增长潜力大"
        assert d["adopted_reasoning"] == "增长前景更重要"


# --- AnalystDecision tests ---


class TestAnalystDecision:
    def test_from_structured_output(self):
        data = MarketAnalystStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.7,
            signal_strength="强",
            invalidation_conditions="跌破10EMA",
        ).model_dump()
        ad = AnalystDecision.from_structured_output(data)
        assert ad.prediction.action == "buy"
        assert ad.prediction.confidence == 0.7
        assert ad.extra["signal_strength"] == "强"

    def test_from_structured_output_defaults(self):
        ad = AnalystDecision.from_structured_output({})
        assert ad.prediction.action == "hold"
        assert ad.extra == {}


# --- ResearchDecision tests ---


class TestResearchDecision:
    def test_from_structured_output(self):
        data = ResearchManagerStructuredOutput(
            direction="bullish",
            action="buy",
            confidence=0.75,
            bull_viewpoints="增长潜力大",
            bear_viewpoints="估值偏高",
            adopted_reasoning="增长前景更重要",
        ).model_dump()
        rd = ResearchDecision.from_structured_output(data)
        assert rd.prediction.action == "buy"
        assert rd.bull_viewpoints == "增长潜力大"
        assert rd.adopted_reasoning == "增长前景更重要"

    def test_from_structured_output_defaults(self):
        rd = ResearchDecision.from_structured_output({})
        assert rd.prediction.action == "hold"
        assert rd.bull_viewpoints == ""


# --- AnalystReports extended tests ---


class TestAnalystReportsExtended:
    def test_defaults_no_structured(self):
        r = AnalystReports()
        assert r.market == ""
        assert r.market_structured is None
        assert r.fundamentals_structured is None

    def test_with_structured(self):
        r = AnalystReports(
            market="市场报告",
            market_structured={"direction": "bullish", "action": "buy"},
        )
        d = r.model_dump()
        assert d["market"] == "市场报告"
        assert d["market_structured"]["direction"] == "bullish"


# --- AnalysisResult with analyst/research structured data ---


class TestAnalysisResultExtended:
    def test_from_context_with_all_structured(self, mock_context):
        ctx = mock_context
        ctx.company_of_interest = "AAPL"
        ctx.trade_date = "2025-01-01"
        ctx.latest_trading_date = "2024-12-31"
        ctx.market_report = "Market is up"
        ctx.fundamentals_report = "Revenue growing"
        ctx.news_report = "New product launch"
        ctx.sentiment_report = "Positive sentiment"

        result = AnalysisResult.from_context(
            ctx,
            market_structured=MarketAnalystStructuredOutput(
                direction="bullish",
                action="buy",
                confidence=0.7,
                signal_strength="强",
            ).model_dump(),
            fundamentals_structured=FundamentalsAnalystStructuredOutput(
                direction="bullish",
                action="buy",
                confidence=0.6,
            ).model_dump(),
            news_structured=NewsAnalystStructuredOutput(
                direction="neutral",
                action="hold",
                confidence=0.5,
                sentiment="neutral",
            ).model_dump(),
            social_media_structured=SocialMediaAnalystStructuredOutput(
                direction="bullish",
                action="buy",
                confidence=0.65,
                sentiment_score=7.5,
            ).model_dump(),
            research_structured=ResearchManagerStructuredOutput(
                direction="bullish",
                action="buy",
                confidence=0.75,
                bull_viewpoints="增长潜力大",
            ).model_dump(),
        )

        assert result.ticker == "AAPL"
        assert "market" in result.analyst_decisions
        assert result.analyst_decisions["market"]["prediction"]["action"] == "buy"
        assert result.analyst_decisions["social_media"]["extra"]["sentiment_score"] == 7.5
        assert result.research_decision.prediction.action == "buy"
        assert result.research_decision.bull_viewpoints == "增长潜力大"
        assert result.analyst_reports.market_structured["direction"] == "bullish"

        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["analyst_decisions"]["market"]["prediction"]["action"] == "buy"
        assert parsed["research_decision"]["prediction"]["action"] == "buy"

    def test_from_context_without_analyst_structured(self, mock_context):
        ctx = mock_context
        ctx.company_of_interest = "AAPL"
        ctx.trade_date = "2025-01-01"
        ctx.latest_trading_date = "2024-12-31"
        ctx.market_report = "Market report"

        result = AnalysisResult.from_context(ctx)
        assert result.analyst_decisions == {}
        assert result.research_decision.prediction.action == "hold"
        assert result.analyst_reports.market_structured is None


# --- Human-facing markdown report cleanup ---


class TestHumanFacingReportCleanup:
    def test_generate_full_report_removes_structured_json_blocks(self, mock_context):
        ctx = mock_context
        ctx.company_of_interest = "AAPL"
        ctx.trade_date = "2025-01-01"
        ctx.latest_trading_date = "2024-12-31"
        ctx.final_trade_decision = '''最终建议：卖出

```json
{
  "direction": "bearish",
  "action": "sell",
  "confidence": 0.8
}
```'''
        ctx.trader_investment_plan = '操作计划：减仓\n\njson { "action": "sell" }'

        report = ctx.generate_full_report_md()

        assert "最终建议：卖出" in report
        assert "操作计划：减仓" in report
        assert "```json" not in report
        assert '"direction"' not in report
        assert '"action"' not in report

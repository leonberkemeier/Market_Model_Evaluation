"""
Tests for Signal Generator.

Key behaviors to verify:
1. No signals generated when opportunities don't meet thresholds
2. All quality gates are enforced
3. Kelly-based position sizing works correctly
4. SELL signals only for held positions
"""

import pytest
from datetime import date
from src.trading.signal_generator import (
    SignalGenerator, 
    SignalGeneratorConfig,
    SignalType,
    SignalReason,
    TradeSignal,
    create_conservative_generator,
    create_aggressive_generator
)
from src.data_structures import ScoreResult, ScorerOutput


# === Fixtures ===

@pytest.fixture
def default_generator():
    """Default signal generator with standard thresholds."""
    return SignalGenerator()


@pytest.fixture
def conservative_generator():
    """Conservative generator with high thresholds."""
    return create_conservative_generator()


@pytest.fixture
def high_quality_score():
    """A score that should pass all quality gates."""
    return ScoreResult(
        ticker="AAPL",
        date=date(2025, 1, 15),
        model_name="xgboost",
        score=85.0,
        p_win=0.65,
        avg_win=0.03,
        avg_loss=0.015,
        ev=0.014,  # (0.65 * 0.03) - (0.35 * 0.015) = 0.0195 - 0.00525 = 0.01425
        confidence=0.75,
        data_points=100
    )


@pytest.fixture
def mediocre_score():
    """A score that's positive but doesn't meet thresholds."""
    return ScoreResult(
        ticker="MSFT",
        date=date(2025, 1, 15),
        model_name="xgboost",
        score=55.0,  # Below threshold
        p_win=0.52,  # Barely above 50%
        avg_win=0.02,
        avg_loss=0.018,
        ev=0.002,  # Positive but low
        confidence=0.45,  # Below threshold
        data_points=50
    )


@pytest.fixture
def negative_ev_score():
    """A score with negative expected value."""
    return ScoreResult(
        ticker="NFLX",
        date=date(2025, 1, 15),
        model_name="xgboost",
        score=35.0,
        p_win=0.40,
        avg_win=0.02,
        avg_loss=0.025,
        ev=-0.007,  # Negative EV
        confidence=0.6,
        data_points=80
    )


@pytest.fixture
def scorer_output_mixed(high_quality_score, mediocre_score, negative_ev_score):
    """Scorer output with mixed quality scores."""
    return ScorerOutput(
        date=date(2025, 1, 15),
        model_name="xgboost",
        scores=[high_quality_score, mediocre_score, negative_ev_score]
    )


# === Test: No Forced Trades ===

class TestNoForcedTrades:
    """Verify that generator doesn't force trades when opportunities are poor."""
    
    def test_no_signals_when_all_scores_mediocre(self, default_generator, mediocre_score):
        """Should return empty list when no scores meet thresholds."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[mediocre_score]
        )
        
        signals = default_generator.generate_signals(output)
        
        assert len(signals) == 0, "Should not generate signals for mediocre opportunities"
    
    def test_no_signals_when_all_negative_ev(self, default_generator, negative_ev_score):
        """Should return empty list when all EVs are negative."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[negative_ev_score]
        )
        
        signals = default_generator.generate_signals(output)
        
        assert len(signals) == 0, "Should not generate signals for negative EV"
    
    def test_no_signals_when_empty_scores(self, default_generator):
        """Should return empty list for empty input."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[]
        )
        
        signals = default_generator.generate_signals(output)
        
        assert len(signals) == 0
    
    def test_only_quality_signals_from_mixed_input(self, default_generator, scorer_output_mixed):
        """Should only return signals for high-quality opportunities."""
        signals = default_generator.generate_signals(scorer_output_mixed)
        
        # Should only have signal for AAPL (high quality)
        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"
        assert signals[0].signal_type == SignalType.BUY


# === Test: Quality Gates ===

class TestQualityGates:
    """Verify each quality gate is enforced."""
    
    def test_ev_gate(self, default_generator):
        """Score with low EV should be rejected."""
        score = ScoreResult(
            ticker="TEST",
            date=date(2025, 1, 15),
            model_name="test",
            score=80.0,  # Good
            p_win=0.60,  # Good
            avg_win=0.008,
            avg_loss=0.007,
            ev=0.002,  # Below 0.005 threshold
            confidence=0.7,
            data_points=50
        )
        output = ScorerOutput(date=date(2025, 1, 15), model_name="test", scores=[score])
        
        signals = default_generator.generate_signals(output)
        assert len(signals) == 0, "Low EV should be rejected"
    
    def test_score_gate(self, default_generator):
        """Score below threshold should be rejected."""
        score = ScoreResult(
            ticker="TEST",
            date=date(2025, 1, 15),
            model_name="test",
            score=65.0,  # Below 70 threshold
            p_win=0.65,
            avg_win=0.03,
            avg_loss=0.015,
            ev=0.015,
            confidence=0.7,
            data_points=50
        )
        output = ScorerOutput(date=date(2025, 1, 15), model_name="test", scores=[score])
        
        signals = default_generator.generate_signals(output)
        assert len(signals) == 0, "Low score should be rejected"
    
    def test_confidence_gate(self, default_generator):
        """Low confidence should be rejected."""
        score = ScoreResult(
            ticker="TEST",
            date=date(2025, 1, 15),
            model_name="test",
            score=80.0,
            p_win=0.65,
            avg_win=0.03,
            avg_loss=0.015,
            ev=0.015,
            confidence=0.5,  # Below 0.6 threshold
            data_points=50
        )
        output = ScorerOutput(date=date(2025, 1, 15), model_name="test", scores=[score])
        
        signals = default_generator.generate_signals(output)
        assert len(signals) == 0, "Low confidence should be rejected"
    
    def test_p_win_gate(self, default_generator):
        """Low win probability should be rejected."""
        score = ScoreResult(
            ticker="TEST",
            date=date(2025, 1, 15),
            model_name="test",
            score=80.0,
            p_win=0.52,  # Below 0.55 threshold
            avg_win=0.05,
            avg_loss=0.02,
            ev=0.016,
            confidence=0.7,
            data_points=50
        )
        output = ScorerOutput(date=date(2025, 1, 15), model_name="test", scores=[score])
        
        signals = default_generator.generate_signals(output)
        assert len(signals) == 0, "Low p_win should be rejected"
    
    def test_data_points_gate(self, default_generator):
        """Insufficient data points should be rejected."""
        score = ScoreResult(
            ticker="TEST",
            date=date(2025, 1, 15),
            model_name="test",
            score=80.0,
            p_win=0.65,
            avg_win=0.03,
            avg_loss=0.015,
            ev=0.015,
            confidence=0.7,
            data_points=10  # Below 20 threshold
        )
        output = ScorerOutput(date=date(2025, 1, 15), model_name="test", scores=[score])
        
        signals = default_generator.generate_signals(output)
        assert len(signals) == 0, "Insufficient data points should be rejected"


# === Test: Signal Generation ===

class TestSignalGeneration:
    """Test correct signal generation for valid opportunities."""
    
    def test_buy_signal_created_for_quality_score(self, default_generator, high_quality_score):
        """High quality score should generate BUY signal."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[high_quality_score]
        )
        
        signals = default_generator.generate_signals(output)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == SignalType.BUY
        assert signal.ticker == "AAPL"
        assert signal.ev == high_quality_score.ev
        assert signal.suggested_weight > 0
    
    def test_sell_signal_for_held_deteriorating_position(self, default_generator, negative_ev_score):
        """Should generate SELL for held position with negative EV."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[negative_ev_score]
        )
        
        # We hold NFLX
        signals = default_generator.generate_signals(
            output, 
            current_holdings=["NFLX"]
        )
        
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SELL
        assert signals[0].ticker == "NFLX"
        assert signals[0].reason == SignalReason.EXIT_DETERIORATING
    
    def test_no_sell_signal_for_unowned_stock(self, default_generator, negative_ev_score):
        """Should NOT generate SELL for stock we don't own."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[negative_ev_score]
        )
        
        # We don't hold NFLX
        signals = default_generator.generate_signals(
            output, 
            current_holdings=["AAPL", "GOOGL"]
        )
        
        assert len(signals) == 0


# === Test: Position Sizing ===

class TestPositionSizing:
    """Test Kelly-based position sizing."""
    
    def test_suggested_weight_is_positive(self, default_generator, high_quality_score):
        """BUY signals should have positive weight."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[high_quality_score]
        )
        
        signals = default_generator.generate_signals(output)
        
        assert signals[0].suggested_weight > 0
        assert signals[0].suggested_weight <= 0.15  # Max position size
    
    def test_quantity_calculated_with_capital_and_price(self, default_generator, high_quality_score):
        """Should calculate share quantity when capital and price provided."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[high_quality_score]
        )
        
        signals = default_generator.generate_signals(
            output,
            capital=100000,
            prices={"AAPL": 180.0}
        )
        
        assert signals[0].suggested_quantity is not None
        assert signals[0].suggested_quantity > 0
    
    def test_sell_signal_has_zero_weight(self, default_generator, negative_ev_score):
        """SELL signals should have zero weight (exit entire position)."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[negative_ev_score]
        )
        
        signals = default_generator.generate_signals(
            output,
            current_holdings=["NFLX"]
        )
        
        assert signals[0].suggested_weight == 0.0


# === Test: Portfolio Constraints ===

class TestPortfolioConstraints:
    """Test portfolio-level constraints."""
    
    def test_max_positions_respected(self, default_generator):
        """Should not exceed max positions."""
        # Create many high-quality scores
        scores = []
        for i in range(30):
            scores.append(ScoreResult(
                ticker=f"STOCK{i}",
                date=date(2025, 1, 15),
                model_name="test",
                score=85.0,
                p_win=0.65,
                avg_win=0.03,
                avg_loss=0.015,
                ev=0.015,
                confidence=0.7,
                data_points=50
            ))
        
        output = ScorerOutput(date=date(2025, 1, 15), model_name="test", scores=scores)
        signals = default_generator.generate_signals(output)
        
        # Default max is 20
        assert len(signals) <= 20
    
    def test_skip_already_held_stocks(self, default_generator, high_quality_score):
        """Should not generate BUY for already held stocks."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[high_quality_score]
        )
        
        signals = default_generator.generate_signals(
            output,
            current_holdings=["AAPL"]  # Already hold AAPL
        )
        
        # Should not generate BUY for AAPL since we already hold it
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) == 0


# === Test: Conservative vs Aggressive ===

class TestGeneratorProfiles:
    """Test different generator profiles."""
    
    def test_conservative_rejects_more(self, conservative_generator, high_quality_score):
        """Conservative generator should be more selective."""
        # Modify score to be borderline
        borderline_score = ScoreResult(
            ticker="TEST",
            date=date(2025, 1, 15),
            model_name="test",
            score=72.0,  # Passes default (70), fails conservative (80)
            p_win=0.58,  # Passes default (55), fails conservative (60)
            avg_win=0.025,
            avg_loss=0.015,
            ev=0.008,  # Passes default (0.5%), fails conservative (1%)
            confidence=0.65,
            data_points=50
        )
        
        output = ScorerOutput(date=date(2025, 1, 15), model_name="test", scores=[borderline_score])
        
        default_gen = SignalGenerator()
        conservative_gen = create_conservative_generator()
        
        default_signals = default_gen.generate_signals(output)
        conservative_signals = conservative_gen.generate_signals(output)
        
        assert len(default_signals) == 1, "Default should accept borderline"
        assert len(conservative_signals) == 0, "Conservative should reject borderline"


# === Test: Opportunity Summary ===

class TestOpportunitySummary:
    """Test the opportunity summary feature."""
    
    def test_summary_with_mixed_scores(self, default_generator, scorer_output_mixed):
        """Summary should correctly analyze opportunity distribution."""
        summary = default_generator.get_opportunity_summary(scorer_output_mixed)
        
        assert summary["status"] == "analyzed"
        assert summary["total_scores"] == 3
        assert summary["positive_ev_count"] >= 1  # At least AAPL
        assert summary["qualified_opportunities"] == 1  # Only AAPL
        assert summary["recommendation"] == "trade"  # Has qualified opportunities
    
    def test_summary_recommends_wait_when_no_opportunities(self, default_generator, mediocre_score):
        """Summary should recommend 'wait' when no quality opportunities."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="test",
            scores=[mediocre_score]
        )
        
        summary = default_generator.get_opportunity_summary(output)
        
        assert summary["qualified_opportunities"] == 0
        assert summary["recommendation"] == "wait"


# === Test: Signal Serialization ===

class TestSignalSerialization:
    """Test signal can be serialized for API transmission."""
    
    def test_to_dict(self, default_generator, high_quality_score):
        """Signal should serialize to dictionary."""
        output = ScorerOutput(
            date=date(2025, 1, 15),
            model_name="xgboost",
            scores=[high_quality_score]
        )
        
        signals = default_generator.generate_signals(output)
        signal_dict = signals[0].to_dict()
        
        assert "ticker" in signal_dict
        assert "signal_type" in signal_dict
        assert signal_dict["signal_type"] == "BUY"
        assert "ev" in signal_dict
        assert "suggested_weight" in signal_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

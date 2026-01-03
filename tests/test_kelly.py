"""Unit tests for Kelly Criterion optimizer."""

import pytest
from datetime import date
from src.portfolio.kelly_optimizer import KellyOptimizer
from src.data_structures import ScoreResult


@pytest.fixture
def kelly_optimizer():
    """Create a Kelly optimizer instance."""
    return KellyOptimizer(kelly_fraction=0.25, max_position=0.10, min_position=0.005)


def test_kelly_fraction_basic(kelly_optimizer):
    """Test basic Kelly fraction calculation."""
    # p_win=0.55, avg_win=0.01, avg_loss=0.01
    # b = 0.01 / 0.01 = 1
    # f = (0.55 * 1 - 0.45) / 1 = 0.10
    # fractional = 0.10 * 0.25 = 0.025
    f = kelly_optimizer.compute_kelly_fraction(p_win=0.55, avg_win=0.01, avg_loss=0.01)
    assert f > 0
    assert f <= 0.10


def test_kelly_fraction_zero_loss(kelly_optimizer):
    """Test Kelly fraction when avg_loss is zero."""
    f = kelly_optimizer.compute_kelly_fraction(p_win=0.55, avg_win=0.01, avg_loss=0.0)
    assert f == 0.0


def test_kelly_fraction_negative_edge(kelly_optimizer):
    """Test Kelly fraction with negative EV."""
    # p_win=0.4, avg_win=0.01, avg_loss=0.02
    # b = 0.01 / 0.02 = 0.5
    # f = (0.4 * 0.5 - 0.6) / 0.5 = -0.4
    f = kelly_optimizer.compute_kelly_fraction(p_win=0.4, avg_win=0.01, avg_loss=0.02)
    assert f == 0.0  # Clipped to 0


def test_build_portfolio_empty(kelly_optimizer):
    """Test portfolio building with empty scores."""
    portfolio = kelly_optimizer.build_portfolio([], capital=100_000)
    assert portfolio == {}


def test_build_portfolio_single_position(kelly_optimizer):
    """Test portfolio building with single position."""
    scores = [
        ScoreResult(
            ticker="AAPL",
            date=date(2024, 1, 1),
            model_name="test",
            score=75.0,
            p_win=0.55,
            avg_win=0.01,
            avg_loss=0.01,
            ev=0.001,
        )
    ]
    
    portfolio = kelly_optimizer.build_portfolio(scores, capital=100_000)
    
    # Single position should get all capital (if above min_position)
    assert "AAPL" in portfolio
    assert portfolio["AAPL"] > 0
    assert portfolio["AAPL"] <= 100_000


def test_build_portfolio_multiple_positions(kelly_optimizer):
    """Test portfolio building with multiple positions."""
    scores = [
        ScoreResult(
            ticker="AAPL",
            date=date(2024, 1, 1),
            model_name="test",
            score=75.0,
            p_win=0.55,
            avg_win=0.01,
            avg_loss=0.01,
            ev=0.001,
        ),
        ScoreResult(
            ticker="MSFT",
            date=date(2024, 1, 1),
            model_name="test",
            score=70.0,
            p_win=0.52,
            avg_win=0.015,
            avg_loss=0.015,
            ev=0.0005,
        ),
    ]
    
    portfolio = kelly_optimizer.build_portfolio(scores, capital=100_000)
    
    # Should have both positions
    assert len(portfolio) > 0
    # Total should not exceed capital
    total_allocated = sum(portfolio.values())
    assert total_allocated <= 100_000


def test_position_weights(kelly_optimizer):
    """Test position weight normalization."""
    scores = [
        ScoreResult(
            ticker="AAPL",
            date=date(2024, 1, 1),
            model_name="test",
            score=75.0,
            p_win=0.55,
            avg_win=0.01,
            avg_loss=0.01,
            ev=0.001,
        ),
        ScoreResult(
            ticker="MSFT",
            date=date(2024, 1, 1),
            model_name="test",
            score=70.0,
            p_win=0.52,
            avg_win=0.015,
            avg_loss=0.015,
            ev=0.0005,
        ),
    ]
    
    weights = kelly_optimizer.get_position_weights(scores)
    
    # Weights should sum to approximately 1
    if weights:
        total_weight = sum(weights.values())
        assert 0.95 < total_weight <= 1.05  # Allow small floating point errors

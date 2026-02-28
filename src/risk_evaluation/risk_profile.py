"""
RiskProfile — standardized output from all risk evaluators.

Every risk evaluator (historical, GARCH, regime-conditional) produces
a RiskProfile per stock. This feeds into Monte Carlo simulation and
Kelly position sizing.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional
import numpy as np


@dataclass
class RiskProfile:
    """
    Risk characteristics for a single stock on a given date.

    This replaces the old ScoreResult's return-prediction fields
    with statistically grounded risk metrics.
    """
    ticker: str
    evaluation_date: date
    evaluator_name: str  # "historical", "garch", "regime_conditional"

    # Volatility
    annualized_vol: float          # Annualized standard deviation of returns
    current_vol: float             # Recent (e.g. 20-day) realized volatility
    vol_regime: str = "normal"     # "low", "normal", "high"

    # Market sensitivity
    beta: float = 1.0              # Beta to market benchmark (SPY)

    # Drawdown risk
    max_drawdown: float = 0.0      # Maximum historical drawdown (negative)
    current_drawdown: float = 0.0  # Current drawdown from peak

    # Return distribution (historical)
    mean_return: float = 0.0       # Annualized mean return
    sharpe: float = 0.0            # Sharpe ratio (annualized)
    skewness: float = 0.0          # Return distribution skewness
    kurtosis: float = 3.0          # Return distribution kurtosis (3.0 = normal)

    # Win/loss stats (for Kelly inputs)
    win_rate: float = 0.5          # Fraction of positive-return days
    avg_win: float = 0.0           # Mean positive daily return
    avg_loss: float = 0.0          # Mean negative daily return (as positive number)
    win_loss_ratio: float = 1.0    # avg_win / avg_loss

    # Value at Risk
    var_95: float = 0.0            # 5th percentile daily return
    var_99: float = 0.0            # 1st percentile daily return
    cvar_95: float = 0.0           # Expected shortfall (mean of worst 5%)

    # Regime
    current_regime: Optional[str] = None  # "bull", "bear", "sideways"
    regime_confidence: float = 0.0

    # Correlation to market
    correlation_to_market: float = 0.0

    # Metadata
    confidence: float = 0.5        # Evaluator's confidence in this profile (0-1)
    data_points: int = 0           # Number of observations used
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    @property
    def expected_value(self) -> float:
        """Expected value per trade (Kelly input)."""
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)

    @property
    def is_tradeable(self) -> bool:
        """Whether this stock has sufficient data and acceptable risk."""
        return (
            self.data_points >= 60
            and self.annualized_vol > 0
            and self.confidence >= 0.3
        )

    def validate(self) -> bool:
        """Validate profile integrity."""
        checks = {
            "vol_positive": self.annualized_vol >= 0,
            "win_rate_range": 0 <= self.win_rate <= 1,
            "avg_win_positive": self.avg_win >= 0,
            "avg_loss_positive": self.avg_loss >= 0,
            "confidence_range": 0 <= self.confidence <= 1,
            "data_points_positive": self.data_points >= 0,
        }
        return all(checks.values())

    def to_kelly_inputs(self) -> Dict[str, float]:
        """Extract inputs needed for Kelly Criterion calculation."""
        return {
            "win_prob": self.win_rate,
            "win_loss_ratio": self.win_loss_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expected_return": self.expected_value,
        }

    def to_monte_carlo_inputs(self) -> Dict[str, float]:
        """Extract inputs needed for Monte Carlo simulation."""
        return {
            "mean_return": self.mean_return / 252,  # Daily
            "volatility": self.annualized_vol / np.sqrt(252),  # Daily
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "current_regime": self.current_regime,
        }

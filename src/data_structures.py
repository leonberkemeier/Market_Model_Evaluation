"""
Data structures for the model_regime_comparison project.
Standardized objects for scoring, portfolio management, and backtest results.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional, List
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""
    LINEAR = "linear"
    CNN = "cnn"
    XGBOOST = "xgboost"
    LLM = "llm"


@dataclass
class ScoreResult:
    """
    Standardized output from any scorer model.
    All models produce this format for consistent comparison.
    """
    ticker: str
    date: date
    model_name: str  # "linear", "cnn", "xgboost", "llm"
    
    # Core EV components
    score: float  # 0-100 normalized percentile rank
    p_win: float  # 0-1 probability of favorable move
    avg_win: float  # % return if right (as decimal, e.g., 0.02 = 2%)
    avg_loss: float  # % return if wrong (as decimal, e.g., 0.01 = 1%)
    ev: float  # Expected value = (p_win * avg_win) - ((1 - p_win) * avg_loss)
    
    # Metadata
    confidence: float = 0.5  # Model confidence 0-1
    data_points: int = 0  # How many observations support this?
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)  # Model-specific data
    
    def validate(self) -> bool:
        """Validate score integrity."""
        checks = {
            'p_win_range': 0 <= self.p_win <= 1,
            'score_range': 0 <= self.score <= 100,
            'payoff_reasonable': abs(self.avg_win) < 1.0 and abs(self.avg_loss) < 1.0,
            'payoff_positive': self.avg_win >= 0 and self.avg_loss >= 0,
            'ev_reasonable': -1.0 < self.ev < 1.0,
            'confidence_range': 0 <= self.confidence <= 1,
            'data_points_positive': self.data_points >= 0,
        }
        return all(checks.values())


@dataclass
class Position:
    """Single position in a portfolio."""
    ticker: str
    quantity: float  # Number of shares
    entry_price: float  # Entry price
    entry_date: date
    sector: str
    
    def value(self, current_price: float) -> float:
        """Calculate current position value."""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return (current_price - self.entry_price) * self.quantity
    
    def return_pct(self, current_price: float) -> float:
        """Calculate return percentage."""
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price


@dataclass
class Portfolio:
    """Portfolio state at a point in time."""
    date: date
    capital: float  # Total capital
    cash: float  # Available cash
    positions: Dict[str, Position] = field(default_factory=dict)  # ticker -> Position
    
    def total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            pos.value(prices.get(pos.ticker, pos.entry_price))
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def get_holdings_list(self) -> List[str]:
        """Get list of tickers currently held."""
        return list(self.positions.keys())
    
    def get_position_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get weight of each position in portfolio."""
        total_value = self.total_value(prices)
        if total_value == 0:
            return {}
        return {
            ticker: pos.value(prices.get(ticker, pos.entry_price)) / total_value
            for ticker, pos in self.positions.items()
        }


@dataclass
class Trade:
    """Single trade execution."""
    date: date
    ticker: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    sector: str
    model_name: Optional[str] = None  # Which model recommended this trade
    reason: str = ""
    
    def cost(self) -> float:
        """Total cost of trade."""
        return self.quantity * self.price


@dataclass
class DailyMetrics:
    """Performance metrics for a single day."""
    date: date
    portfolio_value: float
    daily_return: float  # (value_today - value_yesterday) / value_yesterday
    daily_pnl: float  # $ amount
    holdings_count: int
    cash_position: float
    
    
@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics over a period."""
    model_name: str
    period_start: date
    period_end: date
    sector: Optional[str] = None  # If sector-specific
    
    total_return: float = 0.0  # (ending_value - starting_value) / starting_value
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0  # annual_return / max_drawdown
    
    win_rate: float = 0.0  # % of positive days
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0  # sum_wins / sum_losses
    
    trades_executed: int = 0
    turnover: float = 0.0  # rebalance intensity
    
    metadata: Dict = field(default_factory=dict)


@dataclass
class ScorerOutput:
    """Complete output from scorer for all stocks on a date."""
    date: date
    model_name: str
    scores: List[ScoreResult]  # One per stock
    computed_at: datetime = field(default_factory=datetime.now)
    
    def get_top_n(self, n: int) -> List[ScoreResult]:
        """Get top n scores by percentile rank."""
        sorted_scores = sorted(self.scores, key=lambda s: s.score, reverse=True)
        return sorted_scores[:n]
    
    def get_bottom_n(self, n: int) -> List[ScoreResult]:
        """Get bottom n scores by percentile rank."""
        sorted_scores = sorted(self.scores, key=lambda s: s.score)
        return sorted_scores[:n]


@dataclass
class BacktestResult:
    """Complete backtest results for one model on one period."""
    model_name: str
    period_start: date
    period_end: date
    sector: Optional[str] = None
    
    portfolio_values: List[float] = field(default_factory=list)
    dates: List[date] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    trade_log: List[Trade] = field(default_factory=list)
    daily_metrics: List[DailyMetrics] = field(default_factory=list)
    
    metrics: Optional[PerformanceMetrics] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'sector': self.sector,
            'total_return': self.metrics.total_return if self.metrics else None,
            'sharpe_ratio': self.metrics.sharpe_ratio if self.metrics else None,
            'max_drawdown': self.metrics.max_drawdown if self.metrics else None,
            'trades_executed': self.metrics.trades_executed if self.metrics else None,
        }

"""Kelly Criterion portfolio optimization."""

from typing import Dict, List, Tuple
from ..data_structures import ScoreResult


class KellyOptimizer:
    """
    Kelly Criterion based portfolio optimizer.
    Computes optimal position sizes based on probability and payoff estimates.
    """
    
    def __init__(self, kelly_fraction: float = 0.25, max_position: float = 0.10, min_position: float = 0.005):
        """
        Initialize Kelly optimizer.
        
        Args:
            kelly_fraction: Fractional Kelly to use (0.25 = Kelly/4 for safety)
            max_position: Maximum position size as fraction of capital
            min_position: Minimum position size to include in portfolio
        """
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position
    
    def compute_kelly_fraction(self, p_win: float, avg_win: float, avg_loss: float) -> float:
        """
        Compute Kelly fraction for a single position.
        
        Formula: f* = (p*b - q) / b
        where:
            p = probability of win
            q = probability of loss (1 - p)
            b = odds ratio (avg_win / avg_loss)
        
        Args:
            p_win: Probability of positive return (0-1)
            avg_win: Average win size as % (e.g., 0.02 = 2%)
            avg_loss: Average loss size as % (e.g., 0.01 = 1%)
            
        Returns:
            Kelly fraction (as fraction of capital)
        """
        if avg_loss <= 0:
            return 0.0
        
        q = 1.0 - p_win
        b = avg_win / avg_loss
        
        # Kelly formula
        f = (p_win * b - q) / b
        
        # Apply fractional Kelly for safety
        f = f * self.kelly_fraction
        
        # Clip to bounds
        f = max(0, min(f, self.max_position))
        
        return f
    
    def build_portfolio(self, scores: List[ScoreResult], capital: float) -> Dict[str, float]:
        """
        Build portfolio of positions from scores using Kelly Criterion.
        
        Args:
            scores: List of ScoreResult objects
            capital: Total capital to allocate
            
        Returns:
            Dictionary of ticker -> dollar amount to allocate
        """
        if not scores:
            return {}
        
        # Compute Kelly fraction for each position
        kelly_fractions = []
        for score in scores:
            kelly_f = self.compute_kelly_fraction(score.p_win, score.avg_win, score.avg_loss)
            kelly_fractions.append((score.ticker, kelly_f))
        
        # Normalize so total = 1
        total_kelly = sum(f for _, f in kelly_fractions)
        if total_kelly <= 0:
            # No positive Kelly positions, return empty portfolio
            return {}
        
        weights = [(t, f / total_kelly) for t, f in kelly_fractions]
        
        # Filter to minimum position size and convert to dollars
        portfolio = {}
        for ticker, weight in weights:
            if weight >= self.min_position:
                portfolio[ticker] = weight * capital
        
        return portfolio
    
    def get_position_weights(self, scores: List[ScoreResult]) -> Dict[str, float]:
        """
        Get normalized position weights (sum to 1).
        
        Args:
            scores: List of ScoreResult objects
            
        Returns:
            Dictionary of ticker -> weight (0-1)
        """
        if not scores:
            return {}
        
        # Compute Kelly fraction for each position
        kelly_fractions = []
        for score in scores:
            kelly_f = self.compute_kelly_fraction(score.p_win, score.avg_win, score.avg_loss)
            kelly_fractions.append((score.ticker, kelly_f))
        
        # Normalize so total = 1
        total_kelly = sum(f for _, f in kelly_fractions)
        if total_kelly <= 0:
            return {}
        
        weights = {}
        for ticker, kelly_f in kelly_fractions:
            weight = kelly_f / total_kelly
            if weight >= self.min_position:
                weights[ticker] = weight
        
        return weights

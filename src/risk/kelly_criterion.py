"""
Kelly Criterion for Layer IV Risk Management

Calculates optimal position size based on:
- Win probability
- Win/loss ratio
- Risk constraints

Formula: f* = (p * b - q) / b
Where:
- f* = fraction of capital to bet
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (avg_win / avg_loss)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass

from .monte_carlo import MonteCarloResult

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Results from Kelly Criterion calculation."""
    ticker: str
    kelly_fraction: float  # Optimal position size (fraction of capital)
    kelly_percentage: float  # As percentage
    
    # Inputs
    win_prob: float
    win_loss_ratio: float
    expected_return: float
    
    # Adjusted sizes
    half_kelly: float  # Conservative: 0.5 * kelly
    quarter_kelly: float  # Very conservative: 0.25 * kelly
    regime_adjusted: float  # Adjusted for market regime


class KellyCriterion:
    """
    Kelly Criterion calculator for optimal position sizing.
    
    Process:
    1. Take Monte Carlo simulation result
    2. Calculate win probability and win/loss ratio
    3. Compute Kelly fraction
    4. Apply safety constraints and regime adjustments
    
    Used in Sentinel flow:
    Layer IV → Monte Carlo simulates outcomes → Kelly sizes position → Layer V executes
    """
    
    def __init__(
        self,
        max_position_size: float = 0.15,  # Maximum 15% per position
        use_fractional_kelly: bool = True,  # Use half-Kelly by default
        fractional_multiplier: float = 0.5,  # 50% of full Kelly
        min_win_prob: float = 0.52  # Minimum 52% win rate to trade
    ):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            max_position_size: Maximum position size (as fraction)
            use_fractional_kelly: Use fractional Kelly (more conservative)
            fractional_multiplier: Multiplier for fractional Kelly (0.5 = half-Kelly)
            min_win_prob: Minimum win probability to take position
        """
        self.max_position_size = max_position_size
        self.use_fractional_kelly = use_fractional_kelly
        self.fractional_multiplier = fractional_multiplier
        self.min_win_prob = min_win_prob
        
        logger.info(
            f"KellyCriterion initialized: "
            f"max_size={max_position_size:.2%}, "
            f"fractional={use_fractional_kelly}, "
            f"multiplier={fractional_multiplier}"
        )
    
    def calculate(
        self,
        mc_result: MonteCarloResult,
        regime: Optional[str] = None
    ) -> KellyResult:
        """
        Calculate Kelly position size from Monte Carlo result.
        
        Args:
            mc_result: Monte Carlo simulation result
            regime: Current market regime (for adjustments)
        
        Returns:
            KellyResult with position size and metadata
        """
        # Get win probability and win/loss ratio from MC simulation
        win_prob, win_loss_ratio = self._extract_kelly_inputs(mc_result)
        
        # Calculate full Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(win_prob, win_loss_ratio)
        
        # Apply fractional Kelly if enabled
        if self.use_fractional_kelly:
            kelly_fraction *= self.fractional_multiplier
        
        # Apply regime adjustment
        regime_adjusted = self._adjust_for_regime(kelly_fraction, regime)
        
        # Apply maximum position size constraint
        final_size = min(regime_adjusted, self.max_position_size)
        
        # If win probability too low, set size to 0
        if win_prob < self.min_win_prob:
            final_size = 0.0
            logger.debug(
                f"{mc_result.ticker}: Win prob {win_prob:.2%} < {self.min_win_prob:.2%}, skipping"
            )
        
        # Create result
        result = KellyResult(
            ticker=mc_result.ticker,
            kelly_fraction=final_size,
            kelly_percentage=final_size * 100,
            win_prob=win_prob,
            win_loss_ratio=win_loss_ratio,
            expected_return=mc_result.mean_return,
            half_kelly=kelly_fraction * 0.5 if not self.use_fractional_kelly else final_size,
            quarter_kelly=kelly_fraction * 0.25 if not self.use_fractional_kelly else final_size * 0.5,
            regime_adjusted=regime_adjusted
        )
        
        logger.debug(
            f"Kelly for {mc_result.ticker}: "
            f"p={win_prob:.2%}, "
            f"b={win_loss_ratio:.2f}, "
            f"size={final_size:.2%}"
        )
        
        return result
    
    def calculate_from_stats(
        self,
        ticker: str,
        win_prob: float,
        win_loss_ratio: float,
        expected_return: float,
        regime: Optional[str] = None
    ) -> KellyResult:
        """
        Calculate Kelly directly from statistics (without MC simulation).
        
        Args:
            ticker: Stock ticker
            win_prob: Probability of winning
            win_loss_ratio: Ratio of average win to average loss
            expected_return: Expected return
            regime: Current market regime
        
        Returns:
            KellyResult with position size
        """
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(win_prob, win_loss_ratio)
        
        # Apply fractional Kelly
        if self.use_fractional_kelly:
            kelly_fraction *= self.fractional_multiplier
        
        # Apply regime adjustment
        regime_adjusted = self._adjust_for_regime(kelly_fraction, regime)
        
        # Apply constraints
        final_size = min(regime_adjusted, self.max_position_size)
        
        if win_prob < self.min_win_prob:
            final_size = 0.0
        
        result = KellyResult(
            ticker=ticker,
            kelly_fraction=final_size,
            kelly_percentage=final_size * 100,
            win_prob=win_prob,
            win_loss_ratio=win_loss_ratio,
            expected_return=expected_return,
            half_kelly=kelly_fraction * 0.5 if not self.use_fractional_kelly else final_size,
            quarter_kelly=kelly_fraction * 0.25 if not self.use_fractional_kelly else final_size * 0.5,
            regime_adjusted=regime_adjusted
        )
        
        return result
    
    def calculate_portfolio(
        self,
        mc_results: Dict[str, MonteCarloResult],
        regime: Optional[str] = None,
        normalize: bool = True
    ) -> Dict[str, KellyResult]:
        """
        Calculate Kelly sizes for a portfolio of positions.
        
        Args:
            mc_results: Dict of ticker -> MonteCarloResult
            regime: Current market regime
            normalize: Whether to normalize positions to sum to 1.0
        
        Returns:
            Dict of ticker -> KellyResult
        """
        kelly_results = {}
        
        # Calculate individual Kelly fractions
        for ticker, mc_result in mc_results.items():
            kelly_result = self.calculate(mc_result, regime)
            kelly_results[ticker] = kelly_result
        
        # Normalize if requested (positions sum to 100%)
        if normalize:
            kelly_results = self._normalize_portfolio(kelly_results)
        
        return kelly_results
    
    def _calculate_kelly_fraction(
        self,
        win_prob: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly fraction using the formula: f* = (p*b - q) / b
        
        Args:
            win_prob: Probability of winning (p)
            win_loss_ratio: Win/loss ratio (b)
        
        Returns:
            kelly_fraction: Optimal fraction to bet
        """
        p = win_prob
        q = 1.0 - p
        b = win_loss_ratio
        
        # Kelly formula
        if b <= 0:
            return 0.0
        
        kelly = (p * b - q) / b
        
        # Kelly can be negative (negative edge), cap at 0
        kelly = max(0.0, kelly)
        
        return kelly
    
    def _extract_kelly_inputs(
        self,
        mc_result: MonteCarloResult
    ) -> Tuple[float, float]:
        """
        Extract win probability and win/loss ratio from MC result.
        
        Args:
            mc_result: Monte Carlo simulation result
        
        Returns:
            (win_prob, win_loss_ratio)
        """
        returns = mc_result.simulated_returns
        
        # Win probability (return > 0)
        win_prob = float(np.mean(returns > 0))
        
        # Average win and loss
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        
        if len(wins) > 0 and len(losses) > 0:
            avg_win = float(np.mean(wins))
            avg_loss = float(np.abs(np.mean(losses)))
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        else:
            # Edge case: all wins or all losses
            win_loss_ratio = 1.0
        
        return win_prob, win_loss_ratio
    
    def _adjust_for_regime(
        self,
        kelly_fraction: float,
        regime: Optional[str]
    ) -> float:
        """
        Adjust Kelly fraction based on market regime.
        
        Args:
            kelly_fraction: Base Kelly fraction
            regime: Current market regime
        
        Returns:
            adjusted_fraction: Regime-adjusted Kelly fraction
        """
        if regime is None:
            return kelly_fraction
        
        # Regime multipliers
        regime_multipliers = {
            "bull": 1.0,      # Normal sizing in bull markets
            "sideways": 0.7,  # Reduce size 30% in choppy markets
            "bear": 0.5       # Reduce size 50% in bear markets
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        adjusted = kelly_fraction * multiplier
        
        return adjusted
    
    def _normalize_portfolio(
        self,
        kelly_results: Dict[str, KellyResult]
    ) -> Dict[str, KellyResult]:
        """
        Normalize Kelly fractions so they sum to 1.0 (100%).
        
        Args:
            kelly_results: Dict of ticker -> KellyResult
        
        Returns:
            normalized_results: Dict with normalized Kelly fractions
        """
        # Sum of all Kelly fractions
        total_kelly = sum(r.kelly_fraction for r in kelly_results.values())
        
        if total_kelly == 0:
            return kelly_results  # No positions, nothing to normalize
        
        # Normalize each position
        normalized_results = {}
        for ticker, result in kelly_results.items():
            normalized_fraction = result.kelly_fraction / total_kelly
            
            # Create new result with normalized fraction
            normalized_result = KellyResult(
                ticker=result.ticker,
                kelly_fraction=normalized_fraction,
                kelly_percentage=normalized_fraction * 100,
                win_prob=result.win_prob,
                win_loss_ratio=result.win_loss_ratio,
                expected_return=result.expected_return,
                half_kelly=result.half_kelly / total_kelly if total_kelly > 0 else 0,
                quarter_kelly=result.quarter_kelly / total_kelly if total_kelly > 0 else 0,
                regime_adjusted=result.regime_adjusted / total_kelly if total_kelly > 0 else 0
            )
            
            normalized_results[ticker] = normalized_result
        
        return normalized_results
    
    def calculate_expected_growth(
        self,
        kelly_result: KellyResult
    ) -> float:
        """
        Calculate expected logarithmic growth rate (Kelly objective).
        
        Args:
            kelly_result: Kelly calculation result
        
        Returns:
            expected_growth: Expected log growth rate
        """
        f = kelly_result.kelly_fraction
        p = kelly_result.win_prob
        q = 1.0 - p
        b = kelly_result.win_loss_ratio
        
        # Expected log wealth: E[log(1 + f*R)]
        # Approximation: p*log(1+f*b) + q*log(1-f)
        if f >= 1.0:
            return -np.inf  # Betting everything = ruin
        
        win_term = p * np.log(1 + f * b) if (1 + f * b) > 0 else -np.inf
        loss_term = q * np.log(1 - f) if (1 - f) > 0 else -np.inf
        
        expected_growth = win_term + loss_term
        
        return float(expected_growth)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get Kelly calculator statistics.
        
        Returns:
            dict with calculator configuration
        """
        return {
            "max_position_size": self.max_position_size,
            "use_fractional_kelly": self.use_fractional_kelly,
            "fractional_multiplier": self.fractional_multiplier,
            "min_win_prob": self.min_win_prob,
            "effective_multiplier": (
                self.fractional_multiplier if self.use_fractional_kelly else 1.0
            )
        }


# ============================================================================
# BAYESIAN KELLY CRITERION
# ============================================================================

@dataclass
class BayesianKellyResult:
    """Result from Bayesian Kelly sizing for a single stock."""
    ticker: str
    kelly_fraction: float       # Final position size after all adjustments
    kelly_percentage: float     # As percentage

    # Inputs from Bayesian posterior
    mu: float                   # Posterior mean return
    sigma: float                # Posterior std
    raw_kelly: float            # μ / σ² (before adjustments)

    # Adjustments
    confidence_score: float     # 1 / (1 + ci_width), from BayesianPosterior
    regime_multiplier: float
    fractional_multiplier: float


class BayesianKelly:
    """
    Bayesian Kelly Criterion using posterior predictive distributions.

    Uses the continuous Kelly formula:  f* = μ / σ²
    with a confidence haircut from the credible interval width:
        final_f = f* × confidence_score × fractional_multiplier × regime_multiplier

    The confidence_score = 1/(1 + ci_width) naturally shrinks positions
    when the model is uncertain (wide credible intervals).
    """

    def __init__(
        self,
        max_position_size: float = 0.15,
        fractional_multiplier: float = 0.25,
    ):
        self.max_position_size = max_position_size
        self.fractional_multiplier = fractional_multiplier

        # Regime multipliers (same as classic Kelly)
        self.regime_multipliers = {
            "bull": 1.0,
            "sideways": 0.7,
            "bear": 0.5,
        }

        logger.info(
            f"BayesianKelly initialized: max_size={max_position_size:.2%}, "
            f"frac_mult={fractional_multiplier}"
        )

    def calculate(
        self,
        posterior: 'BayesianPosterior',
        regime: Optional[str] = None,
    ) -> BayesianKellyResult:
        """
        Calculate Bayesian Kelly position size from a posterior.

        Args:
            posterior: BayesianPosterior from Tier 3.
            regime: Current market regime for adjustment.

        Returns:
            BayesianKellyResult with final position size.
        """
        mu = posterior.mu
        sigma = posterior.sigma

        # Continuous Kelly: f* = μ / σ²
        if sigma <= 0:
            raw_kelly = 0.0
        else:
            raw_kelly = mu / (sigma ** 2)

        # Cap negative Kelly at 0 (no short positions in this framework)
        raw_kelly = max(0.0, raw_kelly)

        # Apply confidence haircut
        confidence = posterior.confidence_score

        # Regime multiplier
        regime_mult = self.regime_multipliers.get(regime, 1.0) if regime else 1.0

        # Final fraction
        final_f = (
            raw_kelly
            * confidence
            * self.fractional_multiplier
            * regime_mult
        )

        # Cap at max position size
        final_f = min(final_f, self.max_position_size)

        return BayesianKellyResult(
            ticker=posterior.ticker,
            kelly_fraction=final_f,
            kelly_percentage=final_f * 100,
            mu=mu,
            sigma=sigma,
            raw_kelly=raw_kelly,
            confidence_score=confidence,
            regime_multiplier=regime_mult,
            fractional_multiplier=self.fractional_multiplier,
        )

    def calculate_portfolio(
        self,
        posteriors: Dict[str, 'BayesianPosterior'],
        regime: Optional[str] = None,
        normalize: bool = True,
        max_positions: int = 20,
        min_fraction: float = 0.005,
    ) -> Dict[str, BayesianKellyResult]:
        """
        Calculate Bayesian Kelly sizes for a portfolio.

        Args:
            posteriors: Dict of ticker -> BayesianPosterior.
            regime: Current market regime.
            normalize: Normalize weights to sum to 1.0.
            max_positions: Maximum number of positions.
            min_fraction: Minimum Kelly fraction to include.

        Returns:
            Dict of ticker -> BayesianKellyResult.
        """
        results = {}
        for ticker, posterior in posteriors.items():
            result = self.calculate(posterior, regime)
            if result.kelly_fraction >= min_fraction:
                results[ticker] = result

        # Take top N by Kelly fraction
        if len(results) > max_positions:
            sorted_r = sorted(
                results.items(), key=lambda x: x[1].kelly_fraction, reverse=True
            )
            results = dict(sorted_r[:max_positions])

        # Normalize to sum to 1.0
        if normalize and results:
            total = sum(r.kelly_fraction for r in results.values())
            if total > 0:
                normalized = {}
                for ticker, r in results.items():
                    norm_f = r.kelly_fraction / total
                    normalized[ticker] = BayesianKellyResult(
                        ticker=r.ticker,
                        kelly_fraction=norm_f,
                        kelly_percentage=norm_f * 100,
                        mu=r.mu,
                        sigma=r.sigma,
                        raw_kelly=r.raw_kelly,
                        confidence_score=r.confidence_score,
                        regime_multiplier=r.regime_multiplier,
                        fractional_multiplier=r.fractional_multiplier,
                    )
                results = normalized

        return results


def calculate_optimal_f(
    returns: np.ndarray,
    method: str = "kelly"
) -> float:
    """
    Calculate optimal position size from historical returns.
    
    Args:
        returns: Array of historical returns
        method: Method to use ("kelly", "sharpe", "max_return")
    
    Returns:
        optimal_f: Optimal position size
    """
    if method == "kelly":
        # Estimate Kelly from returns
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        p = len(wins) / len(returns)
        q = 1.0 - p
        avg_win = np.mean(wins)
        avg_loss = np.abs(np.mean(losses))
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        kelly = (p * b - q) / b
        return max(0.0, kelly)
    
    elif method == "sharpe":
        # Size proportional to Sharpe ratio
        if len(returns) < 2:
            return 0.0
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        # Map Sharpe to position size (Sharpe 2.0 = 100% position)
        optimal = sharpe / 2.0
        return max(0.0, min(1.0, optimal))
    
    elif method == "max_return":
        # Maximize expected return (full position if positive expected return)
        expected_return = np.mean(returns)
        return 1.0 if expected_return > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")

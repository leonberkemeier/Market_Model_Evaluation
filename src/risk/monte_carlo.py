"""
Monte Carlo Simulation for Layer IV Risk Management

Simulates possible future outcomes based on expert predictions to estimate:
- Forecast uncertainty (confidence intervals)
- Value at Risk (VaR)
- Expected Shortfall (CVaR)
- Probability of loss

Used to inform Kelly Criterion position sizing.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    ticker: str
    predicted_return: float  # Mean prediction from expert
    simulated_returns: np.ndarray  # Array of simulated returns
    
    # Summary statistics
    mean_return: float
    median_return: float
    std_return: float
    
    # Risk metrics
    var_95: float  # Value at Risk (5th percentile)
    var_99: float  # Value at Risk (1st percentile)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    prob_loss: float  # Probability of negative return
    
    # Confidence intervals
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float


class MonteCarloSimulator:
    """
    Monte Carlo simulator for forecast uncertainty estimation.
    
    Process:
    1. Take expert prediction (mean return)
    2. Estimate prediction uncertainty (historical error)
    3. Simulate N possible outcomes
    4. Calculate risk metrics
    
    Used in Sentinel flow:
    Layer III → Expert predicts return
    Layer IV → Monte Carlo estimates uncertainty → Kelly sizes position
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        use_historical_error: bool = True,
        default_volatility: float = 0.02
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of Monte Carlo paths to simulate
            use_historical_error: Use historical prediction errors for uncertainty
            default_volatility: Default prediction error std (if no history)
        """
        self.n_simulations = n_simulations
        self.use_historical_error = use_historical_error
        self.default_volatility = default_volatility
        
        # Historical prediction errors (ticker -> errors)
        self.prediction_errors: Dict[str, List[float]] = {}
        
        logger.info(f"MonteCarloSimulator initialized with {n_simulations} simulations")
    
    def update_prediction_errors(
        self,
        ticker: str,
        predicted_return: float,
        actual_return: float
    ) -> None:
        """
        Update historical prediction errors for a ticker.
        
        Args:
            ticker: Stock ticker
            predicted_return: What the model predicted
            actual_return: What actually happened
        """
        error = actual_return - predicted_return
        
        if ticker not in self.prediction_errors:
            self.prediction_errors[ticker] = []
        
        self.prediction_errors[ticker].append(error)
        
        # Keep only recent errors (last 252 trades = 1 year)
        if len(self.prediction_errors[ticker]) > 252:
            self.prediction_errors[ticker] = self.prediction_errors[ticker][-252:]
    
    def simulate(
        self,
        ticker: str,
        predicted_return: float,
        prediction_volatility: Optional[float] = None,
        regime: Optional[str] = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for a single prediction.
        
        Args:
            ticker: Stock ticker
            predicted_return: Expert's predicted return
            prediction_volatility: Optional override for prediction uncertainty
            regime: Current market regime (adjusts uncertainty)
        
        Returns:
            MonteCarloResult with simulation outcomes and risk metrics
        """
        # Estimate prediction uncertainty
        if prediction_volatility is not None:
            vol = prediction_volatility
        else:
            vol = self._estimate_prediction_volatility(ticker, regime)
        
        # Simulate returns: Normal(predicted_return, vol)
        simulated_returns = np.random.normal(
            loc=predicted_return,
            scale=vol,
            size=self.n_simulations
        )
        
        # Calculate summary statistics
        mean_return = float(np.mean(simulated_returns))
        median_return = float(np.median(simulated_returns))
        std_return = float(np.std(simulated_returns))
        
        # Calculate risk metrics
        var_95 = float(np.percentile(simulated_returns, 5))  # 5th percentile
        var_99 = float(np.percentile(simulated_returns, 1))  # 1st percentile
        
        # Conditional VaR (Expected Shortfall) - mean of worst 5%
        worst_5_pct = simulated_returns[simulated_returns <= var_95]
        cvar_95 = float(np.mean(worst_5_pct)) if len(worst_5_pct) > 0 else var_95
        
        # Probability of loss
        prob_loss = float(np.mean(simulated_returns < 0))
        
        # Confidence intervals
        ci_95_lower = float(np.percentile(simulated_returns, 2.5))
        ci_95_upper = float(np.percentile(simulated_returns, 97.5))
        ci_99_lower = float(np.percentile(simulated_returns, 0.5))
        ci_99_upper = float(np.percentile(simulated_returns, 99.5))
        
        result = MonteCarloResult(
            ticker=ticker,
            predicted_return=predicted_return,
            simulated_returns=simulated_returns,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            prob_loss=prob_loss,
            ci_95_lower=ci_95_lower,
            ci_95_upper=ci_95_upper,
            ci_99_lower=ci_99_lower,
            ci_99_upper=ci_99_upper
        )
        
        logger.debug(
            f"MC simulation for {ticker}: "
            f"pred={predicted_return:.4f}, "
            f"VaR95={var_95:.4f}, "
            f"prob_loss={prob_loss:.2%}"
        )
        
        return result
    
    def simulate_portfolio(
        self,
        predictions: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate portfolio of multiple positions.
        
        Args:
            predictions: Dict of ticker -> predicted return
            weights: Dict of ticker -> portfolio weight (if None, equal weight)
            regime: Current market regime
        
        Returns:
            dict with portfolio simulation results
        """
        tickers = list(predictions.keys())
        n_positions = len(tickers)
        
        if weights is None:
            weights = {ticker: 1.0 / n_positions for ticker in tickers}
        
        # Simulate each position
        position_simulations = {}
        for ticker in tickers:
            result = self.simulate(
                ticker=ticker,
                predicted_return=predictions[ticker],
                regime=regime
            )
            position_simulations[ticker] = result
        
        # Aggregate portfolio returns
        portfolio_returns = np.zeros(self.n_simulations)
        for ticker in tickers:
            portfolio_returns += (
                weights[ticker] * position_simulations[ticker].simulated_returns
            )
        
        # Portfolio risk metrics
        portfolio_var_95 = float(np.percentile(portfolio_returns, 5))
        portfolio_var_99 = float(np.percentile(portfolio_returns, 1))
        
        worst_5_pct = portfolio_returns[portfolio_returns <= portfolio_var_95]
        portfolio_cvar_95 = float(np.mean(worst_5_pct)) if len(worst_5_pct) > 0 else portfolio_var_95
        
        return {
            "portfolio_returns": portfolio_returns,
            "mean_return": float(np.mean(portfolio_returns)),
            "std_return": float(np.std(portfolio_returns)),
            "var_95": portfolio_var_95,
            "var_99": portfolio_var_99,
            "cvar_95": portfolio_cvar_95,
            "prob_loss": float(np.mean(portfolio_returns < 0)),
            "position_results": position_simulations
        }
    
    def _estimate_prediction_volatility(
        self,
        ticker: str,
        regime: Optional[str] = None
    ) -> float:
        """
        Estimate prediction uncertainty for a ticker.
        
        Args:
            ticker: Stock ticker
            regime: Current market regime (adjusts volatility)
        
        Returns:
            vol: Standard deviation of prediction error
        """
        if self.use_historical_error and ticker in self.prediction_errors:
            errors = self.prediction_errors[ticker]
            if len(errors) >= 20:  # Need minimum history
                vol = float(np.std(errors))
            else:
                vol = self.default_volatility
        else:
            vol = self.default_volatility
        
        # Adjust for regime (more uncertainty in volatile regimes)
        if regime == "bear":
            vol *= 1.5  # 50% more uncertainty in bear markets
        elif regime == "sideways":
            vol *= 1.2  # 20% more uncertainty in choppy markets
        # Bull regime uses base volatility
        
        return vol
    
    def get_risk_adjusted_return(
        self,
        result: MonteCarloResult,
        risk_aversion: float = 2.0
    ) -> float:
        """
        Calculate risk-adjusted return (mean - risk_aversion * variance).
        
        Args:
            result: Monte Carlo simulation result
            risk_aversion: Risk aversion coefficient (higher = more conservative)
        
        Returns:
            risk_adjusted_return: Expected return adjusted for risk
        """
        # Mean-variance utility: E[R] - (lambda/2) * Var[R]
        variance = result.std_return ** 2
        risk_adjusted = result.mean_return - (risk_aversion / 2) * variance
        
        return float(risk_adjusted)
    
    def get_kelly_inputs(
        self,
        result: MonteCarloResult,
        threshold: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate inputs for Kelly Criterion from simulation.
        
        Args:
            result: Monte Carlo simulation result
            threshold: Return threshold for "win" (default: 0 = positive return)
        
        Returns:
            (win_prob, avg_win_loss_ratio): Inputs for Kelly formula
        """
        returns = result.simulated_returns
        
        # Probability of winning (return > threshold)
        win_prob = float(np.mean(returns > threshold))
        
        # Average win and loss
        wins = returns[returns > threshold]
        losses = returns[returns <= threshold]
        
        if len(wins) > 0 and len(losses) > 0:
            avg_win = float(np.mean(wins))
            avg_loss = float(np.abs(np.mean(losses)))
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        else:
            win_loss_ratio = 1.0
        
        return win_prob, win_loss_ratio
    
    def clear_history(self, ticker: Optional[str] = None) -> None:
        """
        Clear prediction error history.
        
        Args:
            ticker: Specific ticker to clear (if None, clears all)
        """
        if ticker is None:
            self.prediction_errors.clear()
            logger.info("Cleared all prediction error history")
        elif ticker in self.prediction_errors:
            del self.prediction_errors[ticker]
            logger.info(f"Cleared prediction error history for {ticker}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get simulator statistics.
        
        Returns:
            dict with simulator stats
        """
        stats_dict = {
            "n_simulations": self.n_simulations,
            "default_volatility": self.default_volatility,
            "n_tickers_tracked": len(self.prediction_errors),
            "avg_history_length": (
                np.mean([len(errors) for errors in self.prediction_errors.values()])
                if self.prediction_errors else 0
            )
        }
        
        return stats_dict


def calculate_var_backtest(
    predictions: pd.Series,
    actuals: pd.Series,
    var_level: float = 0.05
) -> Dict[str, float]:
    """
    Backtest VaR accuracy (how often losses exceed VaR).
    
    Args:
        predictions: Series of VaR predictions
        actuals: Series of actual returns
        var_level: VaR confidence level (0.05 = 95% VaR)
    
    Returns:
        dict with backtest results
    """
    # Count violations (actual < VaR)
    violations = (actuals < predictions).sum()
    total = len(actuals)
    violation_rate = violations / total
    
    # Expected violation rate = var_level (e.g., 5%)
    expected_rate = var_level
    
    # Statistical test (binomial)
    p_value = stats.binom_test(violations, total, var_level, alternative='two-sided')
    
    return {
        "violations": int(violations),
        "total": int(total),
        "violation_rate": float(violation_rate),
        "expected_rate": float(expected_rate),
        "p_value": float(p_value),
        "test_passed": p_value > 0.05  # Fail to reject null (VaR is accurate)
    }

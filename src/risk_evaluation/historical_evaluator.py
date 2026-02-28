"""
Historical Risk Evaluator

Computes risk profiles from rolling historical statistics.
Simple, robust baseline — no model assumptions beyond the data itself.

Calculates: volatility, beta, drawdowns, Sharpe, win/loss stats, VaR.
"""

from datetime import date
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from .base_evaluator import BaseRiskEvaluator
from .risk_profile import RiskProfile


class HistoricalEvaluator(BaseRiskEvaluator):
    """
    Risk evaluation using rolling historical statistics.

    The simplest evaluator — uses only historical price data to compute
    risk metrics. No parametric assumptions beyond what the data shows.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        short_vol_window: int = 20,
        risk_free_rate: float = 0.05,
    ):
        """
        Args:
            lookback_days: Full lookback for historical stats
            short_vol_window: Short window for current/recent volatility
            risk_free_rate: Annualized risk-free rate for Sharpe calculation
        """
        super().__init__(name="historical", lookback_days=lookback_days)
        self.short_vol_window = short_vol_window
        self.risk_free_rate = risk_free_rate

    def evaluate(
        self,
        ticker: str,
        prices: pd.DataFrame,
        evaluation_date: date,
        market_prices: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> RiskProfile:
        """Compute risk profile from historical price data."""

        # Use close prices, trim to lookback window
        close = self._get_close(prices).iloc[-self.lookback_days:]
        returns = close.pct_change().dropna()

        if len(returns) < 30:
            raise ValueError(f"{ticker}: Only {len(returns)} return observations")

        # --- Volatility ---
        annualized_vol = float(returns.std() * np.sqrt(252))
        current_vol = float(
            returns.iloc[-self.short_vol_window:].std() * np.sqrt(252)
        )
        vol_regime = self._classify_vol_regime(current_vol, annualized_vol)

        # --- Beta & correlation to market ---
        beta, corr_to_market = self._compute_beta(returns, market_prices)

        # --- Drawdown ---
        max_dd, current_dd = self._compute_drawdowns(close)

        # --- Return distribution ---
        mean_return = float(returns.mean() * 252)
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = (
            float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))
            if excess_returns.std() > 0 else 0.0
        )
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns, fisher=False))  # Excess=False → normal=3

        # --- Win/loss stats ---
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0.5
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # --- VaR ---
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))
        worst_5pct = returns[returns <= var_95]
        cvar_95 = float(worst_5pct.mean()) if len(worst_5pct) > 0 else var_95

        # --- Confidence ---
        # More data → higher confidence, capped at 1.0
        confidence = min(len(returns) / self.lookback_days, 1.0)

        return RiskProfile(
            ticker=ticker,
            evaluation_date=evaluation_date,
            evaluator_name=self.name,
            annualized_vol=annualized_vol,
            current_vol=current_vol,
            vol_regime=vol_regime,
            beta=beta,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            mean_return=mean_return,
            sharpe=sharpe,
            skewness=skewness,
            kurtosis=kurtosis,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            correlation_to_market=corr_to_market,
            confidence=confidence,
            data_points=len(returns),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_close(self, prices: pd.DataFrame) -> pd.Series:
        """Extract close price series from DataFrame."""
        if "adjusted_close" in prices.columns:
            return prices["adjusted_close"].dropna()
        if "close" in prices.columns:
            return prices["close"].dropna()
        if "Close" in prices.columns:
            return prices["Close"].dropna()
        raise ValueError(f"No close price column found in {list(prices.columns)}")

    def _classify_vol_regime(
        self, current_vol: float, long_vol: float
    ) -> str:
        """Classify current volatility relative to historical."""
        if long_vol == 0:
            return "normal"
        ratio = current_vol / long_vol
        if ratio < 0.75:
            return "low"
        if ratio > 1.25:
            return "high"
        return "normal"

    def _compute_beta(
        self,
        returns: pd.Series,
        market_prices: Optional[pd.DataFrame],
    ) -> tuple[float, float]:
        """Compute beta and correlation to market benchmark."""
        if market_prices is None or market_prices.empty:
            return 1.0, 0.0

        market_close = self._get_close(market_prices)
        market_returns = market_close.pct_change().dropna()

        # Align dates
        aligned = pd.DataFrame({
            "stock": returns,
            "market": market_returns,
        }).dropna()

        if len(aligned) < 30:
            return 1.0, 0.0

        cov = aligned["stock"].cov(aligned["market"])
        var_market = aligned["market"].var()
        beta = float(cov / var_market) if var_market > 0 else 1.0
        corr = float(aligned["stock"].corr(aligned["market"]))

        return beta, corr

    def _compute_drawdowns(self, close: pd.Series) -> tuple[float, float]:
        """Compute maximum and current drawdown."""
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax

        max_dd = float(drawdown.min())  # Most negative
        current_dd = float(drawdown.iloc[-1])

        return max_dd, current_dd

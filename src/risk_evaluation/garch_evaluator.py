"""
GARCH Risk Evaluator

Uses GARCH(1,1) to forecast volatility. Better than historical at
capturing volatility clustering — periods of high vol tend to persist.

Falls back to historical volatility if GARCH fitting fails.
"""

from datetime import date
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from .base_evaluator import BaseRiskEvaluator
from .historical_evaluator import HistoricalEvaluator
from .risk_profile import RiskProfile

try:
    from arch import arch_model
except ImportError:
    arch_model = None


class GARCHEvaluator(BaseRiskEvaluator):
    """
    Risk evaluation using GARCH(1,1) volatility model.

    GARCH captures volatility clustering: if today is volatile,
    tomorrow is likely volatile too. This gives better forward-looking
    vol estimates than simple historical standard deviation.
    """

    def __init__(
        self,
        lookback_days: int = 504,
        forecast_horizon: int = 21,
        risk_free_rate: float = 0.05,
    ):
        """
        Args:
            lookback_days: History for GARCH fitting (2 years default)
            forecast_horizon: Days ahead to forecast volatility
            risk_free_rate: Annualized risk-free rate for Sharpe
        """
        super().__init__(name="garch", lookback_days=lookback_days)
        self.forecast_horizon = forecast_horizon
        self.risk_free_rate = risk_free_rate
        # Fallback evaluator for non-vol metrics
        self._historical = HistoricalEvaluator(
            lookback_days=lookback_days,
            risk_free_rate=risk_free_rate,
        )

    def evaluate(
        self,
        ticker: str,
        prices: pd.DataFrame,
        evaluation_date: date,
        market_prices: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> RiskProfile:
        """Compute risk profile with GARCH-forecasted volatility."""

        # Get baseline profile from historical evaluator
        profile = self._historical.evaluate(
            ticker=ticker,
            prices=prices,
            evaluation_date=evaluation_date,
            market_prices=market_prices,
        )

        # Overlay GARCH vol forecast
        garch_vol = self._fit_garch(prices)

        if garch_vol is not None:
            profile.current_vol = garch_vol
            profile.vol_regime = self._classify_vol_regime(
                garch_vol, profile.annualized_vol
            )
            profile.evaluator_name = self.name
            profile.metadata["garch_forecast_vol"] = garch_vol
            profile.metadata["garch_forecast_horizon"] = self.forecast_horizon
        else:
            # GARCH failed, keep historical values but note it
            profile.evaluator_name = f"{self.name}_fallback"
            profile.confidence *= 0.8  # Reduce confidence on fallback

        return profile

    def _fit_garch(self, prices: pd.DataFrame) -> Optional[float]:
        """
        Fit GARCH(1,1) and return forecasted annualized volatility.

        Returns None if fitting fails.
        """
        if arch_model is None:
            self.logger.warning("arch package not installed, falling back to historical")
            return None

        try:
            close = self._historical._get_close(prices)
            returns = close.pct_change().dropna().iloc[-self.lookback_days:]

            if len(returns) < 100:
                return None

            # Scale returns to percentage for numerical stability
            returns_pct = returns * 100

            model = arch_model(
                returns_pct,
                vol="Garch",
                p=1,
                q=1,
                dist="normal",
            )

            result = model.fit(disp="off", show_warning=False)

            # Forecast variance
            forecast = result.forecast(horizon=self.forecast_horizon)
            # Mean forecasted variance over horizon
            forecast_variance = forecast.variance.iloc[-1].mean()

            # Convert back from percentage to decimal, annualize
            daily_vol = np.sqrt(forecast_variance) / 100
            annualized_vol = float(daily_vol * np.sqrt(252))

            return annualized_vol

        except Exception as e:
            self.logger.debug(f"GARCH fitting failed: {e}")
            return None

    def _classify_vol_regime(self, current_vol: float, long_vol: float) -> str:
        """Classify current volatility relative to historical."""
        if long_vol == 0:
            return "normal"
        ratio = current_vol / long_vol
        if ratio < 0.75:
            return "low"
        if ratio > 1.25:
            return "high"
        return "normal"

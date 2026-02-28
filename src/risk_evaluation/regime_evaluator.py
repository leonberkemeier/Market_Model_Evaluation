"""
Regime-Conditional Risk Evaluator

Uses HMM regime detection to compute risk stats conditional on the
current market regime. In a bear regime, uses bear-period historical
stats instead of the full history.

This captures the key insight that risk characteristics are
non-stationary — a stock's vol/drawdown/win-rate behave differently
in bull vs bear markets.
"""

from datetime import date
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats

from .base_evaluator import BaseRiskEvaluator
from .historical_evaluator import HistoricalEvaluator
from .risk_profile import RiskProfile
from ..regime.hmm_detector import HMMRegimeDetector


class RegimeConditionalEvaluator(BaseRiskEvaluator):
    """
    Risk evaluation conditional on market regime.

    Process:
    1. Detect current regime (bull/bear/sideways) using HMM
    2. Filter historical data to periods matching current regime
    3. Compute risk stats from regime-matched data only
    4. Blend with unconditional stats (to avoid overfitting to small samples)
    """

    def __init__(
        self,
        lookback_days: int = 504,
        risk_free_rate: float = 0.05,
        regime_blend_weight: float = 0.7,
        detector: Optional[HMMRegimeDetector] = None,
    ):
        """
        Args:
            lookback_days: Full history for regime detection + stats
            risk_free_rate: Annualized risk-free rate
            regime_blend_weight: Weight for regime-conditional stats (0-1).
                0.7 = 70% regime stats + 30% unconditional stats.
            detector: Pre-trained HMM detector. If None, creates and fits one.
        """
        super().__init__(name="regime_conditional", lookback_days=lookback_days)
        self.risk_free_rate = risk_free_rate
        self.regime_blend_weight = regime_blend_weight
        self.detector = detector
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
        """Compute regime-conditional risk profile."""

        # Get unconditional baseline
        baseline = self._historical.evaluate(
            ticker=ticker,
            prices=prices,
            evaluation_date=evaluation_date,
            market_prices=market_prices,
        )

        # Detect regime from market prices (or stock prices if no market data)
        regime_source = market_prices if market_prices is not None else prices
        regime, regime_confidence, regime_history = self._detect_regime(regime_source)

        if regime is None or regime_history is None:
            # Can't detect regime, return baseline with name change
            baseline.evaluator_name = f"{self.name}_fallback"
            baseline.confidence *= 0.7
            return baseline

        # Compute regime-conditional stats
        close = self._historical._get_close(prices).iloc[-self.lookback_days:]
        returns = close.pct_change().dropna()

        # Align regime history with returns
        aligned_regime = regime_history.reindex(returns.index, method="ffill")
        regime_mask = aligned_regime["regime"] == regime

        regime_returns = returns[regime_mask]

        # Need minimum data in regime to be meaningful
        if len(regime_returns) < 20:
            baseline.current_regime = regime
            baseline.regime_confidence = regime_confidence
            baseline.evaluator_name = f"{self.name}_insufficient_regime_data"
            baseline.confidence *= 0.6
            return baseline

        # Compute regime-conditional metrics
        regime_vol = float(regime_returns.std() * np.sqrt(252))
        regime_mean = float(regime_returns.mean() * 252)

        wins = regime_returns[regime_returns > 0]
        losses = regime_returns[regime_returns < 0]
        regime_win_rate = float(len(wins) / len(regime_returns))
        regime_avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        regime_avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        regime_wl_ratio = regime_avg_win / regime_avg_loss if regime_avg_loss > 0 else 1.0

        regime_var_95 = float(np.percentile(regime_returns, 5))

        # Blend regime-conditional with unconditional
        w = self.regime_blend_weight
        profile = RiskProfile(
            ticker=ticker,
            evaluation_date=evaluation_date,
            evaluator_name=self.name,
            annualized_vol=_blend(regime_vol, baseline.annualized_vol, w),
            current_vol=_blend(regime_vol, baseline.current_vol, w),
            vol_regime=baseline.vol_regime,
            beta=baseline.beta,
            max_drawdown=baseline.max_drawdown,
            current_drawdown=baseline.current_drawdown,
            mean_return=_blend(regime_mean, baseline.mean_return, w),
            sharpe=baseline.sharpe,  # Keep unconditional Sharpe
            skewness=float(stats.skew(regime_returns)),
            kurtosis=float(stats.kurtosis(regime_returns, fisher=False)),
            win_rate=_blend(regime_win_rate, baseline.win_rate, w),
            avg_win=_blend(regime_avg_win, baseline.avg_win, w),
            avg_loss=_blend(regime_avg_loss, baseline.avg_loss, w),
            win_loss_ratio=_blend(regime_wl_ratio, baseline.win_loss_ratio, w),
            var_95=_blend(regime_var_95, baseline.var_95, w),
            var_99=baseline.var_99,
            cvar_95=baseline.cvar_95,
            current_regime=regime,
            regime_confidence=regime_confidence,
            correlation_to_market=baseline.correlation_to_market,
            confidence=baseline.confidence * min(regime_confidence, 1.0),
            data_points=len(regime_returns),
            metadata={
                "unconditional_data_points": baseline.data_points,
                "regime_data_points": len(regime_returns),
                "regime_blend_weight": w,
                "regime_vol": regime_vol,
                "unconditional_vol": baseline.annualized_vol,
            },
        )

        return profile

    def _detect_regime(
        self,
        prices: pd.DataFrame,
    ) -> tuple[Optional[str], float, Optional[pd.DataFrame]]:
        """
        Detect current market regime.

        Returns:
            (regime_name, confidence, regime_history_df) or (None, 0, None) on failure
        """
        try:
            close = self._historical._get_close(prices)

            if len(close) < 100:
                return None, 0.0, None

            # Create or reuse detector
            if self.detector is None:
                self.detector = HMMRegimeDetector(n_states=3)
                self.detector.fit(close, verbose=False)

            # Current regime
            result = self.detector.predict_regime(
                close, return_probabilities=True
            )

            # Full regime history
            history = self.detector.predict_regime_history(close)

            return result["regime"], result["confidence"], history

        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return None, 0.0, None


def _blend(regime_val: float, unconditional_val: float, weight: float) -> float:
    """Blend regime-conditional and unconditional values."""
    return weight * regime_val + (1 - weight) * unconditional_val

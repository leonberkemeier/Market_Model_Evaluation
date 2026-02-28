"""
Base Risk Evaluator — abstract interface for all risk evaluation models.

All evaluators (historical, GARCH, regime-conditional) implement this
interface to produce RiskProfiles for stocks.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from .risk_profile import RiskProfile


class BaseRiskEvaluator(ABC):
    """
    Abstract base class for risk evaluators.

    Each evaluator takes historical price data and produces a RiskProfile
    per stock, containing volatility, drawdown, win/loss stats, and VaR
    metrics needed for Monte Carlo simulation and Kelly sizing.
    """

    def __init__(self, name: str, lookback_days: int = 252):
        """
        Args:
            name: Evaluator identifier (e.g. "historical", "garch")
            lookback_days: Default lookback window for calculations
        """
        self.name = name
        self.lookback_days = lookback_days
        self.logger = logger.bind(module=f"risk_eval_{name}")

    @abstractmethod
    def evaluate(
        self,
        ticker: str,
        prices: pd.DataFrame,
        evaluation_date: date,
        market_prices: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> RiskProfile:
        """
        Evaluate risk for a single stock.

        Args:
            ticker: Stock ticker symbol
            prices: DataFrame with OHLCV data (date-indexed)
            evaluation_date: Date of evaluation
            market_prices: Optional benchmark prices (SPY) for beta/correlation
            **kwargs: Evaluator-specific parameters

        Returns:
            RiskProfile for this stock
        """
        pass

    def evaluate_universe(
        self,
        universe: Dict[str, pd.DataFrame],
        evaluation_date: date,
        market_prices: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, RiskProfile]:
        """
        Evaluate risk for all stocks in the universe.

        Args:
            universe: Dict of ticker -> price DataFrame
            evaluation_date: Date of evaluation
            market_prices: Optional benchmark prices for beta/correlation

        Returns:
            Dict of ticker -> RiskProfile
        """
        profiles = {}
        failed = []

        for ticker, prices in universe.items():
            try:
                if prices.empty or len(prices) < 30:
                    self.logger.warning(
                        f"{ticker}: Insufficient data ({len(prices)} rows), skipping"
                    )
                    failed.append(ticker)
                    continue

                profile = self.evaluate(
                    ticker=ticker,
                    prices=prices,
                    evaluation_date=evaluation_date,
                    market_prices=market_prices,
                    **kwargs,
                )

                if profile.validate():
                    profiles[ticker] = profile
                else:
                    self.logger.warning(f"{ticker}: Profile validation failed")
                    failed.append(ticker)

            except Exception as e:
                self.logger.error(f"{ticker}: Evaluation failed — {e}")
                failed.append(ticker)

        self.logger.info(
            f"Evaluated {len(profiles)}/{len(universe)} stocks "
            f"({len(failed)} failed: {failed[:5]}{'...' if len(failed) > 5 else ''})"
        )
        return profiles

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', lookback={self.lookback_days})"

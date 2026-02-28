"""
Strategy Comparator

Runs multiple allocation strategies on the same data and compares
their performance metrics side-by-side.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from ..portfolio.strategy_runner import StrategyRunner, AllocationResult


@dataclass
class StrategyMetrics:
    """Performance summary for a single strategy."""
    strategy_name: str
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    avg_positions: float = 0.0
    turnover: float = 0.0       # Average fraction of portfolio traded per rebalance
    total_trades: int = 0


@dataclass
class ComparisonResult:
    """Output from comparing multiple strategies."""
    strategies: Dict[str, StrategyMetrics]
    equity_curves: Dict[str, pd.Series]  # strategy_name -> NAV time series
    allocation_history: Dict[str, List[AllocationResult]]
    best_return: Optional[str] = None
    best_sharpe: Optional[str] = None
    lowest_drawdown: Optional[str] = None


class StrategyComparator:
    """
    Compares multiple allocation strategies on the same dataset.

    Usage:
        comparator = StrategyComparator()
        comparator.add_strategy("historical_kelly", runner1)
        comparator.add_strategy("equal_weight", runner2)
        result = comparator.compare(universe, rebalance_dates)
    """

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.strategies: Dict[str, StrategyRunner] = {}
        self.logger = logger.bind(module="comparator")

    def add_strategy(self, name: str, runner: StrategyRunner):
        """Register a strategy for comparison."""
        self.strategies[name] = runner

    def compare(
        self,
        universe: Dict[str, pd.DataFrame],
        rebalance_dates: List[date],
        market_prices: Optional[pd.DataFrame] = None,
    ) -> ComparisonResult:
        """
        Run all strategies across rebalance dates and compare.

        Args:
            universe: Dict of ticker -> full price history DataFrame
            rebalance_dates: Dates on which to rebalance
            market_prices: Benchmark prices for beta calculations

        Returns:
            ComparisonResult with metrics and equity curves
        """
        equity_curves = {}
        allocation_history = {}
        metrics = {}

        for name, runner in self.strategies.items():
            self.logger.info(f"Running strategy: {name}")

            allocations = []
            nav_series = []
            prev_weights = {}

            capital = self.initial_capital

            for i, rebal_date in enumerate(rebalance_dates):
                # Trim universe to data available up to rebal_date
                trimmed = self._trim_universe(universe, rebal_date)

                # Run strategy
                allocation = runner.run(
                    universe=trimmed,
                    evaluation_date=rebal_date,
                    market_prices=market_prices,
                )
                allocations.append(allocation)

                # Simulate returns until next rebalance
                if i + 1 < len(rebalance_dates):
                    next_date = rebalance_dates[i + 1]
                else:
                    # Last period: simulate to end of available data
                    next_date = None

                period_return = self._simulate_period(
                    weights=allocation.weights,
                    universe=universe,
                    start_date=rebal_date,
                    end_date=next_date,
                )

                capital *= (1 + period_return)
                nav_series.append((rebal_date, capital))
                prev_weights = allocation.weights

            # Build NAV series
            nav_df = pd.Series(
                {d: v for d, v in nav_series},
                name=name,
            )
            equity_curves[name] = nav_df

            # Compute metrics
            metrics[name] = self._compute_metrics(name, nav_df, allocations)
            allocation_history[name] = allocations

        # Find best performers
        best_return = max(metrics, key=lambda k: metrics[k].total_return) if metrics else None
        best_sharpe = max(metrics, key=lambda k: metrics[k].sharpe_ratio) if metrics else None
        lowest_dd = min(metrics, key=lambda k: abs(metrics[k].max_drawdown)) if metrics else None

        return ComparisonResult(
            strategies=metrics,
            equity_curves=equity_curves,
            allocation_history=allocation_history,
            best_return=best_return,
            best_sharpe=best_sharpe,
            lowest_drawdown=lowest_dd,
        )

    def _trim_universe(
        self, universe: Dict[str, pd.DataFrame], as_of: date
    ) -> Dict[str, pd.DataFrame]:
        """Trim price data to only include data up to as_of date."""
        trimmed = {}
        for ticker, prices in universe.items():
            mask = prices.index.date <= as_of if hasattr(prices.index, 'date') else True
            subset = prices.loc[mask] if not isinstance(mask, bool) else prices
            if len(subset) >= 30:
                trimmed[ticker] = subset
        return trimmed

    def _simulate_period(
        self,
        weights: Dict[str, float],
        universe: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: Optional[date],
    ) -> float:
        """
        Simulate portfolio return over a period given target weights.

        Simple approach: weighted average of individual stock returns.
        """
        if not weights:
            return 0.0

        total_return = 0.0

        for ticker, weight in weights.items():
            if ticker not in universe:
                continue

            prices = universe[ticker]
            try:
                # Get close prices in the period
                if hasattr(prices.index, 'date'):
                    mask = prices.index.date >= start_date
                    if end_date:
                        mask &= prices.index.date < end_date
                    period_prices = prices.loc[mask]
                else:
                    period_prices = prices

                if "close" in period_prices.columns:
                    close = period_prices["close"]
                elif "Close" in period_prices.columns:
                    close = period_prices["Close"]
                elif "adjusted_close" in period_prices.columns:
                    close = period_prices["adjusted_close"]
                else:
                    continue

                if len(close) < 2:
                    continue

                stock_return = (close.iloc[-1] / close.iloc[0]) - 1
                total_return += weight * stock_return

            except Exception:
                continue

        return total_return

    def _compute_metrics(
        self,
        name: str,
        nav: pd.Series,
        allocations: List[AllocationResult],
    ) -> StrategyMetrics:
        """Compute performance metrics from NAV series."""
        if len(nav) < 2:
            return StrategyMetrics(strategy_name=name)

        returns = nav.pct_change().dropna()

        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
        n_years = max(len(nav) / 52, 0.1)  # Approximate years (weekly rebalance)
        ann_return = (1 + total_return) ** (1 / n_years) - 1

        vol = float(returns.std() * np.sqrt(52))  # Annualized (weekly)
        sharpe = ann_return / vol if vol > 0 else 0.0

        # Drawdown
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_dd = float(drawdown.min())

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        win_rate = float((returns > 0).mean())
        avg_positions = np.mean([a.n_positions for a in allocations])

        return StrategyMetrics(
            strategy_name=name,
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            volatility=vol,
            win_rate=win_rate,
            avg_positions=avg_positions,
        )

"""
Strategy Runner

Orchestrates a full allocation strategy:
1. Risk evaluator produces RiskProfiles for the stock universe
2. Monte Carlo simulates forward paths using those profiles
3. Kelly criterion sizes positions
4. Output: target portfolio weights + trade signals

Each strategy is defined by its risk evaluator choice.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from ..risk_evaluation.base_evaluator import BaseRiskEvaluator
from ..risk_evaluation.risk_profile import RiskProfile
from ..risk.monte_carlo import MonteCarloSimulator
from ..risk.kelly_criterion import KellyCriterion


@dataclass
class AllocationResult:
    """Output of a strategy run for a single rebalance date."""
    strategy_name: str
    evaluation_date: date
    weights: Dict[str, float]        # ticker -> target weight (0 to 1)
    risk_profiles: Dict[str, RiskProfile]
    kelly_fractions: Dict[str, float]  # Raw Kelly fractions before normalization
    metadata: Dict = field(default_factory=dict)

    @property
    def n_positions(self) -> int:
        return sum(1 for w in self.weights.values() if w > 0)

    @property
    def top_positions(self) -> List[Tuple[str, float]]:
        """Top 10 positions by weight."""
        sorted_w = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_w[:10]


class StrategyRunner:
    """
    Runs a risk-based allocation strategy.

    Strategies:
    - "historical_kelly": HistoricalEvaluator → MC → Kelly
    - "garch_kelly": GARCHEvaluator → MC → Kelly
    - "regime_kelly": RegimeConditionalEvaluator → MC → Kelly
    - "equal_weight": 1/N allocation (baseline)
    - "risk_parity": Inverse-volatility weighting (baseline)
    """

    def __init__(
        self,
        strategy_name: str,
        evaluator: Optional[BaseRiskEvaluator] = None,
        kelly: Optional[KellyCriterion] = None,
        mc_simulator: Optional[MonteCarloSimulator] = None,
        max_positions: int = 20,
        min_kelly_fraction: float = 0.005,
    ):
        """
        Args:
            strategy_name: Name for this strategy (used as model_name in Trading_Simulator)
            evaluator: Risk evaluator instance (None for baselines)
            kelly: Kelly criterion calculator
            mc_simulator: Monte Carlo simulator
            max_positions: Maximum number of positions in portfolio
            min_kelly_fraction: Minimum Kelly fraction to include a position
        """
        self.strategy_name = strategy_name
        self.evaluator = evaluator
        self.kelly = kelly or KellyCriterion()
        self.mc_simulator = mc_simulator or MonteCarloSimulator()
        self.max_positions = max_positions
        self.min_kelly_fraction = min_kelly_fraction
        self.logger = logger.bind(module=f"strategy_{strategy_name}")

    def run(
        self,
        universe: Dict[str, pd.DataFrame],
        evaluation_date: date,
        market_prices: Optional[pd.DataFrame] = None,
        current_regime: Optional[str] = None,
    ) -> AllocationResult:
        """
        Run the strategy to produce target allocations.

        Args:
            universe: Dict of ticker -> price DataFrame
            evaluation_date: Date of evaluation
            market_prices: SPY or benchmark prices for beta
            current_regime: Override regime (otherwise detected by evaluator)

        Returns:
            AllocationResult with target weights
        """
        self.logger.info(
            f"Running {self.strategy_name} on {len(universe)} stocks "
            f"for {evaluation_date}"
        )

        # Baseline strategies don't need evaluators
        if self.strategy_name == "equal_weight":
            return self._equal_weight(universe, evaluation_date)

        if self.strategy_name == "risk_parity":
            return self._risk_parity(universe, evaluation_date)

        # Risk-model-based strategies
        if self.evaluator is None:
            raise ValueError(f"Strategy {self.strategy_name} requires a risk evaluator")

        # Step 1: Evaluate risk profiles
        profiles = self.evaluator.evaluate_universe(
            universe=universe,
            evaluation_date=evaluation_date,
            market_prices=market_prices,
        )

        if not profiles:
            self.logger.warning("No valid risk profiles, returning empty allocation")
            return AllocationResult(
                strategy_name=self.strategy_name,
                evaluation_date=evaluation_date,
                weights={},
                risk_profiles={},
                kelly_fractions={},
            )

        # Step 2: Filter to tradeable stocks
        tradeable = {t: p for t, p in profiles.items() if p.is_tradeable}
        self.logger.info(f"{len(tradeable)}/{len(profiles)} stocks are tradeable")

        # Step 3: Compute Kelly fractions
        kelly_fractions = {}
        for ticker, profile in tradeable.items():
            inputs = profile.to_kelly_inputs()
            result = self.kelly.calculate_from_stats(
                ticker=ticker,
                win_prob=inputs["win_prob"],
                win_loss_ratio=inputs["win_loss_ratio"],
                expected_return=inputs["expected_return"],
                regime=current_regime or profile.current_regime,
            )
            kelly_fractions[ticker] = result.kelly_fraction

        # Step 4: Filter and normalize to weights
        # Keep only positive Kelly fractions above minimum
        positive = {
            t: f for t, f in kelly_fractions.items()
            if f >= self.min_kelly_fraction
        }

        # Take top N by Kelly fraction
        if len(positive) > self.max_positions:
            sorted_k = sorted(positive.items(), key=lambda x: x[1], reverse=True)
            positive = dict(sorted_k[:self.max_positions])

        # Normalize to sum to 1.0
        total = sum(positive.values())
        weights = {t: f / total for t, f in positive.items()} if total > 0 else {}

        self.logger.info(
            f"Allocation: {len(weights)} positions, "
            f"top 3: {list(weights.items())[:3]}"
        )

        return AllocationResult(
            strategy_name=self.strategy_name,
            evaluation_date=evaluation_date,
            weights=weights,
            risk_profiles=profiles,
            kelly_fractions=kelly_fractions,
        )

    # ------------------------------------------------------------------
    # Baseline strategies
    # ------------------------------------------------------------------

    def _equal_weight(
        self, universe: Dict[str, pd.DataFrame], evaluation_date: date
    ) -> AllocationResult:
        """1/N equal weight allocation."""
        tickers = list(universe.keys())
        n = min(len(tickers), self.max_positions)
        selected = tickers[:n]
        weight = 1.0 / n if n > 0 else 0.0

        return AllocationResult(
            strategy_name="equal_weight",
            evaluation_date=evaluation_date,
            weights={t: weight for t in selected},
            risk_profiles={},
            kelly_fractions={},
        )

    def _risk_parity(
        self, universe: Dict[str, pd.DataFrame], evaluation_date: date
    ) -> AllocationResult:
        """Inverse-volatility weighted allocation."""
        inv_vols = {}

        for ticker, prices in universe.items():
            try:
                close = prices.get("close", prices.get("Close", prices.get("adjusted_close")))
                if close is None or len(close) < 60:
                    continue
                returns = close.pct_change().dropna()
                vol = returns.std() * np.sqrt(252)
                if vol > 0:
                    inv_vols[ticker] = 1.0 / vol
            except Exception:
                continue

        # Take top N by inverse vol (least volatile)
        if len(inv_vols) > self.max_positions:
            sorted_iv = sorted(inv_vols.items(), key=lambda x: x[1], reverse=True)
            inv_vols = dict(sorted_iv[:self.max_positions])

        total = sum(inv_vols.values())
        weights = {t: iv / total for t, iv in inv_vols.items()} if total > 0 else {}

        return AllocationResult(
            strategy_name="risk_parity",
            evaluation_date=evaluation_date,
            weights=weights,
            risk_profiles={},
            kelly_fractions={},
        )

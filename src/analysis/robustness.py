"""
Robustness / Sensitivity Analysis

Runs the full Bayesian Kelly pipeline twice — once with VIX as the macro
feature, once with FEDFUNDS — and compares portfolio performance.

This demonstrates that the framework isn't overfit to a single indicator
and investigates how information quality drives allocation decisions.

Usage:
    python main.py --robustness
"""

from datetime import date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

from src.config.config import (
    INITIAL_CAPITAL, TRAINING_START_DATE, BACKTEST_END_DATE,
    TARGET_NUM_POSITIONS, BAYESIAN_MC_N_SIMS, KELLY_FRACTION,
)
from src.data.data_loader import DataLoader
from src.portfolio.strategy_runner import BayesianStrategyRunner, AllocationResult
from src.comparison.strategy_comparator import StrategyComparator, StrategyMetrics


class RobustnessAnalyzer:
    """
    Runs the Bayesian Kelly pipeline with different macro features
    and compares the resulting portfolio performance.

    Primary comparison: VIX (market-implied volatility, daily, forward-looking)
    vs FEDFUNDS (policy rate, monthly, backward-looking).

    Outputs:
        - Sharpe ratio delta
        - Max drawdown delta
        - Mean confidence score delta (how much the model's certainty changes)
        - Summary table logged and saved to CSV
    """

    def __init__(
        self,
        loader: DataLoader,
        universe: Dict[str, pd.DataFrame],
        market_prices: Optional[pd.DataFrame],
        bond_spread: pd.Series,
    ):
        self.loader = loader
        self.universe = universe
        self.market_prices = market_prices
        self.bond_spread = bond_spread
        self.logger = logger.bind(module="robustness")

    def _load_macro(self, name: str) -> pd.Series:
        """Load macro feature by name."""
        if name == "VIX":
            return self.loader.load_vix(TRAINING_START_DATE, BACKTEST_END_DATE)
        elif name == "FEDFUNDS":
            return self.loader.load_fedfunds(TRAINING_START_DATE, BACKTEST_END_DATE)
        else:
            raise ValueError(f"Unknown macro feature: {name}")

    def _run_single(
        self,
        macro_name: str,
        macro_feature: pd.Series,
        rebalance_dates: List[date],
    ) -> StrategyMetrics:
        """Run Bayesian Kelly with a specific macro feature and return metrics."""
        runner = BayesianStrategyRunner(
            macro_feature_name=macro_name,
            max_positions=TARGET_NUM_POSITIONS,
            n_simulations=BAYESIAN_MC_N_SIMS,
            fractional_kelly=KELLY_FRACTION,
        )

        comparator = StrategyComparator(initial_capital=INITIAL_CAPITAL)
        comparator.add_strategy(f"bayesian_kelly_{macro_name}", runner)

        result = comparator.compare(
            universe=self.universe,
            rebalance_dates=rebalance_dates,
            market_prices=self.market_prices,
            macro_feature=macro_feature,
            bond_spread=self.bond_spread,
        )

        strategy_name = f"bayesian_kelly_{macro_name}"
        return result.strategies.get(strategy_name, StrategyMetrics(strategy_name=strategy_name))

    def run(
        self,
        rebalance_dates: List[date],
        output_dir: Optional[str] = None,
    ) -> Dict[str, StrategyMetrics]:
        """
        Run the full robustness comparison.

        Args:
            rebalance_dates: Dates on which to rebalance.
            output_dir: Directory to save results CSV (optional).

        Returns:
            Dict of macro_name -> StrategyMetrics.
        """
        self.logger.info("=" * 70)
        self.logger.info("ROBUSTNESS ANALYSIS: VIX vs FEDFUNDS")
        self.logger.info("=" * 70)

        results = {}

        for macro_name in ["VIX", "FEDFUNDS"]:
            self.logger.info(f"\n--- Running with MACRO_FEATURE = {macro_name} ---")

            macro_feature = self._load_macro(macro_name)

            if macro_feature.empty:
                self.logger.warning(
                    f"No data for {macro_name}, skipping this variant"
                )
                continue

            self.logger.info(f"Loaded {macro_name}: {len(macro_feature)} observations")

            metrics = self._run_single(macro_name, macro_feature, rebalance_dates)
            results[macro_name] = metrics

            self.logger.info(
                f"{macro_name}: Return={metrics.total_return:+.2%}, "
                f"Sharpe={metrics.sharpe_ratio:.3f}, "
                f"MaxDD={metrics.max_drawdown:.2%}, "
                f"Vol={metrics.volatility:.2%}"
            )

        # --- Comparison ---
        if len(results) == 2:
            vix_m = results["VIX"]
            fed_m = results["FEDFUNDS"]

            self.logger.info("\n" + "=" * 70)
            self.logger.info("ROBUSTNESS SUMMARY")
            self.logger.info("=" * 70)

            delta_sharpe = vix_m.sharpe_ratio - fed_m.sharpe_ratio
            delta_return = vix_m.total_return - fed_m.total_return
            delta_dd = abs(vix_m.max_drawdown) - abs(fed_m.max_drawdown)

            self.logger.info(f"{'Metric':<25} {'VIX':>12} {'FEDFUNDS':>12} {'Delta':>12}")
            self.logger.info("-" * 65)
            self.logger.info(
                f"{'Total Return':<25} {vix_m.total_return:>+11.2%} "
                f"{fed_m.total_return:>+11.2%} {delta_return:>+11.2%}"
            )
            self.logger.info(
                f"{'Sharpe Ratio':<25} {vix_m.sharpe_ratio:>12.3f} "
                f"{fed_m.sharpe_ratio:>12.3f} {delta_sharpe:>+12.3f}"
            )
            self.logger.info(
                f"{'Max Drawdown':<25} {vix_m.max_drawdown:>11.2%} "
                f"{fed_m.max_drawdown:>11.2%} {delta_dd:>+11.2%}"
            )
            self.logger.info(
                f"{'Volatility':<25} {vix_m.volatility:>11.2%} "
                f"{fed_m.volatility:>11.2%} "
                f"{vix_m.volatility - fed_m.volatility:>+11.2%}"
            )
            self.logger.info(
                f"{'Avg Positions':<25} {vix_m.avg_positions:>12.1f} "
                f"{fed_m.avg_positions:>12.1f} "
                f"{vix_m.avg_positions - fed_m.avg_positions:>+12.1f}"
            )

            if abs(delta_sharpe) < 0.1:
                self.logger.info(
                    "\nConclusion: Sharpe is robust to macro feature choice "
                    "(|ΔSharpe| < 0.1). Framework is not overfit to VIX."
                )
            elif delta_sharpe > 0:
                self.logger.info(
                    f"\nConclusion: VIX improves Sharpe by {delta_sharpe:+.3f}. "
                    "Forward-looking vol information adds value."
                )
            else:
                self.logger.info(
                    f"\nConclusion: FEDFUNDS improves Sharpe by {-delta_sharpe:+.3f}. "
                    "Policy rate signal dominates in this period."
                )

            # Save to CSV
            if output_dir:
                self._save_results(results, output_dir)

        return results

    def _save_results(
        self,
        results: Dict[str, StrategyMetrics],
        output_dir: str,
    ):
        """Save robustness results to CSV."""
        rows = []
        for macro_name, m in results.items():
            rows.append({
                "macro_feature": macro_name,
                "total_return": m.total_return,
                "annualized_return": m.annualized_return,
                "sharpe_ratio": m.sharpe_ratio,
                "max_drawdown": m.max_drawdown,
                "calmar_ratio": m.calmar_ratio,
                "volatility": m.volatility,
                "win_rate": m.win_rate,
                "avg_positions": m.avg_positions,
            })

        df = pd.DataFrame(rows)
        path = Path(output_dir) / "robustness_vix_vs_fedfunds.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        self.logger.info(f"Results saved to {path}")

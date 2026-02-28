"""
Main orchestration script for model_regime_comparison project.

Pipeline: Data → Risk Evaluation → Monte Carlo → Kelly → Backtest → Visualize

Usage:
    python main.py                          # Run all strategies comparison
    python main.py --strategy historical_kelly  # Run single strategy
    python main.py --evaluate-only          # Just compute risk profiles
"""

import sys
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import (
    LOG_FILE, LOG_LEVEL, LOG_FORMAT,
    DATABASE_URL, BENCHMARK_TICKER,
    BACKTEST_START_DATE, BACKTEST_END_DATE, TRAINING_START_DATE,
    INITIAL_CAPITAL, REBALANCE_FREQUENCY, STRATEGIES,
    HISTORICAL_LOOKBACK, GARCH_LOOKBACK, REGIME_LOOKBACK,
    RISK_FREE_RATE, SHORT_VOL_WINDOW,
    KELLY_FRACTION, MAX_KELLY_FRACTION, TARGET_NUM_POSITIONS,
    ALL_TICKERS, RESULTS_DIR,
)
from src.data.data_loader import DataLoader
from src.risk_evaluation import (
    HistoricalEvaluator, GARCHEvaluator, RegimeConditionalEvaluator,
)
from src.risk.kelly_criterion import KellyCriterion
from src.portfolio.strategy_runner import StrategyRunner
from src.comparison.strategy_comparator import StrategyComparator


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(LOG_FILE, format=LOG_FORMAT, level=LOG_LEVEL)
    logger.add(sys.stdout, format=LOG_FORMAT, level=LOG_LEVEL)


def load_universe(
    loader: DataLoader,
    tickers: List[str],
    start_date: date,
    end_date: date,
) -> Dict:
    """Load price data for the stock universe from the aggregator DB."""
    universe = {}
    for ticker in tickers:
        prices = loader.load_stock_prices(ticker, start_date, end_date)
        if not prices.empty and len(prices) >= 60:
            universe[ticker] = prices

    logger.info(f"Loaded {len(universe)}/{len(tickers)} tickers with sufficient data")
    return universe


def build_evaluator(name: str):
    """Build a risk evaluator by name."""
    if name == "historical":
        return HistoricalEvaluator(
            lookback_days=HISTORICAL_LOOKBACK,
            short_vol_window=SHORT_VOL_WINDOW,
            risk_free_rate=RISK_FREE_RATE,
        )
    elif name == "garch":
        return GARCHEvaluator(
            lookback_days=GARCH_LOOKBACK,
            risk_free_rate=RISK_FREE_RATE,
        )
    elif name == "regime_conditional":
        return RegimeConditionalEvaluator(
            lookback_days=REGIME_LOOKBACK,
            risk_free_rate=RISK_FREE_RATE,
        )
    return None


def build_strategy_runners() -> Dict[str, StrategyRunner]:
    """Build StrategyRunner instances for all configured strategies."""
    kelly = KellyCriterion(
        max_position_size=MAX_KELLY_FRACTION,
        use_fractional_kelly=True,
        fractional_multiplier=KELLY_FRACTION,
    )

    runners = {}
    for name, config in STRATEGIES.items():
        evaluator = None
        if config.get("evaluator"):
            evaluator = build_evaluator(config["evaluator"])

        runners[name] = StrategyRunner(
            strategy_name=name,
            evaluator=evaluator,
            kelly=kelly,
            max_positions=TARGET_NUM_POSITIONS,
        )

    return runners


def generate_rebalance_dates(
    start: date, end: date, frequency: str = "weekly"
) -> List[date]:
    """Generate rebalance dates between start and end."""
    dates = []
    current = start
    delta = {
        "daily": timedelta(days=1),
        "weekly": timedelta(weeks=1),
        "monthly": timedelta(days=30),
    }.get(frequency, timedelta(weeks=1))

    while current <= end:
        # Skip weekends
        if current.weekday() < 5:
            dates.append(current)
        current += delta

    return dates


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Risk-Based Strategy Comparison")
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Run a single strategy (e.g. historical_kelly)"
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Only compute risk profiles, don't run backtest"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override stock universe (e.g. AAPL MSFT GOOGL)"
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 80)
    logger.info("Risk-Based Strategy Comparison Pipeline")
    logger.info("=" * 80)

    # --- Step 1: Load data ---
    logger.info("Step 1: Loading price data from financial_data_aggregator")
    loader = DataLoader()
    tickers = args.tickers or ALL_TICKERS

    universe = load_universe(
        loader, tickers,
        start_date=TRAINING_START_DATE,
        end_date=BACKTEST_END_DATE,
    )

    if not universe:
        logger.error("No data loaded. Check database connection and ticker universe.")
        sys.exit(1)

    # Load benchmark
    market_prices = loader.load_stock_prices(
        BENCHMARK_TICKER, TRAINING_START_DATE, BACKTEST_END_DATE
    )
    if market_prices.empty:
        logger.warning(f"No benchmark data for {BENCHMARK_TICKER}, beta will default to 1.0")
        market_prices = None

    # --- Step 2: Evaluate only mode ---
    if args.evaluate_only:
        logger.info("Step 2: Computing risk profiles (evaluate-only mode)")
        evaluator = HistoricalEvaluator(
            lookback_days=HISTORICAL_LOOKBACK,
            risk_free_rate=RISK_FREE_RATE,
        )
        profiles = evaluator.evaluate_universe(
            universe=universe,
            evaluation_date=date.today(),
            market_prices=market_prices,
        )
        logger.info(f"Computed {len(profiles)} risk profiles")
        for ticker, p in list(profiles.items())[:5]:
            logger.info(
                f"  {ticker}: vol={p.annualized_vol:.1%} beta={p.beta:.2f} "
                f"sharpe={p.sharpe:.2f} win_rate={p.win_rate:.1%} dd={p.max_drawdown:.1%}"
            )
        return

    # --- Step 3: Build strategies ---
    logger.info("Step 2: Building strategy runners")
    runners = build_strategy_runners()

    if args.strategy:
        if args.strategy not in runners:
            logger.error(f"Unknown strategy: {args.strategy}. Available: {list(runners.keys())}")
            sys.exit(1)
        runners = {args.strategy: runners[args.strategy]}

    logger.info(f"Strategies: {list(runners.keys())}")

    # --- Step 4: Run comparison ---
    logger.info("Step 3: Running strategy comparison")
    rebalance_dates = generate_rebalance_dates(
        BACKTEST_START_DATE, BACKTEST_END_DATE, REBALANCE_FREQUENCY
    )
    logger.info(f"Rebalance dates: {len(rebalance_dates)} ({REBALANCE_FREQUENCY})")

    comparator = StrategyComparator(initial_capital=INITIAL_CAPITAL)
    for name, runner in runners.items():
        comparator.add_strategy(name, runner)

    result = comparator.compare(
        universe=universe,
        rebalance_dates=rebalance_dates,
        market_prices=market_prices,
    )

    # --- Step 5: Report results ---
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    for name, metrics in result.strategies.items():
        logger.info(
            f"{name:25s} | Return: {metrics.total_return:+7.1%} | "
            f"Sharpe: {metrics.sharpe_ratio:5.2f} | "
            f"MaxDD: {metrics.max_drawdown:7.1%} | "
            f"Positions: {metrics.avg_positions:.0f}"
        )

    if result.best_sharpe:
        logger.info(f"Best Sharpe: {result.best_sharpe}")
    if result.best_return:
        logger.info(f"Best Return: {result.best_return}")

    # --- Step 6: Visualization ---
    try:
        from src.visualization.dashboard import Dashboard

        charts_dir = str(RESULTS_DIR / "charts")
        Path(charts_dir).mkdir(parents=True, exist_ok=True)

        dash = Dashboard(output_dir=charts_dir)
        dash.plot_equity_curves(result)
        dash.plot_drawdowns(result)
        dash.plot_metrics_comparison(result)
        logger.info(f"Charts saved to {charts_dir}/")
    except ImportError:
        logger.warning("plotly not installed, skipping visualization")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()

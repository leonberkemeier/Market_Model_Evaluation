"""
Run Sentinel Daily Pipeline

This script orchestrates the full Sentinel trading pipeline:
1. Layer I: Feature Engineering
2. Layer II: Regime Detection (HMM)
3. Layer III: Expert Model Predictions
4. Layer IV: Risk Management (Monte Carlo + Kelly)
5. Layer V: Execution via Trading Simulator

Usage:
    # Run with default config
    python scripts/run_sentinel_daily.py
    
    # Dry run (no execution)
    python scripts/run_sentinel_daily.py --dry-run
    
    # Custom config
    python scripts/run_sentinel_daily.py --config config/custom_config.yaml
    
    # Verbose logging
    python scripts/run_sentinel_daily.py --verbose
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import yaml
import json
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.execution.scheduler import SentinelScheduler
from src.execution.api_client import TradingSimulatorClient
from src.execution.portfolio_manager import SentinelPortfolioManager
from src.regime.hmm_detector import HMMRegimeDetector
from src.risk.monte_carlo import MonteCarloSimulator
from src.risk.kelly_criterion import KellyCriterion


def setup_logging(log_config: dict, verbose: bool = False):
    """
    Setup logging configuration.
    
    Args:
        log_config: Logging config from YAML
        verbose: If True, use DEBUG level
    """
    log_dir = Path(log_config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / log_config.get("log_file", "sentinel_pipeline.log")
    log_level = logging.DEBUG if verbose else getattr(
        logging, 
        log_config.get("level", "INFO").upper()
    )
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: level={logging.getLevelName(log_level)}, file={log_file}")
    
    return logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML
    
    Returns:
        Configuration dict
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def load_expert_model(model_config: dict, project_root: Path):
    """
    Load an expert model from disk.
    
    Args:
        model_config: Model config dict with 'type' and 'path'
        project_root: Project root directory
    
    Returns:
        Loaded expert model
    """
    model_path = project_root / model_config["path"]
    model_type = model_config["type"]
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    logging.info(f"Loaded {model_type} model from {model_path}")
    
    return model


def initialize_components(config: dict, project_root: Path, logger: logging.Logger):
    """
    Initialize all Sentinel components from config.
    
    Args:
        config: Configuration dict
        project_root: Project root directory
        logger: Logger instance
    
    Returns:
        tuple of (scheduler, api_client, portfolio_manager)
    """
    logger.info("Initializing Sentinel components...")
    
    # Initialize Trading Simulator API client
    api_config = config["trading_simulator"]
    api_client = TradingSimulatorClient(
        base_url=api_config["base_url"],
        timeout=api_config["timeout_seconds"]
    )
    logger.info(f"Trading Simulator API client initialized: {api_config['base_url']}")
    
    # Initialize Portfolio Manager
    portfolio_config = config["portfolio"]
    portfolio_manager = SentinelPortfolioManager(
        api_client=api_client,
        initial_capital=portfolio_config["initial_capital"],
        min_rebalance_threshold=portfolio_config["min_rebalance_threshold"]
    )
    logger.info(f"Portfolio Manager initialized: ${portfolio_config['initial_capital']:,.2f} capital")
    
    # Load HMM Regime Detector
    regime_config = config["regime"]
    regime_model_path = project_root / regime_config["model_path"]
    
    if regime_model_path.exists():
        with open(regime_model_path, "rb") as f:
            hmm_detector = pickle.load(f)
        logger.info(f"HMM Regime Detector loaded from {regime_model_path}")
    else:
        logger.warning(f"HMM model not found at {regime_model_path}, using default")
        hmm_detector = HMMRegimeDetector(n_regimes=3)
    
    # Load Expert Models
    experts_config = config["experts"]
    active_sectors = experts_config["active_sectors"]
    models_config = experts_config["models"]
    
    sector_experts = {}
    for sector in active_sectors:
        if sector not in models_config:
            logger.warning(f"No model config found for sector: {sector}, skipping")
            continue
        
        try:
            model = load_expert_model(models_config[sector], project_root)
            sector_experts[sector] = model
            logger.info(f"Expert loaded for sector: {sector} ({models_config[sector]['type']})")
        except Exception as e:
            logger.error(f"Failed to load expert for {sector}: {e}")
            continue
    
    if not sector_experts:
        raise ValueError("No expert models loaded! Check config and model paths.")
    
    # Initialize Monte Carlo Simulator
    mc_config = config["monte_carlo"]
    monte_carlo = MonteCarloSimulator(
        n_simulations=mc_config["n_simulations"],
        random_seed=mc_config.get("random_seed")
    )
    logger.info(f"Monte Carlo Simulator initialized: {mc_config['n_simulations']} simulations")
    
    # Initialize Kelly Criterion
    kelly_config = config["kelly"]
    kelly = KellyCriterion(
        max_position_size=kelly_config["max_position_size"],
        use_fractional_kelly=kelly_config["use_fractional_kelly"],
        fractional_multiplier=kelly_config["fractional_multiplier"],
        min_win_prob=kelly_config["min_win_prob"]
    )
    logger.info(
        f"Kelly Criterion initialized: max_pos={kelly_config['max_position_size']:.1%}, "
        f"fractional={kelly_config['fractional_multiplier']}"
    )
    
    # Get sector tickers
    sector_tickers = config.get("sector_tickers", {})
    # Filter to only active sectors
    sector_tickers = {
        sector: tickers
        for sector, tickers in sector_tickers.items()
        if sector in active_sectors
    }
    
    # Initialize Sentinel Scheduler
    scheduler = SentinelScheduler(
        sector_experts=sector_experts,
        hmm_detector=hmm_detector,
        portfolio_manager=portfolio_manager,
        monte_carlo=monte_carlo,
        kelly=kelly,
        sector_tickers=sector_tickers
    )
    
    logger.info("All components initialized successfully")
    
    return scheduler, api_client, portfolio_manager


def save_results(result: dict, output_dir: Path, logger: logging.Logger):
    """
    Save pipeline results to JSON.
    
    Args:
        result: Pipeline result dict
        output_dir: Output directory
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sentinel_run_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")


def print_summary(result: dict, logger: logging.Logger):
    """
    Print execution summary.
    
    Args:
        result: Pipeline result dict
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("SENTINEL PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"Date: {result['date']}")
    logger.info(f"Dry Run: {result['dry_run']}")
    
    regime = result["regime"]
    logger.info(f"\nRegime: {regime['regime'].upper()} (confidence: {regime['confidence']:.2%})")
    
    predictions = result["predictions"]
    logger.info(f"\nPredictions: {sum(predictions.values())} total across {len(predictions)} sectors")
    for sector, count in predictions.items():
        logger.info(f"  - {sector}: {count} predictions")
    
    position_sizes = result["position_sizes"]
    logger.info(f"\nPosition Sizes: {len(position_sizes)} positions")
    for ticker, size in sorted(position_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  - {ticker}: {size:.2%}")
    
    portfolio = result["portfolio"]
    logger.info(f"\nPortfolio:")
    logger.info(f"  - Total Value: ${portfolio['total_value']:,.2f}")
    logger.info(f"  - Cash Balance: ${portfolio['cash_balance']:,.2f}")
    logger.info(f"  - Positions: {portfolio['n_positions']}")
    logger.info(f"  - Return: {portfolio['total_return_pct']:.2%}")
    
    logger.info(f"\nOrders Executed: {result['n_orders']}")
    
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Sentinel daily trading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/sentinel_config.yaml",
        help="Path to configuration YAML file (default: config/sentinel_config.yaml)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without executing trades (testing mode)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = project_root / args.config
    config = load_config(str(config_path))
    
    # Setup logging
    logger = setup_logging(config["logging"], verbose=args.verbose)
    
    logger.info("=" * 80)
    logger.info("SENTINEL DAILY PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info("=" * 80)
    
    try:
        # Initialize components
        scheduler, api_client, portfolio_manager = initialize_components(
            config, 
            project_root, 
            logger
        )
        
        # Get scheduler statistics
        stats = scheduler.get_statistics()
        logger.info(f"Scheduler stats: {json.dumps(stats, indent=2)}")
        
        # Run pipeline
        logger.info("\nStarting pipeline execution...\n")
        
        result = scheduler.run_daily_pipeline(
            run_date=datetime.now(),
            dry_run=args.dry_run or config["execution"].get("dry_run", False)
        )
        
        # Print summary
        print_summary(result, logger)
        
        # Save results
        output_dir = project_root / args.output_dir
        save_results(result, output_dir, logger)
        
        logger.info("\nPipeline execution completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

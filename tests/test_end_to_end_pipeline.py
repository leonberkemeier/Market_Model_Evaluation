"""
Integration Test for Sentinel End-to-End Pipeline

This test validates that all 5 layers of the Sentinel pipeline work together.

Usage:
    python tests/test_end_to_end_pipeline.py
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.execution.scheduler import SentinelScheduler
from src.execution.api_client import TradingSimulatorClient
from src.execution.portfolio_manager import SentinelPortfolioManager
from src.regime.hmm_detector import HMMRegimeDetector
from src.experts.linear_model import LinearExpert
from src.experts.xgboost_model import XGBoostExpert
from src.risk.monte_carlo import MonteCarloSimulator
from src.risk.kelly_criterion import KellyCriterion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline_initialization():
    """Test that all pipeline components can be initialized."""
    logger.info("=" * 80)
    logger.info("TEST: Pipeline Initialization")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("Initializing API client...")
    api_client = TradingSimulatorClient(
        base_url="http://localhost:8000",
        timeout=30
    )
    
    logger.info("Initializing portfolio manager...")
    portfolio_manager = SentinelPortfolioManager(
        api_client=api_client,
        initial_capital=100000.0,
        min_rebalance_threshold=0.05
    )
    
    logger.info("Initializing HMM regime detector...")
    hmm_detector = HMMRegimeDetector(n_regimes=3)
    
    logger.info("Initializing expert models...")
    linear_expert = LinearExpert(
        sector="finance",
        ridge_alpha=1.0,
        max_features=50
    )
    
    xgboost_expert = XGBoostExpert(
        sector="tech",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01
    )
    
    sector_experts = {
        "finance": linear_expert,
        "tech": xgboost_expert
    }
    
    logger.info("Initializing Monte Carlo simulator...")
    monte_carlo = MonteCarloSimulator(
        n_simulations=10000,
        random_seed=42
    )
    
    logger.info("Initializing Kelly Criterion...")
    kelly = KellyCriterion(
        max_position_size=0.15,
        use_fractional_kelly=True,
        fractional_multiplier=0.5,
        min_win_prob=0.52
    )
    
    logger.info("Initializing Sentinel scheduler...")
    scheduler = SentinelScheduler(
        sector_experts=sector_experts,
        hmm_detector=hmm_detector,
        portfolio_manager=portfolio_manager,
        monte_carlo=monte_carlo,
        kelly=kelly
    )
    
    logger.info("✓ All components initialized successfully")
    
    # Get statistics
    stats = scheduler.get_statistics()
    logger.info(f"\nScheduler Statistics:")
    logger.info(f"  - Sectors: {stats['n_sectors']} ({', '.join(stats['sectors'])})")
    logger.info(f"  - Total tickers: {stats['n_tickers_total']}")
    logger.info(f"  - Tickers per sector: {stats['tickers_per_sector']}")
    
    return scheduler


def test_pipeline_dry_run(scheduler: SentinelScheduler):
    """Test that the pipeline can run in dry-run mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Pipeline Dry Run")
    logger.info("=" * 80)
    
    logger.info("Running pipeline in dry-run mode (no real trades)...")
    
    try:
        result = scheduler.run_daily_pipeline(dry_run=True)
        
        logger.info("✓ Pipeline completed successfully")
        
        # Validate result structure
        assert "date" in result
        assert "regime" in result
        assert "predictions" in result
        assert "position_sizes" in result
        assert "portfolio" in result
        assert result["dry_run"] is True
        
        logger.info(f"\nPipeline Result Summary:")
        logger.info(f"  - Date: {result['date']}")
        logger.info(f"  - Regime: {result['regime']['regime']} (confidence: {result['regime']['confidence']:.2%})")
        logger.info(f"  - Predictions: {sum(result['predictions'].values())} total")
        logger.info(f"  - Position sizes: {len(result['position_sizes'])} positions")
        logger.info(f"  - Orders executed: {result['n_orders']}")
        
        # Show top positions
        if result['position_sizes']:
            logger.info(f"\n  Top positions:")
            sorted_positions = sorted(
                result['position_sizes'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for ticker, size in sorted_positions[:5]:
                logger.info(f"    - {ticker}: {size:.2%}")
        
        logger.info("\n✓ Pipeline dry run test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline dry run test failed: {e}", exc_info=True)
        return False


def main():
    """Run all integration tests."""
    logger.info("\n" + "=" * 80)
    logger.info("SENTINEL PIPELINE INTEGRATION TESTS")
    logger.info("=" * 80)
    
    try:
        # Test 1: Initialization
        scheduler = test_pipeline_initialization()
        
        # Test 2: Dry run
        success = test_pipeline_dry_run(scheduler)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        if success:
            logger.info("✓ All tests passed!")
            logger.info("\nThe end-to-end pipeline is working correctly.")
            logger.info("You can now:")
            logger.info("  1. Train your expert models (Linear, XGBoost, CNN, LLM)")
            logger.info("  2. Run the pipeline with real models")
            logger.info("  3. Execute: python scripts/run_sentinel_daily.py --dry-run")
            return 0
        else:
            logger.error("✗ Some tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"✗ Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
⚠️ DEPRECATED: Integration Test for OLD 4-Scorer Pipeline

This test validates the OLD Sentinel pipeline using:
- HMM Regime Detection (3 states)
- Expert Models (Linear, CNN, XGBoost)
- Monte Carlo + Kelly Criterion

STATUS: DEPRECATED as of April 10, 2026

FOR NEW CODE: Use test_end_to_end_pipeline_NEW.py instead

The NEW test validates the 7-phase pipeline:
1. Load data from financial_data_aggregator DB
2. Markov Chain: Detect market regime (5 states)
3. Enhanced Monte Carlo: Simulate returns + compute risk metrics
4. LLM Asset Selector: Score assets based on regime + metrics
5. Risk Profile Portfolio: Construct allocation respecting constraints
6. Validation: Verify portfolio meets all constraints
7. Export: Save results and create audit trail

See AUDIT_AND_CLEANUP_REPORT.md for details on the transition.

---

This test is kept for backward compatibility but should not be run
in production. It tests the deprecated 4-scorer architecture.

Usage (deprecated):
    python tests/test_end_to_end_pipeline.py

Usage (NEW):
    python tests/test_end_to_end_pipeline_NEW.py
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ⚠️ LEGACY IMPORTS - These modules are deprecated
from src.execution.scheduler import SentinelScheduler
from src.execution.api_client import TradingSimulatorClient
from src.execution.portfolio_manager import SentinelPortfolioManager
from src.regime.hmm_detector import HMMRegimeDetector
from src.experts.linear_model import LinearModel
from src.experts.xgboost_model import XGBoostModel
from src.risk.monte_carlo import MonteCarloSimulator
from src.risk.kelly_criterion import KellyCriterion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

logger.warning(
    "⚠️ DEPRECATED TEST: test_end_to_end_pipeline.py is testing the OLD 4-scorer pipeline.\n"
    "For NEW pipeline tests, use: test_end_to_end_pipeline_NEW.py"
)


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
        portfolio_id=1,  # Test portfolio ID
        rebalance_tolerance=0.05
    )
    
    logger.info("Initializing HMM regime detector...")
    hmm_detector = HMMRegimeDetector(n_states=3)
    
    logger.info("Initializing expert models...")
    linear_expert = LinearModel(sector="finance")
    
    xgboost_expert = XGBoostModel(sector="tech")
    
    sector_experts = {
        "finance": linear_expert,
        "tech": xgboost_expert
    }
    
    logger.info("Initializing Monte Carlo simulator...")
    monte_carlo = MonteCarloSimulator(n_simulations=10000)
    
    logger.info("Initializing Kelly Criterion...")
    kelly = KellyCriterion()
    
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
            logger.info("\n⚠️ NOTE: This is testing the DEPRECATED 4-scorer pipeline.")
            logger.info("For NEW pipeline, run: python tests/test_end_to_end_pipeline_NEW.py")
            logger.info("\nThe end-to-end pipeline is working correctly.")
            logger.info("You can now:")
            logger.info("  1. Train your expert models (Linear, XGBoost, CNN, LLM) [DEPRECATED]")
            logger.info("  2. Use NEW pipeline: src.analysis.analysis_pipeline.AnalysisPipeline")
            logger.info("  3. NEW pipeline: Markov → Enhanced MC → LLM → Risk Profiles")
            return 0
        else:
            logger.error("✗ Some tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"✗ Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

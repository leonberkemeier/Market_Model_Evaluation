"""
Sentinel Scheduler - End-to-End Pipeline Orchestration

Orchestrates the full 5-layer Sentinel pipeline:
- Layer I: Feature Engineering
- Layer II: Regime Detection
- Layer III: Expert Models
- Layer IV: Monte Carlo + Kelly
- Layer V: Execution
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from ..regime import HMMRegimeDetector
from ..experts import BaseExpert
from ..risk import MonteCarloSimulator, KellyCriterion
from .api_client import TradingSimulatorClient
from .portfolio_manager import SentinelPortfolioManager

logger = logging.getLogger(__name__)


class SentinelScheduler:
    """
    Orchestrates the full Sentinel pipeline.
    
    Workflow:
    1. Layer I: Load and engineer features
    2. Layer II: Detect market regime
    3. Layer III: Generate predictions from sector experts
    4. Layer IV: Calculate position sizes (Monte Carlo + Kelly)
    5. Layer V: Execute trades via Trading Simulator
    
    Usage:
        scheduler = SentinelScheduler(
            sector_experts={"tech": xgboost_model, "finance": linear_model},
            hmm_detector=hmm,
            portfolio_manager=manager
        )
        
        result = scheduler.run_daily_pipeline(date=datetime.today())
    """
    
    def __init__(
        self,
        sector_experts: Dict[str, BaseExpert],
        hmm_detector: HMMRegimeDetector,
        portfolio_manager: SentinelPortfolioManager,
        monte_carlo: Optional[MonteCarloSimulator] = None,
        kelly: Optional[KellyCriterion] = None,
        sector_tickers: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize Sentinel scheduler.
        
        Args:
            sector_experts: Dict of sector -> expert model
            hmm_detector: HMM regime detector
            portfolio_manager: Portfolio manager for execution
            monte_carlo: Monte Carlo simulator (creates default if None)
            kelly: Kelly Criterion calculator (creates default if None)
            sector_tickers: Dict of sector -> list of tickers (uses defaults if None)
        """
        self.sector_experts = sector_experts
        self.hmm_detector = hmm_detector
        self.portfolio_manager = portfolio_manager
        
        # Initialize risk management
        self.monte_carlo = monte_carlo or MonteCarloSimulator(n_simulations=10000)
        self.kelly = kelly or KellyCriterion(
            max_position_size=0.15,
            use_fractional_kelly=True,
            fractional_multiplier=0.5,
            min_win_prob=0.52
        )
        
        # Default sector tickers (can be overridden)
        self.sector_tickers = sector_tickers or self._get_default_sector_tickers()
        
        logger.info(
            f"SentinelScheduler initialized with {len(self.sector_experts)} experts "
            f"covering {len(self.sector_tickers)} sectors"
        )
    
    def run_daily_pipeline(
        self,
        run_date: Optional[datetime] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run full Sentinel pipeline for a given date.
        
        Args:
            run_date: Date to run pipeline for (uses today if None)
            dry_run: If True, skip execution (testing mode)
        
        Returns:
            dict with execution summary
        """
        run_date = run_date or datetime.now()
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Running Sentinel pipeline for {run_date.date()}")
        
        try:
            # Layer II: Detect regime
            logger.info("Layer II: Detecting market regime...")
            regime_result = self._detect_regime()
            regime = regime_result["regime"]
            regime_confidence = regime_result["confidence"]
            
            logger.info(
                f"Current regime: {regime.upper()} "
                f"(confidence: {regime_confidence:.2%})"
            )
            
            # Layer III: Get predictions from experts
            logger.info("Layer III: Generating predictions from sector experts...")
            predictions = self._generate_predictions(regime)
            
            logger.info(f"Generated {sum(len(p) for p in predictions.values())} predictions")
            
            # Layer IV: Calculate position sizes
            logger.info("Layer IV: Calculating position sizes (Monte Carlo + Kelly)...")
            position_sizes = self._calculate_position_sizes(predictions, regime)
            
            logger.info(f"Calculated position sizes for {len(position_sizes)} tickers")
            
            # Normalize positions to sum to 100%
            position_sizes = self._normalize_positions(position_sizes)
            
            # Layer V: Execute trades
            if not dry_run:
                logger.info("Layer V: Executing rebalance...")
                orders = self.portfolio_manager.rebalance_portfolio(
                    target_positions=position_sizes,
                    dry_run=False
                )
                logger.info(f"Executed {len(orders)} orders")
            else:
                logger.info("Layer V: Skipping execution (dry run mode)")
                orders = []
            
            # Get portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            performance_summary = self.portfolio_manager.get_performance_summary()
            
            # Build result
            result = {
                "date": run_date.isoformat(),
                "dry_run": dry_run,
                "regime": {
                    "regime": regime,
                    "confidence": regime_confidence,
                    "probabilities": regime_result.get("probabilities", {})
                },
                "predictions": {
                    sector: len(preds) 
                    for sector, preds in predictions.items()
                },
                "position_sizes": position_sizes,
                "n_orders": len(orders),
                "portfolio": {
                    "total_value": portfolio_summary["total_value"],
                    "cash_balance": portfolio_summary["cash_balance"],
                    "n_positions": portfolio_summary["n_positions"],
                    "total_return_pct": portfolio_summary["total_return_pct"]
                },
                "performance": performance_summary,
                "orders": [
                    {
                        "ticker": o.ticker,
                        "action": o.order_type,
                        "quantity": o.quantity,
                        "price": o.price,
                        "order_id": o.order_id
                    }
                    for o in orders
                ]
            }
            
            logger.info(
                f"Pipeline complete: {len(orders)} orders, "
                f"Portfolio value: ${portfolio_summary['total_value']:,.2f}, "
                f"Return: {portfolio_summary['total_return_pct']:.2%}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _detect_regime(self) -> Dict[str, Any]:
        """
        Detect current market regime using HMM.
        
        Returns:
            dict with regime, confidence, and probabilities
        """
        # For now, return a placeholder
        # TODO: Integrate with actual price data loading
        
        # This should load SPY prices and predict regime
        # regime_result = self.hmm_detector.predict_regime(spy_prices)
        
        # Placeholder for now
        regime_result = {
            "regime": "bull",
            "confidence": 0.85,
            "probabilities": {
                "bull": 0.85,
                "bear": 0.10,
                "sideways": 0.05
            }
        }
        
        logger.debug(f"Regime detection: {regime_result}")
        return regime_result
    
    def _generate_predictions(
        self,
        regime: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate predictions from all sector experts.
        
        Args:
            regime: Current market regime
        
        Returns:
            dict of sector -> list of predictions
        """
        predictions = {}
        
        for sector, expert in self.sector_experts.items():
            tickers = self.sector_tickers.get(sector, [])
            if not tickers:
                logger.warning(f"No tickers defined for sector: {sector}")
                continue
            
            logger.debug(f"Generating predictions for {sector} ({len(tickers)} tickers)")
            
            # TODO: Load actual features for these tickers
            # For now, return placeholder predictions
            
            sector_predictions = []
            for ticker in tickers:
                # Placeholder prediction
                prediction = {
                    "ticker": ticker,
                    "predicted_return": 0.02,  # 2% expected return
                    "confidence": 0.75
                }
                sector_predictions.append(prediction)
            
            predictions[sector] = sector_predictions
        
        return predictions
    
    def _calculate_position_sizes(
        self,
        predictions: Dict[str, List[Dict[str, Any]]],
        regime: str
    ) -> Dict[str, float]:
        """
        Calculate position sizes using Monte Carlo + Kelly.
        
        Args:
            predictions: Sector predictions
            regime: Current market regime
        
        Returns:
            dict of ticker -> Kelly fraction
        """
        position_sizes = {}
        
        for sector, sector_preds in predictions.items():
            for pred in sector_preds:
                ticker = pred["ticker"]
                predicted_return = pred["predicted_return"]
                
                # Skip if predicted return is too low
                if predicted_return < 0.005:  # Less than 0.5%
                    logger.debug(f"Skipping {ticker}: predicted return too low ({predicted_return:.2%})")
                    continue
                
                # Monte Carlo simulation
                mc_result = self.monte_carlo.simulate(
                    ticker=ticker,
                    predicted_return=predicted_return,
                    regime=regime
                )
                
                # Kelly position sizing
                kelly_result = self.kelly.calculate(mc_result, regime=regime)
                
                # Only include if Kelly fraction > 0
                if kelly_result.kelly_fraction > 0:
                    position_sizes[ticker] = kelly_result.kelly_fraction
                    
                    logger.debug(
                        f"{ticker}: Kelly={kelly_result.kelly_fraction:.2%} "
                        f"(win_prob={kelly_result.win_prob:.2%}, "
                        f"win_loss_ratio={kelly_result.win_loss_ratio:.2f})"
                    )
        
        return position_sizes
    
    def _normalize_positions(
        self,
        position_sizes: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize position sizes to sum to 100%.
        
        Args:
            position_sizes: Raw Kelly fractions
        
        Returns:
            Normalized position sizes
        """
        if not position_sizes:
            return {}
        
        total = sum(position_sizes.values())
        
        if total == 0:
            logger.warning("Total position size is 0, no positions to take")
            return {}
        
        normalized = {
            ticker: size / total
            for ticker, size in position_sizes.items()
        }
        
        logger.debug(f"Normalized {len(normalized)} positions (sum={sum(normalized.values()):.2%})")
        
        return normalized
    
    def _get_default_sector_tickers(self) -> Dict[str, List[str]]:
        """
        Get default sector ticker mappings.
        
        Returns:
            dict of sector -> list of tickers
        """
        return {
            "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMD", "INTC", "CRM", "ADBE"],
            "finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP", "USB", "PNC"],
            "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"],
            "commodities": ["GLD", "SLV", "USO", "DBC"],
            "cyclicals": ["CAT", "DE", "BA", "HON", "MMM", "GE", "UPS", "FDX", "DAL", "UAL"]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            dict with scheduler stats
        """
        return {
            "n_sectors": len(self.sector_experts),
            "sectors": list(self.sector_experts.keys()),
            "n_tickers_total": sum(len(tickers) for tickers in self.sector_tickers.values()),
            "tickers_per_sector": {
                sector: len(tickers)
                for sector, tickers in self.sector_tickers.items()
            },
            "monte_carlo": self.monte_carlo.get_statistics(),
            "kelly": self.kelly.get_statistics()
        }

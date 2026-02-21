#!/usr/bin/env python3
"""
Daily Signal Orchestrator

Ties together the full pipeline:
1. Load price data from financial_data_aggregator
2. Compute features for stock universe
3. Score stocks using ML models
4. Generate trade signals (only high-quality opportunities)
5. Send signals to Trading_Simulator for execution
6. Log results

Usage:
    # Run with default settings (dry run)
    python run_daily_signals.py
    
    # Execute signals for real
    python run_daily_signals.py --execute
    
    # Run specific model
    python run_daily_signals.py --model xgboost --execute
    
    # Use conservative signal generator
    python run_daily_signals.py --profile conservative
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_structures import ScoreResult, ScorerOutput
from src.trading import (
    SignalGenerator,
    SignalGeneratorConfig,
    create_conservative_generator,
    create_moderate_generator,
    create_aggressive_generator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DailySignalOrchestrator:
    """
    Orchestrates the daily signal generation and execution pipeline.
    
    Workflow:
    1. Load stock universe and prices (optionally filtered by screener)
    2. Compute features
    3. Run model scorers
    4. Generate signals (with quality gates)
    5. Execute via Trading_Simulator API
    """
    
    def __init__(
        self,
        model_name: str = "xgboost",
        signal_profile: str = "moderate",
        trading_api_url: str = "http://localhost:8001",
        initial_capital: float = 100000.0,
        use_screener: bool = False,
        screener_strategies: List[str] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            model_name: Which model to use ("linear", "cnn", "xgboost", "llm")
            signal_profile: Signal generator profile ("conservative", "moderate", "aggressive")
            trading_api_url: Trading Simulator API URL
            initial_capital: Starting capital for new portfolios
        """
        self.model_name = model_name
        self.trading_api_url = trading_api_url
        self.initial_capital = initial_capital
        self.use_screener = use_screener
        self.screener_strategies = screener_strategies or ["dividend", "volatility"]
        
        # Initialize signal generator based on profile
        if signal_profile == "conservative":
            self.signal_generator = create_conservative_generator()
        elif signal_profile == "aggressive":
            self.signal_generator = create_aggressive_generator()
        else:
            self.signal_generator = create_moderate_generator()
        
        screener_info = f", screener={self.screener_strategies}" if use_screener else ""
        logger.info(f"Initialized orchestrator: model={model_name}, profile={signal_profile}{screener_info}")
    
    def run(
        self,
        target_date: Optional[date] = None,
        dry_run: bool = True,
        tickers: Optional[List[str]] = None
    ) -> Dict:
        """
        Run the full daily pipeline.
        
        Args:
            target_date: Date to generate signals for (default: today)
            dry_run: If True, simulate without executing trades
            tickers: Optional list of tickers to process (default: full universe)
            
        Returns:
            Summary dictionary with results
        """
        target_date = target_date or date.today()
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting daily signal run for {target_date}")
        
        results = {
            "date": target_date.isoformat(),
            "model_name": self.model_name,
            "dry_run": dry_run,
            "status": "started",
            "steps": {}
        }
        
        try:
            # Step 0: Get screener stats if using screener
            if self.use_screener:
                screener_stats = self._get_screener_stats()
                results["screener"] = screener_stats
                logger.info(f"Screener: {screener_stats.get('dividend_opportunities', 0)} dividend, "
                           f"{screener_stats.get('volatility_opportunities', 0)} volatility opportunities")
            
            # Step 1: Load data (filtered by screener if enabled)
            logger.info("Step 1: Loading price data...")
            prices, universe = self._load_data(target_date, tickers)
            results["steps"]["load_data"] = {
                "status": "success",
                "tickers_loaded": len(universe),
                "source": "screener" if self.use_screener else "default"
            }
            
            # Step 2: Compute features
            logger.info("Step 2: Computing features...")
            features = self._compute_features(universe, prices, target_date)
            results["steps"]["compute_features"] = {
                "status": "success",
                "features_computed": len(features)
            }
            
            # Step 3: Score stocks
            logger.info("Step 3: Scoring stocks...")
            scorer_output = self._score_stocks(universe, features, target_date)
            results["steps"]["score_stocks"] = {
                "status": "success",
                "stocks_scored": len(scorer_output.scores)
            }
            
            # Step 4: Generate signals
            logger.info("Step 4: Generating signals...")
            current_holdings = self._get_current_holdings()
            signals = self.signal_generator.generate_signals(
                scorer_output=scorer_output,
                current_holdings=current_holdings,
                capital=self.initial_capital,
                prices=prices
            )
            
            # Get opportunity summary
            summary = self.signal_generator.get_opportunity_summary(scorer_output)
            results["steps"]["generate_signals"] = {
                "status": "success",
                "total_scores": summary["total_scores"],
                "qualified_opportunities": summary["qualified_opportunities"],
                "signals_generated": len(signals),
                "recommendation": summary["recommendation"]
            }
            
            # Log signal details
            if signals:
                logger.info(f"Generated {len(signals)} signals:")
                for s in signals:
                    logger.info(f"  {s.signal_type.value} {s.ticker}: "
                               f"EV={s.ev:.4f}, Score={s.score:.1f}, "
                               f"Weight={s.suggested_weight:.2%}")
            else:
                logger.info("No signals generated - no opportunities met quality thresholds")
            
            # Step 5: Execute signals
            logger.info("Step 5: Executing signals...")
            execution_result = self._execute_signals(signals, dry_run)
            results["steps"]["execute_signals"] = execution_result
            
            results["status"] = "completed"
            logger.info(f"Daily run completed: {execution_result.get('executed', 0)} trades executed")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _get_screener_stats(self) -> Dict:
        """Get statistics from stock screener."""
        try:
            from src.integrations import get_screener_stats
            return get_screener_stats()
        except Exception as e:
            logger.warning(f"Could not get screener stats: {e}")
            return {"error": str(e)}
    
    def _load_data(
        self,
        target_date: date,
        tickers: Optional[List[str]] = None
    ) -> tuple:
        """Load price data from financial_data_aggregator."""
        # First, determine the stock universe
        if self.use_screener:
            universe = self._get_screener_universe()
            if not universe:
                logger.warning("Screener returned no tickers, falling back to default")
                universe = tickers or self._get_default_universe()
        else:
            universe = tickers or self._get_default_universe()
        
        try:
            from src.data_loaders.data_loader import DataLoader
            loader = DataLoader()
            
            # Load prices for each ticker
            prices = {}
            for ticker in universe:
                try:
                    price_data = loader.load_stock_prices(
                        ticker,
                        start=(target_date - timedelta(days=365)).isoformat(),
                        end=target_date.isoformat()
                    )
                    if price_data is not None and not price_data.empty:
                        # Get latest price
                        latest = price_data.iloc[-1]
                        prices[ticker] = float(latest.get('close', latest.get('Close', 0)))
                except Exception as e:
                    logger.warning(f"Failed to load price for {ticker}: {e}")
            
            return prices, list(prices.keys())
            
        except ImportError:
            logger.warning("DataLoader not available, using mock data")
            return self._get_mock_prices(tickers), universe
    
    def _get_screener_universe(self) -> List[str]:
        """Get stock universe from screener based on configured strategies."""
        try:
            from src.integrations import ScreenerClient
            client = ScreenerClient()
            
            tickers = set()
            if "dividend" in self.screener_strategies:
                dividend_opps = client.get_dividend_opportunities(limit=50)
                tickers.update(opp["ticker"] for opp in dividend_opps)
            
            if "volatility" in self.screener_strategies:
                vol_opps = client.get_volatility_opportunities(limit=50)
                tickers.update(opp["ticker"] for opp in vol_opps)
            
            logger.info(f"Screener returned {len(tickers)} unique tickers")
            return list(tickers)
            
        except Exception as e:
            logger.warning(f"Failed to get screener universe: {e}")
            return []
    
    def _compute_features(
        self,
        universe: List[str],
        prices: Dict[str, float],
        target_date: date
    ) -> Dict[str, Dict]:
        """Compute features for all stocks."""
        features = {}
        
        try:
            from src.feature_engineering.feature_aggregator import FeatureAggregator
            aggregator = FeatureAggregator()
            
            # Batch compute features
            for ticker in universe:
                try:
                    ticker_features = aggregator.compute_all_features(
                        ticker=ticker,
                        date=target_date
                    )
                    if ticker_features:
                        features[ticker] = ticker_features
                except Exception as e:
                    logger.warning(f"Failed to compute features for {ticker}: {e}")
                    
        except ImportError:
            logger.warning("FeatureAggregator not available, using mock features")
            features = {ticker: {"mock": True} for ticker in universe}
        
        return features
    
    def _score_stocks(
        self,
        universe: List[str],
        features: Dict[str, Dict],
        target_date: date
    ) -> ScorerOutput:
        """
        Score all stocks using the configured model.
        
        Note: This uses placeholder scores until real models are implemented.
        """
        scores = []
        
        # Try to load real scorer
        try:
            if self.model_name == "linear":
                from src.scorers.linear_scorer import LinearScorer
                scorer = LinearScorer()
            elif self.model_name == "cnn":
                from src.scorers.cnn_scorer import CNNScorer
                scorer = CNNScorer()
            elif self.model_name == "xgboost":
                from src.scorers.xgboost_scorer import XGBoostScorer
                scorer = XGBoostScorer()
            elif self.model_name == "llm":
                from src.scorers.llm_scorer import LLMScorer
                scorer = LLMScorer()
            else:
                raise ImportError(f"Unknown model: {self.model_name}")
            
            # Score each stock
            for ticker in universe:
                if ticker in features:
                    score = scorer.score(ticker, target_date, features[ticker])
                    if score.validate():
                        scores.append(score)
                        
        except ImportError:
            logger.warning(f"Scorer for {self.model_name} not implemented, using placeholder scores")
            scores = self._generate_placeholder_scores(universe, target_date)
        
        return ScorerOutput(
            date=target_date,
            model_name=self.model_name,
            scores=scores
        )
    
    def _generate_placeholder_scores(
        self,
        universe: List[str],
        target_date: date
    ) -> List[ScoreResult]:
        """
        Generate placeholder scores for testing.
        
        In production, these would come from trained models.
        This creates a mix of scores to test the signal generator.
        """
        import random
        random.seed(42)  # Reproducible for testing
        
        scores = []
        for ticker in universe:
            # Generate varied scores to test quality gates
            p_win = random.uniform(0.45, 0.70)
            avg_win = random.uniform(0.01, 0.04)
            avg_loss = random.uniform(0.008, 0.025)
            ev = (p_win * avg_win) - ((1 - p_win) * avg_loss)
            confidence = random.uniform(0.4, 0.85)
            
            # Normalize score based on EV
            if ev > 0.02:
                score = random.uniform(75, 95)
            elif ev > 0.005:
                score = random.uniform(55, 75)
            elif ev > 0:
                score = random.uniform(40, 55)
            else:
                score = random.uniform(20, 40)
            
            scores.append(ScoreResult(
                ticker=ticker,
                date=target_date,
                model_name=self.model_name,
                score=score,
                p_win=p_win,
                avg_win=avg_win,
                avg_loss=avg_loss,
                ev=ev,
                confidence=confidence,
                data_points=random.randint(30, 200)
            ))
        
        return scores
    
    def _get_current_holdings(self) -> List[str]:
        """Get current holdings from Trading_Simulator."""
        try:
            from src.trading import create_client, _HAS_TRADING_CLIENT
            
            if not _HAS_TRADING_CLIENT:
                logger.debug("TradingClient not available, assuming no holdings")
                return []
            
            client = create_client(self.trading_api_url)
            if not client.health_check():
                logger.warning("Trading API not available, assuming no holdings")
                return []
            
            # Get holdings for this model's portfolio
            portfolios = client.list_portfolios()
            for p in portfolios:
                if p.get("model_name") == self.model_name:
                    return client.get_current_holdings_tickers(p["id"])
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to get holdings: {e}")
            return []
    
    def _execute_signals(
        self,
        signals: List,
        dry_run: bool
    ) -> Dict:
        """Execute signals via Trading_Simulator API."""
        if not signals:
            return {
                "status": "skipped",
                "reason": "no_signals",
                "executed": 0,
                "failed": 0
            }
        
        try:
            from src.trading import create_client, _HAS_TRADING_CLIENT
            
            if not _HAS_TRADING_CLIENT:
                logger.warning("TradingClient not available (install requests)")
                return {
                    "status": "skipped",
                    "reason": "client_unavailable",
                    "executed": 0,
                    "failed": 0,
                    "signals_would_execute": [s.to_dict() for s in signals]
                }
            
            client = create_client(self.trading_api_url)
            
            # Check API health
            if not client.health_check():
                logger.warning("Trading API not available")
                return {
                    "status": "skipped",
                    "reason": "api_unavailable",
                    "executed": 0,
                    "failed": 0
                }
            
            # Get or create portfolio for this model
            portfolio_id = client.get_or_create_model_portfolio(
                self.model_name,
                self.initial_capital
            )
            
            if portfolio_id is None:
                return {
                    "status": "failed",
                    "reason": "portfolio_creation_failed",
                    "executed": 0,
                    "failed": 0
                }
            
            # Execute signals
            result = client.execute_signals(
                signals=signals,
                portfolio_id=portfolio_id,
                dry_run=dry_run
            )
            
            result["status"] = "success"
            return result
            
        except ImportError:
            logger.warning("TradingClient not available (install requests)")
            return {
                "status": "skipped",
                "reason": "client_unavailable",
                "executed": 0,
                "failed": 0,
                "signals_would_execute": [s.to_dict() for s in signals]
            }
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {
                "status": "failed",
                "reason": str(e),
                "executed": 0,
                "failed": len(signals)
            }
    
    def _get_default_universe(self) -> List[str]:
        """Default stock universe for testing."""
        return [
            # Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            # Finance
            "JPM", "BAC", "WFC", "GS", "MS",
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV",
            # Consumer
            "WMT", "PG", "KO", "PEP",
            # Industrial
            "CAT", "BA", "GE", "MMM"
        ]
    
    def _get_mock_prices(self, tickers: Optional[List[str]] = None) -> Dict[str, float]:
        """Mock prices for testing when database unavailable."""
        universe = tickers or self._get_default_universe()
        # Realistic-ish mock prices
        mock_prices = {
            "AAPL": 185.0, "MSFT": 420.0, "GOOGL": 175.0, "AMZN": 185.0,
            "META": 500.0, "NVDA": 880.0, "TSLA": 175.0, "JPM": 195.0,
            "BAC": 35.0, "WFC": 55.0, "GS": 380.0, "MS": 95.0,
            "JNJ": 155.0, "PFE": 27.0, "UNH": 520.0, "ABBV": 175.0,
            "WMT": 165.0, "PG": 165.0, "KO": 62.0, "PEP": 175.0,
            "CAT": 340.0, "BA": 185.0, "GE": 165.0, "MMM": 95.0
        }
        return {t: mock_prices.get(t, 100.0) for t in universe}


def main():
    parser = argparse.ArgumentParser(
        description="Run daily signal generation and execution pipeline"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["linear", "cnn", "xgboost", "llm"],
        default="xgboost",
        help="Model to use for scoring (default: xgboost)"
    )
    parser.add_argument(
        "--profile", "-p",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Signal generator profile (default: moderate)"
    )
    parser.add_argument(
        "--execute", "-e",
        action="store_true",
        help="Execute trades (default is dry run)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8001",
        help="Trading Simulator API URL"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital for new portfolios"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD), default is today"
    )
    parser.add_argument(
        "--use-screener", "-s",
        action="store_true",
        help="Use stock-screener to filter universe"
    )
    parser.add_argument(
        "--screener-strategy",
        choices=["dividend", "volatility", "both"],
        default="both",
        help="Screener strategy to use (default: both)"
    )
    
    args = parser.parse_args()
    
    # Parse date if provided
    target_date = None
    if args.date:
        target_date = date.fromisoformat(args.date)
    
    # Determine screener strategies
    if args.screener_strategy == "both":
        screener_strategies = ["dividend", "volatility"]
    else:
        screener_strategies = [args.screener_strategy]
    
    # Initialize and run
    orchestrator = DailySignalOrchestrator(
        model_name=args.model,
        signal_profile=args.profile,
        trading_api_url=args.api_url,
        initial_capital=args.capital,
        use_screener=args.use_screener,
        screener_strategies=screener_strategies
    )
    
    dry_run = not args.execute
    results = orchestrator.run(target_date=target_date, dry_run=dry_run)
    
    # Print summary
    print("\n" + "="*60)
    print("DAILY SIGNAL RUN SUMMARY")
    print("="*60)
    print(f"Date:     {results['date']}")
    print(f"Model:    {results['model_name']}")
    print(f"Status:   {results['status']}")
    print(f"Dry Run:  {results['dry_run']}")
    
    if "screener" in results and "error" not in results["screener"]:
        print(f"\nScreener:")
        print(f"  Dividend opportunities: {results['screener'].get('dividend_opportunities', 0)}")
        print(f"  Volatility opportunities: {results['screener'].get('volatility_opportunities', 0)}")
    
    if "steps" in results:
        print("\nSteps:")
        for step, data in results["steps"].items():
            status = data.get("status", "unknown")
            print(f"  {step}: {status}")
            
            if step == "generate_signals":
                print(f"    - Qualified opportunities: {data.get('qualified_opportunities', 0)}")
                print(f"    - Signals generated: {data.get('signals_generated', 0)}")
                print(f"    - Recommendation: {data.get('recommendation', 'N/A')}")
            
            if step == "execute_signals":
                print(f"    - Executed: {data.get('executed', 0)}")
                print(f"    - Failed: {data.get('failed', 0)}")
    
    if "error" in results:
        print(f"\nError: {results['error']}")
    
    print("="*60)
    
    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

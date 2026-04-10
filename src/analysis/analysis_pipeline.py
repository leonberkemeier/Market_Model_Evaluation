"""
Analysis Pipeline Orchestrator

Coordinates the complete workflow:
1. Markov Chain → Detect current market regime
2. Monte Carlo → Compute risk metrics for all assets
3. LLM → Filter and score asset recommendations
4. Portfolio Constructor → Build optimal allocation per risk profile
5. Export → Send to Trading Simulator

Usage:
    pipeline = AnalysisPipeline(database_url, markov_model_path)
    portfolio = pipeline.run(risk_profile=RiskProfileType.MODERATE, budget=100000)
"""

from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from loguru import logger
import pandas as pd
import json

from ..regime.markov_chain_detector import MarkovChainRegimeDetector, MarkovRegimeState
from ..risk.enhanced_monte_carlo import MonteCarloSimulator, MonteCarloMetrics
from ..portfolio.risk_profiles import (
    RiskProfileType, RiskProfileRegistry, LLMPortfolioConstructor,
    PortfolioAllocation
)
from ..data.data_loader import DataLoader
from ..advisory.llm_asset_selector import LLMAssetSelector
from ..integrations.trading_simulator_client import TradingSimulatorClient


@dataclass
class PipelineExecutionResult:
    """Complete result from pipeline execution."""
    execution_id: str
    execution_date: datetime
    
    # Stage results
    markov_regime_state: MarkovRegimeState
    monte_carlo_results: Dict[str, MonteCarloMetrics]
    llm_recommendations: Dict[str, float]
    portfolio_allocation: PortfolioAllocation
    
    # Metadata
    budget: float
    risk_profile: RiskProfileType
    current_regime: str
    assets_analyzed: int
    assets_recommended: int
    
    # Audit trail
    execution_log: List[str] = field(default_factory=list)


class AnalysisPipeline:
    """
    Complete analysis pipeline from data to portfolio.
    
    Process:
    1. Load data from aggregator
    2. Run Markov chain regime detection
    3. Run Monte Carlo simulations (regime-aware)
    4. Query LLM for asset filtering
    5. Construct portfolio per risk profile
    6. Validate and export
    """
    
    def __init__(
        self,
        database_url: str,
        markov_model_path: Optional[Path] = None,
        llm_host: str = "http://localhost:11434",
        trading_sim_url: str = "http://localhost:8001",
        n_mc_simulations: int = 10000,
        mc_horizon_days: int = 20,
    ):
        """
        Initialize analysis pipeline.
        
        Args:
            database_url: SQLite database URL from financial_data_aggregator
            markov_model_path: Path to saved Markov model
            llm_host: Ollama LLM endpoint
            trading_sim_url: Trading Simulator backend URL
            n_mc_simulations: Number of Monte Carlo paths
            mc_horizon_days: Monte Carlo forward-looking period (days)
        """
        self.database_url = database_url
        self.llm_host = llm_host
        self.trading_sim_url = trading_sim_url
        
        self.logger = logger.bind(module="analysis_pipeline")
        
        # Initialize components
        self.data_loader = DataLoader(database_url)
        
        self.markov_detector = MarkovChainRegimeDetector(
            n_states=5,
            model_path=markov_model_path
        )
        
        self.monte_carlo = MonteCarloSimulator(
            n_simulations=n_mc_simulations,
            horizon_days=mc_horizon_days,
            markov_detector=self.markov_detector
        )
        
        self.llm_selector = LLMAssetSelector(llm_host=llm_host)
        
        self.portfolio_constructor = LLMPortfolioConstructor()
        
        self.trading_sim_client = TradingSimulatorClient(
            base_url=trading_sim_url
        )
        
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Pipeline initialized (ID: {self.execution_id})")
    
    def run(
        self,
        risk_profile: RiskProfileType,
        budget: float,
        stock_universe: Optional[List[str]] = None,
        force_retrain_markov: bool = False,
        send_to_simulator: bool = True,
    ) -> PipelineExecutionResult:
        """
        Run complete analysis pipeline.
        
        Args:
            risk_profile: Risk profile type
            budget: Portfolio budget ($)
            stock_universe: List of tickers to analyze (default: all)
            force_retrain_markov: Force retraining Markov model
            send_to_simulator: Send result to Trading Simulator
            
        Returns:
            PipelineExecutionResult with complete analysis
        """
        execution_log = []
        
        try:
            execution_log.append(f"Pipeline starting: risk_profile={risk_profile}, budget=${budget:.0f}")
            
            # PHASE 1: Load Data
            self.logger.info("PHASE 1: Loading data...")
            market_prices = self._load_market_data(stock_universe)
            execution_log.append(f"Loaded prices for {len(market_prices)} assets")
            
            # PHASE 2: Markov Chain Regime Detection
            self.logger.info("PHASE 2: Running Markov chain regime detection...")
            markov_regime = self._run_markov_chain(market_prices, force_retrain_markov)
            execution_log.append(f"Detected regime: {markov_regime.current_regime}")
            
            # PHASE 3: Monte Carlo Simulations
            self.logger.info("PHASE 3: Running Monte Carlo simulations...")
            mc_results = self._run_monte_carlo(market_prices, markov_regime)
            execution_log.append(f"Simulated {len(mc_results)} assets")
            
            # PHASE 4: LLM Asset Filtering
            self.logger.info("PHASE 4: Querying LLM for asset recommendations...")
            llm_recommendations = self._query_llm(mc_results, markov_regime, risk_profile)
            execution_log.append(f"LLM recommended {len(llm_recommendations)} assets")
            
            # PHASE 5: Portfolio Construction
            self.logger.info("PHASE 5: Constructing portfolio...")
            asset_metadata = self._build_asset_metadata(market_prices)
            portfolio = self._construct_portfolio(
                budget, risk_profile, mc_results, llm_recommendations,
                markov_regime, asset_metadata
            )
            execution_log.append(f"Portfolio constructed with {len(portfolio.positions)} positions")
            
            # PHASE 6: Validation
            self.logger.info("PHASE 6: Validating portfolio...")
            is_valid, violations = portfolio.validate_vs_profile()
            if is_valid:
                execution_log.append("✓ Portfolio validation passed")
            else:
                execution_log.append(f"⚠ Portfolio validation issues: {violations}")
            
            # PHASE 7: Send to Simulator (Optional)
            if send_to_simulator:
                self.logger.info("PHASE 7: Sending portfolio to Trading Simulator...")
                self._send_to_simulator(portfolio)
                execution_log.append("✓ Portfolio sent to Trading Simulator")
            
            # Create result
            result = PipelineExecutionResult(
                execution_id=self.execution_id,
                execution_date=datetime.now(),
                markov_regime_state=markov_regime,
                monte_carlo_results=mc_results,
                llm_recommendations=llm_recommendations,
                portfolio_allocation=portfolio,
                budget=budget,
                risk_profile=risk_profile,
                current_regime=markov_regime.current_regime,
                assets_analyzed=len(market_prices),
                assets_recommended=len(llm_recommendations),
                execution_log=execution_log,
            )
            
            self.logger.info(f"✓ Pipeline completed successfully")
            return result
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            execution_log.append(f"ERROR: {str(e)}")
            raise
    
    def _load_market_data(self, stock_universe: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """
        Load market data for all assets.
        
        Args:
            stock_universe: List of tickers (None = load all)
            
        Returns:
            Dict mapping ticker → price series
        """
        if stock_universe is None:
            # Load default universe from config
            from ..config.config import ALL_TICKERS, TRAINING_START_DATE, BACKTEST_END_DATE
            stock_universe = ALL_TICKERS
            start_date = TRAINING_START_DATE
            end_date = BACKTEST_END_DATE
        else:
            start_date = pd.Timestamp.now() - pd.Timedelta(days=365)
            end_date = pd.Timestamp.now()
        
        market_prices = {}
        
        for ticker in stock_universe:
            try:
                prices = self.data_loader.load_stock_prices(
                    ticker, start_date.date() if hasattr(start_date, 'date') else start_date,
                    end_date.date() if hasattr(end_date, 'date') else end_date
                )
                if not prices.empty and len(prices) >= 100:
                    market_prices[ticker] = prices['close']
            except Exception as e:
                self.logger.warning(f"Failed to load {ticker}: {e}")
        
        self.logger.info(f"Loaded prices for {len(market_prices)} assets")
        return market_prices
    
    def _run_markov_chain(
        self,
        market_prices: Dict[str, pd.Series],
        force_retrain: bool = False,
    ) -> MarkovRegimeState:
        """
        Run Markov chain regime detection.
        
        Args:
            market_prices: Dict of ticker → prices
            force_retrain: Force retraining model
            
        Returns:
            Current market regime state
        """
        # Use benchmark (broad market index) for regime detection
        benchmark_ticker = "SPY"  # S&P 500
        if benchmark_ticker not in market_prices:
            # Fallback: use first available ticker
            benchmark_ticker = list(market_prices.keys())[0]
        
        benchmark_prices = market_prices[benchmark_ticker]
        
        # Try to load model
        if not force_retrain and self.markov_detector.load_model():
            self.logger.info("Loaded pre-trained Markov model")
        else:
            self.logger.info("Training new Markov model...")
            self.markov_detector.fit(benchmark_prices)
        
        # Detect current regime
        regime_state = self.markov_detector.detect_current_regime(benchmark_prices)
        
        return regime_state
    
    def _run_monte_carlo(
        self,
        market_prices: Dict[str, pd.Series],
        markov_regime: MarkovRegimeState,
    ) -> Dict[str, MonteCarloMetrics]:
        """
        Run Monte Carlo simulations for all assets.
        
        Args:
            market_prices: Dict of ticker → prices
            markov_regime: Current regime
            
        Returns:
            Dict mapping ticker → MonteCarloMetrics
        """
        mc_results = {}
        
        for ticker, prices in market_prices.items():
            try:
                metrics = self.monte_carlo.simulate_asset(
                    prices, ticker, markov_regime
                )
                mc_results[ticker] = metrics
            except Exception as e:
                self.logger.warning(f"MC simulation failed for {ticker}: {e}")
        
        self.logger.info(f"Completed MC simulations for {len(mc_results)} assets")
        return mc_results
    
    def _query_llm(
        self,
        mc_results: Dict[str, MonteCarloMetrics],
        markov_regime: MarkovRegimeState,
        risk_profile: RiskProfileType,
    ) -> Dict[str, float]:
        """
        Query LLM for asset recommendations.
        
        Args:
            mc_results: Monte Carlo results
            markov_regime: Current regime
            risk_profile: Risk profile
            
        Returns:
            Dict mapping ticker → recommendation score (0-1)
        """
        try:
            recommendations = self.llm_selector.get_recommendations(
                mc_results=mc_results,
                current_regime=markov_regime,
                risk_profile=risk_profile,
            )
            return recommendations
        except Exception as e:
            self.logger.warning(f"LLM query failed: {e}. Using MC scores as fallback.")
            # Fallback: use MC metrics to score
            fallback_scores = {}
            for ticker, mc in mc_results.items():
                # Simple scoring: higher mean return + lower ES
                score = (0.5 + mc.mean_return * 10) / (1 + abs(mc.es_95))
                fallback_scores[ticker] = min(1.0, max(0.0, score))
            return fallback_scores
    
    def _build_asset_metadata(self, market_prices: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """Build metadata for each asset."""
        metadata = {}
        
        for ticker in market_prices.keys():
            try:
                company = self.data_loader.load_company_metadata(ticker)
                metadata[ticker] = {
                    "type": self._infer_asset_type(ticker),
                    "sector": company.get("sector", ""),
                    "industry": company.get("industry", ""),
                    "country": company.get("country", ""),
                }
            except Exception as e:
                self.logger.debug(f"Failed to load metadata for {ticker}: {e}")
                metadata[ticker] = {"type": self._infer_asset_type(ticker)}
        
        return metadata
    
    def _infer_asset_type(self, ticker: str) -> str:
        """Infer asset type from ticker."""
        if ticker in ["BTC", "ETH", "ADA"]:
            return "crypto"
        elif ticker.startswith("^"):
            return "bond"
        elif ticker in ["GLD", "OIL", "CORN"]:
            return "commodity"
        else:
            return "stock"
    
    def _construct_portfolio(
        self,
        budget: float,
        risk_profile: RiskProfileType,
        mc_results: Dict[str, MonteCarloMetrics],
        llm_recommendations: Dict[str, float],
        markov_regime: MarkovRegimeState,
        asset_metadata: Dict[str, Dict],
    ) -> PortfolioAllocation:
        """Construct portfolio."""
        portfolio = self.portfolio_constructor.construct_portfolio(
            budget=budget,
            risk_profile=risk_profile,
            mc_results=mc_results,
            llm_recommendations=llm_recommendations,
            current_regime=markov_regime,
            asset_metadata=asset_metadata,
        )
        return portfolio
    
    def _send_to_simulator(self, portfolio: PortfolioAllocation) -> None:
        """Send portfolio to Trading Simulator."""
        try:
            self.trading_sim_client.create_portfolio(portfolio)
            self.logger.info("✓ Portfolio sent to simulator")
        except Exception as e:
            self.logger.warning(f"Failed to send to simulator: {e}")
    
    def save_results(self, result: PipelineExecutionResult, output_dir: Path) -> None:
        """
        Save pipeline results to disk.
        
        Args:
            result: Pipeline execution result
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save execution metadata
        metadata = {
            "execution_id": result.execution_id,
            "execution_date": result.execution_date.isoformat(),
            "risk_profile": str(result.risk_profile),
            "current_regime": result.current_regime,
            "budget": result.budget,
            "assets_analyzed": result.assets_analyzed,
            "assets_recommended": result.assets_recommended,
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save execution log
        with open(output_dir / "execution_log.txt", "w") as f:
            f.write("\n".join(result.execution_log))
        
        # Save portfolio
        portfolio_dict = asdict(result.portfolio_allocation)
        with open(output_dir / "portfolio.json", "w") as f:
            json.dump(portfolio_dict, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_dir}")

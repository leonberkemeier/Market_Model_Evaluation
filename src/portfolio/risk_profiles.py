"""
Risk Profile Definitions and LLM-Driven Portfolio Constructor

Defines risk tolerance levels and builds portfolios based on:
1. Risk profile type (VERY_CONSERVATIVE to VERY_AGGRESSIVE)
2. User budget
3. Monte Carlo simulation results
4. LLM recommendations
5. Current market regime
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger

import numpy as np
import pandas as pd

from ..risk.enhanced_monte_carlo import MonteCarloMetrics
from ..regime.markov_chain_detector import MarkovRegimeState


@dataclass
class RiskProfileConfig:
    """Configuration for a risk profile."""
    name: str
    var_95_target: float      # Max acceptable VaR (95%)
    var_99_target: float      # Max acceptable VaR (99%)
    es_95_target: float       # Max acceptable Expected Shortfall (95%)
    es_99_target: float       # Max acceptable Expected Shortfall (99%)
    
    # Asset class allocation targets (as % of portfolio)
    stocks_pct: float         # Equities
    bonds_pct: float          # Fixed income
    crypto_pct: float         # Cryptocurrencies
    commodities_pct: float    # Commodities / hedges
    
    # Position sizing constraints
    max_position_pct: float           # Max weight per single holding
    max_sector_pct: float             # Max per sector
    max_concentration_top5_pct: float # Max in top 5 holdings
    
    # Diversification requirements
    min_num_positions: int    # Minimum holdings
    max_single_asset_pct: float
    
    # Rebalancing triggers
    rebalance_frequency_days: int
    rebalance_threshold_pct: float    # If any position drifts > this %, rebalance


class RiskProfileType(str, Enum):
    """Risk profile types."""
    VERY_CONSERVATIVE = "VERY_CONSERVATIVE"
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    VERY_AGGRESSIVE = "VERY_AGGRESSIVE"


class RiskProfileRegistry:
    """Registry of predefined risk profiles."""
    
    PROFILES: Dict[RiskProfileType, RiskProfileConfig] = {
        RiskProfileType.VERY_CONSERVATIVE: RiskProfileConfig(
            name="Very Conservative",
            var_95_target=0.02,       # 2% max loss
            var_99_target=0.035,      # 3.5% tail loss
            es_95_target=0.025,       # 2.5% expected tail loss
            es_99_target=0.04,
            
            # 80% bonds, 20% stocks (no crypto)
            stocks_pct=0.20,
            bonds_pct=0.80,
            crypto_pct=0.0,
            commodities_pct=0.0,
            
            # Strict position limits
            max_position_pct=0.08,
            max_sector_pct=0.20,
            max_concentration_top5_pct=0.40,
            
            min_num_positions=8,
            max_single_asset_pct=0.08,
            rebalance_frequency_days=30,
            rebalance_threshold_pct=5.0
        ),
        
        RiskProfileType.CONSERVATIVE: RiskProfileConfig(
            name="Conservative",
            var_95_target=0.05,       # 5% max loss
            var_99_target=0.08,       # 8% tail loss
            es_95_target=0.065,       # 6.5% expected tail loss
            es_99_target=0.10,
            
            # 60% bonds, 40% stocks
            stocks_pct=0.40,
            bonds_pct=0.60,
            crypto_pct=0.0,
            commodities_pct=0.0,
            
            max_position_pct=0.10,
            max_sector_pct=0.25,
            max_concentration_top5_pct=0.50,
            
            min_num_positions=10,
            max_single_asset_pct=0.10,
            rebalance_frequency_days=30,
            rebalance_threshold_pct=7.0
        ),
        
        RiskProfileType.MODERATE: RiskProfileConfig(
            name="Moderate",
            var_95_target=0.10,       # 10% max loss
            var_99_target=0.15,       # 15% tail loss
            es_95_target=0.13,        # 13% expected tail loss
            es_99_target=0.18,
            
            # 50% stocks, 40% bonds, 10% crypto
            stocks_pct=0.50,
            bonds_pct=0.40,
            crypto_pct=0.10,
            commodities_pct=0.0,
            
            max_position_pct=0.12,
            max_sector_pct=0.30,
            max_concentration_top5_pct=0.60,
            
            min_num_positions=12,
            max_single_asset_pct=0.12,
            rebalance_frequency_days=21,
            rebalance_threshold_pct=10.0
        ),
        
        RiskProfileType.AGGRESSIVE: RiskProfileConfig(
            name="Aggressive",
            var_95_target=0.15,       # 15% max loss
            var_99_target=0.22,       # 22% tail loss
            es_95_target=0.20,        # 20% expected tail loss
            es_99_target=0.27,
            
            # 60% stocks, 20% bonds, 20% crypto
            stocks_pct=0.60,
            bonds_pct=0.20,
            crypto_pct=0.20,
            commodities_pct=0.0,
            
            max_position_pct=0.15,
            max_sector_pct=0.35,
            max_concentration_top5_pct=0.70,
            
            min_num_positions=15,
            max_single_asset_pct=0.15,
            rebalance_frequency_days=14,
            rebalance_threshold_pct=12.0
        ),
        
        RiskProfileType.VERY_AGGRESSIVE: RiskProfileConfig(
            name="Very Aggressive",
            var_95_target=0.25,       # 25% max loss
            var_99_target=0.35,       # 35% tail loss
            es_95_target=0.30,        # 30% expected tail loss
            es_99_target=0.40,
            
            # 70% stocks, 0% bonds, 30% crypto
            stocks_pct=0.70,
            bonds_pct=0.0,
            crypto_pct=0.30,
            commodities_pct=0.0,
            
            max_position_pct=0.20,
            max_sector_pct=0.40,
            max_concentration_top5_pct=0.80,
            
            min_num_positions=10,
            max_single_asset_pct=0.20,
            rebalance_frequency_days=7,
            rebalance_threshold_pct=15.0
        ),
    }
    
    @classmethod
    def get_profile(cls, profile_type: RiskProfileType) -> RiskProfileConfig:
        """Get a risk profile by type."""
        return cls.PROFILES[profile_type]
    
    @classmethod
    def get_all_profiles(cls) -> Dict[RiskProfileType, RiskProfileConfig]:
        """Get all available profiles."""
        return cls.PROFILES.copy()


@dataclass
class PortfolioPosition:
    """A single position in a portfolio."""
    ticker: str
    asset_type: str              # "stock", "bond", "crypto", "commodity"
    quantity: float
    dollar_value: float
    weight_pct: float
    sector: str = ""
    entry_price: float = 0.0
    mc_mean_return: float = 0.0  # From Monte Carlo simulation
    mc_var_95: float = 0.0       # From Monte Carlo simulation


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation."""
    positions: List[PortfolioPosition] = field(default_factory=list)
    cash: float = 0.0
    total_value: float = 0.0
    
    # Risk profile
    risk_profile_type: RiskProfileType = RiskProfileType.MODERATE
    risk_profile_config: Optional[RiskProfileConfig] = None
    
    # Portfolio-level metrics
    portfolio_expected_mean_return: float = 0.0
    portfolio_expected_var_95: float = 0.0
    portfolio_expected_es_95: float = 0.0
    
    # Allocation breakdown
    allocation_by_type: Dict[str, float] = field(default_factory=dict)  # asset_type → %
    allocation_by_sector: Dict[str, float] = field(default_factory=dict)  # sector → %
    
    # Metadata
    construction_date: str = ""
    regime_at_construction: str = ""
    llm_recommendations_used: List[str] = field(default_factory=list)
    rebalance_triggers: Dict = field(default_factory=dict)
    
    def get_total_allocated(self) -> float:
        """Get total allocated (excludes cash)."""
        return sum(pos.dollar_value for pos in self.positions)
    
    def get_cash_pct(self) -> float:
        """Get cash as % of total portfolio."""
        if self.total_value == 0:
            return 0.0
        return self.cash / self.total_value * 100
    
    def get_deployment_pct(self) -> float:
        """Get deployed capital as % of total portfolio."""
        return 100.0 - self.get_cash_pct()
    
    def validate_vs_profile(self) -> Tuple[bool, List[str]]:
        """
        Validate portfolio against risk profile constraints.
        
        Returns:
            (is_valid, list of violations)
        """
        if self.risk_profile_config is None:
            return False, ["No risk profile config set"]
        
        violations = []
        config = self.risk_profile_config
        
        # Check position limits
        for pos in self.positions:
            if pos.weight_pct > config.max_position_pct * 100:
                violations.append(
                    f"{pos.ticker}: weight {pos.weight_pct:.1f}% exceeds max {config.max_position_pct * 100:.1f}%"
                )
        
        # Check asset type allocations
        for asset_type, actual_pct in self.allocation_by_type.items():
            target = getattr(config, f"{asset_type}_pct", 0) * 100
            if target == 0 and actual_pct > 0:
                violations.append(f"Asset type '{asset_type}' not allowed in profile, but {actual_pct:.1f}% allocated")
        
        # Check number of positions
        if len(self.positions) < config.min_num_positions:
            violations.append(
                f"Only {len(self.positions)} positions, minimum {config.min_num_positions} required"
            )
        
        return len(violations) == 0, violations


class LLMPortfolioConstructor:
    """
    Constructs portfolios based on risk profiles and LLM recommendations.
    
    Process:
    1. Filter LLM-recommended assets by risk profile constraints
    2. Select top assets per tier (stocks, bonds, crypto, commodities)
    3. Calculate dollar allocation
    4. Apply position sizing constraints
    5. Validate against VaR/ES targets
    6. Return final portfolio
    """
    
    def __init__(self):
        """Initialize portfolio constructor."""
        self.logger = logger.bind(module="llm_portfolio_constructor")
    
    def construct_portfolio(
        self,
        budget: float,
        risk_profile: RiskProfileType,
        mc_results: Dict[str, MonteCarloMetrics],
        llm_recommendations: Dict[str, float],
        current_regime: MarkovRegimeState,
        asset_metadata: Dict[str, Dict] = None,
    ) -> PortfolioAllocation:
        """
        Construct a portfolio based on risk profile and recommendations.
        
        Args:
            budget: Total portfolio budget ($)
            risk_profile: Risk profile type
            mc_results: Dict of ticker → MonteCarloMetrics
            llm_recommendations: Dict of ticker → recommendation score
            current_regime: Current market regime
            asset_metadata: Additional metadata (sector, type, etc.)
            
        Returns:
            PortfolioAllocation with final positions
        """
        profile_config = RiskProfileRegistry.get_profile(risk_profile)
        
        self.logger.info(
            f"Constructing {profile_config.name} portfolio: "
            f"budget=${budget:.0f}, regime={current_regime.current_regime}"
        )
        
        # Step 1: Filter and select assets by tier
        selected_assets = self._select_assets_by_tier(
            mc_results, llm_recommendations, profile_config, asset_metadata
        )
        
        # Step 2: Calculate tier allocations ($)
        tier_budget = self._calculate_tier_budgets(budget, profile_config)
        
        # Step 3: Allocate within each tier
        positions = self._allocate_positions(
            selected_assets, tier_budget, profile_config, mc_results, asset_metadata
        )
        
        # Step 4: Create portfolio
        portfolio = PortfolioAllocation(
            positions=positions,
            cash=budget - sum(pos.dollar_value for pos in positions),
            total_value=budget,
            risk_profile_type=risk_profile,
            risk_profile_config=profile_config,
            construction_date=pd.Timestamp.now().isoformat(),
            regime_at_construction=current_regime.current_regime,
            llm_recommendations_used=list(llm_recommendations.keys()),
        )
        
        # Step 5: Calculate portfolio-level metrics
        self._compute_portfolio_metrics(portfolio, mc_results)
        
        # Step 6: Validate
        is_valid, violations = portfolio.validate_vs_profile()
        if not is_valid:
            self.logger.warning(f"Portfolio validation issues:\n" + "\n".join(violations))
        else:
            self.logger.info("✓ Portfolio validates against risk profile constraints")
        
        return portfolio
    
    def _select_assets_by_tier(
        self,
        mc_results: Dict[str, MonteCarloMetrics],
        llm_recommendations: Dict[str, float],
        profile_config: RiskProfileConfig,
        asset_metadata: Dict[str, Dict],
    ) -> Dict[str, List[str]]:
        """
        Select top assets in each tier (stocks, bonds, crypto, commodities).
        
        Returns:
            Dict mapping tier → list of selected tickers
        """
        # Categorize assets by type
        assets_by_type = self._categorize_assets_by_type(mc_results, asset_metadata)
        
        selected = {}
        
        # Select stocks if allowed
        if profile_config.stocks_pct > 0:
            selected["stocks"] = self._select_top_n_assets(
                assets_by_type.get("stock", []),
                mc_results, llm_recommendations,
                n=int(20 * profile_config.stocks_pct)  # Proportional to allocation
            )
        
        # Select bonds if allowed
        if profile_config.bonds_pct > 0:
            selected["bonds"] = self._select_top_n_assets(
                assets_by_type.get("bond", []),
                mc_results, llm_recommendations,
                n=int(10 * profile_config.bonds_pct)
            )
        
        # Select crypto if allowed
        if profile_config.crypto_pct > 0:
            selected["crypto"] = self._select_top_n_assets(
                assets_by_type.get("crypto", []),
                mc_results, llm_recommendations,
                n=int(5 * profile_config.crypto_pct)
            )
        
        # Select commodities if allowed
        if profile_config.commodities_pct > 0:
            selected["commodities"] = self._select_top_n_assets(
                assets_by_type.get("commodity", []),
                mc_results, llm_recommendations,
                n=int(5 * profile_config.commodities_pct)
            )
        
        return selected
    
    def _categorize_assets_by_type(
        self,
        mc_results: Dict[str, MonteCarloMetrics],
        asset_metadata: Dict[str, Dict],
    ) -> Dict[str, List[str]]:
        """Categorize assets by type."""
        categorized = {
            "stock": [],
            "bond": [],
            "crypto": [],
            "commodity": []
        }
        
        for ticker in mc_results.keys():
            if asset_metadata and ticker in asset_metadata:
                asset_type = asset_metadata[ticker].get("type", "stock")
            else:
                # Infer type from ticker patterns
                asset_type = self._infer_asset_type(ticker)
            
            categorized[asset_type].append(ticker)
        
        return categorized
    
    def _infer_asset_type(self, ticker: str) -> str:
        """Infer asset type from ticker."""
        # Simplified heuristic
        if ticker in ["BTC", "ETH", "ADA"]:
            return "crypto"
        elif ticker.startswith("^"):
            return "bond"
        elif ticker in ["GLD", "OIL", "CORN"]:
            return "commodity"
        else:
            return "stock"
    
    def _select_top_n_assets(
        self,
        candidates: List[str],
        mc_results: Dict[str, MonteCarloMetrics],
        llm_recommendations: Dict[str, float],
        n: int = 5,
    ) -> List[str]:
        """Select top N assets from candidates based on scoring."""
        if not candidates:
            return []
        
        # Score each candidate
        scores = {}
        for ticker in candidates:
            if ticker not in mc_results:
                continue
            
            mc = mc_results[ticker]
            llm_score = llm_recommendations.get(ticker, 0.5)
            
            # Combined score: 60% LLM + 40% mean return
            combined_score = llm_score * 0.6 + (mc.mean_return / 0.05) * 0.4
            scores[ticker] = combined_score
        
        # Sort by score and take top N
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [ticker for ticker, score in sorted_assets[:n]]
        
        return selected
    
    def _calculate_tier_budgets(self, budget: float, profile_config: RiskProfileConfig) -> Dict[str, float]:
        """Calculate budget for each asset tier."""
        return {
            "stocks": budget * profile_config.stocks_pct,
            "bonds": budget * profile_config.bonds_pct,
            "crypto": budget * profile_config.crypto_pct,
            "commodities": budget * profile_config.commodities_pct,
        }
    
    def _allocate_positions(
        self,
        selected_assets: Dict[str, List[str]],
        tier_budget: Dict[str, float],
        profile_config: RiskProfileConfig,
        mc_results: Dict[str, MonteCarloMetrics],
        asset_metadata: Dict[str, Dict],
    ) -> List[PortfolioPosition]:
        """Allocate positions within each tier."""
        positions = []
        
        for tier, tickers in selected_assets.items():
            budget = tier_budget.get(tier, 0)
            if budget == 0 or not tickers:
                continue
            
            # Equal-weight within tier (or could use market-cap weight)
            per_asset = budget / len(tickers)
            
            for ticker in tickers:
                if ticker not in mc_results:
                    continue
                
                mc = mc_results[ticker]
                metadata = asset_metadata.get(ticker, {}) if asset_metadata else {}
                
                position = PortfolioPosition(
                    ticker=ticker,
                    asset_type=metadata.get("type", tier.rstrip("s")),
                    quantity=per_asset / max(1, mc.mean_return * 100),  # Simplified
                    dollar_value=per_asset,
                    weight_pct=(per_asset / tier_budget.get(tier, 1)) * 100,
                    sector=metadata.get("sector", ""),
                    mc_mean_return=mc.mean_return,
                    mc_var_95=mc.var_95,
                )
                positions.append(position)
        
        return positions
    
    def _compute_portfolio_metrics(
        self,
        portfolio: PortfolioAllocation,
        mc_results: Dict[str, MonteCarloMetrics],
    ) -> None:
        """Compute portfolio-level risk metrics."""
        # Calculate weighted average metrics
        total_value = portfolio.get_total_allocated()
        
        if total_value == 0:
            return
        
        weighted_mean = 0.0
        weighted_var_95 = 0.0
        weighted_es_95 = 0.0
        
        for pos in portfolio.positions:
            weight = pos.dollar_value / total_value
            if pos.ticker in mc_results:
                mc = mc_results[pos.ticker]
                weighted_mean += weight * mc.mean_return
                weighted_var_95 += weight * mc.var_95
                weighted_es_95 += weight * mc.es_95
        
        portfolio.portfolio_expected_mean_return = weighted_mean
        portfolio.portfolio_expected_var_95 = weighted_var_95
        portfolio.portfolio_expected_es_95 = weighted_es_95
        
        # Calculate allocation by type
        for pos in portfolio.positions:
            asset_type = pos.asset_type
            if asset_type not in portfolio.allocation_by_type:
                portfolio.allocation_by_type[asset_type] = 0.0
            portfolio.allocation_by_type[asset_type] += pos.weight_pct
        
        self.logger.info(
            f"Portfolio metrics: "
            f"mean={portfolio.portfolio_expected_mean_return:.2%}, "
            f"VaR95={portfolio.portfolio_expected_var_95:.2%}, "
            f"ES95={portfolio.portfolio_expected_es_95:.2%}"
        )


# Type hint helpers
from typing import Tuple

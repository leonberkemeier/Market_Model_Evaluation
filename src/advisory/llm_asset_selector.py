"""
LLM-based Asset Selector

Uses Ollama or remote LLM to filter assets and make intelligent recommendations
based on Monte Carlo simulation results and current market regime.
"""

from typing import Dict, List, Optional
from loguru import logger
import requests
import json

from ..regime.markov_chain_detector import MarkovRegimeState
from ..portfolio.risk_profiles import RiskProfileType, RiskProfileRegistry
from ..risk.enhanced_monte_carlo import MonteCarloMetrics


class LLMAssetSelector:
    """Selects and filters assets using LLM intelligence."""
    
    def __init__(self, llm_host: str = "http://localhost:11434"):
        """
        Initialize LLM asset selector.
        
        Args:
            llm_host: Ollama host URL
        """
        self.llm_host = llm_host
        self.logger = logger.bind(module="llm_asset_selector")
    
    def get_recommendations(
        self,
        mc_results: Dict[str, MonteCarloMetrics],
        current_regime: MarkovRegimeState,
        risk_profile: RiskProfileType,
        top_k: int = 20,
    ) -> Dict[str, float]:
        """
        Get LLM-based asset recommendations.
        
        Args:
            mc_results: Monte Carlo results for all assets
            current_regime: Current market regime
            risk_profile: Risk profile type
            top_k: Number of top recommendations to return
            
        Returns:
            Dict mapping ticker → recommendation score (0-1)
        """
        try:
            # Build prompt
            prompt = self._build_prompt(mc_results, current_regime, risk_profile)
            
            # Query LLM
            llm_response = self._query_llm(prompt)
            
            # Parse response
            recommendations = self._parse_response(llm_response, len(mc_results), top_k)
            
            return recommendations
        
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}. Returning empty recommendations.")
            return {}
    
    def _build_prompt(
        self,
        mc_results: Dict[str, MonteCarloMetrics],
        current_regime: MarkovRegimeState,
        risk_profile: RiskProfileType,
    ) -> str:
        """Build prompt for LLM."""
        profile_config = RiskProfileRegistry.get_profile(risk_profile)
        
        # Summarize Monte Carlo results
        assets_summary = []
        for ticker, mc in list(mc_results.items())[:30]:  # Top 30
            assets_summary.append(
                f"- {ticker}: mean={mc.mean_return:.2%}, "
                f"VaR95={mc.var_95:.2%}, ES95={mc.es_95:.2%}, "
                f"regime_suitability={mc.regime_suitability.get(current_regime.current_regime, 0.5):.2f}"
            )
        
        prompt = f"""
You are an intelligent portfolio manager analyzing assets for investment.

CURRENT MARKET CONDITIONS:
- Regime: {current_regime.current_regime}
- Regime confidence: {current_regime.regime_probability:.2%}
- Time in regime: {current_regime.time_in_regime.days} days
- Regime features: {current_regime.regime_features}

INVESTOR PROFILE:
- Risk profile: {risk_profile.value}
- Max acceptable VaR(95%): {profile_config.var_95_target:.2%}
- Max acceptable ES(95%): {profile_config.es_95_target:.2%}
- Target stocks: {profile_config.stocks_pct:.0%}
- Target bonds: {profile_config.bonds_pct:.0%}
- Target crypto: {profile_config.crypto_pct:.0%}

AVAILABLE ASSETS (Monte Carlo Simulation Results):
{chr(10).join(assets_summary)}

TASK:
1. Identify which assets best fit the current regime and risk profile
2. Consider how each asset performs in the {current_regime.current_regime} regime
3. Ensure recommended assets fit the risk constraints
4. Score each recommended asset from 0.0 (bad fit) to 1.0 (excellent fit)

RESPOND WITH JSON:
{{
    "recommended_assets": ["TICKER1", "TICKER2", ...],
    "scores": {{"TICKER1": 0.85, "TICKER2": 0.72, ...}},
    "rationale": "Brief explanation of selections"
}}
"""
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query LLM."""
        try:
            response = requests.post(
                f"{self.llm_host}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise
    
    def _parse_response(
        self,
        response: str,
        total_assets: int,
        top_k: int,
    ) -> Dict[str, float]:
        """Parse LLM response."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
            
            parsed = json.loads(json_str)
            scores = parsed.get("scores", {})
            
            # Return top K
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_scores[:top_k])
        
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Return uniform scores as fallback
            return {f"ASSET_{i}": 0.5 for i in range(min(top_k, total_assets))}

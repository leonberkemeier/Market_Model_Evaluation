"""
Macro/Inter-Market Features (Cluster C from Roadmap)

Sector-specific macro indicators:
- Yield Curve (Finance): 10Y-2Y spread
- DXY (Commodities/Crypto): US Dollar Index
- Copper/Gold Ratio (Cyclicals): Economic growth indicator
- Real Interest Rates (Tech): Inflation-adjusted rates
- VIX (All): Volatility index
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class MacroFeatures:
    """
    Inter-market and macro features that provide context for sector models.
    
    These are "satellite" features that feed into sector experts and help
    the Managing Model understand broader market conditions.
    """
    
    def __init__(self):
        """Initialize macro features calculator."""
        self.logger = logger.bind(module="macro_features")
    
    def calculate_yield_curve_spread(
        self,
        ten_year_yield: pd.Series,
        two_year_yield: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate Yield Curve Spread (10Y - 2Y).
        
        Critical for Finance sector expert:
        - Positive spread: Normal (banks profitable)
        - Inverted spread (negative): Recession signal (banks struggle)
        - Steepening: Economic expansion
        - Flattening: Economic slowdown
        
        Args:
            ten_year_yield: Series of 10-year treasury yields
            two_year_yield: Series of 2-year treasury yields
            
        Returns:
            Dict with yield_spread, spread_trend, inversion_signal
        """
        if ten_year_yield.empty or two_year_yield.empty:
            return {
                "yield_spread": np.nan,
                "spread_trend": np.nan,
                "inversion_signal": 0.0
            }
        
        try:
            # Calculate spread
            spread = ten_year_yield - two_year_yield
            
            current_spread = float(spread.iloc[-1])
            
            # Spread trend (30-day change)
            if len(spread) >= 30:
                spread_trend = float(spread.iloc[-1] - spread.iloc[-30])
            else:
                spread_trend = np.nan
            
            # Inversion signal
            if current_spread < 0:
                inversion_signal = -1.0  # Inverted - recession warning
            elif current_spread < 0.25:
                inversion_signal = -0.5  # Flattening - caution
            elif current_spread > 1.5:
                inversion_signal = 1.0  # Very steep - strong growth
            else:
                inversion_signal = 0.0  # Normal
            
            return {
                "yield_spread": current_spread,
                "spread_trend": spread_trend,
                "inversion_signal": inversion_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating yield curve spread: {e}")
            return {
                "yield_spread": np.nan,
                "spread_trend": np.nan,
                "inversion_signal": 0.0
            }
    
    def calculate_dxy_features(
        self,
        dxy: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate DXY (US Dollar Index) features.
        
        Essential for Commodities and Crypto (priced in USD):
        - Rising DXY: Bearish for commodities/crypto
        - Falling DXY: Bullish for commodities/crypto
        
        Args:
            dxy: Series of DXY values
            window: Rolling window for calculations (default: 20)
            
        Returns:
            Dict with dxy_level, dxy_trend, dxy_momentum, dollar_strength
        """
        if dxy.empty or len(dxy) < window:
            return {
                "dxy_level": np.nan,
                "dxy_trend": np.nan,
                "dxy_momentum": np.nan,
                "dollar_strength": 0.0
            }
        
        try:
            current_dxy = float(dxy.iloc[-1])
            
            # DXY trend (30-day change)
            if len(dxy) >= 30:
                dxy_30d_ago = dxy.iloc[-30]
                dxy_trend = float((current_dxy - dxy_30d_ago) / dxy_30d_ago)
            else:
                dxy_trend = np.nan
            
            # DXY momentum (rate of change)
            dxy_returns = dxy.pct_change()
            dxy_momentum = float(dxy_returns.rolling(window=window).mean().iloc[-1])
            
            # Dollar strength signal
            if dxy_trend > 0.05:  # Strong uptrend
                dollar_strength = 1.0  # Bearish for commodities/crypto
            elif dxy_trend < -0.05:  # Strong downtrend
                dollar_strength = -1.0  # Bullish for commodities/crypto
            else:
                dollar_strength = 0.0  # Neutral
            
            return {
                "dxy_level": current_dxy,
                "dxy_trend": dxy_trend,
                "dxy_momentum": dxy_momentum,
                "dollar_strength": dollar_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating DXY features: {e}")
            return {
                "dxy_level": np.nan,
                "dxy_trend": np.nan,
                "dxy_momentum": np.nan,
                "dollar_strength": 0.0
            }
    
    def calculate_copper_gold_ratio(
        self,
        copper_prices: pd.Series,
        gold_prices: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate Copper-to-Gold Ratio.
        
        Famous leading indicator for Cyclicals/Industrial sector:
        - Rising ratio: Economic expansion (industrial demand up)
        - Falling ratio: Economic contraction (flight to safety)
        
        Copper = "Dr. Copper" (has PhD in economics)
        Gold = Safe haven
        
        Args:
            copper_prices: Series of copper prices
            gold_prices: Series of gold prices
            
        Returns:
            Dict with copper_gold_ratio, ratio_trend, economic_signal
        """
        if copper_prices.empty or gold_prices.empty:
            return {
                "copper_gold_ratio": np.nan,
                "ratio_trend": np.nan,
                "economic_signal": 0.0
            }
        
        try:
            # Calculate ratio
            ratio = copper_prices / gold_prices
            
            current_ratio = float(ratio.iloc[-1])
            
            # Ratio trend (60-day change, as this is a slow indicator)
            if len(ratio) >= 60:
                ratio_60d_ago = ratio.iloc[-60]
                ratio_trend = float((current_ratio - ratio_60d_ago) / ratio_60d_ago)
            else:
                ratio_trend = np.nan
            
            # Economic signal
            if ratio_trend > 0.10:  # Strong rise
                economic_signal = 1.0  # Economic expansion
            elif ratio_trend < -0.10:  # Strong fall
                economic_signal = -1.0  # Economic contraction
            else:
                economic_signal = 0.0  # Neutral
            
            return {
                "copper_gold_ratio": current_ratio,
                "ratio_trend": ratio_trend,
                "economic_signal": economic_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating copper-gold ratio: {e}")
            return {
                "copper_gold_ratio": np.nan,
                "ratio_trend": np.nan,
                "economic_signal": 0.0
            }
    
    def calculate_real_rates(
        self,
        nominal_rate: pd.Series,
        inflation_rate: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate Real Interest Rates (Nominal - Inflation).
        
        Critical for Tech sector expert:
        - High real rates: Bearish for growth stocks (discount future cash flows more)
        - Low/negative real rates: Bullish for growth stocks
        
        Args:
            nominal_rate: Series of nominal interest rates (e.g., 10Y yield)
            inflation_rate: Series of inflation rates (e.g., CPI YoY)
            
        Returns:
            Dict with real_rate, real_rate_trend, tech_valuation_pressure
        """
        if nominal_rate.empty or inflation_rate.empty:
            return {
                "real_rate": np.nan,
                "real_rate_trend": np.nan,
                "tech_valuation_pressure": 0.0
            }
        
        try:
            # Calculate real rate
            real_rate = nominal_rate - inflation_rate
            
            current_real_rate = float(real_rate.iloc[-1])
            
            # Real rate trend (30-day change)
            if len(real_rate) >= 30:
                real_rate_30d_ago = real_rate.iloc[-30]
                real_rate_trend = float(current_real_rate - real_rate_30d_ago)
            else:
                real_rate_trend = np.nan
            
            # Tech valuation pressure
            if current_real_rate > 2.0:  # High real rates
                tech_valuation_pressure = 1.0  # Bearish for tech
            elif current_real_rate < 0.0:  # Negative real rates
                tech_valuation_pressure = -1.0  # Bullish for tech
            else:
                tech_valuation_pressure = 0.0  # Neutral
            
            return {
                "real_rate": current_real_rate,
                "real_rate_trend": real_rate_trend,
                "tech_valuation_pressure": tech_valuation_pressure
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating real rates: {e}")
            return {
                "real_rate": np.nan,
                "real_rate_trend": np.nan,
                "tech_valuation_pressure": 0.0
            }
    
    def calculate_vix_features(
        self,
        vix: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate VIX (Volatility Index) features.
        
        VIX is the "fear gauge" - critical for all sectors:
        - VIX < 15: Complacency
        - VIX 15-25: Normal
        - VIX > 25: Fear/uncertainty
        - VIX > 40: Panic
        
        Args:
            vix: Series of VIX values
            window: Rolling window (default: 20)
            
        Returns:
            Dict with vix_level, vix_percentile, fear_signal, vix_spike
        """
        if vix.empty or len(vix) < window:
            return {
                "vix_level": np.nan,
                "vix_percentile": np.nan,
                "fear_signal": 0.0,
                "vix_spike": False
            }
        
        try:
            current_vix = float(vix.iloc[-1])
            
            # VIX percentile (where is current VIX in historical distribution?)
            vix_percentile = float(vix.rank(pct=True).iloc[-1])
            
            # Fear signal
            if current_vix < 15:
                fear_signal = -1.0  # Complacency - potential reversal
            elif current_vix > 30:
                fear_signal = 1.0  # High fear - reduce exposure
            else:
                fear_signal = 0.0  # Normal
            
            # VIX spike detection (>20% increase in 5 days)
            if len(vix) >= 5:
                vix_5d_ago = vix.iloc[-5]
                vix_change = (current_vix - vix_5d_ago) / vix_5d_ago
                vix_spike = vix_change > 0.20
            else:
                vix_spike = False
            
            return {
                "vix_level": current_vix,
                "vix_percentile": vix_percentile,
                "fear_signal": fear_signal,
                "vix_spike": vix_spike
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VIX features: {e}")
            return {
                "vix_level": np.nan,
                "vix_percentile": np.nan,
                "fear_signal": 0.0,
                "vix_spike": False
            }
    
    def calculate_oil_features(
        self,
        oil_prices: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate Oil Price features.
        
        Important for Cyclicals and Energy sectors:
        - Rising oil: Inflation pressure, good for energy
        - Falling oil: Deflation signal, bad for energy
        
        Args:
            oil_prices: Series of crude oil prices (e.g., WTI)
            window: Rolling window (default: 20)
            
        Returns:
            Dict with oil_price, oil_momentum, oil_volatility, energy_signal
        """
        if oil_prices.empty or len(oil_prices) < window:
            return {
                "oil_price": np.nan,
                "oil_momentum": np.nan,
                "oil_volatility": np.nan,
                "energy_signal": 0.0
            }
        
        try:
            current_oil = float(oil_prices.iloc[-1])
            
            # Oil momentum
            oil_returns = oil_prices.pct_change()
            oil_momentum = float(oil_returns.rolling(window=window).mean().iloc[-1])
            
            # Oil volatility
            oil_volatility = float(oil_returns.rolling(window=window).std().iloc[-1])
            
            # Energy signal
            if oil_momentum > 0.01:  # Strong uptrend
                energy_signal = 1.0  # Bullish for energy sector
            elif oil_momentum < -0.01:  # Strong downtrend
                energy_signal = -1.0  # Bearish for energy sector
            else:
                energy_signal = 0.0  # Neutral
            
            return {
                "oil_price": current_oil,
                "oil_momentum": oil_momentum,
                "oil_volatility": oil_volatility,
                "energy_signal": energy_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating oil features: {e}")
            return {
                "oil_price": np.nan,
                "oil_momentum": np.nan,
                "oil_volatility": np.nan,
                "energy_signal": 0.0
            }


# Example usage
if __name__ == "__main__":
    # Generate sample macro data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    # Sample data
    ten_year_yield = pd.Series(3.5 + np.cumsum(np.random.normal(0, 0.05, 300)), index=dates)
    two_year_yield = pd.Series(3.8 + np.cumsum(np.random.normal(0, 0.05, 300)), index=dates)
    dxy = pd.Series(100 + np.cumsum(np.random.normal(0, 0.5, 300)), index=dates)
    copper = pd.Series(4.0 + np.cumsum(np.random.normal(0, 0.1, 300)), index=dates)
    gold = pd.Series(1900 + np.cumsum(np.random.normal(0, 10, 300)), index=dates)
    inflation = pd.Series(3.0 + np.random.normal(0, 0.2, 300), index=dates)
    vix = pd.Series(np.abs(18 + np.cumsum(np.random.normal(0, 2, 300))), index=dates)
    oil = pd.Series(75 + np.cumsum(np.random.normal(0, 3, 300)), index=dates)
    
    calc = MacroFeatures()
    
    # Test individual features
    print("=== Yield Curve Spread ===")
    print(calc.calculate_yield_curve_spread(ten_year_yield, two_year_yield))
    
    print("\n=== DXY Features ===")
    print(calc.calculate_dxy_features(dxy))
    
    print("\n=== Copper/Gold Ratio ===")
    print(calc.calculate_copper_gold_ratio(copper, gold))
    
    print("\n=== Real Interest Rates ===")
    print(calc.calculate_real_rates(ten_year_yield, inflation))
    
    print("\n=== VIX Features ===")
    print(calc.calculate_vix_features(vix))
    
    print("\n=== Oil Features ===")
    print(calc.calculate_oil_features(oil))

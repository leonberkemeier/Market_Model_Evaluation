"""
Microstructure Features (Cluster B from Roadmap)

Features that capture the relationship between volume and price:
- VPIN (Volume-Synchronized Probability of Informed Trading): Toxic order flow detection
- Amihud Illiquidity: Price impact measurement
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class MicrostructureFeatures:
    """
    Order flow and market microstructure features.
    
    These features go beyond simple OHLCV and look at HOW volume and price interact.
    Critical for understanding liquidity risk and informed trading.
    """
    
    def __init__(self):
        """Initialize microstructure features calculator."""
        self.logger = logger.bind(module="microstructure_features")
    
    def calculate_vpin(
        self,
        prices: pd.DataFrame,
        volumes: pd.Series,
        bucket_size: Optional[int] = None,
        window: int = 50
    ) -> Dict[str, float]:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
        
        VPIN measures "toxic" order flow - when market makers are being "run over"
        by informed traders. High VPIN = High risk of adverse selection.
        
        This is a sophisticated measure that tells you:
        - When smart money is moving aggressively
        - When liquidity is actually an illusion
        - When you should widen your Kelly bet or stay out
        
        Args:
            prices: DataFrame with open, close columns
            volumes: Series of volumes
            bucket_size: Volume bucket size (default: mean volume * window)
            window: Number of buckets for VPIN calculation (default: 50)
            
        Returns:
            Dict with vpin, vpin_trend, toxic_flow_signal
        """
        if prices.empty or volumes.empty or len(prices) < window:
            return {
                "vpin": np.nan,
                "vpin_trend": np.nan,
                "toxic_flow_signal": np.nan
            }
        
        try:
            # Extract prices
            if isinstance(prices, pd.DataFrame):
                open_prices = prices['open']
                close_prices = prices['close']
            else:
                # If just a series, use as close and lag for open
                close_prices = prices
                open_prices = prices.shift(1)
            
            # Classify volume as buy or sell based on price movement
            # If close > open: buy volume, else: sell volume
            buy_volume = np.where(close_prices > open_prices, volumes, 0)
            sell_volume = np.where(close_prices <= open_prices, volumes, 0)
            
            # Order imbalance per bar
            order_imbalance = np.abs(buy_volume - sell_volume)
            
            # Set bucket size if not provided
            if bucket_size is None:
                bucket_size = int(volumes.mean() * window)
            
            # Create volume buckets
            cumulative_volume = volumes.cumsum()
            bucket_indices = (cumulative_volume // bucket_size).astype(int)
            
            # Aggregate order imbalance by bucket
            df_temp = pd.DataFrame({
                'bucket': bucket_indices,
                'order_imbalance': order_imbalance,
                'volume': volumes
            })
            
            bucket_aggregates = df_temp.groupby('bucket').agg({
                'order_imbalance': 'sum',
                'volume': 'sum'
            })
            
            # Calculate VPIN for each bucket
            if len(bucket_aggregates) < window:
                return {
                    "vpin": np.nan,
                    "vpin_trend": np.nan,
                    "toxic_flow_signal": np.nan
                }
            
            # VPIN = Average(|Buy - Sell| / Total Volume) over window
            bucket_aggregates['vpin'] = (
                bucket_aggregates['order_imbalance'] / 
                bucket_aggregates['volume'].rolling(window=window).sum()
            )
            
            # Current VPIN (latest bucket)
            current_vpin = float(bucket_aggregates['vpin'].iloc[-1])
            
            # VPIN trend (is it increasing?)
            if len(bucket_aggregates) >= window:
                recent_vpin = bucket_aggregates['vpin'].iloc[-window:].mean()
                older_vpin = bucket_aggregates['vpin'].iloc[-2*window:-window].mean()
                vpin_trend = float(recent_vpin - older_vpin)
            else:
                vpin_trend = np.nan
            
            # Toxic flow signal: 1 if VPIN > 75th percentile, -1 if < 25th percentile
            vpin_75 = bucket_aggregates['vpin'].quantile(0.75)
            vpin_25 = bucket_aggregates['vpin'].quantile(0.25)
            
            if current_vpin > vpin_75:
                toxic_flow_signal = 1.0  # High toxic flow - be cautious
            elif current_vpin < vpin_25:
                toxic_flow_signal = -1.0  # Low toxic flow - liquidity is real
            else:
                toxic_flow_signal = 0.0  # Normal
            
            return {
                "vpin": current_vpin,
                "vpin_trend": vpin_trend,
                "toxic_flow_signal": toxic_flow_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VPIN: {e}")
            return {
                "vpin": np.nan,
                "vpin_trend": np.nan,
                "toxic_flow_signal": np.nan
            }
    
    def calculate_amihud_illiquidity(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate Amihud Illiquidity Ratio.
        
        Formula: |Return| / Volume
        
        This tells you "how much price moves per unit of volume traded."
        High illiquidity = High risk for Kelly Criterion because:
        - Your trade will move the price against you
        - Exit costs are high
        - Should use smaller position sizes
        
        Args:
            prices: Series of closing prices
            volumes: Series of volumes
            window: Rolling window for averaging (default: 20)
            
        Returns:
            Dict with amihud_illiquidity, illiquidity_trend, liquidity_risk
        """
        if prices.empty or volumes.empty or len(prices) < window + 1:
            return {
                "amihud_illiquidity": np.nan,
                "illiquidity_trend": np.nan,
                "liquidity_risk": np.nan
            }
        
        try:
            # Calculate returns
            returns = prices.pct_change().abs()
            
            # Amihud ratio: |Return| / Volume
            # Multiply by 1e6 to scale (standard in literature)
            amihud_ratio = (returns / volumes) * 1e6
            
            # Rolling average
            amihud_rolling = amihud_ratio.rolling(window=window).mean()
            
            # Current illiquidity
            current_illiquidity = float(amihud_rolling.iloc[-1])
            
            # Illiquidity trend (recent vs older)
            if len(amihud_rolling) >= 2 * window:
                recent_illiquidity = amihud_rolling.iloc[-window:].mean()
                older_illiquidity = amihud_rolling.iloc[-2*window:-window].mean()
                illiquidity_trend = float(recent_illiquidity - older_illiquidity)
            else:
                illiquidity_trend = np.nan
            
            # Liquidity risk: normalized position (0-1)
            # 0 = Very liquid, 1 = Very illiquid
            illiquidity_95 = amihud_rolling.quantile(0.95)
            illiquidity_5 = amihud_rolling.quantile(0.05)
            
            if illiquidity_95 > illiquidity_5:
                liquidity_risk = float(
                    (current_illiquidity - illiquidity_5) / 
                    (illiquidity_95 - illiquidity_5)
                )
                liquidity_risk = max(0, min(1, liquidity_risk))  # Clip to 0-1
            else:
                liquidity_risk = 0.5
            
            return {
                "amihud_illiquidity": current_illiquidity,
                "illiquidity_trend": illiquidity_trend,
                "liquidity_risk": liquidity_risk
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Amihud illiquidity: {e}")
            return {
                "amihud_illiquidity": np.nan,
                "illiquidity_trend": np.nan,
                "liquidity_risk": np.nan
            }
    
    def calculate_volume_price_correlation(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate rolling correlation between price changes and volume.
        
        Positive correlation: Volume confirms price moves (healthy trend)
        Negative correlation: Divergence (potential reversal)
        
        Args:
            prices: Series of closing prices
            volumes: Series of volumes
            window: Rolling window (default: 20)
            
        Returns:
            Dict with volume_price_corr, trend_confirmation
        """
        if prices.empty or volumes.empty or len(prices) < window + 1:
            return {
                "volume_price_corr": np.nan,
                "trend_confirmation": np.nan
            }
        
        try:
            # Calculate price changes
            price_changes = prices.pct_change()
            
            # Calculate volume changes
            volume_changes = volumes.pct_change()
            
            # Rolling correlation
            rolling_corr = price_changes.rolling(window=window).corr(volume_changes)
            
            current_corr = float(rolling_corr.iloc[-1])
            
            # Trend confirmation:
            # > 0.3: Strong confirmation
            # -0.3 to 0.3: Neutral
            # < -0.3: Divergence (warning)
            if current_corr > 0.3:
                trend_confirmation = 1.0
            elif current_corr < -0.3:
                trend_confirmation = -1.0
            else:
                trend_confirmation = 0.0
            
            return {
                "volume_price_corr": current_corr,
                "trend_confirmation": trend_confirmation
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume-price correlation: {e}")
            return {
                "volume_price_corr": np.nan,
                "trend_confirmation": np.nan
            }
    
    def calculate_effective_spread(
        self,
        prices: pd.DataFrame,
        volumes: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Estimate effective spread from high-low range weighted by volume.
        
        Effective spread = (High - Low) / Mid-price * 100
        
        High spread = High transaction costs
        
        Args:
            prices: DataFrame with high, low columns
            volumes: Series of volumes
            window: Rolling window (default: 20)
            
        Returns:
            Dict with effective_spread, spread_cost
        """
        if prices.empty or volumes.empty or len(prices) < window:
            return {
                "effective_spread": np.nan,
                "spread_cost": np.nan
            }
        
        try:
            high = prices['high'] if isinstance(prices, pd.DataFrame) else prices
            low = prices['low'] if isinstance(prices, pd.DataFrame) else prices
            
            # Mid-price
            mid_price = (high + low) / 2
            
            # Effective spread (as percentage)
            spread = ((high - low) / mid_price) * 100
            
            # Volume-weighted average spread
            volume_weighted_spread = (spread * volumes).rolling(window=window).sum() / volumes.rolling(window=window).sum()
            
            current_spread = float(volume_weighted_spread.iloc[-1])
            
            # Spread cost: How much above median?
            median_spread = volume_weighted_spread.median()
            spread_cost = float((current_spread - median_spread) / (median_spread + 1e-6))
            
            return {
                "effective_spread": current_spread,
                "spread_cost": spread_cost
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating effective spread: {e}")
            return {
                "effective_spread": np.nan,
                "spread_cost": np.nan
            }
    
    def calculate_all_microstructure_features(
        self,
        prices: pd.DataFrame,
        volumes: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate all microstructure features at once.
        
        Args:
            prices: DataFrame with open, high, low, close columns
            volumes: Series of volumes
            
        Returns:
            Dict with all microstructure features
        """
        features = {}
        
        # VPIN
        features.update(self.calculate_vpin(prices, volumes))
        
        # Amihud Illiquidity
        close_prices = prices['close'] if isinstance(prices, pd.DataFrame) else prices
        features.update(self.calculate_amihud_illiquidity(close_prices, volumes))
        
        # Volume-Price Correlation
        features.update(self.calculate_volume_price_correlation(close_prices, volumes))
        
        # Effective Spread
        features.update(self.calculate_effective_spread(prices, volumes))
        
        return features


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    # Create sample OHLCV data
    close_prices = 100 + np.cumsum(np.random.normal(0, 2, 300))
    open_prices = close_prices + np.random.normal(0, 0.5, 300)
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 1, 300))
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 1, 300))
    volumes = np.random.lognormal(15, 0.5, 300)
    
    prices_df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }, index=dates)
    
    volumes_series = pd.Series(volumes, index=dates)
    
    calc = MicrostructureFeatures()
    
    # Test individual features
    print("=== VPIN ===")
    print(calc.calculate_vpin(prices_df, volumes_series))
    
    print("\n=== Amihud Illiquidity ===")
    print(calc.calculate_amihud_illiquidity(prices_df['close'], volumes_series))
    
    print("\n=== Volume-Price Correlation ===")
    print(calc.calculate_volume_price_correlation(prices_df['close'], volumes_series))
    
    print("\n=== Effective Spread ===")
    print(calc.calculate_effective_spread(prices_df, volumes_series))
    
    print("\n=== All Microstructure Features ===")
    all_features = calc.calculate_all_microstructure_features(prices_df, volumes_series)
    for key, value in all_features.items():
        print(f"{key}: {value:.4f}" if not np.isnan(value) else f"{key}: NaN")

"""Advanced technical indicators for regime detection and trend analysis."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class AdvancedTechnicalFeatures:
    """Advanced technical indicators (intermediate complexity)."""
    
    def __init__(self):
        """Initialize advanced features calculator."""
        self.logger = logger.bind(module="advanced_technical_features")
    
    def calculate_adx(self, prices: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """
        Calculate Average Directional Index (ADX) and Directional Indicators.
        
        ADX measures trend strength (0-100):
        - <25: Weak trend
        - 25-50: Strong trend
        - >50: Very strong trend
        
        Args:
            prices: DataFrame with high, low, close columns
            period: ADX period (default: 14)
            
        Returns:
            Dict with adx_14, di_plus, di_minus
        """
        if prices.empty or len(prices) < period + 1:
            return {
                "adx_14": np.nan,
                "di_plus": np.nan,
                "di_minus": np.nan
            }
        
        try:
            high = prices['high'] if isinstance(prices, pd.DataFrame) else prices
            low = prices['low'] if isinstance(prices, pd.DataFrame) else prices
            close = prices['close'] if isinstance(prices, pd.DataFrame) else prices
            
            # Calculate true range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Directional movements
            up = high.diff()
            down = -low.diff()
            
            # Determine positive and negative directional movement
            pos_dm = np.where((up > down) & (up > 0), up, 0)
            neg_dm = np.where((down > up) & (down > 0), down, 0)
            
            # Smooth directional movements
            pos_dm_smooth = pd.Series(pos_dm).rolling(window=period).sum()
            neg_dm_smooth = pd.Series(neg_dm).rolling(window=period).sum()
            
            # Directional indicators
            di_plus = 100 * pos_dm_smooth / atr
            di_minus = 100 * neg_dm_smooth / atr
            
            # ADX calculation
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=period).mean()
            
            return {
                "adx_14": float(adx.iloc[-1]),
                "di_plus": float(di_plus.iloc[-1]),
                "di_minus": float(di_minus.iloc[-1])
            }
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return {
                "adx_14": np.nan,
                "di_plus": np.nan,
                "di_minus": np.nan
            }
    
    def calculate_donchian_channels(
        self,
        prices: pd.DataFrame,
        periods: list = None
    ) -> Dict[str, float]:
        """
        Calculate Donchian Channels (highest high and lowest low).
        
        Useful for breakout detection and support/resistance.
        
        Args:
            prices: DataFrame with high, low columns
            periods: List of periods (default: [20, 50, 200])
            
        Returns:
            Dict with donchian_{period}_high and _low for each period
        """
        if prices.empty:
            periods = periods or [20, 50, 200]
            return {f"donchian_{p}_high": np.nan for p in periods} | \
                   {f"donchian_{p}_low": np.nan for p in periods}
        
        if periods is None:
            periods = [20, 50, 200]
        
        try:
            high = prices['high'] if isinstance(prices, pd.DataFrame) else prices
            low = prices['low'] if isinstance(prices, pd.DataFrame) else prices
            
            features = {}
            
            for period in periods:
                if len(prices) >= period:
                    highest = high.rolling(window=period).max()
                    lowest = low.rolling(window=period).min()
                    
                    features[f"donchian_{period}_high"] = float(highest.iloc[-1])
                    features[f"donchian_{period}_low"] = float(lowest.iloc[-1])
                else:
                    features[f"donchian_{period}_high"] = np.nan
                    features[f"donchian_{period}_low"] = np.nan
            
            return features
        except Exception as e:
            self.logger.error(f"Error calculating Donchian: {e}")
            periods = periods or [20, 50, 200]
            return {f"donchian_{p}_high": np.nan for p in periods} | \
                   {f"donchian_{p}_low": np.nan for p in periods}
    
    def calculate_mfi(
        self,
        prices: pd.DataFrame,
        volumes: pd.Series,
        period: int = 14
    ) -> Dict[str, float]:
        """
        Calculate Money Flow Index (MFI) - Volume-weighted momentum.
        
        Similar to RSI but incorporates volume.
        MFI 0-100 scale:
        - <20: Oversold
        - >80: Overbought
        
        Args:
            prices: DataFrame with high, low, close columns
            volumes: Series of volumes
            period: MFI period (default: 14)
            
        Returns:
            Dict with mfi_14
        """
        if prices.empty or len(prices) < period + 1 or volumes.empty:
            return {"mfi_14": np.nan}
        
        try:
            high = prices['high'] if isinstance(prices, pd.DataFrame) else prices
            low = prices['low'] if isinstance(prices, pd.DataFrame) else prices
            close = prices['close'] if isinstance(prices, pd.DataFrame) else prices
            
            # Calculate typical price
            tp = (high + low + close) / 3
            
            # Money flow
            mf = tp * volumes
            
            # Positive and negative money flow
            positive_mf = np.where(tp > tp.shift(1), mf, 0)
            negative_mf = np.where(tp < tp.shift(1), mf, 0)
            
            # Sum over period
            positive_mf_sum = pd.Series(positive_mf).rolling(window=period).sum()
            negative_mf_sum = pd.Series(negative_mf).rolling(window=period).sum()
            
            # Money flow ratio
            mfr = positive_mf_sum / negative_mf_sum
            
            # MFI
            mfi = 100 - (100 / (1 + mfr))
            
            return {"mfi_14": float(mfi.iloc[-1])}
        except Exception as e:
            self.logger.error(f"Error calculating MFI: {e}")
            return {"mfi_14": np.nan}
    
    def calculate_obv(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        ema_period: int = 21
    ) -> Dict[str, float]:
        """
        Calculate On-Balance Volume (OBV) - Cumulative volume indicator.
        
        OBV increases with up days, decreases with down days.
        Useful for confirming trends and divergences.
        
        Args:
            prices: Series of closing prices
            volumes: Series of volumes
            ema_period: Period for EMA smoothing (default: 21)
            
        Returns:
            Dict with obv and obv_ema_21
        """
        if prices.empty or volumes.empty or len(prices) < ema_period:
            return {
                "obv": np.nan,
                "obv_ema_21": np.nan
            }
        
        try:
            # Calculate OBV
            obv_values = []
            obv = 0
            
            for i in range(len(prices)):
                if i == 0:
                    obv = volumes.iloc[i] if prices.iloc[i] > 0 else 0
                else:
                    if prices.iloc[i] > prices.iloc[i-1]:
                        obv += volumes.iloc[i]
                    elif prices.iloc[i] < prices.iloc[i-1]:
                        obv -= volumes.iloc[i]
                
                obv_values.append(obv)
            
            obv_series = pd.Series(obv_values)
            
            # EMA smoothing
            obv_ema = obv_series.ewm(span=ema_period).mean()
            
            return {
                "obv": float(obv_series.iloc[-1]),
                "obv_ema_21": float(obv_ema.iloc[-1])
            }
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return {
                "obv": np.nan,
                "obv_ema_21": np.nan
            }
    
    def calculate_stochastic(
        self,
        prices: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, float]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Measures where price closes relative to recent range.
        0-100 scale:
        - <20: Oversold
        - >80: Overbought
        
        Args:
            prices: DataFrame with high, low, close columns
            k_period: K period (default: 14)
            d_period: D period smoothing (default: 3)
            
        Returns:
            Dict with stochastic_k and stochastic_d
        """
        if prices.empty or len(prices) < k_period + d_period:
            return {
                "stochastic_k": np.nan,
                "stochastic_d": np.nan
            }
        
        try:
            high = prices['high'] if isinstance(prices, pd.DataFrame) else prices
            low = prices['low'] if isinstance(prices, pd.DataFrame) else prices
            close = prices['close'] if isinstance(prices, pd.DataFrame) else prices
            
            # Calculate K
            highest_high = high.rolling(window=k_period).max()
            lowest_low = low.rolling(window=k_period).min()
            
            k_numerator = close - lowest_low
            k_denominator = highest_high - lowest_low
            
            stochastic_k = 100 * (k_numerator / k_denominator)
            
            # Calculate D (SMA of K)
            stochastic_d = stochastic_k.rolling(window=d_period).mean()
            
            return {
                "stochastic_k": float(stochastic_k.iloc[-1]),
                "stochastic_d": float(stochastic_d.iloc[-1])
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return {
                "stochastic_k": np.nan,
                "stochastic_d": np.nan
            }
    
    def calculate_all_advanced_features(
        self,
        prices_data,
        volumes: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate all advanced technical features.
        
        Args:
            prices_data: DataFrame with OHLCV or dict with close prices
            volumes: Series of volumes (optional, extracted from DataFrame if available)
            
        Returns:
            Dict with all advanced features
        """
        # Handle both DataFrame and dict inputs
        if isinstance(prices_data, pd.DataFrame):
            if prices_data.empty:
                return {}
            prices_df = prices_data
            if volumes is None and 'volume' in prices_data.columns:
                volumes = prices_data['volume']
            close_prices = prices_data['close'] if 'close' in prices_data.columns else prices_data.iloc[:, -2]
        else:
            return {}
        
        features = {}
        
        # ADX
        features.update(self.calculate_adx(prices_df))
        
        # Donchian Channels
        features.update(self.calculate_donchian_channels(prices_df))
        
        # MFI (requires volume)
        if volumes is not None and len(volumes) > 0:
            features.update(self.calculate_mfi(prices_df, volumes))
        else:
            features["mfi_14"] = np.nan
        
        # OBV (requires volume)
        if volumes is not None and len(volumes) > 0:
            features.update(self.calculate_obv(close_prices, volumes))
        else:
            features["obv"] = np.nan
            features["obv_ema_21"] = np.nan
        
        # Stochastic
        features.update(self.calculate_stochastic(prices_df))
        
        # Ensure all values are scalar
        for key in features:
            val = features[key]
            if isinstance(val, (pd.Series, pd.Index, np.ndarray)):
                features[key] = float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
            elif not pd.isna(val):
                features[key] = float(val)
            else:
                features[key] = np.nan
        
        return features


if __name__ == "__main__":
    logger.add(lambda msg: print(msg, end=''))
    
    # Example usage
    calc = AdvancedTechnicalFeatures()
    
    # Create sample data
    import yfinance as yf
    data = yf.download("AAPL", period="1y", progress=False)
    
    features = calc.calculate_all_advanced_features(data)
    
    print("\n" + "=" * 80)
    print("ADVANCED TECHNICAL FEATURES FOR AAPL")
    print("=" * 80)
    
    for key, value in sorted(features.items()):
        if pd.notna(value):
            print(f"{key:30s}: {value:12.4f}")
        else:
            print(f"{key:30s}: NaN")

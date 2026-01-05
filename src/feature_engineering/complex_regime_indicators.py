"""Complex regime indicators for advanced market analysis."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats
from loguru import logger


class ComplexRegimeIndicators:
    """High-complexity regime detection and analysis indicators."""
    
    def __init__(self):
        """Initialize complex indicators calculator."""
        self.logger = logger.bind(module="complex_regime_indicators")
    
    def calculate_order_flow_imbalance(
        self,
        prices: pd.DataFrame,
        volumes: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate Order Flow Imbalance (OFI).
        
        Captures buy/sell pressure independent of price movement.
        
        Args:
            prices: DataFrame with open, close columns
            volumes: Series of volumes
            window: Rolling window for trend (default: 20)
            
        Returns:
            Dict with ofi (current) and ofi_trend (smoothed)
        """
        if prices.empty or volumes.empty or len(prices) < window:
            return {
                "ofi": np.nan,
                "ofi_trend": np.nan
            }
        
        try:
            open_prices = prices['open'] if isinstance(prices, pd.DataFrame) else prices
            close_prices = prices['close'] if isinstance(prices, pd.DataFrame) else prices
            
            # Buy volume: when close > open
            buy_volume = np.where(close_prices > open_prices, volumes, 0)
            
            # Sell volume: when close < open
            sell_volume = np.where(close_prices < open_prices, volumes, 0)
            
            # OFI: (Buy - Sell) / Total
            total_volume = buy_volume + sell_volume
            ofi = (buy_volume - sell_volume) / (total_volume + 1e-6)
            
            # Current OFI
            current_ofi = float(ofi[-1])
            
            # OFI trend (smoothed)
            ofi_trend = pd.Series(ofi).rolling(window=window).mean()
            current_ofi_trend = float(ofi_trend.iloc[-1])
            
            return {
                "ofi": current_ofi,
                "ofi_trend": current_ofi_trend
            }
        except Exception as e:
            self.logger.error(f"Error calculating OFI: {e}")
            return {
                "ofi": np.nan,
                "ofi_trend": np.nan
            }
    
    def calculate_volatility_regime(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate volatility regime (low/medium/high).
        
        3-state regime detection using quantile-based thresholds.
        
        Args:
            prices: Series of closing prices
            window: Rolling window for volatility (default: 20)
            
        Returns:
            Dict with vol_regime (0/1/2), confidence, mean_reversion_strength
        """
        if prices.empty or len(prices) < window:
            return {
                "vol_regime": np.nan,
                "vol_regime_confidence": np.nan,
                "mean_reversion_strength": np.nan
            }
        
        try:
            returns = prices.pct_change().dropna()
            rolling_vol = returns.rolling(window=window).std()
            
            # Define regime boundaries using quantiles
            vol_33 = rolling_vol.quantile(0.33)
            vol_67 = rolling_vol.quantile(0.67)
            
            current_vol = rolling_vol.iloc[-1]
            
            # Assign regime
            if current_vol < vol_33:
                regime = 0  # Low volatility
                distance = current_vol - 0  # Distance from boundary
                boundary_range = vol_33 - 0
            elif current_vol < vol_67:
                regime = 1  # Medium volatility
                distance = min(current_vol - vol_33, vol_67 - current_vol)
                boundary_range = (vol_67 - vol_33) / 2
            else:
                regime = 2  # High volatility
                distance = current_vol - vol_67
                boundary_range = rolling_vol.max() - vol_67
            
            # Confidence: how far from boundary (0-1)
            confidence = min(abs(distance) / (boundary_range + 1e-6), 1.0)
            
            # Mean reversion strength (inverse relationship with volatility)
            # High vol = weak mean reversion, Low vol = strong mean reversion
            mean_reversion = 1.0 - (current_vol / rolling_vol.max())
            
            return {
                "vol_regime": float(regime),
                "vol_regime_confidence": float(confidence),
                "mean_reversion_strength": float(mean_reversion)
            }
        except Exception as e:
            self.logger.error(f"Error calculating volatility regime: {e}")
            return {
                "vol_regime": np.nan,
                "vol_regime_confidence": np.nan,
                "mean_reversion_strength": np.nan
            }
    
    def calculate_hurst_exponent(
        self,
        prices: pd.Series,
        min_lag: int = 10,
        max_lag: int = 100
    ) -> Dict[str, float]:
        """
        Calculate Hurst Exponent via Rescaled Range (R/S) Analysis.
        
        H < 0.5: Mean-reverting (anti-persistent)
        H = 0.5: Random walk
        H > 0.5: Trending (persistent)
        
        Args:
            prices: Series of closing prices
            min_lag: Minimum lag for analysis (default: 10)
            max_lag: Maximum lag for analysis (default: 100)
            
        Returns:
            Dict with hurst_exponent, is_trending, is_mean_reverting, trend_strength
        """
        if prices.empty or len(prices) < max_lag + 10:
            return {
                "hurst_exponent": np.nan,
                "is_trending": False,
                "is_mean_reverting": False,
                "trend_strength": np.nan
            }
        
        try:
            returns = prices.pct_change().dropna().values
            
            lags = range(min_lag, max_lag, 5)
            tau = []
            
            for lag in lags:
                # Divide into chunks
                chunks = []
                for i in range(0, len(returns) - lag, lag):
                    chunks.append(returns[i:i+lag])
                
                if not chunks:
                    continue
                
                # Calculate R/S for each chunk
                rs_values = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    
                    # Mean-adjusted returns
                    mean_centered = chunk - np.mean(chunk)
                    
                    # Cumulative sum
                    cumsum = np.cumsum(mean_centered)
                    
                    # Range
                    R = np.max(cumsum) - np.min(cumsum)
                    
                    # Standard deviation
                    S = np.std(chunk, ddof=1)
                    
                    if S > 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(np.mean(rs_values))
            
            if len(tau) < 2:
                return {
                    "hurst_exponent": np.nan,
                    "is_trending": False,
                    "is_mean_reverting": False,
                    "trend_strength": np.nan
                }
            
            # Log-log linear regression to find exponent
            log_lags = np.log(list(lags)[:len(tau)])
            log_tau = np.log(tau)
            
            # Fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_tau)
            hurst = slope
            
            # Classify
            is_trending = hurst > 0.55
            is_mean_reverting = hurst < 0.45
            trend_strength = abs(hurst - 0.5) * 2  # Normalize to 0-1
            
            return {
                "hurst_exponent": float(hurst),
                "is_trending": bool(is_trending),
                "is_mean_reverting": bool(is_mean_reverting),
                "trend_strength": float(trend_strength)
            }
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return {
                "hurst_exponent": np.nan,
                "is_trending": False,
                "is_mean_reverting": False,
                "trend_strength": np.nan
            }
    
    def calculate_market_entropy(
        self,
        prices: pd.Series,
        window: int = 20,
        bins: int = 3
    ) -> Dict[str, float]:
        """
        Calculate Market Entropy (approximate entropy).
        
        High entropy: Random, unpredictable
        Low entropy: Structured, predictable
        
        Args:
            prices: Series of closing prices
            window: Window for entropy calculation (default: 20)
            bins: Number of quantization bins (default: 3)
            
        Returns:
            Dict with entropy, predictability, is_efficient
        """
        if prices.empty or len(prices) < window:
            return {
                "market_entropy": np.nan,
                "predictability": np.nan,
                "is_efficient": False
            }
        
        try:
            returns = prices.pct_change().dropna().values
            
            # Normalize returns
            mean = np.mean(returns)
            std = np.std(returns)
            
            if std == 0:
                return {
                    "market_entropy": np.nan,
                    "predictability": np.nan,
                    "is_efficient": False
                }
            
            normalized = (returns - mean) / std
            
            # Quantize into bins
            quantized = np.digitize(normalized, bins=np.linspace(-3, 3, bins))
            
            # Count pattern frequencies (using 2-step patterns)
            patterns = {}
            pattern_length = 2
            
            for i in range(len(quantized) - pattern_length):
                pattern = tuple(quantized[i:i+pattern_length])
                patterns[pattern] = patterns.get(pattern, 0) + 1
            
            if not patterns:
                return {
                    "market_entropy": np.nan,
                    "predictability": np.nan,
                    "is_efficient": False
                }
            
            # Calculate Shannon entropy
            total = sum(patterns.values())
            entropy = -sum((count/total) * np.log(count/total) 
                          for count in patterns.values() if count > 0)
            
            # Normalize entropy (0-1 scale)
            max_entropy = np.log(len(patterns))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0
            
            # Predictability is inverse of entropy
            predictability = 1.0 - normalized_entropy
            
            # Market is efficient if entropy is high
            is_efficient = normalized_entropy > 0.7
            
            return {
                "market_entropy": float(normalized_entropy),
                "predictability": float(predictability),
                "is_efficient": bool(is_efficient)
            }
        except Exception as e:
            self.logger.error(f"Error calculating market entropy: {e}")
            return {
                "market_entropy": np.nan,
                "predictability": np.nan,
                "is_efficient": False
            }
    
    def calculate_autocorrelation_decay(
        self,
        prices: pd.Series,
        max_lags: int = 20
    ) -> Dict[str, float]:
        """
        Calculate Autocorrelation Decay (mean-reversion indicator).
        
        Fast decay: Mean-reverting
        Slow decay: Trending/Persistent
        
        Args:
            prices: Series of closing prices
            max_lags: Maximum lags to analyze (default: 20)
            
        Returns:
            Dict with autocorr_decay, mean_reversion_indicator
        """
        if prices.empty or len(prices) < max_lags + 5:
            return {
                "autocorr_decay": np.nan,
                "mean_reversion_indicator": np.nan
            }
        
        try:
            returns = prices.pct_change().dropna().values
            
            # Calculate autocorrelations
            autocorr_values = []
            for lag in range(1, max_lags):
                if len(returns) > lag:
                    acf = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    autocorr_values.append(acf)
            
            if len(autocorr_values) < 2:
                return {
                    "autocorr_decay": np.nan,
                    "mean_reversion_indicator": np.nan
                }
            
            # Estimate decay rate using first two lags
            # Exponential decay: acf = A * exp(-lambda * lag)
            # lambda = -ln(acf[1] / acf[0])
            
            if abs(autocorr_values[0]) > 0.01:  # Avoid division by zero
                decay_rate = -np.log(abs(autocorr_values[1]) / (abs(autocorr_values[0]) + 1e-6))
            else:
                decay_rate = np.nan
            
            # Higher decay = faster mean reversion
            # Clamp between 0 and 1 for interpretability
            mean_reversion = min(max(decay_rate, 0), 1) if not np.isnan(decay_rate) else np.nan
            
            return {
                "autocorr_decay": float(decay_rate) if not np.isnan(decay_rate) else np.nan,
                "mean_reversion_indicator": float(mean_reversion)
            }
        except Exception as e:
            self.logger.error(f"Error calculating autocorrelation decay: {e}")
            return {
                "autocorr_decay": np.nan,
                "mean_reversion_indicator": np.nan
            }
    
    def calculate_regime_composite_score(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate composite regime score assessing model suitability.
        
        Args:
            prices: Series of closing prices
            volumes: Series of volumes (optional)
            
        Returns:
            Dict with regime_score and model suitability scores
        """
        # Get all regime indicators
        vol_regime = self.calculate_volatility_regime(prices)
        hurst = self.calculate_hurst_exponent(prices)
        entropy = self.calculate_market_entropy(prices)
        autocorr = self.calculate_autocorrelation_decay(prices)
        
        try:
            # Extract values with NaN handling
            vol_reg = vol_regime.get("vol_regime", 1)
            is_trending = hurst.get("is_trending", False)
            is_mean_rev = hurst.get("is_mean_reverting", False)
            pred = entropy.get("predictability", 0.5)
            mr_ind = autocorr.get("mean_reversion_indicator", 0.5)
            
            # Handle NaN values
            if pd.isna(vol_reg):
                vol_reg = 1
            if pd.isna(pred):
                pred = 0.5
            if pd.isna(mr_ind):
                mr_ind = 0.5
            
            # Model suitability scoring
            # Linear: Excels in mean-reverting, low-vol
            linear_score = float(mr_ind) * (1 - vol_reg / 2) if vol_reg < 3 else 0
            
            # CNN: Excels in trending, medium-high vol
            cnn_score = float(is_trending) * (vol_reg / 2) if vol_reg > 0 else 0
            
            # XGBoost: Balanced, works everywhere
            xgboost_score = 0.7  # Baseline strong performance
            
            # LLM: Excels when market is efficient (high entropy)
            llm_score = (1 - pred) if pred < 1 else 0
            
            # Composite regime score
            regime_score = (is_trending * 0.33) + (vol_reg / 2 * 0.33) + (pred * 0.34)
            
            return {
                "regime_score": float(regime_score),
                "linear_suitability": float(linear_score),
                "cnn_suitability": float(cnn_score),
                "xgboost_suitability": float(xgboost_score),
                "llm_suitability": float(llm_score)
            }
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return {
                "regime_score": np.nan,
                "linear_suitability": np.nan,
                "cnn_suitability": np.nan,
                "xgboost_suitability": np.nan,
                "llm_suitability": np.nan
            }
    
    def calculate_all_complex_indicators(
        self,
        prices_data,
        volumes: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate all complex regime indicators.
        
        Args:
            prices_data: DataFrame with OHLCV or Series of prices
            volumes: Series of volumes (optional)
            
        Returns:
            Dict with all complex regime features
        """
        # Handle DataFrame input
        if isinstance(prices_data, pd.DataFrame):
            if prices_data.empty:
                return {}
            prices_df = prices_data
            close_prices = prices_data['close'] if 'close' in prices_data.columns else prices_data.iloc[:, -2]
            if volumes is None and 'volume' in prices_data.columns:
                volumes = prices_data['volume']
        else:
            close_prices = prices_data
            prices_df = None
        
        if close_prices.empty or len(close_prices) < 100:
            return {}
        
        features = {}
        
        # OFI (requires OHLCV)
        if prices_df is not None and volumes is not None:
            features.update(self.calculate_order_flow_imbalance(prices_df, volumes))
        else:
            features["ofi"] = np.nan
            features["ofi_trend"] = np.nan
        
        # Volatility regime
        features.update(self.calculate_volatility_regime(close_prices))
        
        # Hurst exponent
        features.update(self.calculate_hurst_exponent(close_prices))
        
        # Market entropy
        features.update(self.calculate_market_entropy(close_prices))
        
        # Autocorrelation decay
        features.update(self.calculate_autocorrelation_decay(close_prices))
        
        # Composite score
        features.update(self.calculate_regime_composite_score(close_prices, volumes))
        
        # Ensure all numeric values are scalar
        for key in features:
            val = features[key]
            if isinstance(val, (pd.Series, pd.Index, np.ndarray)):
                features[key] = float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
            elif isinstance(val, (bool, np.bool_)):
                features[key] = float(val)
            elif not pd.isna(val) and not isinstance(val, bool):
                features[key] = float(val)
        
        return features


if __name__ == "__main__":
    logger.add(lambda msg: print(msg, end=''))
    
    # Example usage
    calc = ComplexRegimeIndicators()
    
    import yfinance as yf
    data = yf.download("AAPL", period="2y", progress=False)
    
    features = calc.calculate_all_complex_indicators(data)
    
    print("\n" + "=" * 80)
    print("COMPLEX REGIME INDICATORS FOR AAPL")
    print("=" * 80)
    
    for key, value in sorted(features.items()):
        if isinstance(value, bool):
            print(f"{key:35s}: {str(value)}")
        elif pd.notna(value):
            print(f"{key:35s}: {value:12.4f}")
        else:
            print(f"{key:35s}: NaN")

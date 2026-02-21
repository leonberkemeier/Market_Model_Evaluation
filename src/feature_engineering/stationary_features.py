"""
Stationary Memory Features (Cluster A from Roadmap)

These features preserve price memory while achieving stationarity:
- Fractional Differentiation (d): Most important feature for memory preservation
- Hurst Exponent: Regime detection (trending vs mean-reverting)
- Entropy (Shannon/Approximate): Signal randomness/noise measurement
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
from scipy import stats


class StationaryFeatures:
    """
    Advanced statistical features for maintaining price memory without non-stationarity.
    
    Core principle: Raw prices are non-stationary, raw returns lose memory.
    These features balance both concerns.
    """
    
    def __init__(self):
        """Initialize stationary features calculator."""
        self.logger = logger.bind(module="stationary_features")
    
    def fractional_difference(
        self,
        prices: pd.Series,
        d: float = 0.5,
        threshold: float = 1e-5
    ) -> Dict[str, float]:
        """
        Calculate Fractional Differentiation to preserve memory while achieving stationarity.
        
        This is THE most important feature for financial time series. Unlike integer
        differentiation (d=1, which is just returns), fractional d (0.3-0.7) keeps
        the "trend memory" while making the series stationary.
        
        d = 0.3: High memory retention (good for trending assets like Crypto)
        d = 0.5: Balanced (recommended default)
        d = 0.7: Low memory retention (good for mean-reverting assets like Finance)
        
        Args:
            prices: Series of closing prices
            d: Differentiation order (typically 0.3-0.7)
            threshold: Minimum weight threshold for computational efficiency
            
        Returns:
            Dict with frac_diff_mean, frac_diff_std, frac_diff_current
        """
        if prices.empty or len(prices) < 10:
            return {
                "frac_diff_mean": np.nan,
                "frac_diff_std": np.nan,
                "frac_diff_current": np.nan
            }
        
        try:
            # Calculate weights for fractional differentiation
            # w_k = (-1)^k * Gamma(d+1) / (Gamma(k+1) * Gamma(d-k+1))
            weights = [1.0]
            k = 1
            while True:
                w = -weights[-1] * (d - k + 1) / k
                if abs(w) < threshold:
                    break
                weights.append(w)
                k += 1
            
            weights = np.array(weights[::-1])
            
            # Apply fractional differentiation
            frac_diff = pd.Series(index=prices.index, dtype=float)
            
            for i in range(len(weights), len(prices)):
                frac_diff.iloc[i] = np.dot(
                    weights,
                    prices.iloc[i-len(weights)+1:i+1].values
                )
            
            # Remove NaN values
            frac_diff = frac_diff.dropna()
            
            if len(frac_diff) == 0:
                return {
                    "frac_diff_mean": np.nan,
                    "frac_diff_std": np.nan,
                    "frac_diff_current": np.nan
                }
            
            return {
                "frac_diff_mean": float(frac_diff.mean()),
                "frac_diff_std": float(frac_diff.std()),
                "frac_diff_current": float(frac_diff.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating fractional differentiation: {e}")
            return {
                "frac_diff_mean": np.nan,
                "frac_diff_std": np.nan,
                "frac_diff_current": np.nan
            }
    
    def calculate_hurst_exponent(
        self,
        prices: pd.Series,
        min_lag: int = 10,
        max_lag: int = 100
    ) -> Dict[str, float]:
        """
        Calculate Hurst Exponent via Rescaled Range (R/S) Analysis.
        
        The Hurst exponent determines market regime:
        H < 0.5: Mean-reverting (anti-persistent) → Good for pairs trading, Finance sector
        H = 0.5: Random walk → No edge
        H > 0.5: Trending (persistent) → Good for momentum strategies, Crypto
        
        Use this as a REGIME SWITCH INPUT to tell models whether to be
        contrarian or momentum-following.
        
        Args:
            prices: Series of closing prices
            min_lag: Minimum lag for R/S analysis (default: 10)
            max_lag: Maximum lag for R/S analysis (default: 100)
            
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
            
            lags = range(min_lag, min(max_lag, len(returns) // 2), 5)
            tau = []
            
            for lag in lags:
                # Divide returns into chunks
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
                    
                    # R/S ratio
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
            
            # Linear regression on log-log plot
            lags_used = list(lags)[:len(tau)]
            log_lags = np.log(lags_used)
            log_tau = np.log(tau)
            
            # Hurst exponent is the slope
            slope, intercept = np.polyfit(log_lags, log_tau, 1)
            hurst = float(slope)
            
            # Classify regime
            is_trending = hurst > 0.55  # Persistent/trending
            is_mean_reverting = hurst < 0.45  # Anti-persistent/mean-reverting
            
            # Trend strength: distance from random walk (0.5)
            trend_strength = float(abs(hurst - 0.5) * 2)  # Scale to 0-1
            
            return {
                "hurst_exponent": hurst,
                "is_trending": is_trending,
                "is_mean_reverting": is_mean_reverting,
                "trend_strength": trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return {
                "hurst_exponent": np.nan,
                "is_trending": False,
                "is_mean_reverting": False,
                "trend_strength": np.nan
            }
    
    def calculate_shannon_entropy(
        self,
        prices: pd.Series,
        bins: int = 50,
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate Shannon Entropy to measure signal randomness.
        
        High entropy = High noise/low confidence → Lower Kelly bet
        Low entropy = Structure present → Can bet more confidently
        
        Use this to dynamically adjust position sizes. When entropy spikes,
        the model should reduce exposure.
        
        Args:
            prices: Series of closing prices
            bins: Number of bins for histogram (default: 50)
            window: Rolling window (None = use all data)
            
        Returns:
            Dict with shannon_entropy, normalized_entropy, signal_quality
        """
        if prices.empty or len(prices) < 10:
            return {
                "shannon_entropy": np.nan,
                "normalized_entropy": np.nan,
                "signal_quality": np.nan
            }
        
        try:
            returns = prices.pct_change().dropna()
            
            if window and len(returns) > window:
                returns = returns.iloc[-window:]
            
            # Create histogram
            counts, _ = np.histogram(returns, bins=bins)
            
            # Calculate probabilities (remove zeros)
            probabilities = counts[counts > 0] / len(returns)
            
            # Shannon entropy: -Σ(p * log(p))
            shannon_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            
            # Normalized entropy (0-1 scale)
            max_entropy = np.log2(bins)  # Maximum possible entropy
            normalized_entropy = float(shannon_entropy / max_entropy)
            
            # Signal quality: inverse of normalized entropy
            # 1.0 = Very structured (low entropy)
            # 0.0 = Very noisy (high entropy)
            signal_quality = float(1.0 - normalized_entropy)
            
            return {
                "shannon_entropy": shannon_entropy,
                "normalized_entropy": normalized_entropy,
                "signal_quality": signal_quality
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Shannon entropy: {e}")
            return {
                "shannon_entropy": np.nan,
                "normalized_entropy": np.nan,
                "signal_quality": np.nan
            }
    
    def calculate_approximate_entropy(
        self,
        prices: pd.Series,
        m: int = 2,
        r: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Approximate Entropy (ApEn) - more robust for short time series.
        
        ApEn measures the predictability of fluctuations:
        - Low ApEn: Regular, predictable patterns
        - High ApEn: Irregular, unpredictable (high noise)
        
        More computationally expensive than Shannon but better for financial data.
        
        Args:
            prices: Series of closing prices
            m: Pattern length (default: 2)
            r: Tolerance (default: 0.2 * std of returns)
            
        Returns:
            Dict with approximate_entropy, pattern_regularity
        """
        if prices.empty or len(prices) < 50:
            return {
                "approximate_entropy": np.nan,
                "pattern_regularity": np.nan
            }
        
        try:
            returns = prices.pct_change().dropna().values
            N = len(returns)
            
            # Set tolerance if not provided
            if r is None:
                r = 0.2 * np.std(returns)
            
            def _maxdist(x_i, x_j, m):
                """Calculate maximum distance between patterns."""
                return max([abs(x_i[k] - x_j[k]) for k in range(m)])
            
            def _phi(m):
                """Calculate phi for pattern length m."""
                patterns = np.array([[returns[j] for j in range(i, i + m)] 
                                    for i in range(N - m + 1)])
                C = []
                
                for i in range(len(patterns)):
                    # Count matching patterns
                    matches = sum([1 for j in range(len(patterns)) 
                                  if _maxdist(patterns[i], patterns[j], m) <= r])
                    C.append(matches / (N - m + 1))
                
                return np.sum(np.log(C)) / (N - m + 1)
            
            # Calculate ApEn
            approximate_entropy = float(_phi(m) - _phi(m + 1))
            
            # Pattern regularity: inverse of ApEn (scaled to 0-1)
            # Typical ApEn range is 0-2
            pattern_regularity = float(max(0, min(1, 1 - approximate_entropy / 2)))
            
            return {
                "approximate_entropy": approximate_entropy,
                "pattern_regularity": pattern_regularity
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating approximate entropy: {e}")
            return {
                "approximate_entropy": np.nan,
                "pattern_regularity": np.nan
            }
    
    def calculate_all_stationary_features(
        self,
        prices: pd.Series,
        d_values: list = None
    ) -> Dict[str, float]:
        """
        Calculate all stationary features at once.
        
        Args:
            prices: Series of closing prices
            d_values: List of d values for fractional diff (default: [0.3, 0.5, 0.7])
            
        Returns:
            Dict with all stationary features
        """
        if d_values is None:
            d_values = [0.3, 0.5, 0.7]
        
        features = {}
        
        # Fractional differentiation (multiple d values)
        for d in d_values:
            frac_diff = self.fractional_difference(prices, d=d)
            for key, value in frac_diff.items():
                features[f"{key}_d{int(d*10)}"] = value
        
        # Hurst exponent
        features.update(self.calculate_hurst_exponent(prices))
        
        # Shannon entropy
        features.update(self.calculate_shannon_entropy(prices))
        
        # Approximate entropy
        features.update(self.calculate_approximate_entropy(prices))
        
        return features


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    # Create trending series for testing
    trend = np.linspace(100, 200, 300)
    noise = np.random.normal(0, 5, 300)
    prices = pd.Series(trend + noise, index=dates)
    
    calc = StationaryFeatures()
    
    # Test individual features
    print("=== Fractional Differentiation ===")
    print(calc.fractional_difference(prices, d=0.5))
    
    print("\n=== Hurst Exponent ===")
    print(calc.calculate_hurst_exponent(prices))
    
    print("\n=== Shannon Entropy ===")
    print(calc.calculate_shannon_entropy(prices))
    
    print("\n=== Approximate Entropy ===")
    print(calc.calculate_approximate_entropy(prices))
    
    print("\n=== All Stationary Features ===")
    all_features = calc.calculate_all_stationary_features(prices)
    for key, value in all_features.items():
        print(f"{key}: {value:.4f}" if not np.isnan(value) else f"{key}: NaN")

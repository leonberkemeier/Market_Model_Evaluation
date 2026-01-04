"""Technical feature calculations for stock price analysis."""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from loguru import logger


class TechnicalFeatures:
    """Calculate technical features from price data."""
    
    def __init__(self):
        """Initialize technical features calculator."""
        self.logger = logger.bind(module="technical_features")
    
    def calculate_momentum(
        self,
        prices: pd.Series,
        windows: list = None
    ) -> Dict[str, float]:
        """
        Calculate momentum (returns) over multiple windows.
        
        Args:
            prices: Series of closing prices
            windows: List of lookback windows (default: [5, 20, 60])
            
        Returns:
            Dict with momentum values for each window
        """
        if windows is None:
            windows = [5, 20, 60]
        
        if len(prices) < max(windows):
            return {f"momentum_{w}d": np.nan for w in windows}
        
        momentum = {}
        for window in windows:
            # Calculate return over window
            pct_change = float((prices.iloc[-1] - prices.iloc[-window]) / prices.iloc[-window])
            momentum[f"momentum_{window}d"] = pct_change
        
        return momentum
    
    def calculate_volatility(
        self,
        prices: pd.Series,
        windows: list = None
    ) -> Dict[str, float]:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            prices: Series of closing prices
            windows: List of lookback windows (default: [20, 60])
            
        Returns:
            Dict with volatility values for each window
        """
        if windows is None:
            windows = [20, 60]
        
        if len(prices) < max(windows):
            return {f"volatility_{w}d": np.nan for w in windows}
        
        volatility = {}
        for window in windows:
            returns = prices.pct_change().iloc[-window:]
            vol = float(returns.std())
            volatility[f"volatility_{window}d"] = vol
        
        return volatility
    
    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI ranges from 0-100:
        - < 30: Oversold
        - > 70: Overbought
        - 30-70: Normal
        
        Args:
            prices: Series of closing prices
            period: RSI period (default: 14)
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return np.nan
        
        # Calculate changes
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            Dict with MACD line, signal line, and histogram
        """
        if len(prices) < slow + signal:
            return {
                "macd_line": np.nan,
                "macd_signal": np.nan,
                "macd_histogram": np.nan
            }
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        macd_signal = macd_line.ewm(span=signal).mean()
        
        # Histogram
        macd_histogram = macd_line - macd_signal
        
        return {
            "macd_line": float(macd_line.iloc[-1]),
            "macd_signal": float(macd_signal.iloc[-1]),
            "macd_histogram": float(macd_histogram.iloc[-1])
        }
    
    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of closing prices
            period: SMA period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            
        Returns:
            Dict with upper band, middle band, lower band, and %B
        """
        if len(prices) < period:
            return {
                "bb_upper": np.nan,
                "bb_middle": np.nan,
                "bb_lower": np.nan,
                "bb_percent_b": np.nan
            }
        
        # Calculate SMA and std
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        # Bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # %B (where price is within bands)
        current_price = prices.iloc[-1]
        percent_b = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        return {
            "bb_upper": float(upper_band.iloc[-1]),
            "bb_middle": float(sma.iloc[-1]),
            "bb_lower": float(lower_band.iloc[-1]),
            "bb_percent_b": float(percent_b)
        }
    
    def calculate_price_to_sma(
        self,
        prices: pd.Series,
        periods: list = None
    ) -> Dict[str, float]:
        """
        Calculate price to SMA ratio.
        
        Ratio > 1: Price above SMA (uptrend)
        Ratio < 1: Price below SMA (downtrend)
        
        Args:
            prices: Series of closing prices
            periods: List of SMA periods (default: [20, 50, 200])
            
        Returns:
            Dict with price/SMA ratios
        """
        if periods is None:
            periods = [20, 50, 200]
        
        if prices.empty:
            return {f"price_to_sma_{p}": np.nan for p in periods}
        
        current_price = prices.iloc[-1]
        ratios = {}
        
        for period in periods:
            if len(prices) >= period:
                sma = prices.rolling(window=period).mean().iloc[-1]
                ratio = float(current_price / sma)
                ratios[f"price_to_sma_{period}"] = ratio
            else:
                ratios[f"price_to_sma_{period}"] = np.nan
        
        return ratios
    
    def calculate_volume_trend(
        self,
        volume: pd.Series,
        window: int = 20
    ) -> float:
        """
        Calculate volume trend: current volume / average volume.
        
        > 1: Above average volume
        < 1: Below average volume
        
        Args:
            volume: Series of trading volumes
            window: SMA window (default: 20)
            
        Returns:
            Volume trend ratio
        """
        if len(volume) < window:
            return np.nan
        
        avg_volume = float(volume.rolling(window=window).mean().iloc[-1])
        current_volume = float(volume.iloc[-1])
        
        if avg_volume == 0:
            return np.nan
        
        return float(current_volume / avg_volume)
    
    def calculate_all_technical_features(
        self,
        prices_data,
        momentum_windows: list = None,
        volatility_windows: list = None,
        sma_periods: list = None
    ) -> Dict[str, float]:
        """
        Calculate all technical features at once.
        
        Args:
            prices_data: DataFrame with OHLCV or Series of closing prices
            momentum_windows: Momentum windows (default: [5, 20, 60])
            volatility_windows: Volatility windows (default: [20, 60])
            sma_periods: SMA periods (default: [20, 50, 200])
            
        Returns:
            Dict with all technical features
        """
        if momentum_windows is None:
            momentum_windows = [5, 20, 60]
        if volatility_windows is None:
            volatility_windows = [20, 60]
        if sma_periods is None:
            sma_periods = [20, 50, 200]
        
        # Extract close and volume
        if isinstance(prices_data, pd.DataFrame):
            if prices_data.empty:
                return {}
            prices = prices_data['close'] if 'close' in prices_data.columns else prices_data.iloc[:, -2]
            volumes = prices_data['volume'] if 'volume' in prices_data.columns else None
        else:
            prices = prices_data
            volumes = None
        
        if prices.empty:
            return {}
        
        features = {}
        
        # Momentum
        features.update(self.calculate_momentum(prices, momentum_windows))
        
        # Volatility
        features.update(self.calculate_volatility(prices, volatility_windows))
        
        # RSI
        features["rsi_14"] = self.calculate_rsi(prices, period=14)
        
        # MACD
        features.update(self.calculate_macd(prices))
        
        # Bollinger Bands
        features.update(self.calculate_bollinger_bands(prices))
        
        # Price to SMA
        sma_features = self.calculate_price_to_sma(prices, sma_periods)
        # Rename keys for consistency
        sma_renamed = {k.replace('price_sma_', 'price_to_sma_'): v for k, v in sma_features.items()}
        features.update(sma_renamed)
        
        # Volume trend
        if volumes is not None and len(volumes) > 0:
            features["volume_trend"] = self.calculate_volume_trend(volumes)
        else:
            features["volume_trend"] = np.nan
        
        # Ensure all values are scalar
        for key in features:
            val = features[key]
            if isinstance(val, (pd.Series, pd.Index, np.ndarray)):
                features[key] = float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
            elif pd.isna(val):
                features[key] = np.nan
            else:
                features[key] = float(val)
        
        return features


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    logger.add(lambda msg: print(msg, end=''))
    
    # Download sample data
    data = yf.download("AAPL", period="1y", progress=False)
    prices = data['Close']
    volumes = data['Volume']
    
    # Calculate features
    calc = TechnicalFeatures()
    features = calc.calculate_all_technical_features(prices, volumes)
    
    print("\n" + "=" * 80)
    print("TECHNICAL FEATURES FOR AAPL")
    print("=" * 80)
    for key, value in features.items():
        if pd.notna(value):
            print(f"{key:25s}: {value:12.4f}")
        else:
            print(f"{key:25s}: NaN")

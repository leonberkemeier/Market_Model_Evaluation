"""Base class for feature calculators."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import date
import pandas as pd


class BaseFeatureCalculator(ABC):
    """
    Abstract base class for feature calculators.
    Each subclass implements specific feature engineering logic.
    """
    
    def __init__(self, name: str):
        """
        Initialize calculator.
        
        Args:
            name: Descriptive name for this calculator
        """
        self.name = name
        self.cache = {}
    
    @abstractmethod
    def calculate(self, ticker: str, date: date, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate features for a stock on a specific date.
        
        Args:
            ticker: Stock ticker
            date: Date to calculate features for
            data: DataFrame with OHLCV data up to date (inclusive)
            
        Returns:
            Dictionary of feature_name -> value
        """
        pass
    
    def validate_data(self, data: pd.DataFrame, min_rows: int = 1) -> bool:
        """
        Validate that input data has required columns and sufficient rows.
        
        Args:
            data: Input DataFrame
            min_rows: Minimum required rows
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = {'close', 'high', 'low', 'open', 'volume'}
        if not required_columns.issubset(data.columns):
            return False
        return len(data) >= min_rows
    
    def clear_cache(self):
        """Clear internal cache."""
        self.cache = {}


class TechnicalFeatureCalculator(BaseFeatureCalculator):
    """Calculate technical indicators (momentum, volatility, moving averages)."""
    
    def __init__(self):
        super().__init__("technical_features")
    
    def calculate(self, ticker: str, date: date, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical features.
        
        Args:
            ticker: Stock ticker
            date: Date to calculate for
            data: OHLCV DataFrame
            
        Returns:
            Dict of technical features
        """
        if not self.validate_data(data, min_rows=60):
            return {}
        
        features = {}
        
        # Momentum features
        for window in [5, 20, 60]:
            if len(data) >= window:
                momentum = (data['close'].iloc[-1] - data['close'].iloc[-window]) / data['close'].iloc[-window]
                features[f'momentum_{window}d'] = momentum
        
        # Volatility
        for window in [20, 60]:
            if len(data) >= window:
                volatility = data['close'].iloc[-window:].pct_change().std()
                features[f'volatility_{window}d'] = volatility
        
        # Moving averages (price / SMA ratio)
        for window in [20, 50, 200]:
            if len(data) >= window:
                sma = data['close'].iloc[-window:].mean()
                if sma > 0:
                    features[f'price_to_sma_{window}'] = data['close'].iloc[-1] / sma
        
        # Volume trend
        if len(data) >= 20:
            volume_avg = data['volume'].iloc[-20:].mean()
            if volume_avg > 0:
                features['volume_trend'] = data['volume'].iloc[-1] / volume_avg
        
        return features


class FundamentalFeatureCalculator(BaseFeatureCalculator):
    """Calculate fundamental features from financial statements."""
    
    def __init__(self):
        super().__init__("fundamental_features")
    
    def calculate(self, ticker: str, date: date, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate fundamental features.
        
        Note: This is a placeholder. Real implementation would query
        financial statements from database or API.
        
        Args:
            ticker: Stock ticker
            date: Date to calculate for
            data: OHLCV DataFrame (not used for fundamentals)
            
        Returns:
            Dict of fundamental features
        """
        # TODO: Implement actual fundamental feature calculation
        # This would query PE ratio, debt/equity, growth rates, etc.
        features = {
            'pe_ratio': 0.0,  # Placeholder
            'debt_equity': 0.0,
            'profit_margin': 0.0,
        }
        return features


class SentimentFeatureCalculator(BaseFeatureCalculator):
    """Calculate sentiment features from text analysis."""
    
    def __init__(self):
        super().__init__("sentiment_features")
    
    def calculate(self, ticker: str, date: date, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate sentiment features.
        
        Note: This is a placeholder. Real implementation would analyze
        SEC filings, news, etc.
        
        Args:
            ticker: Stock ticker
            date: Date to calculate for
            data: OHLCV DataFrame (not used for sentiment)
            
        Returns:
            Dict of sentiment features
        """
        # TODO: Implement actual sentiment analysis
        # This would parse SEC filings, news articles, etc.
        features = {
            'risk_keywords': 0.0,  # Placeholder
            'insider_sentiment': 0.0,
        }
        return features

"""Feature aggregator and caching for batch feature computation."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, date
from loguru import logger
import json
import pickle

from src.feature_engineering.technical_features import TechnicalFeatures
from src.feature_engineering.fundamental_features import FundamentalFeatures
from src.data.data_loader import DataLoader


class FeatureAggregator:
    """
    Aggregates technical and fundamental features.
    
    Provides caching and batch computation for efficient feature calculation
    across multiple stocks and dates.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize feature aggregator.
        
        Args:
            cache_dir: Directory for caching computed features
        """
        self.logger = logger.bind(module="feature_aggregator")
        self.technical = TechnicalFeatures()
        self.fundamental = FundamentalFeatures()
        self.data_loader = DataLoader()
        
        # Cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "cache" / "features"
        else:
            cache_dir = Path(cache_dir)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, ticker: str, cache_date: date) -> Path:
        """Get cache file path for ticker and date."""
        filename = f"{ticker}_{cache_date.isoformat()}.pkl"
        return self.cache_dir / filename
    
    def _load_from_cache(self, ticker: str, cache_date: date) -> Optional[Dict]:
        """Load cached features from disk."""
        cache_path = self._get_cache_path(ticker, cache_date)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.logger.debug(f"Loaded cache for {ticker} on {cache_date}")
                return data
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {ticker}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, ticker: str, cache_date: date, features: Dict) -> None:
        """Save computed features to cache."""
        cache_path = self._get_cache_path(ticker, cache_date)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            self.logger.debug(f"Cached features for {ticker} on {cache_date}")
        except Exception as e:
            self.logger.warning(f"Failed to cache features for {ticker}: {e}")
    
    def compute_technical_features(
        self,
        ticker: str,
        prices: pd.DataFrame,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Compute technical features for a stock.
        
        Args:
            ticker: Stock ticker symbol
            prices: DataFrame with OHLCV data
            use_cache: Whether to use cached features
            
        Returns:
            Dict with technical features
        """
        if prices.empty:
            self.logger.warning(f"No price data for {ticker}")
            return {}
        
        # Get most recent date
        if isinstance(prices.index, pd.DatetimeIndex):
            most_recent = prices.index[-1].date()
        else:
            most_recent = pd.to_datetime(prices.index[-1]).date()
        
        # Check cache
        if use_cache:
            cached = self._load_from_cache(ticker, most_recent)
            if cached and "technical" in cached:
                return cached["technical"]
        
        # Compute technical features
        try:
            tech_features = self.technical.calculate_all_technical_features(prices)
            return tech_features
        except Exception as e:
            self.logger.error(f"Error computing technical features for {ticker}: {e}")
            return {}
    
    def compute_fundamental_features(
        self,
        ticker: str,
        sector_performance: Optional[Dict[str, float]] = None,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Compute fundamental features for a stock.
        
        Args:
            ticker: Stock ticker symbol
            sector_performance: Sector performance dict
            use_cache: Whether to use cached features
            
        Returns:
            Dict with fundamental features
        """
        # Check cache
        if use_cache:
            cached = self._load_from_cache(ticker, date.today())
            if cached and "fundamental" in cached:
                return cached["fundamental"]
        
        # Compute fundamental features
        try:
            fund_features = self.fundamental.get_all_fundamental_features(
                ticker,
                sector_performance=sector_performance
            )
            return fund_features
        except Exception as e:
            self.logger.error(f"Error computing fundamental features for {ticker}: {e}")
            return {}
    
    def compute_all_features(
        self,
        ticker: str,
        prices: pd.DataFrame,
        sector_performance: Optional[Dict[str, float]] = None,
        use_cache: bool = True,
        save_cache: bool = True
    ) -> Dict[str, float]:
        """
        Compute all features (technical + fundamental) for a stock.
        
        Args:
            ticker: Stock ticker symbol
            prices: DataFrame with OHLCV data
            sector_performance: Sector performance dict
            use_cache: Whether to use cached features
            save_cache: Whether to save computed features to cache
            
        Returns:
            Dict with all features
        """
        all_features = {}
        
        # Technical features
        tech_features = self.compute_technical_features(ticker, prices, use_cache=False)
        all_features.update(tech_features)
        
        # Fundamental features
        fund_features = self.compute_fundamental_features(
            ticker,
            sector_performance=sector_performance,
            use_cache=False
        )
        all_features.update(fund_features)
        
        # Save to cache
        if save_cache:
            most_recent = (
                prices.index[-1].date()
                if isinstance(prices.index, pd.DatetimeIndex)
                else pd.to_datetime(prices.index[-1]).date()
            )
            self._save_to_cache(ticker, most_recent, {
                "technical": tech_features,
                "fundamental": fund_features
            })
        
        return all_features
    
    def compute_batch_features(
        self,
        tickers: List[str],
        prices_dict: Dict[str, pd.DataFrame],
        sector_performance: Optional[Dict[str, float]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute features for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            prices_dict: Dict mapping ticker -> price DataFrame
            sector_performance: Sector performance dict
            use_cache: Whether to use cached features
            
        Returns:
            DataFrame with features (rows=tickers, columns=feature names)
        """
        results = []
        
        for ticker in tickers:
            if ticker not in prices_dict:
                self.logger.warning(f"No price data for {ticker}")
                continue
            
            prices = prices_dict[ticker]
            features = self.compute_all_features(
                ticker,
                prices,
                sector_performance=sector_performance,
                use_cache=use_cache
            )
            
            features["ticker"] = ticker
            results.append(features)
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            self.logger.info(f"Computed features for {len(results)} stocks")
            return df
        else:
            self.logger.warning("No features computed")
            return pd.DataFrame()
    
    def compute_features_by_date(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        prices: pd.DataFrame,
        sector_performance: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute features for multiple dates (rolling window).
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            prices: DataFrame with OHLCV data
            sector_performance: Sector performance dict
            
        Returns:
            DataFrame with features indexed by date
        """
        if prices.empty:
            self.logger.warning(f"No price data for {ticker}")
            return pd.DataFrame()
        
        # Convert index to datetime if needed
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        
        results = []
        
        # Filter to dates within range
        mask = (prices.index.date >= start_date) & (prices.index.date <= end_date)
        date_prices = prices[mask]
        
        # Compute features for each date (using data up to that date)
        for date_idx in date_prices.index:
            # Get all data up to this date
            historical_prices = prices[:date_idx]
            
            if len(historical_prices) < 20:  # Need minimum data
                continue
            
            # Compute features
            features = self.compute_all_features(
                ticker,
                historical_prices,
                sector_performance=sector_performance,
                save_cache=False
            )
            
            features["date"] = date_idx.date()
            results.append(features)
        
        if results:
            df = pd.DataFrame(results)
            df.set_index("date", inplace=True)
            self.logger.info(f"Computed {len(df)} date-based feature sets for {ticker}")
            return df
        else:
            self.logger.warning(f"No date-based features computed for {ticker}")
            return pd.DataFrame()
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> None:
        """
        Clear cached features.
        
        Args:
            older_than_days: Only delete cache older than this many days
        """
        import time
        
        cleared_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if older_than_days is not None:
                file_time = cache_file.stat().st_mtime
                file_age_days = (current_time - file_time) / (24 * 3600)
                if file_age_days < older_than_days:
                    continue
            
            try:
                cache_file.unlink()
                cleared_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to delete {cache_file}: {e}")
        
        self.logger.info(f"Cleared {cleared_count} cache files")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "num_cached_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


if __name__ == "__main__":
    logger.add(lambda msg: print(msg, end=''))
    
    # Example usage
    aggregator = FeatureAggregator()
    
    print("\n" + "=" * 80)
    print("FEATURE AGGREGATOR EXAMPLE")
    print("=" * 80)
    
    # Example: Compute features for a single stock
    print("\nNote: Ensure stock price data is available in the database")
    print("Run: python -c \"from src.feature_engineering.feature_aggregator import FeatureAggregator; agg = FeatureAggregator(); print(agg.get_cache_stats())\"")
    
    stats = aggregator.get_cache_stats()
    print(f"\nCache statistics:")
    print(f"  Cached files: {stats['num_cached_files']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    print(f"  Cache dir: {stats['cache_dir']}")

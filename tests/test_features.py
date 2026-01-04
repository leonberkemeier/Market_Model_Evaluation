"""Unit tests for feature engineering modules."""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

from src.feature_engineering.technical_features import TechnicalFeatures
from src.feature_engineering.fundamental_features import FundamentalFeatures
from src.feature_engineering.feature_aggregator import FeatureAggregator


class TestTechnicalFeatures:
    """Test cases for TechnicalFeatures class."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        close_vals = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)))
        high_vals = pd.Series(close_vals + np.abs(np.random.normal(0, 0.5, 200)))
        low_vals = pd.Series(close_vals - np.abs(np.random.normal(0, 0.5, 200)))
        open_vals = close_vals.shift(1)
        open_vals.iloc[0] = close_vals.iloc[0]
        volume_vals = pd.Series(np.random.uniform(1e6, 1e7, 200))
        
        df = pd.DataFrame({
            'open': open_vals.values,
            'high': high_vals.values,
            'low': low_vals.values,
            'close': close_vals.values,
            'volume': volume_vals.values
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def tech_features(self):
        """Create TechnicalFeatures instance."""
        return TechnicalFeatures()
    
    def test_initialization(self, tech_features):
        """Test TechnicalFeatures initialization."""
        assert tech_features is not None
    
    def test_momentum_calculation(self, tech_features, sample_prices):
        """Test momentum calculation."""
        # Extract close prices from DataFrame
        prices = sample_prices['close']
        momentum = tech_features.calculate_momentum(prices)
        
        assert momentum is not None
        assert isinstance(momentum, dict)
        assert all(k in momentum for k in ['momentum_5d', 'momentum_20d', 'momentum_60d'])
        assert all(isinstance(v, (float, int)) or pd.isna(v) for v in momentum.values())
    
    def test_volatility_calculation(self, tech_features, sample_prices):
        """Test volatility calculation."""
        # Extract close prices from DataFrame
        prices = sample_prices['close']
        volatility = tech_features.calculate_volatility(prices)
        
        assert volatility is not None
        assert isinstance(volatility, dict)
        assert all(k in volatility for k in ['volatility_20d', 'volatility_60d'])
        assert all(v >= 0 or pd.isna(v) for v in volatility.values())  # Volatility is non-negative
    
    def test_rsi_calculation(self, tech_features, sample_prices):
        """Test RSI calculation."""
        # Extract close prices from DataFrame
        prices = sample_prices['close']
        rsi = tech_features.calculate_rsi(prices)
        
        assert rsi is not None
        assert isinstance(rsi, (float, int)) or pd.isna(rsi)
        if not pd.isna(rsi):
            assert 0 <= rsi <= 100  # RSI is 0-100 scale
    
    def test_macd_calculation(self, tech_features, sample_prices):
        """Test MACD calculation."""
        # Extract close prices from DataFrame
        prices = sample_prices['close']
        macd = tech_features.calculate_macd(prices)
        
        assert macd is not None
        assert isinstance(macd, dict)
        assert 'macd_line' in macd
        assert 'macd_signal' in macd
        assert 'macd_histogram' in macd
    
    def test_bollinger_bands_calculation(self, tech_features, sample_prices):
        """Test Bollinger Bands calculation."""
        # Extract close prices from DataFrame
        prices = sample_prices['close']
        bb = tech_features.calculate_bollinger_bands(prices)
        
        assert bb is not None
        assert isinstance(bb, dict)
        assert all(k in bb for k in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_percent_b'])
        # Check relationships only if values are not NaN
        if not pd.isna(bb['bb_lower']) and not pd.isna(bb['bb_upper']):
            assert bb['bb_lower'] <= bb['bb_middle'] <= bb['bb_upper']
    
    def test_price_to_sma_calculation(self, tech_features, sample_prices):
        """Test price to SMA ratio calculation."""
        # Extract close prices from DataFrame
        prices = sample_prices['close']
        ratios = tech_features.calculate_price_to_sma(prices)
        
        assert ratios is not None
        assert isinstance(ratios, dict)
        assert all(k in ratios for k in ['price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200'])
        assert all(v > 0 or pd.isna(v) for v in ratios.values())  # Ratios should be positive
    
    def test_volume_trend_calculation(self, tech_features, sample_prices):
        """Test volume trend calculation."""
        # Extract volume from DataFrame
        volume = sample_prices['volume']
        volume_trend = tech_features.calculate_volume_trend(volume)
        
        assert volume_trend is not None
        assert isinstance(volume_trend, (float, int)) or pd.isna(volume_trend)
        if not pd.isna(volume_trend):
            assert volume_trend > 0
    
    def test_all_technical_features(self, tech_features, sample_prices):
        """Test all technical features calculation."""
        all_features = tech_features.calculate_all_technical_features(sample_prices)
        
        assert all_features is not None
        assert isinstance(all_features, dict)
        assert len(all_features) > 10  # Should have many features
        assert all(isinstance(v, (int, float)) or pd.isna(v) for v in all_features.values())
    
    def test_empty_dataframe(self, tech_features):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        result = tech_features.calculate_all_technical_features(empty_df)
        
        assert isinstance(result, dict)
        assert len(result) > 0 or result == {}  # Should return empty or with NaN values


class TestFundamentalFeatures:
    """Test cases for FundamentalFeatures class."""
    
    @pytest.fixture
    def fund_features(self):
        """Create FundamentalFeatures instance."""
        return FundamentalFeatures()
    
    def test_initialization(self, fund_features):
        """Test FundamentalFeatures initialization."""
        assert fund_features is not None
        assert fund_features.data_loader is not None
    
    def test_sector_rotation_calculation(self, fund_features):
        """Test sector rotation calculation."""
        sector_performance = {
            'Technology': 0.10,
            'Finance': 0.05,
            'Energy': -0.02,
            'Healthcare': 0.08
        }
        
        rotation = fund_features.calculate_sector_rotation('Technology', sector_performance)
        
        assert isinstance(rotation, float)
        assert -1 <= rotation <= 1
        assert rotation > 0  # Technology is above average
    
    def test_sector_rotation_missing_sector(self, fund_features):
        """Test sector rotation with missing sector."""
        sector_performance = {'Tech': 0.10}
        rotation = fund_features.calculate_sector_rotation('Finance', sector_performance)
        
        assert rotation == 0.0
    
    def test_calculate_growth_metrics(self, fund_features):
        """Test growth metrics calculation."""
        revenue_history = [1000, 1100, 1200]
        earnings_history = [100, 110, 120]
        
        metrics = fund_features.calculate_growth_metrics(
            'TEST',
            revenue_history=revenue_history,
            earnings_history=earnings_history
        )
        
        assert 'revenue_growth' in metrics
        assert 'earnings_growth' in metrics
        assert metrics['revenue_growth'] > 0
        assert metrics['earnings_growth'] > 0
    
    def test_calculate_growth_metrics_insufficient_data(self, fund_features):
        """Test growth metrics with insufficient data."""
        metrics = fund_features.calculate_growth_metrics(
            'TEST',
            revenue_history=[1000],
            earnings_history=[100]
        )
        
        assert pd.isna(metrics['revenue_growth'])
        assert pd.isna(metrics['earnings_growth'])
    
    def test_estimate_financial_health(self, fund_features):
        """Test financial health estimation."""
        financial_metrics = {
            'pe_ratio': 15.0,
            'debt_equity_ratio': 0.5,
            'roe': 0.15
        }
        
        health = fund_features.estimate_financial_health('TEST', financial_metrics)
        
        assert 'pe_ratio' in health
        assert health['pe_ratio'] == 15.0
        assert 'debt_equity_ratio' in health
        assert health['debt_equity_ratio'] == 0.5
    
    def test_get_all_fundamental_features(self, fund_features):
        """Test getting all fundamental features."""
        sector_performance = {'Technology': 0.10}
        
        features = fund_features.get_all_fundamental_features(
            'TEST',
            sector_performance=sector_performance
        )
        
        assert isinstance(features, dict)
        assert 'has_data' in features
        assert 'has_filings' in features


class TestFeatureAggregator:
    """Test cases for FeatureAggregator class."""
    
    @pytest.fixture
    def aggregator(self, tmp_path):
        """Create FeatureAggregator with temporary cache directory."""
        return FeatureAggregator(cache_dir=str(tmp_path / "cache"))
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        close_vals = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)))
        high_vals = pd.Series(close_vals + np.abs(np.random.normal(0, 0.5, 200)))
        low_vals = pd.Series(close_vals - np.abs(np.random.normal(0, 0.5, 200)))
        open_vals = close_vals.shift(1)
        open_vals.iloc[0] = close_vals.iloc[0]
        volume_vals = pd.Series(np.random.uniform(1e6, 1e7, 200))
        
        df = pd.DataFrame({
            'open': open_vals.values,
            'high': high_vals.values,
            'low': low_vals.values,
            'close': close_vals.values,
            'volume': volume_vals.values
        }, index=dates)
        
        return df
    
    def test_initialization(self, aggregator):
        """Test FeatureAggregator initialization."""
        assert aggregator is not None
        assert aggregator.cache_dir.exists()
    
    def test_cache_path_generation(self, aggregator):
        """Test cache path generation."""
        test_date = date(2023, 1, 15)
        cache_path = aggregator._get_cache_path('AAPL', test_date)
        
        assert 'AAPL' in str(cache_path)
        assert '2023-01-15' in str(cache_path)
        assert cache_path.suffix == '.pkl'
    
    def test_cache_save_and_load(self, aggregator):
        """Test saving and loading from cache."""
        test_date = date(2023, 1, 15)
        test_data = {'feature1': 1.5, 'feature2': 2.5}
        
        # Save to cache
        aggregator._save_to_cache('TEST', test_date, test_data)
        
        # Load from cache
        loaded_data = aggregator._load_from_cache('TEST', test_date)
        
        assert loaded_data == test_data
    
    def test_load_nonexistent_cache(self, aggregator):
        """Test loading non-existent cache."""
        result = aggregator._load_from_cache('NONEXISTENT', date(2023, 1, 1))
        assert result is None
    
    def test_compute_technical_features(self, aggregator, sample_prices):
        """Test computing technical features."""
        # Note: compute_technical_features in aggregator wraps the DataFrame
        features = aggregator.compute_technical_features('TEST', sample_prices, use_cache=False)
        
        assert isinstance(features, dict)
        # May be empty if insufficient data, but should not error
        assert features is not None
    
    def test_compute_all_features(self, aggregator, sample_prices):
        """Test computing all features."""
        features = aggregator.compute_all_features(
            'TEST',
            sample_prices,
            use_cache=False,
            save_cache=False
        )
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_compute_batch_features(self, aggregator, sample_prices):
        """Test batch feature computation."""
        prices_dict = {
            'AAPL': sample_prices,
            'MSFT': sample_prices.copy()
        }
        
        features_df = aggregator.compute_batch_features(
            ['AAPL', 'MSFT'],
            prices_dict,
            use_cache=False
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 2
        assert 'ticker' in features_df.columns
    
    def test_compute_features_by_date(self, aggregator, sample_prices):
        """Test computing features by date."""
        features_by_date = aggregator.compute_features_by_date(
            'TEST',
            date(2023, 2, 1),
            date(2023, 2, 28),
            sample_prices
        )
        
        assert isinstance(features_by_date, pd.DataFrame)
        if len(features_by_date) > 0:
            assert features_by_date.index.name == 'date'
    
    def test_get_cache_stats(self, aggregator):
        """Test getting cache statistics."""
        stats = aggregator.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'num_cached_files' in stats
        assert 'total_size_bytes' in stats
        assert 'total_size_mb' in stats
        assert 'cache_dir' in stats
    
    def test_clear_cache(self, aggregator, sample_prices):
        """Test clearing cache."""
        # Save some data to cache
        aggregator.compute_all_features(
            'TEST1',
            sample_prices,
            save_cache=True
        )
        aggregator.compute_all_features(
            'TEST2',
            sample_prices,
            save_cache=True
        )
        
        stats_before = aggregator.get_cache_stats()
        
        # Clear cache
        aggregator.clear_cache()
        
        stats_after = aggregator.get_cache_stats()
        assert stats_after['num_cached_files'] < stats_before['num_cached_files']


class TestFeatureIntegration:
    """Integration tests for feature engineering."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        np.random.seed(42)
        
        close_vals = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 250)))
        high_vals = pd.Series(close_vals + np.abs(np.random.normal(0, 0.5, 250)))
        low_vals = pd.Series(close_vals - np.abs(np.random.normal(0, 0.5, 250)))
        open_vals = close_vals.shift(1)
        open_vals.iloc[0] = close_vals.iloc[0]
        volume_vals = pd.Series(np.random.uniform(1e6, 1e7, 250))
        
        df = pd.DataFrame({
            'open': open_vals.values,
            'high': high_vals.values,
            'low': low_vals.values,
            'close': close_vals.values,
            'volume': volume_vals.values
        }, index=dates)
        
        return df
    
    def test_full_feature_pipeline(self, sample_prices, tmp_path):
        """Test complete feature engineering pipeline."""
        aggregator = FeatureAggregator(cache_dir=str(tmp_path / "cache"))
        
        # Compute features
        features = aggregator.compute_all_features(
            'TEST',
            sample_prices,
            save_cache=True
        )
        
        # Verify features are computed
        assert len(features) > 0
        
        # Verify cache was created
        stats = aggregator.get_cache_stats()
        assert stats['num_cached_files'] > 0
        
        # Verify features can be loaded from cache
        features_cached = aggregator.compute_all_features(
            'TEST',
            sample_prices,
            use_cache=True
        )
        
        assert len(features_cached) > 0
    
    def test_feature_consistency(self, sample_prices, tmp_path):
        """Test that features are consistent across multiple computations."""
        aggregator = FeatureAggregator(cache_dir=str(tmp_path / "cache"))
        
        # Compute twice without cache
        features1 = aggregator.compute_all_features(
            'TEST',
            sample_prices,
            use_cache=False,
            save_cache=False
        )
        features2 = aggregator.compute_all_features(
            'TEST',
            sample_prices,
            use_cache=False,
            save_cache=False
        )
        
        # Compare numeric features
        for key in features1:
            if isinstance(features1[key], (int, float)):
                if not (pd.isna(features1[key]) and pd.isna(features2.get(key))):
                    assert features1[key] == features2.get(key), f"Feature {key} mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

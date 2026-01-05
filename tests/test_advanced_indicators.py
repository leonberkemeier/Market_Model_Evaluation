"""Unit tests for advanced and complex technical indicators."""
import pytest
import pandas as pd
import numpy as np
from datetime import date

from src.feature_engineering.advanced_technical_features import AdvancedTechnicalFeatures
from src.feature_engineering.complex_regime_indicators import ComplexRegimeIndicators


class TestAdvancedTechnicalFeatures:
    """Tests for advanced technical indicators."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        close_vals = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)))
        high_vals = pd.Series(close_vals + np.abs(np.random.normal(0, 0.5, 200)))
        low_vals = pd.Series(close_vals - np.abs(np.random.normal(0, 0.5, 200)))
        open_vals = close_vals.shift(1)
        open_vals.iloc[0] = close_vals.iloc[0]
        volume = np.random.uniform(1e6, 1e7, 200)
        
        return pd.DataFrame({
            'open': open_vals.values,
            'high': high_vals.values,
            'low': low_vals.values,
            'close': close_vals.values,
            'volume': volume
        }, index=dates)
    
    @pytest.fixture
    def calc(self):
        """Create calculator instance."""
        return AdvancedTechnicalFeatures()
    
    def test_adx_calculation(self, calc, sample_data):
        """Test ADX calculation."""
        result = calc.calculate_adx(sample_data)
        
        assert isinstance(result, dict)
        assert 'adx_14' in result
        assert 'di_plus' in result
        assert 'di_minus' in result
        assert all(isinstance(v, (float, int)) or pd.isna(v) for v in result.values())
    
    def test_donchian_channels(self, calc, sample_data):
        """Test Donchian channels calculation."""
        result = calc.calculate_donchian_channels(sample_data)
        
        assert isinstance(result, dict)
        assert 'donchian_20_high' in result
        assert 'donchian_20_low' in result
        assert 'donchian_200_high' in result
        # High should be >= low
        if not pd.isna(result['donchian_20_high']) and not pd.isna(result['donchian_20_low']):
            assert result['donchian_20_high'] >= result['donchian_20_low']
    
    def test_mfi_calculation(self, calc, sample_data):
        """Test MFI calculation."""
        result = calc.calculate_mfi(sample_data, sample_data['volume'])
        
        assert isinstance(result, dict)
        assert 'mfi_14' in result
        if not pd.isna(result['mfi_14']):
            assert 0 <= result['mfi_14'] <= 100
    
    def test_obv_calculation(self, calc, sample_data):
        """Test OBV calculation."""
        result = calc.calculate_obv(sample_data['close'], sample_data['volume'])
        
        assert isinstance(result, dict)
        assert 'obv' in result
        assert 'obv_ema_21' in result
    
    def test_stochastic_calculation(self, calc, sample_data):
        """Test Stochastic oscillator."""
        result = calc.calculate_stochastic(sample_data)
        
        assert isinstance(result, dict)
        assert 'stochastic_k' in result
        assert 'stochastic_d' in result
        if not pd.isna(result['stochastic_k']):
            assert 0 <= result['stochastic_k'] <= 100
    
    def test_all_advanced_features(self, calc, sample_data):
        """Test all advanced features calculation."""
        result = calc.calculate_all_advanced_features(sample_data)
        
        assert isinstance(result, dict)
        assert len(result) > 10  # Should have many features
        # Check no Series objects
        assert all(not isinstance(v, pd.Series) for v in result.values())
    
    def test_empty_dataframe(self, calc):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        result = calc.calculate_all_advanced_features(empty_df)
        
        assert isinstance(result, dict)


class TestComplexRegimeIndicators:
    """Tests for complex regime indicators."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        close_vals = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 300)))
        open_vals = close_vals.shift(1)
        open_vals.iloc[0] = close_vals.iloc[0]
        high_vals = pd.Series(close_vals + np.abs(np.random.normal(0, 0.5, 300)))
        low_vals = pd.Series(close_vals - np.abs(np.random.normal(0, 0.5, 300)))
        volume = np.random.uniform(1e6, 1e7, 300)
        
        return pd.DataFrame({
            'open': open_vals.values,
            'close': close_vals.values,
            'high': high_vals.values,
            'low': low_vals.values,
            'volume': volume
        }, index=dates)
    
    @pytest.fixture
    def calc(self):
        """Create calculator instance."""
        return ComplexRegimeIndicators()
    
    def test_ofi_calculation(self, calc, sample_data):
        """Test OFI calculation."""
        result = calc.calculate_order_flow_imbalance(sample_data, sample_data['volume'])
        
        assert isinstance(result, dict)
        assert 'ofi' in result
        assert 'ofi_trend' in result
        if not pd.isna(result['ofi']):
            assert -1 <= result['ofi'] <= 1
    
    def test_volatility_regime(self, calc, sample_data):
        """Test volatility regime calculation."""
        result = calc.calculate_volatility_regime(sample_data['close'])
        
        assert isinstance(result, dict)
        assert 'vol_regime' in result
        assert 'vol_regime_confidence' in result
        assert 'mean_reversion_strength' in result
        
        if not pd.isna(result['vol_regime']):
            assert result['vol_regime'] in [0, 1, 2]
    
    def test_hurst_exponent(self, calc, sample_data):
        """Test Hurst exponent calculation."""
        result = calc.calculate_hurst_exponent(sample_data['close'])
        
        assert isinstance(result, dict)
        assert 'hurst_exponent' in result
        assert 'is_trending' in result
        assert 'is_mean_reverting' in result
        assert 'trend_strength' in result
        
        if not pd.isna(result['hurst_exponent']):
            assert 0 <= result['hurst_exponent'] <= 1
    
    def test_market_entropy(self, calc, sample_data):
        """Test market entropy calculation."""
        result = calc.calculate_market_entropy(sample_data['close'])
        
        assert isinstance(result, dict)
        assert 'market_entropy' in result
        assert 'predictability' in result
        assert 'is_efficient' in result
        
        if not pd.isna(result['market_entropy']):
            assert 0 <= result['market_entropy'] <= 1
            assert 0 <= result['predictability'] <= 1
    
    def test_autocorrelation_decay(self, calc, sample_data):
        """Test autocorrelation decay."""
        result = calc.calculate_autocorrelation_decay(sample_data['close'])
        
        assert isinstance(result, dict)
        assert 'autocorr_decay' in result
        assert 'mean_reversion_indicator' in result
    
    def test_regime_composite_score(self, calc, sample_data):
        """Test composite regime score."""
        result = calc.calculate_regime_composite_score(sample_data['close'])
        
        assert isinstance(result, dict)
        assert 'regime_score' in result
        assert 'linear_suitability' in result
        assert 'cnn_suitability' in result
        assert 'xgboost_suitability' in result
        assert 'llm_suitability' in result
    
    def test_all_complex_indicators(self, calc, sample_data):
        """Test all complex indicators calculation."""
        result = calc.calculate_all_complex_indicators(sample_data)
        
        assert isinstance(result, dict)
        assert len(result) > 15  # Should have many features
        # Check no Series objects
        assert all(not isinstance(v, pd.Series) for v in result.values())
    
    def test_insufficient_data(self, calc):
        """Test handling of insufficient data."""
        small_data = pd.Series([100, 101, 102])
        result = calc.calculate_hurst_exponent(small_data)
        
        assert isinstance(result, dict)
        assert pd.isna(result['hurst_exponent'])


class TestIntegrationAdvancedIndicators:
    """Integration tests for all indicators together."""
    
    @pytest.fixture
    def full_data(self):
        """Create full year of data."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        close_vals = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 252)))
        high_vals = pd.Series(close_vals + np.abs(np.random.normal(0, 0.5, 252)))
        low_vals = pd.Series(close_vals - np.abs(np.random.normal(0, 0.5, 252)))
        open_vals = close_vals.shift(1)
        open_vals.iloc[0] = close_vals.iloc[0]
        volume = np.random.uniform(1e6, 1e7, 252)
        
        return pd.DataFrame({
            'open': open_vals.values,
            'high': high_vals.values,
            'low': low_vals.values,
            'close': close_vals.values,
            'volume': volume
        }, index=dates)
    
    def test_combined_feature_set(self, full_data):
        """Test complete advanced + complex feature set."""
        adv = AdvancedTechnicalFeatures()
        comp = ComplexRegimeIndicators()
        
        # Calculate both
        advanced_feats = adv.calculate_all_advanced_features(full_data)
        complex_feats = comp.calculate_all_complex_indicators(full_data)
        
        # Combine
        all_features = {**advanced_feats, **complex_feats}
        
        # Should have 11 advanced + 20+ complex = 30+
        assert len(all_features) > 25
        
        # All should be numeric or NaN
        for v in all_features.values():
            assert isinstance(v, (int, float)) or pd.isna(v) or isinstance(v, bool)
    
    def test_feature_consistency(self, full_data):
        """Test that features are consistent across runs."""
        adv = AdvancedTechnicalFeatures()
        
        result1 = adv.calculate_all_advanced_features(full_data)
        result2 = adv.calculate_all_advanced_features(full_data)
        
        # Should be identical
        for key in result1:
            if isinstance(result1[key], float) and not pd.isna(result1[key]):
                assert result1[key] == result2[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

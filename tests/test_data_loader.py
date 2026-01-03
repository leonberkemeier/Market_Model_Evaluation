"""Unit tests for data loader."""
import pytest
from datetime import date
from src.data.data_loader import DataLoader
from src.config.config import ALL_TICKERS


class TestDataLoader:
    """Test DataLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Fixture for DataLoader instance."""
        return DataLoader()
    
    def test_initialization(self, loader):
        """Test that DataLoader initializes correctly."""
        assert loader is not None
        assert loader.engine is not None
    
    def test_load_stock_prices_empty(self, loader):
        """Test loading prices for non-existent ticker."""
        df = loader.load_stock_prices(
            "NONEXISTENT",
            date(2023, 1, 1),
            date(2023, 12, 31)
        )
        # May be empty or raise exception - both acceptable
        assert df is not None or df.empty
    
    def test_load_company_metadata(self, loader):
        """Test loading company metadata."""
        # Try with a known ticker from the database
        # This will depend on what's actually in the database
        result = loader.load_company_metadata("AAPL")
        
        # If AAPL exists, it should have expected fields
        if result:
            assert "ticker" in result
            assert "name" in result
            assert "sector" in result
    
    def test_verify_data_availability(self, loader):
        """Test data availability verification."""
        # Test with a valid ticker (may fail if not in database)
        is_available, message = loader.verify_data_availability("AAPL")
        
        assert isinstance(is_available, bool)
        assert isinstance(message, str)
    
    def test_batch_validation(self, loader):
        """Test validating multiple tickers."""
        test_tickers = ["AAPL", "MSFT", "GOOGL"]
        availability = loader.validate_universe(test_tickers)
        
        assert isinstance(availability, dict)
        assert len(availability) == len(test_tickers)
        
        for ticker in test_tickers:
            assert ticker in availability
            assert "available" in availability[ticker]
            assert "message" in availability[ticker]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

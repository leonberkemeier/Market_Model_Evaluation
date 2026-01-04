"""Fundamental feature extraction from financial data and SEC filings."""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import date
from loguru import logger

from src.data.data_loader import DataLoader


class FundamentalFeatures:
    """Extract fundamental features from company data and filings."""
    
    def __init__(self):
        """Initialize fundamental features extractor."""
        self.logger = logger.bind(module="fundamental_features")
        self.data_loader = DataLoader()
    
    def get_company_features(self, ticker: str) -> Dict[str, float]:
        """
        Get fundamental features for a company from database.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with fundamental features
        """
        features = {}
        
        # Load company metadata
        metadata = self.data_loader.load_company_metadata(ticker)
        
        if not metadata:
            self.logger.warning(f"No metadata found for {ticker}")
            return {
                "has_data": 0,
                "sector_encoded": np.nan,
                "industry_encoded": np.nan
            }
        
        features["has_data"] = 1
        features["sector"] = metadata.get("sector", "Unknown")
        features["industry"] = metadata.get("industry", "Unknown")
        features["country"] = metadata.get("country", "Unknown")
        
        if metadata.get("exchange_code"):
            features["exchange"] = metadata["exchange_code"]
        
        return features
    
    def calculate_sector_rotation(
        self,
        sector: str,
        sector_performance: Dict[str, float]
    ) -> float:
        """
        Calculate sector rotation signal.
        
        Measures relative strength of sector vs broad market.
        
        Args:
            sector: Company sector
            sector_performance: Dict of sector -> performance
            
        Returns:
            Sector rotation score (-1 to +1)
        """
        if sector not in sector_performance:
            return 0.0
        
        sector_perf = sector_performance[sector]
        avg_perf = np.mean(list(sector_performance.values()))
        
        if avg_perf == 0:
            return 0.0
        
        # Normalize to -1 to +1 range
        rotation = (sector_perf - avg_perf) / (abs(avg_perf) + 0.01)
        return np.clip(rotation, -1, 1)
    
    def extract_filing_metrics(
        self,
        ticker: str
    ) -> Dict[str, float]:
        """
        Extract metrics from SEC filings.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with filing-based metrics
        """
        features = {}
        
        # Load latest filings
        filings = self.data_loader.load_sec_filings(ticker, limit=3)
        
        if filings.empty:
            self.logger.debug(f"No SEC filings found for {ticker}")
            return {
                "has_filings": 0,
                "avg_filing_size": np.nan,
                "filing_frequency": 0
            }
        
        features["has_filings"] = 1
        features["filing_count"] = len(filings)
        
        # Average filing size (proxy for company complexity/detail)
        avg_size = filings["filing_size"].mean() if "filing_size" in filings else np.nan
        features["avg_filing_size"] = avg_size
        
        # Most recent filing date (proxy for reporting timeliness)
        if "filing_date" in filings:
            latest_date = pd.to_datetime(filings["filing_date"]).max()
            features["days_since_latest_filing"] = (date.today() - latest_date.date()).days
        
        return features
    
    def estimate_financial_health(
        self,
        ticker: str,
        known_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Estimate financial health metrics.
        
        This would typically pull from company financial statements.
        For now, we estimate from available data.
        
        Args:
            ticker: Stock ticker symbol
            known_metrics: Pre-calculated metrics (P/E, debt/equity, etc.)
            
        Returns:
            Dict with financial health estimates
        """
        features = {}
        
        if known_metrics is None:
            known_metrics = {}
        
        # These would normally come from financial statements
        # For now, provide structure for future data integration
        
        financial_health_fields = [
            ("pe_ratio", "Price-to-Earnings ratio"),
            ("debt_equity_ratio", "Debt-to-Equity ratio"),
            ("current_ratio", "Current ratio"),
            ("roe", "Return on Equity"),
            ("roa", "Return on Assets"),
            ("asset_turnover", "Asset turnover"),
            ("profit_margin", "Profit margin"),
            ("free_cash_flow_yield", "Free cash flow yield"),
        ]
        
        for metric_name, description in financial_health_fields:
            if metric_name in known_metrics:
                features[metric_name] = known_metrics[metric_name]
            else:
                features[metric_name] = np.nan
        
        return features
    
    def calculate_growth_metrics(
        self,
        ticker: str,
        revenue_history: Optional[list] = None,
        earnings_history: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Calculate growth metrics.
        
        Args:
            ticker: Stock ticker symbol
            revenue_history: List of revenue values over time
            earnings_history: List of earnings values over time
            
        Returns:
            Dict with growth metrics
        """
        features = {}
        
        # Revenue growth
        if revenue_history and len(revenue_history) >= 2:
            valid_revenues = [r for r in revenue_history if r is not None and r > 0]
            if len(valid_revenues) >= 2:
                revenue_growth = (valid_revenues[-1] - valid_revenues[-2]) / valid_revenues[-2]
                features["revenue_growth"] = revenue_growth
            else:
                features["revenue_growth"] = np.nan
        else:
            features["revenue_growth"] = np.nan
        
        # Earnings growth
        if earnings_history and len(earnings_history) >= 2:
            valid_earnings = [e for e in earnings_history if e is not None and e > 0]
            if len(valid_earnings) >= 2:
                earnings_growth = (valid_earnings[-1] - valid_earnings[-2]) / valid_earnings[-2]
                features["earnings_growth"] = earnings_growth
            else:
                features["earnings_growth"] = np.nan
        else:
            features["earnings_growth"] = np.nan
        
        return features
    
    def get_all_fundamental_features(
        self,
        ticker: str,
        sector_performance: Optional[Dict[str, float]] = None,
        financial_metrics: Optional[Dict[str, float]] = None,
        revenue_history: Optional[list] = None,
        earnings_history: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Get all fundamental features for a stock.
        
        Args:
            ticker: Stock ticker symbol
            sector_performance: Sector performance dict
            financial_metrics: Known financial metrics
            revenue_history: Revenue history
            earnings_history: Earnings history
            
        Returns:
            Dict with all fundamental features
        """
        if sector_performance is None:
            sector_performance = {}
        if financial_metrics is None:
            financial_metrics = {}
        
        features = {}
        
        # Company basics
        company_features = self.get_company_features(ticker)
        features.update(company_features)
        
        # Filing metrics
        filing_features = self.extract_filing_metrics(ticker)
        features.update(filing_features)
        
        # Financial health
        health_features = self.estimate_financial_health(ticker, financial_metrics)
        features.update(health_features)
        
        # Growth metrics
        growth_features = self.calculate_growth_metrics(
            ticker,
            revenue_history,
            earnings_history
        )
        features.update(growth_features)
        
        # Sector rotation (if sector is available)
        if "sector" in features and sector_performance:
            sector_rotation = self.calculate_sector_rotation(
                features["sector"],
                sector_performance
            )
            features["sector_rotation"] = sector_rotation
        else:
            features["sector_rotation"] = np.nan
        
        return features


if __name__ == "__main__":
    logger.add(lambda msg: print(msg, end=''))
    
    # Example usage
    extractor = FundamentalFeatures()
    
    print("\n" + "=" * 80)
    print("FUNDAMENTAL FEATURES FOR AAPL")
    print("=" * 80)
    
    features = extractor.get_all_fundamental_features("AAPL")
    
    for key, value in features.items():
        if isinstance(value, (int, float)) and not pd.isna(value):
            print(f"{key:30s}: {value:12.4f}")
        else:
            print(f"{key:30s}: {value}")

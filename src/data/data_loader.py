"""Data loader for pulling from Financial Data Aggregator database."""
from datetime import date
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from loguru import logger

from src.config.config import DATABASE_URL, ALL_TICKERS


class DataLoader:
    """Load stock price and metadata from financial_data_aggregator database."""
    
    def __init__(self):
        """Initialize database connection."""
        self.engine = create_engine(DATABASE_URL)
        logger.info(f"Initialized DataLoader with database: {DATABASE_URL}")
    
    def load_stock_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Load daily stock prices for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        query = """
            SELECT 
                d.date,
                f.open_price as open,
                f.high_price as high,
                f.low_price as low,
                f.close_price as close,
                f.adjusted_close,
                f.volume
            FROM fact_stock_price f
            JOIN dim_company c ON f.company_id = c.company_id
            JOIN dim_date d ON f.date_id = d.date_id
            WHERE c.ticker = :ticker
              AND d.date >= :start_date
              AND d.date <= :end_date
            ORDER BY d.date ASC
        """
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(query),
                    conn,
                    params={
                        "ticker": ticker,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                )
            
            if df.empty:
                logger.warning(f"No price data found for {ticker} in [{start_date}, {end_date}]")
                return pd.DataFrame()
            
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")
            
        except Exception as e:
            logger.error(f"Error loading prices for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def load_company_metadata(self, ticker: str) -> Optional[dict]:
        """
        Load company metadata.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with company info (name, sector, industry, exchange_id) or None
        """
        query = """
            SELECT 
                c.company_id,
                c.ticker,
                c.company_name,
                c.sector,
                c.industry,
                c.exchange_id,
                e.exchange_code,
                e.exchange_name
            FROM dim_company c
            LEFT JOIN dim_exchange e ON c.exchange_id = e.exchange_id
            WHERE c.ticker = :ticker
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"ticker": ticker}).fetchone()
            
            if result:
                return {
                    "company_id": result[0],
                    "ticker": result[1],
                    "name": result[2],
                    "sector": result[3],
                    "industry": result[4],
                    "exchange_id": result[5],
                    "exchange_code": result[6],
                    "exchange_name": result[7],
                }
            else:
                logger.warning(f"No metadata found for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading metadata for {ticker}: {str(e)}")
            return None
    
    def load_sec_filings(
        self,
        ticker: str,
        limit: int = 5,
    ) -> pd.DataFrame:
        """
        Load SEC filing metadata and text.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of filings to return
            
        Returns:
            DataFrame with filing info and text content
        """
        query = """
            SELECT 
                f.filing_id,
                c.ticker,
                ft.filing_type,
                d.date as filing_date,
                f.filing_text,
                f.filing_url,
                f.filing_size
            FROM fact_sec_filing f
            JOIN dim_company c ON f.company_id = c.company_id
            JOIN dim_filing_type ft ON f.filing_type_id = ft.filing_type_id
            JOIN dim_date d ON f.date_id = d.date_id
            WHERE c.ticker = :ticker
              AND f.filing_text IS NOT NULL
            ORDER BY d.date DESC
            LIMIT :limit
        """
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(query),
                    conn,
                    params={"ticker": ticker, "limit": limit}
                )
            
            if df.empty:
                logger.warning(f"No SEC filings found for {ticker}")
                return pd.DataFrame()
            
            df["filing_date"] = pd.to_datetime(df["filing_date"])
            return df
            
        except Exception as e:
            logger.error(f"Error loading SEC filings for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def load_economic_indicator(
        self,
        indicator_code: str,
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """
        Load an economic indicator time series from the aggregator DB.

        Args:
            indicator_code: FRED indicator code (e.g. 'FEDFUNDS', 'VIXCLS')
            start_date: Start date
            end_date: End date

        Returns:
            Date-indexed Series of indicator values
        """
        query = """
            SELECT
                d.date,
                f.value
            FROM fact_economic_indicator f
            JOIN dim_economic_indicator ei ON f.indicator_id = ei.indicator_id
            JOIN dim_date d ON f.date_id = d.date_id
            WHERE ei.indicator_code = :indicator_code
              AND d.date >= :start_date
              AND d.date <= :end_date
            ORDER BY d.date ASC
        """

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(query),
                    conn,
                    params={
                        "indicator_code": indicator_code,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )

            if df.empty:
                logger.warning(
                    f"No data for indicator {indicator_code} in [{start_date}, {end_date}]"
                )
                return pd.Series(dtype=float)

            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.set_index("date")["value"]

        except Exception as e:
            logger.error(f"Error loading indicator {indicator_code}: {e}")
            return pd.Series(dtype=float)

    def load_vix(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """
        Load VIX time series.

        Tries the aggregator DB first (indicator_code='VIXCLS').
        Falls back to yfinance ('^VIX') if not available in DB.

        Returns:
            Date-indexed Series of VIX closing values, forward-filled.
        """
        # Try DB first
        vix = self.load_economic_indicator("VIXCLS", start_date, end_date)

        if not vix.empty:
            logger.info(f"Loaded VIX from DB: {len(vix)} observations")
            return vix.asfreq("B").ffill()

        # Fallback to yfinance
        logger.info("VIX not in DB, falling back to yfinance")
        try:
            import yfinance as yf

            ticker = yf.Ticker("^VIX")
            df = ticker.history(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )

            if df.empty:
                logger.warning("yfinance returned no VIX data")
                return pd.Series(dtype=float)

            vix = df["Close"].rename("VIX")
            vix.index = vix.index.tz_localize(None)
            logger.info(f"Loaded VIX from yfinance: {len(vix)} observations")
            return vix

        except ImportError:
            logger.error("yfinance not installed, cannot load VIX")
            return pd.Series(dtype=float)
        except Exception as e:
            logger.error(f"Error loading VIX from yfinance: {e}")
            return pd.Series(dtype=float)

    def load_fedfunds(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """
        Load Federal Funds Rate from the aggregator DB.

        Always available (FEDFUNDS is in the aggregator pipeline).
        Used as alternative macro feature in robustness tests.

        Returns:
            Date-indexed Series of FEDFUNDS values, forward-filled.
        """
        fedfunds = self.load_economic_indicator("FEDFUNDS", start_date, end_date)

        if fedfunds.empty:
            logger.warning("No FEDFUNDS data found")
            return fedfunds

        logger.info(f"Loaded FEDFUNDS: {len(fedfunds)} observations")
        return fedfunds.asfreq("B").ffill() if not fedfunds.empty else fedfunds

    def load_bond_yields(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Load treasury bond yields and compute the 10Y-2Y spread.

        Queries fact_bond_price joined with dim_bond for treasury yields.
        Pipeline has [3MO, 2Y, 10Y, 30Y].

        Returns:
            Date-indexed DataFrame with columns:
                yield_2y, yield_10y, spread_10y_2y
        """
        query = """
            SELECT
                d.date,
                b.bond_type,
                f.yield_percent
            FROM fact_bond_price f
            JOIN dim_bond b ON f.bond_id = b.bond_id
            JOIN dim_date d ON f.date_id = d.date_id
            WHERE b.bond_type IN ('Treasury', 'Government')
              AND d.date >= :start_date
              AND d.date <= :end_date
            ORDER BY d.date ASC
        """

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(query),
                    conn,
                    params={
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )

            if df.empty:
                logger.warning(f"No bond yield data in [{start_date}, {end_date}]")
                return pd.DataFrame()

            df["date"] = pd.to_datetime(df["date"])
            df["yield_percent"] = pd.to_numeric(df["yield_percent"], errors="coerce")

            # Pivot to get yields by maturity
            pivot = df.pivot_table(
                index="date",
                columns="bond_type",
                values="yield_percent",
                aggfunc="mean",
            )

            # Try to build spread from available columns
            # Bond types may be stored as e.g. 'US_2Y', 'US_10Y' or just maturity labels
            result = pd.DataFrame(index=pivot.index)

            # Look for 2Y and 10Y yields in column names
            for col in pivot.columns:
                col_lower = str(col).lower()
                if "2y" in col_lower or "2yr" in col_lower:
                    result["yield_2y"] = pivot[col]
                elif "10y" in col_lower or "10yr" in col_lower:
                    result["yield_10y"] = pivot[col]

            # If we couldn't match by name, use first two numeric columns
            if "yield_2y" not in result.columns and len(pivot.columns) >= 2:
                result["yield_2y"] = pivot.iloc[:, 0]
                result["yield_10y"] = pivot.iloc[:, 1]

            if "yield_2y" in result.columns and "yield_10y" in result.columns:
                result["spread_10y_2y"] = result["yield_10y"] - result["yield_2y"]
            else:
                logger.warning("Could not compute 10Y-2Y spread, insufficient bond data")
                if not result.empty:
                    result["spread_10y_2y"] = np.nan

            # Forward-fill for daily alignment
            result = result.ffill()
            logger.info(f"Loaded bond yields: {len(result)} observations")
            return result

        except Exception as e:
            logger.error(f"Error loading bond yields: {e}")
            return pd.DataFrame()

    def verify_data_availability(self, ticker: str) -> Tuple[bool, str]:
        """
        Verify that data is available for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (is_available, message)
        """
        # Check company exists
        metadata = self.load_company_metadata(ticker)
        if not metadata:
            return False, f"Company {ticker} not found in database"
        
        # Check prices exist
        prices = self.load_stock_prices(
            ticker,
            date(2022, 1, 1),
            date(2025, 12, 31)
        )
        if prices.empty:
            return False, f"No price data available for {ticker}"
        
        return True, f"Data available for {ticker} ({len(prices)} days)"
    
    def validate_universe(self, tickers: List[str]) -> dict:
        """
        Validate that all tickers in universe have data.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict with availability status for each ticker
        """
        availability = {}
        for ticker in tickers:
            is_available, message = self.verify_data_availability(ticker)
            availability[ticker] = {
                "available": is_available,
                "message": message
            }
            
            if is_available:
                logger.info(f"✓ {ticker}")
            else:
                logger.warning(f"✗ {ticker}: {message}")
        
        available_count = sum(1 for v in availability.values() if v["available"])
        logger.info(f"\n{available_count}/{len(tickers)} tickers have data")
        
        return availability


if __name__ == "__main__":
    # Quick test
    from src.config.config import ALL_TICKERS
    
    logger.add("logs/data_loader_test.log")
    
    loader = DataLoader()
    
    # Test with a few tickers
    test_tickers = ALL_TICKERS[:5]
    availability = loader.validate_universe(test_tickers)

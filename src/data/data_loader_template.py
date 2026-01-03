"""
Template for data loading from financial_data_aggregator SQLite database.

This shows how to connect to the SQLite database and load stock price data,
company metadata, and SEC filings.

NOTE: This is a TEMPLATE. Implement src/data/data_loader.py with actual loading logic.
"""

from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
import os

from config import DATABASE_URL, TRAINING_START, TRAINING_END


def get_db_connection():
    """
    Create database connection from DATABASE_URL.
    
    Returns:
        SQLAlchemy engine instance
    """
    engine = create_engine(DATABASE_URL)
    return engine


def load_stock_prices(
    tickers: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load daily price data for stocks from financial_data_aggregator.
    
    Args:
        tickers: List of stock tickers to load
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        DataFrame with columns: [ticker, date, open, high, low, close, volume]
    """
    engine = get_db_connection()
    
    # Convert dates to strings for SQL
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    
    # Placeholder SQL - adjust based on actual schema
    placeholders = ','.join(['?' for _ in tickers])
    query = f"""
        SELECT 
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume
        FROM fact_stock_price
        WHERE ticker IN ({placeholders})
        AND date >= '{start_str}'
        AND date <= '{end_str}'
        ORDER BY ticker, date
    """
    
    # For SQLite, use parameterized query
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT 
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM fact_stock_price
                WHERE date >= :start_date
                AND date <= :end_date
                ORDER BY ticker, date
            """),
            conn,
            params={"start_date": start_str, "end_date": end_str}
        )
    
    return df


def load_company_metadata(tickers: List[str]) -> pd.DataFrame:
    """
    Load company metadata (sector, industry, etc.) from financial_data_aggregator.
    
    Args:
        tickers: List of stock tickers
        
    Returns:
        DataFrame with columns: [ticker, sector, industry, name, ...]
    """
    engine = get_db_connection()
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT 
                    ticker,
                    sector,
                    industry,
                    name
                FROM dim_company
                WHERE ticker IN :tickers
            """),
            conn,
            params={"tickers": tuple(tickers)}
        )
    
    return df


def load_sec_filings(ticker: str, limit: int = 10) -> pd.DataFrame:
    """
    Load SEC filing documents for LLM analysis.
    
    Args:
        ticker: Stock ticker
        limit: Maximum number of filings to load
        
    Returns:
        DataFrame with columns: [filing_date, filing_type, text, ...]
    """
    engine = get_db_connection()
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT 
                    filing_date,
                    filing_type,
                    text
                FROM fact_sec_filing
                WHERE ticker = :ticker
                ORDER BY filing_date DESC
                LIMIT :limit
            """),
            conn,
            params={"ticker": ticker, "limit": limit}
        )
    
    return df


def validate_data(df: pd.DataFrame, ticker: str) -> bool:
    """
    Validate loaded data has no major gaps.
    
    Args:
        df: DataFrame to validate
        ticker: Ticker being validated (for error messages)
        
    Returns:
        True if data is valid, False if major issues found
    """
    # Check for nulls in critical columns
    critical_cols = ['date', 'close', 'volume']
    for col in critical_cols:
        if df[col].isnull().sum() > 0:
            print(f"WARNING: {ticker} has {df[col].isnull().sum()} nulls in {col}")
            return False
    
    # Check for consecutive dates (allow weekends/holidays)
    df_sorted = df.sort_values('date')
    date_diffs = df_sorted['date'].diff()
    
    # Max expected gap is 4 days (Fri-Mon)
    max_gap = pd.Timedelta(days=4)
    if (date_diffs > max_gap).any():
        print(f"WARNING: {ticker} has date gaps > 4 days")
        return False
    
    return True


# Example usage
if __name__ == "__main__":
    # Example: Load Apple price data for 2024
    tickers = ["AAPL", "MSFT"]
    start = date(2024, 1, 1)
    end = date(2024, 12, 31)
    
    try:
        prices = load_stock_prices(tickers, start, end)
        print(f"Loaded {len(prices)} price points")
        print(prices.head())
        
        # Validate
        for ticker in tickers:
            ticker_data = prices[prices['ticker'] == ticker]
            is_valid = validate_data(ticker_data, ticker)
            print(f"{ticker} valid: {is_valid}")
        
        # Load metadata
        metadata = load_company_metadata(tickers)
        print(f"\nCompany metadata:")
        print(metadata)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure financial_data_aggregator has populated financial_data.db")

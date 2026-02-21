"""
Stock Screener Integration Client.

Connects to the stock-screener SQLite database to fetch screened stocks
for use in the signal generation pipeline.

This enables filtering the trading universe to only include stocks that
pass certain fundamental/technical screening criteria.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ScreenedStock:
    """A stock that passed screening criteria."""
    ticker: str
    name: str
    sector: str
    country: str
    price: float
    market_cap: float
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    volatility: Optional[float] = None
    strategy: str = ""  # "dividend" or "volatility"
    score: float = 0.0  # Screening score (higher = better opportunity)
    reason: str = ""


class ScreenerClient:
    """
    Client for querying the stock-screener database.
    
    Provides filtered stock universes based on screening criteria.
    """
    
    DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "stock-screener" / "stock_screener.db"
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize screener client.
        
        Args:
            db_path: Path to stock_screener.db. Uses default if not provided.
        """
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self._conn = None
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Stock screener database not found: {self.db_path}")
            
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        
        return self._conn
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # === Universe Queries ===
    
    def get_all_tickers(self, limit: int = 500) -> List[str]:
        """
        Get all tickers in the screener database.
        
        Args:
            limit: Maximum number of tickers to return
            
        Returns:
            List of ticker symbols
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT ticker 
            FROM stocks 
            ORDER BY ticker 
            LIMIT ?
        """, (limit,))
        
        return [row['ticker'] for row in cursor.fetchall()]
    
    def get_dividend_stocks(
        self,
        min_yield: float = 0.03,
        max_pe: float = 20.0,
        min_market_cap: float = 1e9,
        limit: int = 100
    ) -> List[ScreenedStock]:
        """
        Get stocks that pass dividend screening criteria.
        
        Args:
            min_yield: Minimum dividend yield (0.03 = 3%)
            max_pe: Maximum P/E ratio
            min_market_cap: Minimum market cap in EUR
            limit: Maximum number of results
            
        Returns:
            List of ScreenedStock objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                s.ticker,
                s.name,
                s.sector,
                s.country,
                d.price_eur,
                d.market_cap_eur,
                d.dividend_yield,
                d.pe_ratio,
                d.beta,
                d.volatility
            FROM stocks s
            JOIN (
                SELECT ticker, MAX(date) as max_date
                FROM stock_data
                GROUP BY ticker
            ) latest ON s.ticker = latest.ticker
            JOIN stock_data d ON s.ticker = d.ticker AND d.date = latest.max_date
            WHERE d.dividend_yield >= ?
              AND d.pe_ratio > 0 
              AND d.pe_ratio <= ?
              AND d.market_cap_eur >= ?
            ORDER BY d.dividend_yield DESC
            LIMIT ?
        """, (min_yield, max_pe, min_market_cap, limit))
        
        results = []
        for row in cursor.fetchall():
            # Calculate a simple score (higher yield + lower PE = better)
            yield_score = min(row['dividend_yield'] / 0.10, 1.0) * 50  # Max 50 points for yield
            pe_score = max(0, (max_pe - row['pe_ratio']) / max_pe) * 50  # Max 50 points for value
            score = yield_score + pe_score
            
            results.append(ScreenedStock(
                ticker=row['ticker'],
                name=row['name'] or row['ticker'],
                sector=row['sector'] or 'Unknown',
                country=row['country'] or 'Unknown',
                price=row['price_eur'] or 0,
                market_cap=row['market_cap_eur'] or 0,
                dividend_yield=row['dividend_yield'],
                pe_ratio=row['pe_ratio'],
                beta=row['beta'],
                volatility=row['volatility'],
                strategy='dividend',
                score=score,
                reason=f"Yield: {row['dividend_yield']*100:.1f}%, P/E: {row['pe_ratio']:.1f}"
            ))
        
        logger.info(f"Found {len(results)} dividend stocks")
        return results
    
    def get_volatility_stocks(
        self,
        min_beta: float = 1.2,
        min_volatility: float = 0.25,
        max_pe: float = 30.0,
        min_market_cap: float = 5e8,
        limit: int = 100
    ) -> List[ScreenedStock]:
        """
        Get high-volatility stocks for momentum trading.
        
        Args:
            min_beta: Minimum beta
            min_volatility: Minimum volatility
            max_pe: Maximum P/E ratio
            min_market_cap: Minimum market cap in EUR
            limit: Maximum number of results
            
        Returns:
            List of ScreenedStock objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                s.ticker,
                s.name,
                s.sector,
                s.country,
                d.price_eur,
                d.market_cap_eur,
                d.dividend_yield,
                d.pe_ratio,
                d.beta,
                d.volatility,
                d.year_high_eur
            FROM stocks s
            JOIN (
                SELECT ticker, MAX(date) as max_date
                FROM stock_data
                GROUP BY ticker
            ) latest ON s.ticker = latest.ticker
            JOIN stock_data d ON s.ticker = d.ticker AND d.date = latest.max_date
            WHERE (d.beta >= ? OR d.volatility >= ?)
              AND d.market_cap_eur >= ?
              AND (d.pe_ratio <= ? OR d.pe_ratio IS NULL OR d.pe_ratio = 0)
            ORDER BY d.beta DESC, d.volatility DESC
            LIMIT ?
        """, (min_beta, min_volatility, min_market_cap, max_pe, limit))
        
        results = []
        for row in cursor.fetchall():
            # Calculate opportunity score based on drop from high
            drop_from_high = 0
            if row['year_high_eur'] and row['price_eur'] and row['year_high_eur'] > 0:
                drop_from_high = (row['price_eur'] - row['year_high_eur']) / row['year_high_eur']
            
            # Score: higher beta/volatility + bigger drop = better opportunity
            beta_score = min((row['beta'] or 0) / 2.0, 1.0) * 40
            vol_score = min((row['volatility'] or 0) / 0.5, 1.0) * 30
            drop_score = min(abs(drop_from_high) / 0.3, 1.0) * 30
            score = beta_score + vol_score + drop_score
            
            results.append(ScreenedStock(
                ticker=row['ticker'],
                name=row['name'] or row['ticker'],
                sector=row['sector'] or 'Unknown',
                country=row['country'] or 'Unknown',
                price=row['price_eur'] or 0,
                market_cap=row['market_cap_eur'] or 0,
                dividend_yield=row['dividend_yield'],
                pe_ratio=row['pe_ratio'],
                beta=row['beta'],
                volatility=row['volatility'],
                strategy='volatility',
                score=score,
                reason=f"Beta: {row['beta']:.2f}, Vol: {(row['volatility'] or 0)*100:.0f}%"
            ))
        
        logger.info(f"Found {len(results)} volatility stocks")
        return results
    
    def get_combined_opportunities(
        self,
        dividend_limit: int = 50,
        volatility_limit: int = 50
    ) -> List[ScreenedStock]:
        """
        Get combined list of dividend and volatility opportunities.
        
        Args:
            dividend_limit: Max dividend stocks
            volatility_limit: Max volatility stocks
            
        Returns:
            Combined list, deduplicated by ticker
        """
        dividend_stocks = self.get_dividend_stocks(limit=dividend_limit)
        volatility_stocks = self.get_volatility_stocks(limit=volatility_limit)
        
        # Combine and deduplicate
        seen = set()
        combined = []
        
        for stock in dividend_stocks + volatility_stocks:
            if stock.ticker not in seen:
                seen.add(stock.ticker)
                combined.append(stock)
        
        # Sort by score
        combined.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Combined universe: {len(combined)} unique stocks")
        return combined
    
    def get_screened_tickers(
        self,
        strategies: List[str] = None,
        min_score: float = 0,
        limit: int = 100
    ) -> List[str]:
        """
        Get just the ticker symbols from screening.
        
        Convenience method for use in signal pipeline.
        
        Args:
            strategies: List of strategies to include ("dividend", "volatility", or both)
            min_score: Minimum screening score
            limit: Maximum number of tickers
            
        Returns:
            List of ticker symbols
        """
        strategies = strategies or ["dividend", "volatility"]
        
        all_stocks = []
        
        if "dividend" in strategies:
            all_stocks.extend(self.get_dividend_stocks(limit=limit))
        
        if "volatility" in strategies:
            all_stocks.extend(self.get_volatility_stocks(limit=limit))
        
        # Deduplicate and filter
        seen = set()
        tickers = []
        
        for stock in sorted(all_stocks, key=lambda x: x.score, reverse=True):
            if stock.ticker not in seen and stock.score >= min_score:
                seen.add(stock.ticker)
                tickers.append(stock.ticker)
                
                if len(tickers) >= limit:
                    break
        
        return tickers
    
    # === Alert Queries ===
    
    def get_recent_alerts(self, days: int = 7) -> List[Dict]:
        """
        Get recent screening alerts.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of alert dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT 
                a.ticker,
                a.strategy,
                a.alert_date,
                a.price_eur,
                a.reason,
                a.metrics,
                s.name,
                s.sector
            FROM alerts a
            LEFT JOIN stocks s ON a.ticker = s.ticker
            WHERE a.alert_date >= ?
            ORDER BY a.alert_timestamp DESC
        """, (cutoff,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # === Statistics ===
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the screener database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM stocks")
        total_stocks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM stock_data")
        stocks_with_data = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(date) FROM stock_data")
        latest_date = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_date >= date('now', '-7 days')")
        recent_alerts = cursor.fetchone()[0]
        
        # Count by strategy
        dividend_count = len(self.get_dividend_stocks(limit=1000))
        volatility_count = len(self.get_volatility_stocks(limit=1000))
        
        return {
            "total_stocks": total_stocks,
            "stocks_with_data": stocks_with_data,
            "latest_data_date": latest_date,
            "recent_alerts_7d": recent_alerts,
            "dividend_opportunities": dividend_count,
            "volatility_opportunities": volatility_count,
            "database_path": str(self.db_path)
        }


# === Convenience Functions ===

def get_screened_universe(
    strategies: List[str] = None,
    limit: int = 100
) -> List[str]:
    """
    Quick function to get screened tickers for signal pipeline.
    
    Usage:
        from src.integrations.screener_client import get_screened_universe
        tickers = get_screened_universe(strategies=["dividend"], limit=50)
    """
    try:
        with ScreenerClient() as client:
            return client.get_screened_tickers(strategies=strategies, limit=limit)
    except FileNotFoundError:
        logger.warning("Stock screener database not found, returning empty list")
        return []
    except Exception as e:
        logger.error(f"Error fetching screened universe: {e}")
        return []


def get_screener_stats() -> Dict:
    """Get screener database statistics."""
    try:
        with ScreenerClient() as client:
            return client.get_database_stats()
    except FileNotFoundError:
        return {"error": "Stock screener database not found"}
    except Exception as e:
        return {"error": str(e)}

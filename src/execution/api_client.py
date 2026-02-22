"""
API Client for Trading Simulator Integration

HTTP client for interacting with the Trading Simulator REST API.
Handles order execution, portfolio queries, and error handling.
"""

from typing import List, Dict, Optional, Any
import requests
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderResponse:
    """Response from order execution."""
    status: str
    ticker: str
    asset_type: str
    order_type: str
    quantity: float
    price: float
    fee: float
    total_cost: float
    timestamp: str
    message: str
    order_id: Optional[int] = None


@dataclass
class Holding:
    """Portfolio holding information."""
    ticker: str
    asset_type: str
    quantity: float
    avg_cost: float
    current_price: float
    total_value: float
    unrealized_pnl: float
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None


@dataclass
class Portfolio:
    """Portfolio information."""
    id: int
    name: str
    cash_balance: float
    total_value: float
    total_return: float
    total_return_pct: float


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None
    total_return: Optional[float] = None


class TradingSimulatorClient:
    """
    Client for Trading Simulator REST API.
    
    Provides methods to:
    - Execute buy/sell orders
    - Query portfolio state
    - Get holdings
    - Retrieve performance metrics
    - Lookup current prices
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Trading Simulator API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        logger.info(f"TradingSimulatorClient initialized: {base_url}")
    
    def place_buy_order(
        self,
        portfolio_id: int,
        ticker: str,
        quantity: float,
        asset_type: str = "stock"
    ) -> OrderResponse:
        """
        Place a buy order.
        
        Args:
            portfolio_id: Portfolio ID
            ticker: Stock ticker
            quantity: Number of shares to buy
            asset_type: Asset type (stock, crypto, bond, commodity)
        
        Returns:
            OrderResponse with execution details
        
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/orders/{portfolio_id}/buy"
        payload = {
            "ticker": ticker,
            "asset_type": asset_type,
            "quantity": quantity
        }
        
        logger.debug(f"Placing buy order: {ticker} x {quantity}")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            order = OrderResponse(**data)
            logger.info(
                f"Buy order executed: {ticker} x {order.quantity} @ ${order.price:.2f} "
                f"(Order #{order.order_id})"
            )
            
            return order
            
        except requests.RequestException as e:
            logger.error(f"Buy order failed for {ticker}: {e}")
            raise
    
    def place_sell_order(
        self,
        portfolio_id: int,
        ticker: str,
        quantity: float,
        asset_type: str = "stock"
    ) -> OrderResponse:
        """
        Place a sell order.
        
        Args:
            portfolio_id: Portfolio ID
            ticker: Stock ticker
            quantity: Number of shares to sell
            asset_type: Asset type (stock, crypto, bond, commodity)
        
        Returns:
            OrderResponse with execution details
        
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/orders/{portfolio_id}/sell"
        payload = {
            "ticker": ticker,
            "asset_type": asset_type,
            "quantity": quantity
        }
        
        logger.debug(f"Placing sell order: {ticker} x {quantity}")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            order = OrderResponse(**data)
            logger.info(
                f"Sell order executed: {ticker} x {order.quantity} @ ${order.price:.2f} "
                f"(Order #{order.order_id})"
            )
            
            return order
            
        except requests.RequestException as e:
            logger.error(f"Sell order failed for {ticker}: {e}")
            raise
    
    def get_holdings(self, portfolio_id: int) -> List[Holding]:
        """
        Get current holdings in portfolio.
        
        Args:
            portfolio_id: Portfolio ID
        
        Returns:
            List of Holding objects
        
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/orders/{portfolio_id}/holdings"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            holdings = [Holding(**h) for h in data]
            logger.debug(f"Retrieved {len(holdings)} holdings for portfolio {portfolio_id}")
            
            return holdings
            
        except requests.RequestException as e:
            logger.error(f"Failed to get holdings for portfolio {portfolio_id}: {e}")
            raise
    
    def get_portfolio(self, portfolio_id: int) -> Portfolio:
        """
        Get portfolio information.
        
        Args:
            portfolio_id: Portfolio ID
        
        Returns:
            Portfolio object with current state
        
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/portfolios/{portfolio_id}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            portfolio = Portfolio(**data)
            logger.debug(
                f"Portfolio {portfolio_id}: "
                f"Value=${portfolio.total_value:.2f}, "
                f"Cash=${portfolio.cash_balance:.2f}, "
                f"Return={portfolio.total_return_pct:.2%}"
            )
            
            return portfolio
            
        except requests.RequestException as e:
            logger.error(f"Failed to get portfolio {portfolio_id}: {e}")
            raise
    
    def get_quote(self, ticker: str) -> float:
        """
        Get current stock price.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            Current price
        
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/orders/quote/{ticker}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            price = float(data["price"])
            logger.debug(f"Quote for {ticker}: ${price:.2f}")
            
            return price
            
        except requests.RequestException as e:
            logger.error(f"Failed to get quote for {ticker}: {e}")
            raise
    
    def get_performance_metrics(self, portfolio_id: int) -> PerformanceMetrics:
        """
        Get portfolio performance metrics.
        
        Args:
            portfolio_id: Portfolio ID
        
        Returns:
            PerformanceMetrics object
        
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/analytics/{portfolio_id}/performance"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            metrics = PerformanceMetrics(**data)
            logger.debug(
                f"Performance metrics for portfolio {portfolio_id}: "
                f"Sharpe={metrics.sharpe_ratio:.2f}, "
                f"MaxDD={metrics.max_drawdown:.2%}"
            )
            
            return metrics
            
        except requests.RequestException as e:
            logger.error(f"Failed to get performance metrics for portfolio {portfolio_id}: {e}")
            raise
    
    def get_transaction_history(
        self,
        portfolio_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history.
        
        Args:
            portfolio_id: Portfolio ID
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
        
        Returns:
            List of transaction dictionaries
        
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/orders/{portfolio_id}/history"
        params = {"skip": skip, "limit": limit}
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            transactions = data.get("transactions", [])
            logger.debug(f"Retrieved {len(transactions)} transactions for portfolio {portfolio_id}")
            
            return transactions
            
        except requests.RequestException as e:
            logger.error(f"Failed to get transaction history for portfolio {portfolio_id}: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if Trading Simulator API is available.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            is_healthy = response.status_code == 200
            
            if is_healthy:
                logger.info("Trading Simulator API is healthy")
            else:
                logger.warning(f"Trading Simulator API returned status {response.status_code}")
            
            return is_healthy
            
        except requests.RequestException as e:
            logger.error(f"Trading Simulator API health check failed: {e}")
            return False
    
    def close(self):
        """Close the session."""
        self.session.close()
        logger.debug("TradingSimulatorClient session closed")

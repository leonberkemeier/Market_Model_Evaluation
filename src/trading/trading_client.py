"""
Trading API Client for Trading Simulator Integration.

Sends trade signals from model_regime_comparison to the Trading_Simulator API.
Handles portfolio management, order execution, and state tracking.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from .signal_generator import TradeSignal, SignalType


logger = logging.getLogger(__name__)


class AssetType(str, Enum):
    """Asset types supported by Trading Simulator."""
    STOCK = "stock"
    CRYPTO = "crypto"
    BOND = "bond"
    COMMODITY = "commodity"


@dataclass
class OrderResult:
    """Result of an order execution attempt."""
    success: bool
    order_id: Optional[int] = None
    ticker: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    total_cost: float = 0.0
    message: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "ticker": self.ticker,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "fee": self.fee,
            "total_cost": self.total_cost,
            "message": self.message,
            "error": self.error
        }


@dataclass
class PortfolioState:
    """Current state of a portfolio."""
    portfolio_id: int
    name: str
    current_cash: float
    nav: float
    total_return_pct: float
    holdings: List[str]  # List of tickers
    model_name: Optional[str] = None


class TradingClientConfig:
    """Configuration for the trading client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: int = 30,
        retry_attempts: int = 3,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.api_key = api_key


class TradingClient:
    """
    HTTP client for Trading Simulator API.
    
    Handles communication between model_regime_comparison and Trading_Simulator.
    Supports portfolio creation, order execution, and state queries.
    """
    
    def __init__(self, config: Optional[TradingClientConfig] = None):
        """
        Initialize trading client.
        
        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self.config = config or TradingClientConfig()
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        if self.config.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        # Cache for portfolio IDs by model name
        self._portfolio_cache: Dict[str, int] = {}
    
    # === Health & Connection ===
    
    def health_check(self) -> bool:
        """
        Check if Trading Simulator API is accessible.
        
        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/health",
                timeout=self.config.timeout
            )
            return response.status_code == 200
        except RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    # === Portfolio Management ===
    
    def create_portfolio(
        self,
        name: str,
        initial_capital: float,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        max_position_size: float = 15.0
    ) -> Optional[int]:
        """
        Create a new portfolio for a model.
        
        Args:
            name: Portfolio name
            initial_capital: Starting capital
            model_name: Associated model name (e.g., "xgboost", "linear")
            description: Optional description
            max_position_size: Max position size as % of portfolio
            
        Returns:
            Portfolio ID if created, None on failure.
        """
        payload = {
            "name": name,
            "initial_capital": initial_capital,
            "model_name": model_name,
            "description": description or f"Auto-managed portfolio for {model_name or 'manual'} model",
            "max_position_size": max_position_size
        }
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/api/portfolios",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code in (200, 201):
                data = response.json()
                portfolio_id = data.get("id")
                
                # Cache the portfolio ID
                if model_name and portfolio_id:
                    self._portfolio_cache[model_name] = portfolio_id
                
                logger.info(f"Created portfolio '{name}' with ID {portfolio_id}")
                return portfolio_id
            else:
                logger.error(f"Failed to create portfolio: {response.status_code} - {response.text}")
                return None
                
        except RequestException as e:
            logger.error(f"Error creating portfolio: {e}")
            return None
    
    def get_or_create_model_portfolio(
        self,
        model_name: str,
        initial_capital: float = 100000.0
    ) -> Optional[int]:
        """
        Get existing portfolio for a model or create one.
        
        Args:
            model_name: Model name (e.g., "xgboost", "linear", "cnn", "llm")
            initial_capital: Starting capital if creating new portfolio
            
        Returns:
            Portfolio ID
        """
        # Check cache first
        if model_name in self._portfolio_cache:
            return self._portfolio_cache[model_name]
        
        # Search for existing portfolio
        portfolios = self.list_portfolios()
        for p in portfolios:
            if p.get("model_name") == model_name:
                portfolio_id = p["id"]
                self._portfolio_cache[model_name] = portfolio_id
                return portfolio_id
        
        # Create new portfolio
        return self.create_portfolio(
            name=f"{model_name.upper()} Model Portfolio",
            initial_capital=initial_capital,
            model_name=model_name,
            description=f"Automated trading portfolio managed by {model_name} model"
        )
    
    def list_portfolios(self) -> List[Dict]:
        """
        List all portfolios.
        
        Returns:
            List of portfolio dictionaries.
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/portfolios",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("portfolios", [])
            else:
                logger.error(f"Failed to list portfolios: {response.status_code}")
                return []
                
        except RequestException as e:
            logger.error(f"Error listing portfolios: {e}")
            return []
    
    def get_portfolio(self, portfolio_id: int) -> Optional[PortfolioState]:
        """
        Get current portfolio state.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            PortfolioState or None on failure.
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/portfolios/{portfolio_id}",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Get holdings
                holdings = self.get_holdings(portfolio_id)
                holding_tickers = [h["ticker"] for h in holdings]
                
                return PortfolioState(
                    portfolio_id=data["id"],
                    name=data["name"],
                    current_cash=float(data["current_cash"]),
                    nav=float(data["nav"]),
                    total_return_pct=float(data["total_return_pct"]),
                    holdings=holding_tickers,
                    model_name=data.get("model_name")
                )
            else:
                logger.error(f"Failed to get portfolio {portfolio_id}: {response.status_code}")
                return None
                
        except RequestException as e:
            logger.error(f"Error getting portfolio: {e}")
            return None
    
    def get_holdings(self, portfolio_id: int) -> List[Dict]:
        """
        Get current holdings for a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            List of holding dictionaries.
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/orders/{portfolio_id}/holdings",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get holdings: {response.status_code}")
                return []
                
        except RequestException as e:
            logger.error(f"Error getting holdings: {e}")
            return []
    
    # === Order Execution ===
    
    def execute_buy(
        self,
        portfolio_id: int,
        ticker: str,
        quantity: int,
        asset_type: AssetType = AssetType.STOCK
    ) -> OrderResult:
        """
        Execute a buy order.
        
        Args:
            portfolio_id: Target portfolio
            ticker: Stock/asset ticker
            quantity: Number of shares
            asset_type: Type of asset
            
        Returns:
            OrderResult with execution details.
        """
        payload = {
            "ticker": ticker,
            "asset_type": asset_type.value,
            "quantity": quantity
        }
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/api/orders/{portfolio_id}/buy",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"BUY executed: {quantity} {ticker} @ ${float(data.get('price', 0)):.2f}")
                
                return OrderResult(
                    success=True,
                    order_id=data.get("order_id"),
                    ticker=data.get("ticker", ticker),
                    order_type="buy",
                    quantity=float(data.get("quantity", quantity)),
                    price=float(data.get("price", 0)),
                    fee=float(data.get("fee", 0)),
                    total_cost=float(data.get("total_cost", 0)),
                    message=data.get("message", "Order executed successfully")
                )
            else:
                error_msg = response.json().get("detail", response.text) if response.text else "Unknown error"
                logger.warning(f"BUY failed for {ticker}: {error_msg}")
                
                return OrderResult(
                    success=False,
                    ticker=ticker,
                    order_type="buy",
                    quantity=quantity,
                    error=error_msg,
                    message=f"Order failed: {error_msg}"
                )
                
        except RequestException as e:
            logger.error(f"Error executing buy: {e}")
            return OrderResult(
                success=False,
                ticker=ticker,
                order_type="buy",
                quantity=quantity,
                error=str(e),
                message=f"Connection error: {e}"
            )
    
    def execute_sell(
        self,
        portfolio_id: int,
        ticker: str,
        quantity: Optional[int] = None,
        asset_type: AssetType = AssetType.STOCK
    ) -> OrderResult:
        """
        Execute a sell order.
        
        Args:
            portfolio_id: Target portfolio
            ticker: Stock/asset ticker
            quantity: Number of shares (None = sell all)
            asset_type: Type of asset
            
        Returns:
            OrderResult with execution details.
        """
        # If no quantity specified, get current holding and sell all
        if quantity is None:
            holdings = self.get_holdings(portfolio_id)
            for h in holdings:
                if h["ticker"].upper() == ticker.upper():
                    quantity = int(float(h["quantity"]))
                    break
            
            if quantity is None or quantity <= 0:
                return OrderResult(
                    success=False,
                    ticker=ticker,
                    order_type="sell",
                    error="No position to sell",
                    message=f"No holding found for {ticker}"
                )
        
        payload = {
            "ticker": ticker,
            "asset_type": asset_type.value,
            "quantity": quantity
        }
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/api/orders/{portfolio_id}/sell",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"SELL executed: {quantity} {ticker} @ ${float(data.get('price', 0)):.2f}")
                
                return OrderResult(
                    success=True,
                    order_id=data.get("order_id"),
                    ticker=data.get("ticker", ticker),
                    order_type="sell",
                    quantity=float(data.get("quantity", quantity)),
                    price=float(data.get("price", 0)),
                    fee=float(data.get("fee", 0)),
                    total_cost=float(data.get("total_cost", 0)),
                    message=data.get("message", "Order executed successfully")
                )
            else:
                error_msg = response.json().get("detail", response.text) if response.text else "Unknown error"
                logger.warning(f"SELL failed for {ticker}: {error_msg}")
                
                return OrderResult(
                    success=False,
                    ticker=ticker,
                    order_type="sell",
                    quantity=quantity or 0,
                    error=error_msg,
                    message=f"Order failed: {error_msg}"
                )
                
        except RequestException as e:
            logger.error(f"Error executing sell: {e}")
            return OrderResult(
                success=False,
                ticker=ticker,
                order_type="sell",
                quantity=quantity or 0,
                error=str(e),
                message=f"Connection error: {e}"
            )
    
    # === Signal Execution ===
    
    def execute_signal(
        self,
        signal: TradeSignal,
        portfolio_id: int,
        asset_type: AssetType = AssetType.STOCK
    ) -> OrderResult:
        """
        Execute a trade signal.
        
        Args:
            signal: TradeSignal from SignalGenerator
            portfolio_id: Target portfolio
            asset_type: Type of asset
            
        Returns:
            OrderResult with execution details.
        """
        if signal.signal_type == SignalType.BUY:
            if signal.suggested_quantity is None or signal.suggested_quantity <= 0:
                # Calculate quantity from weight and portfolio
                state = self.get_portfolio(portfolio_id)
                if state and signal.current_price and signal.current_price > 0:
                    dollar_amount = state.current_cash * signal.suggested_weight
                    quantity = int(dollar_amount / signal.current_price)
                else:
                    return OrderResult(
                        success=False,
                        ticker=signal.ticker,
                        order_type="buy",
                        error="Cannot calculate quantity",
                        message="Missing price or portfolio state"
                    )
            else:
                quantity = signal.suggested_quantity
            
            return self.execute_buy(
                portfolio_id=portfolio_id,
                ticker=signal.ticker,
                quantity=quantity,
                asset_type=asset_type
            )
        
        elif signal.signal_type == SignalType.SELL:
            return self.execute_sell(
                portfolio_id=portfolio_id,
                ticker=signal.ticker,
                quantity=None,  # Sell entire position
                asset_type=asset_type
            )
        
        else:
            return OrderResult(
                success=False,
                ticker=signal.ticker,
                error=f"Unsupported signal type: {signal.signal_type}",
                message="Only BUY and SELL signals are executable"
            )
    
    def execute_signals(
        self,
        signals: List[TradeSignal],
        portfolio_id: int,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute multiple trade signals.
        
        Args:
            signals: List of TradeSignal objects
            portfolio_id: Target portfolio
            dry_run: If True, simulate without executing
            
        Returns:
            Summary of execution results.
        """
        results = {
            "total_signals": len(signals),
            "executed": 0,
            "failed": 0,
            "skipped": 0,
            "buy_orders": [],
            "sell_orders": [],
            "errors": []
        }
        
        if not signals:
            logger.info("No signals to execute")
            return results
        
        # Execute SELL signals first (free up capital)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        for signal in sell_signals:
            if dry_run:
                logger.info(f"[DRY RUN] Would SELL {signal.ticker}")
                results["skipped"] += 1
                continue
            
            result = self.execute_signal(signal, portfolio_id)
            
            if result.success:
                results["executed"] += 1
                results["sell_orders"].append(result.to_dict())
            else:
                results["failed"] += 1
                results["errors"].append({
                    "ticker": signal.ticker,
                    "type": "sell",
                    "error": result.error
                })
        
        for signal in buy_signals:
            if dry_run:
                logger.info(f"[DRY RUN] Would BUY {signal.suggested_quantity or '?'} {signal.ticker}")
                results["skipped"] += 1
                continue
            
            result = self.execute_signal(signal, portfolio_id)
            
            if result.success:
                results["executed"] += 1
                results["buy_orders"].append(result.to_dict())
            else:
                results["failed"] += 1
                results["errors"].append({
                    "ticker": signal.ticker,
                    "type": "buy",
                    "error": result.error
                })
        
        # Log summary
        logger.info(
            f"Signal execution complete: {results['executed']} executed, "
            f"{results['failed']} failed, {results['skipped']} skipped"
        )
        
        return results
    
    # === Utility Methods ===
    
    def get_current_holdings_tickers(self, portfolio_id: int) -> List[str]:
        """
        Get list of tickers currently held in a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            List of ticker symbols.
        """
        holdings = self.get_holdings(portfolio_id)
        return [h["ticker"] for h in holdings]
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# === Convenience Functions ===

def create_client(
    base_url: str = "http://localhost:8001",
    api_key: Optional[str] = None
) -> TradingClient:
    """Create a trading client with common defaults."""
    config = TradingClientConfig(base_url=base_url, api_key=api_key)
    return TradingClient(config)

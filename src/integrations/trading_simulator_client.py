"""
Trading Simulator Integration Client

Communicates with Trading_Simulator backend to:
- Create portfolios
- Execute trades
- Track performance
- Rebalance positions
"""

from typing import Dict, List, Optional
from loguru import logger
import requests
import json

from ..portfolio.risk_profiles import PortfolioAllocation, PortfolioPosition


class TradingSimulatorClient:
    """Client for Trading Simulator backend API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize Trading Simulator client.
        
        Args:
            base_url: Trading Simulator backend URL
        """
        self.base_url = base_url
        self.logger = logger.bind(module="trading_simulator_client")
    
    def health_check(self) -> bool:
        """
        Check if Trading Simulator is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    def create_portfolio(
        self,
        portfolio: PortfolioAllocation,
        portfolio_name: Optional[str] = None,
        description: str = "",
    ) -> Dict:
        """
        Create a new portfolio in Trading Simulator.
        
        Args:
            portfolio: PortfolioAllocation object
            portfolio_name: Name for portfolio (auto-generated if None)
            description: Portfolio description
            
        Returns:
            Response with portfolio_id
        """
        try:
            if portfolio_name is None:
                portfolio_name = f"Model_{portfolio.regime_at_construction}_{portfolio.construction_date}"
            
            # Convert positions to API format
            positions_data = [
                {
                    "ticker": pos.ticker,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "asset_type": pos.asset_type,
                    "sector": pos.sector,
                }
                for pos in portfolio.positions
            ]
            
            payload = {
                "name": portfolio_name,
                "description": description,
                "initial_capital": portfolio.total_value,
                "cash": portfolio.cash,
                "risk_profile": portfolio.risk_profile_type.value,
                "positions": positions_data,
                "metadata": {
                    "regime": portfolio.regime_at_construction,
                    "construction_date": portfolio.construction_date,
                    "llm_recommendations": portfolio.llm_recommendations_used,
                    "expected_var_95": portfolio.portfolio_expected_var_95,
                    "expected_es_95": portfolio.portfolio_expected_es_95,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/portfolios",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            
            result = response.json()
            portfolio_id = result.get("id")
            
            self.logger.info(f"✓ Portfolio created: {portfolio_id}")
            return result
        
        except Exception as e:
            self.logger.error(f"Failed to create portfolio: {e}")
            raise
    
    def get_portfolio(self, portfolio_id: str) -> Dict:
        """
        Get portfolio details.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio data
        """
        try:
            response = requests.get(
                f"{self.base_url}/portfolios/{portfolio_id}",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self.logger.error(f"Failed to get portfolio: {e}")
            raise
    
    def execute_buy_order(
        self,
        portfolio_id: str,
        ticker: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict:
        """
        Execute a buy order.
        
        Args:
            portfolio_id: Portfolio ID
            ticker: Asset ticker
            quantity: Number of shares
            price: Execution price (None = current market price)
            
        Returns:
            Order confirmation
        """
        try:
            payload = {
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
            }
            
            response = requests.post(
                f"{self.base_url}/portfolios/{portfolio_id}/buy",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            
            self.logger.info(f"Buy order executed: {ticker} x {quantity}")
            return response.json()
        
        except Exception as e:
            self.logger.error(f"Buy order failed: {e}")
            raise
    
    def execute_sell_order(
        self,
        portfolio_id: str,
        ticker: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict:
        """
        Execute a sell order.
        
        Args:
            portfolio_id: Portfolio ID
            ticker: Asset ticker
            quantity: Number of shares
            price: Execution price (None = current market price)
            
        Returns:
            Order confirmation
        """
        try:
            payload = {
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
            }
            
            response = requests.post(
                f"{self.base_url}/portfolios/{portfolio_id}/sell",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            
            self.logger.info(f"Sell order executed: {ticker} x {quantity}")
            return response.json()
        
        except Exception as e:
            self.logger.error(f"Sell order failed: {e}")
            raise
    
    def rebalance_portfolio(
        self,
        portfolio_id: str,
        new_allocation: PortfolioAllocation,
    ) -> Dict:
        """
        Rebalance a portfolio to new allocation.
        
        Args:
            portfolio_id: Portfolio ID
            new_allocation: New allocation
            
        Returns:
            Rebalancing summary
        """
        try:
            # Get current portfolio
            current = self.get_portfolio(portfolio_id)
            current_positions = {pos["ticker"]: pos["quantity"] for pos in current["positions"]}
            
            # Calculate trades needed
            trades = []
            for new_pos in new_allocation.positions:
                current_qty = current_positions.get(new_pos.ticker, 0)
                diff = new_pos.quantity - current_qty
                
                if diff > 0:
                    trades.append({"type": "buy", "ticker": new_pos.ticker, "qty": diff})
                elif diff < 0:
                    trades.append({"type": "sell", "ticker": new_pos.ticker, "qty": -diff})
            
            # Execute trades
            for trade in trades:
                if trade["type"] == "buy":
                    self.execute_buy_order(portfolio_id, trade["ticker"], trade["qty"])
                else:
                    self.execute_sell_order(portfolio_id, trade["ticker"], trade["qty"])
            
            self.logger.info(f"Rebalance completed: {len(trades)} trades")
            return {"trades": trades}
        
        except Exception as e:
            self.logger.error(f"Rebalancing failed: {e}")
            raise
    
    def get_portfolio_metrics(self, portfolio_id: str) -> Dict:
        """
        Get portfolio performance metrics.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Metrics (NAV, Sharpe, drawdown, etc.)
        """
        try:
            response = requests.get(
                f"{self.base_url}/portfolios/{portfolio_id}/metrics",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise
    
    def get_holdings(self, portfolio_id: str) -> List[Dict]:
        """
        Get current holdings.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            List of positions
        """
        try:
            response = requests.get(
                f"{self.base_url}/portfolios/{portfolio_id}/holdings",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self.logger.error(f"Failed to get holdings: {e}")
            raise
    
    def get_transactions(self, portfolio_id: str, limit: int = 100) -> List[Dict]:
        """
        Get transaction history.
        
        Args:
            portfolio_id: Portfolio ID
            limit: Maximum transactions to return
            
        Returns:
            List of transactions
        """
        try:
            response = requests.get(
                f"{self.base_url}/portfolios/{portfolio_id}/transactions",
                params={"limit": limit},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self.logger.error(f"Failed to get transactions: {e}")
            raise

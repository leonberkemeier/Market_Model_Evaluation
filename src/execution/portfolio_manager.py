"""
Portfolio Manager for Layer V Execution

Manages capital allocation and order execution for Sentinel system.
Translates Kelly fractions to order quantities and rebalances portfolio.
"""

from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from .api_client import TradingSimulatorClient, OrderResponse, Holding, Portfolio

logger = logging.getLogger(__name__)


@dataclass
class PositionTarget:
    """Target position for a ticker."""
    ticker: str
    target_fraction: float  # Fraction of portfolio (0-1)
    target_value: float  # Dollar value
    target_quantity: float  # Number of shares
    current_quantity: float  # Current shares held
    action: str  # "buy", "sell", or "hold"
    quantity_delta: float  # Shares to buy/sell


class SentinelPortfolioManager:
    """
    Portfolio manager for Sentinel system.
    
    Responsibilities:
    - Translate Kelly fractions to order quantities
    - Execute trades via Trading Simulator API
    - Rebalance portfolio to match target allocations
    - Track execution and maintain portfolio state
    """
    
    def __init__(
        self,
        api_client: TradingSimulatorClient,
        portfolio_id: int,
        rebalance_tolerance: float = 0.10  # 10% tolerance before rebalancing
    ):
        """
        Initialize portfolio manager.
        
        Args:
            api_client: Trading Simulator API client
            portfolio_id: Portfolio ID to manage
            rebalance_tolerance: Tolerance before triggering rebalance (0.10 = 10%)
        """
        self.client = api_client
        self.portfolio_id = portfolio_id
        self.rebalance_tolerance = rebalance_tolerance
        
        logger.info(
            f"SentinelPortfolioManager initialized for portfolio {portfolio_id} "
            f"(tolerance={rebalance_tolerance:.1%})"
        )
    
    def execute_signal(
        self,
        ticker: str,
        position_size: float,
        asset_type: str = "stock"
    ) -> Optional[OrderResponse]:
        """
        Execute a single trading signal.
        
        Args:
            ticker: Stock ticker
            position_size: Target position size as fraction of portfolio (0-1)
            asset_type: Asset type (stock, crypto, bond, commodity)
        
        Returns:
            OrderResponse if order was placed, None if no action needed
        """
        # Get current portfolio state
        portfolio = self.client.get_portfolio(self.portfolio_id)
        holdings = self.client.get_holdings(self.portfolio_id)
        
        # Find current holding if exists
        current_holding = next(
            (h for h in holdings if h.ticker == ticker),
            None
        )
        
        # Calculate target
        target_value = portfolio.total_value * position_size
        
        # Get current price
        try:
            current_price = self.client.get_quote(ticker)
        except Exception as e:
            logger.error(f"Failed to get quote for {ticker}: {e}")
            return None
        
        target_quantity = target_value / current_price if current_price > 0 else 0
        current_quantity = current_holding.quantity if current_holding else 0
        
        # Determine action
        quantity_delta = target_quantity - current_quantity
        
        if abs(quantity_delta) < 0.01:  # Less than 0.01 shares, skip
            logger.debug(f"{ticker}: No action needed (delta={quantity_delta:.4f} shares)")
            return None
        
        # Execute order
        if quantity_delta > 0:
            # Buy
            logger.info(
                f"{ticker}: BUY {quantity_delta:.2f} shares "
                f"(current={current_quantity:.2f}, target={target_quantity:.2f})"
            )
            return self.client.place_buy_order(
                portfolio_id=self.portfolio_id,
                ticker=ticker,
                quantity=abs(quantity_delta),
                asset_type=asset_type
            )
        else:
            # Sell
            logger.info(
                f"{ticker}: SELL {abs(quantity_delta):.2f} shares "
                f"(current={current_quantity:.2f}, target={target_quantity:.2f})"
            )
            return self.client.place_sell_order(
                portfolio_id=self.portfolio_id,
                ticker=ticker,
                quantity=abs(quantity_delta),
                asset_type=asset_type
            )
    
    def rebalance_portfolio(
        self,
        target_positions: Dict[str, float],
        asset_type: str = "stock",
        dry_run: bool = False
    ) -> List[OrderResponse]:
        """
        Rebalance entire portfolio to match target allocations.
        
        Args:
            target_positions: Dict of ticker -> target Kelly fraction
            asset_type: Asset type for all positions
            dry_run: If True, only calculate trades without executing
        
        Returns:
            List of OrderResponse objects for executed orders
        """
        logger.info(
            f"Rebalancing portfolio {self.portfolio_id} "
            f"with {len(target_positions)} target positions"
        )
        
        # Get current state
        portfolio = self.client.get_portfolio(self.portfolio_id)
        holdings = self.client.get_holdings(self.portfolio_id)
        
        # Create holding lookup
        current_holdings = {h.ticker: h for h in holdings}
        
        # Calculate position targets
        position_targets = self._calculate_position_targets(
            portfolio=portfolio,
            current_holdings=current_holdings,
            target_positions=target_positions
        )
        
        # Log rebalancing plan
        logger.info(f"Rebalancing plan for {len(position_targets)} positions:")
        for target in position_targets:
            if target.action != "hold":
                logger.info(
                    f"  {target.ticker}: {target.action.upper()} "
                    f"{abs(target.quantity_delta):.2f} shares "
                    f"(current={target.current_quantity:.2f}, "
                    f"target={target.target_quantity:.2f})"
                )
        
        if dry_run:
            logger.info("Dry run mode: No orders will be executed")
            return []
        
        # Execute orders in two phases:
        # Phase 1: Sell positions (free up cash)
        # Phase 2: Buy positions (use freed cash)
        
        orders = []
        
        # Phase 1: Sells
        for target in position_targets:
            if target.action == "sell":
                try:
                    order = self.client.place_sell_order(
                        portfolio_id=self.portfolio_id,
                        ticker=target.ticker,
                        quantity=abs(target.quantity_delta),
                        asset_type=asset_type
                    )
                    orders.append(order)
                except Exception as e:
                    logger.error(f"Failed to sell {target.ticker}: {e}")
        
        # Phase 2: Buys
        for target in position_targets:
            if target.action == "buy":
                try:
                    order = self.client.place_buy_order(
                        portfolio_id=self.portfolio_id,
                        ticker=target.ticker,
                        quantity=abs(target.quantity_delta),
                        asset_type=asset_type
                    )
                    orders.append(order)
                except Exception as e:
                    logger.error(f"Failed to buy {target.ticker}: {e}")
        
        logger.info(f"Rebalancing complete: {len(orders)} orders executed")
        
        return orders
    
    def _calculate_position_targets(
        self,
        portfolio: Portfolio,
        current_holdings: Dict[str, Holding],
        target_positions: Dict[str, float]
    ) -> List[PositionTarget]:
        """
        Calculate position targets for rebalancing.
        
        Args:
            portfolio: Current portfolio state
            current_holdings: Dict of ticker -> Holding
            target_positions: Dict of ticker -> target fraction
        
        Returns:
            List of PositionTarget objects
        """
        position_targets = []
        
        # All tickers (current + target)
        all_tickers = set(current_holdings.keys()) | set(target_positions.keys())
        
        for ticker in all_tickers:
            target_fraction = target_positions.get(ticker, 0.0)
            current_holding = current_holdings.get(ticker)
            
            # Calculate target value and quantity
            target_value = portfolio.total_value * target_fraction
            
            # Get current price
            try:
                current_price = self.client.get_quote(ticker)
            except Exception as e:
                logger.warning(f"Failed to get quote for {ticker}, skipping: {e}")
                continue
            
            target_quantity = target_value / current_price if current_price > 0 else 0
            
            # Current quantity
            if current_holding:
                current_quantity = current_holding.quantity
                current_value = current_holding.total_value
                current_fraction = current_value / portfolio.total_value
            else:
                current_quantity = 0.0
                current_value = 0.0
                current_fraction = 0.0
            
            # Calculate delta
            quantity_delta = target_quantity - current_quantity
            
            # Determine action based on tolerance
            if abs(quantity_delta) < 0.01:
                action = "hold"
            elif quantity_delta > 0:
                # Check if delta exceeds tolerance
                if current_fraction == 0 or abs(target_fraction - current_fraction) > self.rebalance_tolerance * target_fraction:
                    action = "buy"
                else:
                    action = "hold"
            else:
                # Check if delta exceeds tolerance
                if target_fraction == 0 or abs(current_fraction - target_fraction) > self.rebalance_tolerance * current_fraction:
                    action = "sell"
                else:
                    action = "hold"
            
            position_target = PositionTarget(
                ticker=ticker,
                target_fraction=target_fraction,
                target_value=target_value,
                target_quantity=target_quantity,
                current_quantity=current_quantity,
                action=action,
                quantity_delta=quantity_delta
            )
            
            position_targets.append(position_target)
        
        return position_targets
    
    def liquidate_all_positions(self) -> List[OrderResponse]:
        """
        Sell all positions in the portfolio.
        
        Returns:
            List of OrderResponse objects for executed sell orders
        """
        logger.info(f"Liquidating all positions in portfolio {self.portfolio_id}")
        
        holdings = self.client.get_holdings(self.portfolio_id)
        orders = []
        
        for holding in holdings:
            try:
                logger.info(f"Liquidating {holding.ticker}: {holding.quantity} shares")
                order = self.client.place_sell_order(
                    portfolio_id=self.portfolio_id,
                    ticker=holding.ticker,
                    quantity=holding.quantity,
                    asset_type=holding.asset_type
                )
                orders.append(order)
            except Exception as e:
                logger.error(f"Failed to liquidate {holding.ticker}: {e}")
        
        logger.info(f"Liquidation complete: {len(orders)} positions sold")
        
        return orders
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """
        Get portfolio summary with current state.
        
        Returns:
            dict with portfolio information
        """
        portfolio = self.client.get_portfolio(self.portfolio_id)
        holdings = self.client.get_holdings(self.portfolio_id)
        
        # Calculate position fractions
        positions = []
        for holding in holdings:
            fraction = holding.total_value / portfolio.total_value if portfolio.total_value > 0 else 0
            positions.append({
                "ticker": holding.ticker,
                "quantity": holding.quantity,
                "value": holding.total_value,
                "fraction": fraction,
                "unrealized_pnl": holding.unrealized_pnl
            })
        
        # Sort by value descending
        positions.sort(key=lambda x: x["value"], reverse=True)
        
        summary = {
            "portfolio_id": portfolio.id,
            "portfolio_name": portfolio.name,
            "total_value": portfolio.total_value,
            "cash_balance": portfolio.cash_balance,
            "total_return": portfolio.total_return,
            "total_return_pct": portfolio.total_return_pct,
            "n_positions": len(positions),
            "positions": positions
        }
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, any]:
        """
        Get performance metrics summary.
        
        Returns:
            dict with performance metrics
        """
        try:
            metrics = self.client.get_performance_metrics(self.portfolio_id)
            
            summary = {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "current_drawdown": metrics.current_drawdown,
                "volatility": metrics.volatility,
                "win_rate": metrics.win_rate,
                "total_return": metrics.total_return
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

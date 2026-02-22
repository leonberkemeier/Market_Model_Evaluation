"""Execution module (Layer V)."""

from .api_client import (
    TradingSimulatorClient,
    OrderResponse,
    Holding,
    Portfolio,
    PerformanceMetrics
)
from .portfolio_manager import (
    SentinelPortfolioManager,
    PositionTarget
)

__all__ = [
    "TradingSimulatorClient",
    "OrderResponse",
    "Holding",
    "Portfolio",
    "PerformanceMetrics",
    "SentinelPortfolioManager",
    "PositionTarget"
]

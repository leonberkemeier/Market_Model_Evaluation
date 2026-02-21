"""Trading integration module for model_regime_comparison."""

from .signal_generator import (
    SignalGenerator,
    SignalGeneratorConfig,
    TradeSignal,
    SignalType,
    SignalReason,
    create_conservative_generator,
    create_moderate_generator,
    create_aggressive_generator
)

# Trading client requires 'requests' package
# Import will fail gracefully if not installed
try:
    from .trading_client import (
        TradingClient,
        TradingClientConfig,
        OrderResult,
        PortfolioState,
        AssetType,
        create_client
    )
    _HAS_TRADING_CLIENT = True
except ImportError:
    _HAS_TRADING_CLIENT = False
    TradingClient = None
    TradingClientConfig = None
    OrderResult = None
    PortfolioState = None
    AssetType = None
    create_client = None

__all__ = [
    # Signal Generator (always available)
    "SignalGenerator",
    "SignalGeneratorConfig",
    "TradeSignal",
    "SignalType",
    "SignalReason",
    "create_conservative_generator",
    "create_moderate_generator",
    "create_aggressive_generator",
    # Trading Client (requires 'requests')
    "TradingClient",
    "TradingClientConfig",
    "OrderResult",
    "PortfolioState",
    "AssetType",
    "create_client",
]

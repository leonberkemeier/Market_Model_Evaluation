"""
Configuration module for model_regime_comparison project.
Centralized settings for database, data paths, model hyperparameters, and backtesting parameters.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================
# Reuse financial_data_aggregator database (SQLite)
# Points to: ../financial_data_aggregator/financial_data.db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///financial_data.db"
)

# ============================================================================
# SECTOR & STOCK UNIVERSE CONFIGURATION
# ============================================================================
SECTORS = {
    "Technology": {"count": 50, "filter_marketcap_min": 5e9},
    "Healthcare": {"count": 50, "filter_marketcap_min": 5e9},
    "Finance": {"count": 50, "filter_marketcap_min": 5e9},
    "Energy": {"count": 50, "filter_marketcap_min": 5e9},
    "Industrials": {"count": 50, "filter_marketcap_min": 5e9},
    "Materials": {"count": 50, "filter_marketcap_min": 5e9},
    "Consumer": {"count": 50, "filter_marketcap_min": 5e9},
    "Utilities": {"count": 50, "filter_marketcap_min": 5e9},
}

TOTAL_UNIVERSE_SIZE = sum(s["count"] for s in SECTORS.values())  # 400
MIN_VOLUME_USD = 10_000_000  # $10M minimum daily volume
MIN_HISTORY_DAYS = 252  # Need at least 1 year of data

# ============================================================================
# TIME PERIOD CONFIGURATION
# ============================================================================
TRAINING_START = datetime(2022, 1, 1)
TRAINING_END = datetime(2023, 12, 31)
VALIDATION_START = datetime(2024, 1, 1)
VALIDATION_END = datetime(2024, 12, 31)
BACKTEST_START = datetime(2025, 1, 1)
BACKTEST_END = datetime(2025, 12, 31)

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
TECHNICAL_FEATURES = {
    "momentum_windows": [5, 20, 60],  # days
    "volatility_windows": [20, 60],
    "sma_windows": [20, 50, 200],
    "rsi_period": 14,
    "macd_periods": (12, 26, 9),  # fast, slow, signal
}

# ============================================================================
# MODEL TRAINING PARAMETERS
# ============================================================================

# Linear Regression Scorer
LINEAR_CONFIG = {
    "lookback_period": 252,  # 1 year
    "min_data_points": 100,
    "prediction_horizon": 30,  # days ahead
}

# CNN Scorer
CNN_CONFIG = {
    "sequence_length": 60,  # 60-day windows
    "volatility_multiplier": 1.5,
    "min_confidence": 0.52,
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping_patience": 5,
}

# XGBoost Scorer
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "min_data_points": 50,
    "prediction_horizon": 30,
}

# LLM Scorer
LLM_CONFIG = {
    "cache_frequency": "monthly",  # Update catalysts monthly
    "top_k_rag_results": 5,
    "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "model_name": "llama3.1:8b",
    "embed_model": "nomic-embed-text",
}

# ============================================================================
# KELLY CRITERION & PORTFOLIO CONFIGURATION
# ============================================================================
PORTFOLIO_CONFIG = {
    "initial_capital": 100_000,
    "kelly_fraction": 0.25,  # Fractional Kelly (0.25 = Kelly/4)
    "max_position_size": 0.10,  # 10% max per position
    "min_position_size": 0.005,  # 0.5% minimum to include in portfolio
    "rebalance_frequency": "weekly",  # "daily", "weekly", "monthly"
    "retrain_frequency": "monthly",  # Retrain models monthly
    "benchmark": "SPY",  # For comparison
}

# ============================================================================
# BACKTEST ENGINE PARAMETERS
# ============================================================================
BACKTEST_CONFIG = {
    "slippage": 0.001,  # 0.1% slippage (set to 0 for MVP)
    "commission": 0.0,  # $0 per trade (set to 0 for MVP)
    "calculate_daily_metrics": True,
    "save_trade_log": True,
}

# ============================================================================
# ANALYSIS & VISUALIZATION PARAMETERS
# ============================================================================
ANALYSIS_CONFIG = {
    "correlation_threshold": 0.05,  # Minimum score-return correlation for good signal
    "heatmap_metric": "sharpe_ratio",  # What to show in specialization heatmap
    "color_scheme": "RdYlGn",  # Matplotlib colormap
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
LOG_FILE = LOGS_DIR / f"model_regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ============================================================================
# FEATURE CACHE CONFIGURATION
# ============================================================================
FEATURE_CACHE_CONFIG = {
    "cache_db": CACHE_DIR / "feature_cache.db",
    "ttl_days": 7,  # Recompute features after 7 days
    "compress": True,
}

# ============================================================================
# TESTING CONFIGURATION
# ============================================================================
TEST_CONFIG = {
    "test_universe_size": 10,  # Use 10 stocks for unit tests (faster)
    "test_start_date": datetime(2024, 1, 1),
    "test_end_date": datetime(2024, 3, 31),  # 3 months for quick validation
}

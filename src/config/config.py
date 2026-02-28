"""Configuration management for Model Regime Comparison project."""
import os
from pathlib import Path
from datetime import date, datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
# Connection to financial_data_aggregator database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///~/Desktop/Server/FinancialData/financial_data_aggregator/financial_data.db"
)

# Expand home directory if used
if "~" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("~", os.path.expanduser("~"))

# ============================================================================
# STOCK UNIVERSE DEFINITION (synced with financial_data_aggregator)
# ============================================================================
# 250 stocks total: 50/50 US/Europe per sector
SECTORS = {
    # IT SECTOR (50 stocks: 25 US, 25 Europe)
    "IT": [
        # US Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "ADBE", "CRM",
        "CSCO", "INTC", "AVGO", "QCOM", "AMD", "MU", "ASML", "LRCX", "CDNS", "SNPS",
        "AMAT", "PYPL", "SQ", "RBLX", "CRWD",
        # European Tech
        "SAP", "ASML", "LRCX", "ASLN", "ADSSF",
        "EOAN", "ENXTPA.PA", "OR.PA", "NOKIA", "RMS.L",
        "CASS", "CRDA", "FLDO", "SAP", "EOAN",
        "ASML", "NOKIA", "RMS.L", "CRDA", "JYAFF",
    ],
    
    # FINANCE SECTOR (50 stocks: 25 US, 25 Europe)
    "Finance": [
        # US Finance
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "ICE", "SPGI", "SCHW", "TD",
        "BNY", "USB", "PNC", "CME", "CBOE", "COIN", "HOOD", "IBKR", "SYF", "DFS",
        "AXP", "V", "MA", "AMP", "MSTR",
        # European Finance
        "HSBA", "LLOY", "BARC", "STAN", "NWG",
        "BNP.PA", "SAN.PA", "CA.PA", "CS", "UBS",
        "DB1", "CBK", "DAX", "HNR1", "HNR1S",
        "NOKIA", "SAP", "ENXTPA.PA", "EOAN", "RMS.L",
    ],
    
    # CHEMISTRY/MATERIALS SECTOR (50 stocks: 25 US, 25 Europe)
    "Chemistry": [
        # US Chemicals/Materials
        "DD", "DOW", "LYB", "APD", "SHW", "ECL", "FMC", "ALB", "PPG", "EMN",
        "MLM", "NEM", "SCCO", "FCX", "ARCH", "BTU", "NRG", "AEE", "AES", "CMS",
        "EXC", "NEE", "DUK", "SO", "OKE",
        # European Chemicals/Materials
        "BASF", "BAYER", "SXRT.L", "RELX.L", "ULVR.L",
        "SAF.PA", "ORLY.PA", "VIE.PA", "NOKIA", "NOKIA.HE",
        "LIN", "ECL", "EOAN", "VOD.L", "RDSA.L",
    ],
    
    # COMMODITIES/ENERGY SECTOR (50 stocks: 25 US, 25 Europe)
    "Commodities": [
        # US Commodities/Energy
        "XOM", "CVX", "COP", "EOG", "MPC", "PSX", "VLO", "HES", "OXY", "SLB",
        "HAL", "RIG", "FANG", "PXD", "EQT", "MRO", "WMB", "TRGP", "EPD", "KMI",
        "GLD", "USO", "DBC", "PDBC", "CORN",
        # European Commodities/Energy
        "RDSA.L", "BP.L", "SHELL.L", "ENXTPA.PA", "TTE.PA",
        "ENQ.PA", "NOKIA", "RMS.L", "EOAN", "NEE",
        "NRG", "EOAN", "EXC", "DUK", "SO", "OKE", "KMI", "WMB", "EPD", "MPC",
    ],
    
    # CRYPTO/BLOCKCHAIN SECTOR (50 stocks: 25 US, 25 Europe)
    "Crypto": [
        # US Crypto-related
        "IBIT", "FBTC", "ETHE", "MARA", "CLSK", "RIOT", "MSTR", "COIN", "GBTC",
        "UPST", "SQ", "PYPL", "SOFI", "LMND", "HOOD", "UBER", "LYFT",
        "U", "DDOG", "NET", "CRWD",
        # European Crypto-related / Tech adjacent
        "NOKIA", "SAP", "ASML", "LLOY", "BARC", "STAN", "BNP.PA", "UBS", "CS",
        "HSBA", "DB1", "DXNTAS", "ADSSF", "EOAN",
        "RELX.L", "AZN.L", "ULVR.L", "RDSA.L", "BP.L",
    ],
}

# Flatten to list of all tickers
ALL_TICKERS = [ticker for tickers in SECTORS.values() for ticker in tickers]
NUM_STOCKS = len(ALL_TICKERS)

# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================
# Date range for backtesting
BACKTEST_START_DATE = date(2023, 1, 1)  # Start with 2023 (1 year of training data)
BACKTEST_END_DATE = date(2025, 12, 31)
TRAINING_START_DATE = date(2022, 1, 1)  # Use 2022 for model training

# Portfolio configuration
INITIAL_CAPITAL = 100000  # $100K per portfolio
REBALANCE_FREQUENCY = "weekly"  # "daily", "weekly", "monthly"
MIN_POSITION_SIZE = 0.005  # 0.5% minimum position
MAX_POSITION_SIZE = 0.10  # 10% maximum position
TARGET_NUM_POSITIONS = 15  # Try to have 10-20 positions

# Kelly Criterion configuration
KELLY_FRACTION = 0.25  # Use 1/4 Kelly (fractional Kelly for variance reduction)
MAX_KELLY_FRACTION = 0.10  # Absolute maximum position size

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================
# Technical feature windows
MOMENTUM_WINDOWS = [5, 20, 60]  # Days
VOLATILITY_WINDOWS = [20, 60]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Feature caching
CACHE_FEATURES = True
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"
FEATURE_CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# RISK EVALUATION CONFIGURATION
# ============================================================================
# Lookback windows
HISTORICAL_LOOKBACK = 252     # 1 year for historical evaluator
GARCH_LOOKBACK = 504          # 2 years for GARCH fitting
REGIME_LOOKBACK = 504         # 2 years for regime detection
SHORT_VOL_WINDOW = 20         # Recent volatility window

# Risk-free rate (annualized, for Sharpe calculation)
# Pull from fact_bond_price if available, otherwise use this default
RISK_FREE_RATE = 0.05

# Market benchmark ticker (for beta/correlation)
BENCHMARK_TICKER = "SPY"

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================
# Strategies to run in comparison
STRATEGIES = {
    "historical_kelly": {
        "evaluator": "historical",
        "lookback": HISTORICAL_LOOKBACK,
        "description": "Historical risk stats → Kelly sizing",
    },
    "garch_kelly": {
        "evaluator": "garch",
        "lookback": GARCH_LOOKBACK,
        "description": "GARCH volatility forecast → Kelly sizing",
    },
    "regime_kelly": {
        "evaluator": "regime_conditional",
        "lookback": REGIME_LOOKBACK,
        "description": "Regime-conditional risk stats → Kelly sizing",
    },
    "equal_weight": {
        "evaluator": None,
        "description": "1/N equal weight (baseline)",
    },
    "risk_parity": {
        "evaluator": None,
        "description": "Inverse-volatility weighting (baseline)",
    },
}

# Trading Simulator API (for sending signals)
TRADING_SIMULATOR_URL = os.getenv(
    "TRADING_SIMULATOR_URL", "http://localhost:8000"
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
LOG_FILE = LOGS_DIR / f"model_regime_comparison_{datetime.now().strftime('%Y-%m-%d')}.log"

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================
# Score validation
MIN_P_WIN = 0.0
MAX_P_WIN = 1.0
MIN_AVG_WIN = -1.0
MAX_AVG_WIN = 1.0
MIN_AVG_LOSS = 0.0
MAX_AVG_LOSS = 1.0

# ============================================================================
# API CONFIGURATION
# ============================================================================
# Ollama/LLM API
RAG_CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", "~/Desktop/Server/FinancialData/financial_data_aggregator/data/chromadb")
if "~" in RAG_CHROMA_PATH:
    RAG_CHROMA_PATH = RAG_CHROMA_PATH.replace("~", os.path.expanduser("~"))

# ============================================================================
# EXPORT FOR EASY ACCESS
# ============================================================================
__all__ = [
    "BASE_DIR",
    "SRC_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "LOGS_DIR",
    "DATABASE_URL",
    "SECTORS",
    "ALL_TICKERS",
    "NUM_STOCKS",
    "BACKTEST_START_DATE",
    "BACKTEST_END_DATE",
    "TRAINING_START_DATE",
    "INITIAL_CAPITAL",
    "REBALANCE_FREQUENCY",
    "KELLY_FRACTION",
    "LOG_LEVEL",
]

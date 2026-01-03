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
# STOCK UNIVERSE DEFINITION
# ============================================================================
# 50 stocks per sector = 250 stocks total (can extend to 400)
SECTORS = {
    "IT": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "ADBE", "CRM",
        "CSCO", "INTC", "AVGO", "QCOM", "AMD", "ASML", "MU", "MCHP", "LRCX", "KLAC",
        "AMAT", "CDNS", "SNPS", "PYPL", "SQ", "RBLX", "COIN", "UBER", "LYFT", "ZM",
        "DOCN", "CRWD", "NET", "PSTG", "DBX", "SHOP", "DDOG", "SPLK", "ZS", "OKTA",
        "PANW", "SRG", "TTD", "LI", "NIO", "XPEV", "UPST", "U", "ZSCALER", "ROKU"
    ],
    
    "Finance": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "ICE", "SPGI", "FAST", "SCHW",
        "TD", "BNY", "USB", "PNC", "CME", "CBOE", "MSTR", "AMP", "LPL", "COIN",
        "HOOD", "IBKR", "CPRT", "SYF", "V", "MA", "AXP", "DFS", "NYSE", "LMND",
        "UPST", "APO", "OWL", "CSGP", "RLY", "BX", "KKR", "CTRE", "PDI", "GDDY",
        "PAYC", "PRCT", "COAS", "ALLY", "HWM", "PFSI", "QRVO", "KC", "ISIG", "EVR"
    ],
    
    "Chemistry": [
        "DD", "DOW", "LYB", "APD", "SHW", "ECL", "FMC", "ALB", "PPG", "EMN",
        "MLM", "NEM", "SCCO", "FCX", "ARCH", "BTU", "NRG", "AEE", "AES", "CMS",
        "EXC", "NEE", "DUK", "SO", "OKE", "COP", "CVX", "XOM", "EOG", "MPC",
        "PSX", "VLO", "PLD", "SPG", "WELL", "EQIX", "DLR", "REXR", "PLD", "STAG",
        "COLD", "UMH", "MPW", "GLYPH", "PFSI", "CMCO", "GEL", "VLTR", "HVT", "ORA"
    ],
    
    "Commodities": [
        "GLD", "USO", "DBC", "TLT", "SLV", "DXY", "FXE", "FXY", "FXB", "UUP",
        "DYN", "PDBC", "CORN", "WEAT", "SOYB", "CBOT", "GEVO", "D", "RIO", "VALE",
        "TECK", "CMRE", "DHR", "PH", "AVTR", "TT", "ABG", "PRA", "MPWR", "ETN",
        "RSG", "WM", "AYI", "MWA", "GRC", "MTRN", "MTU", "MATV", "PTC", "TDG",
        "NOC", "BA", "LMT", "RTX", "GD", "LDOS", "HEI", "TXT", "KTOS", "SWIR"
    ],
    
    "Crypto": [
        # Major cryptocurrencies via spot prices or crypto ETFs
        "IBIT",  # iShares Bitcoin Mini Trust
        "FBTC",  # Fidelity Wise Origin Bitcoin Mini Trust
        "ETHE",  # iShares Ethereum Trust
        "MARA",  # Marathon Digital Holdings
        "CLSK",  # CleanSpark
        "RIOT",  # Riot Platforms
        "MSTR",  # MicroStrategy (BTC holdings)
        "COIN",  # Coinbase Global
        "GBTC",  # Grayscale Bitcoin Mini Trust
        "EBIT",  # iShares Ethereum ETF
        "BTCM",  # iShares Bitcoin Trust
        "WLKA",  # Willow Biosciences (often correlated with crypto)
        "CLVS",  # Cleevio (crypto adjacent)
        "ADME",  # Admera Health
        "AREC",  # American Resources Corporation
        "CCCR",  # Cryptocurrency Miners
        "DGHI",  # Data Grail
        "PSHG",  # Pearlman Holdings
        "GRVY",  # Gravity Waveform
        "HALB",  # Halliburton (energy adjacent)
        "MANU",  # Manchester United (blockchain experiments)
        "FLYA",  # Fly Leasing (alternative assets)
        "CLOU",  # Cloudera
        "ARKW",  # ARK Innovation ETF (heavy crypto exposure)
        "DTEC",  # DT Midstream
        "BLOX",  # Overstock (blockchain)
        "BNGO",  # Bionano Genomics (blockchain interest)
        "SIRI",  # SiriusXM (digital distribution)
        "ROKU",  # Roku (platform exposure)
        "SOFI",  # SoFi (fintech/crypto adjacent)
        "UPST",  # Upstart (AI/alternative finance)
        "TDAC",  # Terreno Realty (REIT, alternative)
        "GRAB",  # Grab Holdings (fintech)
        "VROOM", # Vroom (digital marketplace)
        "EXPI",  # eXp Worlds (metaverse)
        "RBLX",  # Roblox (metaverse/crypto adjacent)
        "SAND",  # The Sandbox token (if tradeable)
        "GMER",  # Gaming and Metaverse
        "MTRX",  # Metaverse-related
        "BTCS",  # Bitcoin Shop
        "WLDW",  # Welldyne (blockchain supply chain)
        "ASII",  # Asia II
        "RXT",   # Rexahn Pharmaceuticals
        "YCBD",  # Yucaipa Chris-Craft
        "SYR",   # Syros Pharmaceuticals
        "OTC",   # Over-the-counter trading related
        "PROG",  # Progenics Pharmaceuticals
        "FLRX",  # Flexus Biosciences
        "AEVA",  # AEva Technologies
        "EOSE",  # Eos Energy Enterprises
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
# MODEL CONFIGURATION
# ============================================================================
# Linear Regression
LINEAR_LOOKBACK_PERIOD = 252  # 1 year
LINEAR_MIN_DATA_POINTS = 100

# CNN
CNN_SEQUENCE_LENGTH = 60  # 60-day windows
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 50
CNN_VALIDATION_SPLIT = 0.2
CNN_VOLATILITY_MULTIPLIER = 1.5

# XGBoost
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# LLM + RAG
LLM_CACHE_FREQUENCY = "monthly"  # Update catalysts monthly
LLM_TOP_K_RAG_RESULTS = 5
LLM_MODEL = "llama3.1:8b"
LLM_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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

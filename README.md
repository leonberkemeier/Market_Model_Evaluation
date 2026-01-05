# Model Regime Comparison: Multi-Model Portfolio Backtesting Framework

A comparative framework to test whether different ML architectures have inherent strengths in different market regimes. Each architecture (Linear Regression, CNN, XGBoost, LLM+RAG) independently manages its own portfolio using Kelly Criterion position sizing.

## Quick Start

### Prerequisites
- Python 3.9+
- financial_data_aggregator project (with populated SQLite database)
- Ollama (for LLM features, optional for MVP)

### Installation

1. Clone the repository:
```bash
cd /home/archy/Desktop/Server/FinancialData/model_regime_comparison
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment (optional, uses SQLite by default):
```bash
# If using non-default database location:
export DATABASE_URL="sqlite:////path/to/financial_data.db"
export OLLAMA_HOST=http://localhost:11434
```

### Run Main Script
```bash
python main.py
```

## Project Structure

```
model_regime_comparison/
├── config/                          # Configuration
│   ├── __init__.py
│   └── config.py                   # All settings centralized
├── src/                             # Source code
│   ├── __init__.py
│   ├── data_structures.py           # Core classes (ScoreResult, Portfolio, etc.)
│   ├── feature_engineering/         # Feature computation
│   │   ├── __init__.py
│   │   ├── base_calculator.py       # Abstract base class
│   │   ├── technical_features.py    # 17 technical indicators
│   │   ├── fundamental_features.py  # Company metrics & growth
│   │   └── feature_aggregator.py    # Caching & batch computation
│   ├── scorers/                     # Model implementations
│   │   ├── __init__.py
│   │   ├── base_scorer.py           # Abstract base class
│   │   ├── placeholder_scorers.py   # Implementations (placeholders)
│   │   ├── linear_scorer.py         # TODO: Implement
│   │   ├── cnn_scorer.py            # TODO: Implement
│   │   ├── xgboost_scorer.py        # TODO: Implement
│   │   └── llm_scorer.py            # TODO: Implement
│   ├── portfolio/                   # Portfolio management
│   │   ├── __init__.py
│   │   ├── kelly_optimizer.py       # Kelly Criterion sizing
│   │   ├── portfolio_builder.py     # TODO: Implement
│   │   └── portfolio_manager.py     # TODO: Implement
│   ├── backtest/                    # Backtesting engine
│   │   ├── __init__.py
│   │   ├── engine.py                # Core backtest loop
│   │   ├── performance_metrics.py   # TODO: Implement
│   │   └── validator.py             # TODO: Implement
│   ├── data/                        # Data loading
│   │   ├── __init__.py
│   │   ├── data_loader.py           # ✅ Query financial_data_aggregator DB
│   │   ├── universe_builder.py      # TODO: Implement
│   │   └── cache.py                 # TODO: Implement
│   └── analysis/                    # Analysis & visualization
│       ├── __init__.py
│       ├── comparator.py            # TODO: Implement
│       ├── visualizer.py            # TODO: Implement
│       └── reporter.py              # TODO: Implement
├── models/                          # Trained models
│   ├── linear/                      # Per-stock linear models
│   ├── cnn_global.h5               # Global CNN weights
│   ├── xgboost_global.pkl          # Global XGBoost model
│   └── llm_cache.db                # LLM score cache
├── results/                         # Results & outputs
│   ├── backtest_results.pkl        # Raw simulation data
│   ├── metrics.csv                 # Summary metrics
│   └── visualizations/             # PNG charts
├── tests/                           # Unit & integration tests
│   ├── __init__.py
│   ├── test_data_structures.py
│   ├── test_scorers.py
│   ├── test_kelly.py
│   ├── test_backtest.py
│   ├── test_integration.py
│   ├── test_data_loader.py          # ✅ Data loading tests
│   └── test_features.py             # ✅ Feature engineering tests (29 tests)
├── notebooks/                       # Jupyter notebooks
│   ├── 01_eda.ipynb                # Exploratory analysis
│   ├── 02_training.ipynb           # Model training
│   ├── 03_backtest.ipynb           # Run backtests
│   └── 04_analysis.ipynb           # Final insights
├── main.py                          # Orchestration script
├── requirements.txt                 # Dependencies
├── .gitignore
├── PROJECT.md                       # Detailed project design
├── SCORING_SPECIFICATION.md         # Scoring system specification
└── README.md                        # This file
```

## Architecture Overview

### Four Independent Models

Each model produces a `ScoreResult` (probability, payoff, EV) for every stock:

1. **Linear Regression Scorer** (`src/scorers/linear_scorer.py`)
   - Assumes past patterns repeat
   - Strong in stable, mean-reverting markets (commodities)
   - High win rate, low payoff profile

2. **CNN Scorer** (`src/scorers/cnn_scorer.py`)
   - Learns temporal patterns from 60-day sequences
   - Strong in trending, volatile sectors (tech)
   - Medium win rate, medium payoff

3. **XGBoost Scorer** (`src/scorers/xgboost_scorer.py`)
   - Synthesizes technical + fundamental features
   - Consistent baseline across all sectors
   - Generalist model

4. **LLM+RAG Scorer** (`src/scorers/llm_scorer.py`)
   - Exploits information asymmetry from SEC filings
   - Strong in narrative-driven sectors (biotech, finance)
   - Low win rate, high payoff profile (Power Law)

### Workflow

```
Stock Universe (400 tickers)
         ↓
Feature Engineering (Technical, Fundamental, Sentiment)
         ↓
4 Scorers (all produce ScoreResult)
         ↓
Kelly Criterion Portfolio Builder
         ↓
Backtest Engine (Daily scoring, weekly rebalancing)
         ↓
Performance Metrics & Analysis
         ↓
Visualizations (Heatmaps, Comparisons)
```

## Key Concepts

### Expected Value (EV)
```
EV = (P_win × avg_win) - ((1 - P_win) × avg_loss)
```
- Models that are "wrong" 80% of the time can still be valuable if wins are big enough
- Power Law: Asymmetric payoffs matter more than win rate

### Kelly Criterion Position Sizing
```
f* = (p × b - q) / b
```
Where:
- p = probability of win
- q = probability of loss (1 - p)  
- b = odds ratio (avg_win / avg_loss)
- f* = optimal fraction of capital to risk

We use **fractional Kelly (0.25)** for safety and variance reduction.

### Backtest Structure
- **Training**: 2022-2023 (24 months)
- **Validation**: 2024 (12 months, for hyperparameter tuning)
- **Backtest**: 2025+ (forward-looking)

## Implemented Components

✅ **Phase 1: Foundation** (Complete)
- Configuration system (`config/config.py`)
- Data structures (`src/data_structures.py`)
- Base classes for extensibility
- Kelly Criterion optimizer (`src/portfolio/kelly_optimizer.py`)
- Data loader (`src/data/data_loader.py`) - connects to financial_data_aggregator DB
- 250-stock universe (50 US/Europe per sector: IT, Finance, Chemistry, Commodities, Crypto)
- Placeholder scorer implementations

✅ **Phase 2: Feature Engineering** (Complete)
- Technical Features (`src/feature_engineering/technical_features.py`)
  - 17 features: Momentum (5d/20d/60d), Volatility (20d/60d), RSI, MACD, Bollinger Bands, Price-to-SMA, Volume trend
- Fundamental Features (`src/feature_engineering/fundamental_features.py`)
  - 15+ features: Company metadata, SEC filing metrics, financial health, growth metrics, sector rotation
- Feature Aggregator (`src/feature_engineering/feature_aggregator.py`)
  - Disk-based caching, batch computation, date-based rolling windows
- Comprehensive test suite (29 tests, all passing)

## TODO: Next Steps

### Phase 3: Model Scorers (Priority 1)
- [ ] Implement `src/scorers/linear_scorer.py` - scikit-learn linear regression baseline
- [ ] Implement `src/scorers/cnn_scorer.py` - TensorFlow CNN for temporal patterns
- [ ] Implement `src/scorers/xgboost_scorer.py` - Gradient boosting with technical+fundamental features
- [ ] Implement `src/scorers/llm_scorer.py` - RAG integration with SEC filings
- [ ] Unit tests for each scorer

### Phase 4: Backtest Engine (Priority 2)
- [ ] Implement full `src/backtest/engine.py` - Daily scoring, weekly rebalancing
- [ ] Implement `src/backtest/performance_metrics.py` - Sharpe, Sortino, max drawdown, etc.
- [ ] Implement portfolio manager - Position tracking, rebalancing logic
- [ ] Unit tests for backtest logic

### Phase 5: Analysis & Visualization (Priority 3)
- [ ] Implement specialization heatmap (sector × model strengths)
- [ ] Implement power law scatter plot (win rate vs payoff)
- [ ] Implement comparison metrics table (model × metric)
- [ ] Create Jupyter notebooks for exploration

### Phase 6: Integration & Polish (Priority 4)
- [ ] Integration tests across all components
- [ ] Optimize performance (caching, parallelization)
- [ ] Full documentation and docstrings
- [ ] Example notebooks and guides

## Testing

Run unit tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_scorers.py -v
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Configuration

All settings in `config/config.py`:
- Database connection
- Stock universe definition
- Time periods (training, validation, backtest)
- Model hyperparameters
- Portfolio parameters
- Backtest settings

## Dependencies

Key packages:
- **Data**: pandas, numpy, yfinance
- **ML**: scikit-learn, tensorflow/keras, xgboost
- **Database**: sqlalchemy, psycopg2
- **Backtesting**: Custom implementation
- **Visualization**: matplotlib, seaborn, plotly
- **Logging**: loguru

See `requirements.txt` for complete list.

## Integration with financial_data_aggregator

This project reuses data from `financial_data_aggregator`:
- **SQLite database** (`financial_data.db`) with tables:
  - `fact_stock_price` - Daily OHLCV data
  - `fact_sec_filing` - SEC filing documents
  - `dim_company` - Company metadata and sectors
- **RAG system** for LLM scorer (optional)
- **ChromaDB vector store** for embeddings (optional)
- **Ollama LLM instance** (optional for MVP)

Ensure financial_data_aggregator has populated the SQLite database before running backtests.

## Key Design Decisions

1. **Standardized ScoreResult**: All models produce identical output format for comparison
2. **Modular Architecture**: Each component (features, scorers, portfolio, backtest) is independent
3. **Kelly Criterion**: Mathematically principled position sizing
4. **Fractional Kelly**: Safety factor reduces variance
5. **Monthly Retraining**: Captures market evolution without excessive overhead
6. **Placeholder Implementations**: Focus on architecture, fill in models incrementally

## Performance Targets

- Feature computation: <10s for 400 stocks daily
- Scoring: <5s daily (excluding LLM, which is cached monthly)
- Backtest: Hours for 12-month period
- Memory: <2GB for full simulation

## References

- Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion
- CNN for time series: https://arxiv.org/abs/1810.01257
- XGBoost: https://xgboost.readthedocs.io/
- LLM for finance: https://arxiv.org/abs/2309.17466

## License

Internal use only.

## Questions?

Refer to:
- `PROJECT.md` for overall vision and design
- `SCORING_SPECIFICATION.md` for scoring details
- Inline docstrings in source code

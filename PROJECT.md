# Model Regime Comparison: A Multi-Model Portfolio Backtesting Framework

## 🎯 Executive Summary

This project builds a comparative framework to test whether different ML architectures have inherent strengths in different market regimes. Rather than creating an ensemble model, each architecture (Linear Regression, CNN, XGBoost, LLM+RAG) independently manages its own portfolio using Kelly Criterion position sizing. The goal is to quantify which models exploit which market inefficiencies and validate the thesis that **asymmetric payoffs (Power Law) matter more than win rate**.

**Key Insight**: A model that's wrong 80% of the time is valuable if the 20% wins average +50% and the losses stay at -5%.

---

## 📊 Core Thesis

### Hypothesis
Different ML architectures have inherent inductive biases that naturally align with different market structures:

- **Linear Regression**: Assumes stationarity & repeating patterns → Dominates stable, commodities markets (gold, oil)
- **CNN**: Captures temporal locality & momentum → Excels in trending regimes (tech volatility, meme stocks)
- **XGBoost**: Synthesizes tabular features agnostically → Consistent baseline across all regimes
- **LLM+RAG**: Exploits information asymmetry & narratives → Wins in cyclical/narrative-driven markets (biotech catalysts, chemical regulation)

### Expected Outcomes
By comparing portfolios across 5 market sectors (IT, Chemistry, Finance, Crypto, Commodities), we'll identify:
1. Which model specializes in which sector
2. How win rate ≠ profitability (Power Law validation)
3. When asymmetric payoffs emerge
4. Risk-adjusted performance (Sharpe ratio, max drawdown, Kelly criterion efficiency)

---

## 🏗️ Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FINANCIAL DATA AGGREGATOR                       │
│  (Stock prices, SEC filings, company metadata from existing project)│
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                             │
├─────────────────────────────────────────────────────────────────────┤
│ • Technical features (momentum, volatility, RSI, MACD)              │
│ • Fundamental features (P/E ratio, debt/equity from SEC filings)    │
│ • Sentiment features (LLM analysis of filings + news)              │
│ • Time series sequences (60-day price/volume windows for CNN)       │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
         ┌───────────────────┼───────────────────┐
         ↓                   ↓                   ↓
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   LINEAR    │    │     CNN     │    │  XGBOOST    │
    │ REGRESSION  │    │ TIME SERIES │    │   HYBRID    │
    │   SCORER    │    │   SCORER    │    │   SCORER    │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           ↓                   ↓                   ↓
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ EV Score    │    │ EV Score    │    │ EV Score    │
    │ 0-100       │    │ 0-100       │    │ 0-100       │
    │ P(win), Δ   │    │ P(win), Δ   │    │ P(win), Δ   │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           └───────────────────┼───────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    KELLY PORTFOLIO BUILDER                          │
├─────────────────────────────────────────────────────────────────────┤
│ For each model:                                                     │
│  1. Rank 400 stocks by EV score                                    │
│  2. Compute Kelly fraction for each                                │
│  3. Position size = kelly_fraction / sum(kelly_fractions)          │
│  4. Build 10-20 position portfolio                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
         ┌───────────────────┼───────────────────┐
         ↓                   ↓                   ↓           ↓
    ┌──────────┐        ┌──────────┐       ┌──────────┐  ┌──────────┐
    │ LINEAR   │        │   CNN    │       │ XGBOOST  │  │   LLM    │
    │PORTFOLIO │        │PORTFOLIO │       │PORTFOLIO │  │PORTFOLIO │
    │ $100K    │        │ $100K    │       │ $100K    │  │ $100K    │
    └────┬─────┘        └────┬─────┘       └────┬─────┘  └────┬─────┘
         ↓                   ↓                   ↓            ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    BACKTEST ENGINE                              │
    │  Daily rebalancing (scoring) → Weekly/Monthly position updates  │
    │  Track P&L, win rate, avg win/loss, Sharpe, drawdown           │
    └─────────────────────────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │               ANALYSIS & VISUALIZATION                          │
    │  • Performance by sector (heatmap)                              │
    │  • Model comparison (Sharpe, max drawdown, win rate)            │
    │  • Specialization proof (which model dominates where?)          │
    │  • Correlation matrix (when do models agree?)                   │
    │  • Power Law validation (EV vs actual returns)                  │
    └─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
model_regime_comparison/
├── PROJECT.md                          # This file
├── ARCHITECTURE.md                     # Detailed system design
├── requirements.txt                    # Python dependencies
│
├── config/
│   └── config.py                       # Universe, sectors, date ranges
│
├── src/
│   ├── feature_engineering/
│   │   ├── technical_features.py      # Momentum, volatility, RSI, MACD
│   │   ├── fundamental_features.py    # P/E, debt ratios from SEC
│   │   └── sentiment_features.py      # LLM-based text analysis
│   │
│   ├── scorers/
│   │   ├── base_scorer.py             # Abstract scorer class
│   │   ├── linear_scorer.py           # Linear regression EV scorer
│   │   ├── cnn_scorer.py              # CNN time-series scorer
│   │   ├── xgboost_scorer.py          # XGBoost hybrid scorer
│   │   └── llm_scorer.py              # LLM+RAG narrative scorer
│   │
│   ├── portfolio/
│   │   ├── kelly_optimizer.py         # Kelly criterion math
│   │   ├── portfolio_builder.py       # Position sizing & construction
│   │   └── portfolio_manager.py       # Daily rebalancing logic
│   │
│   ├── backtest/
│   │   ├── backtest_engine.py         # Core backtesting loop
│   │   ├── performance_metrics.py     # Sharpe, drawdown, win rate
│   │   └── trade_executor.py          # Simulated execution
│   │
│   └── data/
│       ├── data_loader.py             # Pull from Financial Data Aggregator DB
│       └── universe_builder.py        # 400-stock universe per sector
│
├── models/
│   ├── linear_model.pkl               # Trained linear regression
│   ├── cnn_model.h5                   # Trained CNN weights
│   ├── xgboost_model.pkl              # Trained XGBoost model
│   └── llm_config.yaml                # RAG + LLM settings
│
├── results/
│   ├── backtest_results.pkl           # Raw backtest data
│   ├── metrics_summary.csv            # Performance metrics by model/sector
│   └── visualizations/
│       ├── performance_heatmap.png
│       ├── sharpe_comparison.png
│       ├── sector_specialization.png
│       └── correlation_matrix.png
│
├── notebooks/
│   ├── 01_eda_and_features.ipynb      # Exploratory analysis
│   ├── 02_model_training.ipynb        # Train each scorer
│   ├── 03_backtest_run.ipynb          # Execute backtests
│   └── 04_analysis.ipynb              # Results analysis
│
├── tests/
│   ├── test_scorers.py                # Unit tests for each scorer
│   ├── test_kelly_optimizer.py        # Kelly math validation
│   └── test_backtest_engine.py        # Backtest correctness
│
└── main.py                             # Orchestration script
```

---

## 🔍 Detailed Component Descriptions

### 1. Feature Engineering

**Technical Features** (all models, especially Linear & CNN):
- Price momentum (5d, 20d, 60d returns)
- Volatility (20d, 60d rolling std)
- RSI, MACD, Bollinger Bands
- Volume trend (Volume/SMA(Volume))
- Price-to-moving-average ratio

**Fundamental Features** (XGBoost advantage):
- Extracted from SEC filings (fact_sec_filing table)
- P/E ratio, Debt/Equity, Free Cash Flow
- Revenue growth, profit margin
- Sector rotation indicators

**Sentiment Features** (LLM+RAG advantage):
- SEC filing sentiment (risk keyword count from fact_filing_analysis)
- Regulatory news sentiment
- Insider transaction signal
- Patent filing momentum (for tech/chemicals)

**Time Series Sequences** (CNN input):
- 60-day price/volume windows
- Normalized OHLCV data
- Technical indicator sequences

### 2. Scorers: EV Calculation per Model

#### Linear Regression Scorer
```python
def score(stock, date):
    # Historical win rate from training
    P_win = empirical_win_rate(stock, lookback_period=252)
    
    # Expected returns if trend continues
    avg_win = avg_historical_positive_return(stock, P_win)
    avg_loss = abs(avg_historical_negative_return(stock, 1-P_win))
    
    # Expected value (in %)
    EV = (P_win * avg_win) - ((1 - P_win) * avg_loss)
    
    # Normalize to 0-100 scale
    return normalize_to_percentile(EV, all_stocks_EV)
```

**Strengths**: Fast, interpretable, works in mean-reverting markets (commodities)
**Weaknesses**: Assumes past patterns repeat; misses regime changes

#### CNN Scorer
```python
def score(stock, date):
    # Get 60-day price/volume sequence
    sequence = get_price_sequence(stock, date, window=60)
    
    # CNN predicts direction probability
    P_win = cnn_model.predict(sequence)  # 0-1
    
    # Expected move from volatility regime
    volatility = calculate_volatility(sequence)
    expected_move = volatility * trend_strength_multiplier
    
    # Estimate avg win/loss from historical regime
    avg_win = expected_move * win_magnitude_factor
    avg_loss = volatility * stop_loss_factor
    
    EV = (P_win * avg_win) - ((1 - P_win) * avg_loss)
    return normalize_to_percentile(EV, all_stocks_EV)
```

**Strengths**: Captures momentum, adapts to volatility regimes
**Weaknesses**: Can chase trends; needs deep history

#### XGBoost Scorer
```python
def score(stock, date):
    # Stack all features: technical + fundamental
    features = {
        'momentum': [5d, 20d, 60d returns],
        'volatility': [20d, 60d std],
        'fundamentals': [P/E, debt/equity, ...],
        'sentiment': [risk_keywords, insider_score],
        'sector': [rotation_signal]
    }
    
    # XGBoost returns probability + importance
    P_win, feature_importance = xgb_model.predict_proba(features)
    
    # Learned from training: features predictive of big wins
    feature_implied_magnitude = sum(
        feature_importance[i] * learned_magnitude[feature_i]
        for i in important_features
    )
    
    avg_win = volatility * feature_implied_magnitude
    avg_loss = volatility * learned_downside_ratio
    
    EV = (P_win * avg_win) - ((1 - P_win) * avg_loss)
    return normalize_to_percentile(EV, all_stocks_EV)
```

**Strengths**: Balanced, captures complex feature interactions
**Weaknesses**: Black box; slower than linear

#### LLM+RAG Scorer

**Architecture**: Leverages existing RAG system from `financial_data_aggregator` project.

**Data Sources**:
- `fact_sec_filing`: Full filing text (100K-500K chars per document)
- `fact_filing_analysis`: Extracted sections, risk keywords, financial mentions
- ChromaDB vector store: Pre-indexed SEC filing chunks with embeddings
- Ollama LLM: Local llama3.1:8b model for analysis

**Implementation**:
```python
def score(stock, date):
    # Check cache first (catalysts don't change daily)
    cached = check_cache(stock, current_month)
    if cached:
        return cached
    
    # Query RAG system for catalysts & risks (via semantic search)
    catalyst_query = f"What are the key business drivers and growth catalysts for {stock}?"
    risk_query = f"What are the main risks and challenges facing {stock}?"
    
    catalyst_context = rag_query(catalyst_query, ticker=stock)
    risk_context = rag_query(risk_query, ticker=stock)
    
    # Parse LLM responses into structured components
    catalyst_analysis = parse_catalyst_response(catalyst_context['answer'])
    risk_analysis = parse_risk_response(risk_context['answer'])
    
    # Extract quantifiable signals
    P_catalyst = catalyst_analysis['probability']  # 0-1, how real is this catalyst?
    catalyst_magnitude = catalyst_analysis['upside_potential']  # e.g., 15-50%
    pricing_gap = catalyst_analysis['already_priced_in']  # 0-1, market awareness
    
    risk_severity = risk_analysis['severity']  # 0-1, downside magnitude
    risk_probability = risk_analysis['probability']  # 0-1, likelihood
    
    # Catalyst only valuable if market doesn't know
    P_win = P_catalyst * (1 - pricing_gap)
    
    # Expected payoffs
    avg_win = catalyst_magnitude / 100  # Convert % to decimal
    avg_loss = (volatility * 0.5) + (risk_severity * 0.1)  # Volatility + risk component
    
    # EV formula
    EV = (P_win * avg_win) - ((1 - P_win) * avg_loss)
    
    # Cache result
    save_cache(stock, current_month, EV)
    
    return normalize_to_percentile(EV, all_stocks_EV)
```

**Strengths**:
- Exploits information asymmetry and narrative-driven opportunities
- Excels in cyclical/regulatory-sensitive sectors (chemistry, biotech, finance)
- Can identify Power Law opportunities (low win rate, high payoff)
- Leverages full SEC filing text for context

**Weaknesses**:
- LLM hallucination risk (mitigated by parsing for specific fields)
- Computational cost (mitigated by monthly caching)
- Requires quality RAG index (pre-built in financial_data_aggregator)
- Slower than other models (~5-10 sec per stock monthly)

**Integration with Financial Data Aggregator**:
- Reuses: `RAGSystem` class from `rag_demo.py`
- Reuses: ChromaDB index with SEC filing chunks
- Reuses: Ollama embeddings (nomic-embed-text) and LLM (llama3.1:8b)
- Reuses: Database connection (fact_sec_filing, dim_company)

### 3. Kelly Criterion Portfolio Builder

```python
def build_portfolio(stocks_and_scores, capital=100000, kelly_fraction=0.25):
    """
    Kelly Criterion: f* = (p*b - q) / b
    where p = P(win), q = P(loss), b = odds ratio
    
    For each stock:
        kelly_f = (EV / avg_loss) 
        
    Fractional Kelly (0.25 here) = kelly_f / 4 for variance reduction
    """
    
    kelly_fractions = []
    for stock, score in stocks_and_scores:
        # Get EV components
        P_win, avg_win, avg_loss = extract_ev_components(score)
        
        # Kelly formula
        kelly_f = (P_win * avg_win - (1 - P_win) * avg_loss) / avg_loss
        kelly_f = max(0, min(kelly_f, 0.10))  # Clip to 0-10%
        kelly_f *= 0.25  # Fractional Kelly
        
        kelly_fractions.append((stock, kelly_f))
    
    # Normalize to sum to 1
    total_kelly = sum(f for _, f in kelly_fractions)
    weights = [(s, f / total_kelly) for s, f in kelly_fractions]
    
    # Only include positions > 0.5% (sparse portfolio)
    positions = {s: w for s, w in weights if w > 0.005}
    
    # Dollar allocate
    portfolio = {s: weight * capital for s, weight in positions.items()}
    
    return portfolio
```

**Result**: 10-20 position portfolio, dynamically sized by EV/risk

### 4. Backtest Engine

```python
def backtest(model_scorer, stock_universe, start_date, end_date, 
             rebalance_frequency='weekly'):
    """
    Daily: Compute scores for all stocks
    Weekly/Monthly: Rebalance portfolio based on scores
    Track: Daily P&L, positions, trades
    """
    
    portfolio_value = []
    daily_returns = []
    trade_log = []
    
    for date in date_range(start_date, end_date):
        # Daily: Score all stocks
        scores = {stock: model_scorer.score(stock, date) 
                 for stock in stock_universe}
        
        # Check if rebalance day
        if is_rebalance_date(date, rebalance_frequency):
            # Build new portfolio
            new_portfolio = build_portfolio(scores)
            trades = rebalance(current_portfolio, new_portfolio)
            trade_log.extend(trades)
            current_portfolio = new_portfolio
        
        # Compute daily returns
        daily_return = compute_portfolio_return(current_portfolio, date)
        daily_returns.append(daily_return)
        portfolio_value.append(portfolio_value[-1] * (1 + daily_return))
    
    # Compute metrics
    metrics = {
        'total_return': (portfolio_value[-1] - initial_capital) / initial_capital,
        'annualized_return': annualized(daily_returns),
        'sharpe_ratio': sharpe(daily_returns),
        'max_drawdown': max_drawdown(portfolio_value),
        'win_rate': len([r for r in daily_returns if r > 0]) / len(daily_returns),
        'avg_win': mean([r for r in daily_returns if r > 0]),
        'avg_loss': mean([r for r in daily_returns if r < 0]),
        'trades': len(trade_log),
        'turnover': sum_trade_value / avg_portfolio_value
    }
    
    return {
        'portfolio_value': portfolio_value,
        'daily_returns': daily_returns,
        'metrics': metrics,
        'trade_log': trade_log
    }
```

### 5. Analysis & Visualization

**Key Metrics Tracked**:
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.5 for good signal)
- **Max Drawdown**: Largest peak-to-trough decline (resilience test)
- **Win Rate**: % of positive days (expect low for LLM, high for Linear)
- **Avg Win/Loss**: Expected value validation
- **Calmar Ratio**: Return / Max Drawdown
- **Information Ratio**: Alpha vs. benchmark

**Comparisons**:
1. **Performance by Sector Heatmap**
   - Rows: Models (Linear, CNN, XGBoost, LLM)
   - Columns: Sectors (IT, Chemistry, Finance, Crypto, Commodities)
   - Values: Sharpe ratio or annualized return
   - Insight: Where does each model dominate?

2. **Specialization Analysis**
   - For each sector, which model has highest Sharpe?
   - For each model, which sector has highest Sharpe?
   - Cross-sector volatility (consistency of model)

3. **Power Law Validation**
   - Scatter: Win Rate vs. Sharpe Ratio
   - Hypothesis: LLM low win rate but high Sharpe
   - Linear high win rate but lower Sharpe

4. **Correlation Matrix**
   - When do models agree on rankings?
   - Sectors where models converge (efficient markets)
   - Sectors where models diverge (inefficient)

---

## 📈 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Design feature engineering pipeline
- [ ] Connect to Financial Data Aggregator (load stock prices, SEC filings)
- [ ] Build 400-stock universe (50 per sector)
- [ ] Implement technical features (momentum, volatility)
- [ ] Implement fundamental features (from SEC data)
- [ ] Implement Kelly criterion math + unit tests

### Phase 2: Model Scorers (Weeks 3-5)
- [ ] Train Linear Regression scorer (historical win rate baseline)
- [ ] Train CNN scorer (60-day sequences, LSTM/Conv1D)
- [ ] Train XGBoost scorer (feature synthesis)
- [ ] Build LLM+RAG scorer (Ollama local, ChromaDB, query SEC filings)
- [ ] Validate each scorer independently

### Phase 3: Portfolio & Backtest (Weeks 6-7)
- [ ] Implement Kelly portfolio builder
- [ ] Build backtest engine (daily scoring, weekly/monthly rebalancing)
- [ ] Run backtests for all 4 models × 5 sectors (20 total)
- [ ] Validate P&L calculations, trade logging

### Phase 4: Analysis & Insights (Week 8)
- [ ] Generate performance metrics
- [ ] Create visualizations (heatmaps, comparisons)
- [ ] Write analysis: which model specializes where?
- [ ] Validate Power Law hypothesis
- [ ] Document findings

### Phase 5: Polish & Documentation (Week 9)
- [ ] Write comprehensive README
- [ ] Create Jupyter notebooks for reproducibility
- [ ] Add unit tests + integration tests
- [ ] Prepare for presentation/portfolio

---

## 🎯 Success Criteria

**Technical**:
- ✅ All 4 models backtest without errors
- ✅ Metrics (Sharpe, drawdown, win rate) calculated correctly
- ✅ Kelly sizing works (sparse portfolios, no shorting)
- ✅ Results reproducible across runs

**Analytical**:
- ✅ Clear specialization pattern (model X dominates sector Y)
- ✅ Power Law validated (LLM wins fewer but bigger)
- ✅ Risk metrics meaningful (Sharpe ratio correlates with actual performance)

**Presentation**:
- ✅ Compelling visualizations (heatmap tells story)
- ✅ Clear narrative (why does each model win where it does?)
- ✅ Portfolio-ready (clean code, documented, reproducible)

---

## 🔄 Data Flow & Dependencies

```
Financial Data Aggregator
    ├── fact_stock_price (daily OHLCV)
    ├── dim_company (ticker, sector, industry)
    ├── fact_sec_filing (full filing text)
    └── fact_filing_analysis (extracted metrics)
              ↓
    Feature Engineering
    ├── Technical features (momentum, volatility)
    ├── Fundamental features (P/E, debt/equity)
    └── Sentiment features (risk keywords, news)
              ↓
    Model Training (historical data, e.g., 2022-2024)
    ├── Linear: win rate, avg returns per regime
    ├── CNN: trend detection, volatility sensitivity
    ├── XGBoost: feature importance, outcome prediction
    └── LLM: RAG index on SEC filings, catalyst detection
              ↓
    Backtesting (2024-2025)
    ├── Daily scoring (scores based on trained models)
    ├── Weekly/Monthly rebalancing (Kelly sizing)
    └── Performance tracking (P&L, metrics)
              ↓
    Analysis (visualization, insights)
```

---

## 🔗 Integration with Financial Data Aggregator

This project **leverages the existing Financial Data Aggregator** infrastructure:

### Data Reuse
**What we reuse from `financial_data_aggregator`**:
- ✅ **Stock price data**: `fact_stock_price` table (daily OHLCV)
- ✅ **Company metadata**: `dim_company`, `dim_exchange`, `dim_date` dimensions
- ✅ **SEC filing text**: `fact_sec_filing` with full 100K-500K character documents
- ✅ **Extracted analysis**: `fact_filing_analysis` with risk keywords, section counts
- ✅ **RAG system**: Pre-built `RAGSystem` class from `rag_demo.py`
- ✅ **Vector store**: ChromaDB index with SEC filing chunks pre-indexed
- ✅ **LLM/Embeddings**: Ollama setup (llama3.1:8b, nomic-embed-text)

### Code Reuse
**What we import/adapt**:
- `config.config`: Database URL, Ollama host, RAG settings
- `src.models.facts` & `src.models.dimensions`: SQLAlchemy ORM models
- `src.extractors.sec_edgar`: Filing fetching and text extraction logic
- `src.analyzers.filing_analyzer`: Section parsing, text normalization
- `RAGSystem` class: Vector search, embeddings, LLM querying

### Setup Required
```bash
# In financial_data_aggregator directory:
1. python sec_etl_pipeline.py  # Ensure SEC filings are indexed
2. python rag_demo.py --init   # Initialize ChromaDB embeddings

# Copy to model_regime_comparison:
- Share database connection (same DATABASE_URL)
- Share Ollama instance (same OLLAMA_HOST)
- Share ChromaDB vector store (same RAG_CHROMA_PATH)
```

### Architecture Diagram
```
┌─────────────────────────────────────────────┐
│  Financial Data Aggregator (Existing)       │
├─────────────────────────────────────────────┤
│  • Stock prices (fact_stock_price)          │
│  • SEC filings (fact_sec_filing)            │
│  • RAG system + ChromaDB vector store       │
│  • Ollama LLM + embeddings                  │
└────────────────┬────────────────────────────┘
                 │ (Data connection)
                 ↓
┌─────────────────────────────────────────────┐
│  Model Regime Comparison (New Project)      │
├─────────────────────────────────────────────┤
│  • Feature engineering pipeline             │
│  • 4 model scorers (Linear, CNN, XGB, LLM) │
│  • Kelly portfolio builder                  │
│  • Backtest engine                          │
│  • Analysis & visualization                 │
└─────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

**Core**:
- Python 3.9+
- pandas, numpy (data manipulation)
- SQLAlchemy (database access)
- scikit-learn (linear regression, preprocessing)

**Models**:
- TensorFlow/Keras (CNN)
- XGBoost (gradient boosting)
- Ollama (local LLM, embeddings) ← Shared from financial_data_aggregator
- ChromaDB (vector store for RAG) ← Shared from financial_data_aggregator
- BeautifulSoup (SEC filing parsing)

**Backtest & Analytics**:
- Custom backtest engine (simple, interpretable)
- matplotlib, seaborn, plotly (visualization)
- scipy (portfolio optimization)

**Testing & Logging**:
- pytest (unit tests)
- loguru (logging)
- Git (version control)

**External Dependencies**:
- `financial_data_aggregator` database and RAG system (must be running)

---

## 📝 Key Assumptions & Risks

**Assumptions**:
1. 400-stock universe is stationary (no survivorship bias)
2. Historical win rates predict forward performance
3. LLM can access SEC filings without future leakage
4. Market inefficiencies exist in these sectors

**Risks**:
1. **Overfitting**: Models may overfit training data. Mitigate with walk-forward validation.
2. **Regime change**: If market structure shifts, models fail. Mitigate with regular retraining.
3. **LLM hallucination**: LLM may generate false catalysts. Validate with domain expert review.
4. **Data leakage**: Backtest must not use future data. Enforce clean date separation.
5. **Computational cost**: Training + backtesting 20 scenarios. Budget 48+ hours. Mitigate with parallel execution.

---

## 🎓 Interview Narrative

"I built a framework to test whether different ML architectures have inherent strengths in different market regimes. Rather than ensembling models, each architecture independently managed a portfolio using Kelly Criterion position sizing. Linear models dominated commodities (mean-reverting behavior); LLMs excelled in cyclicals where narrative drove valuation; CNNs captured momentum in volatile sectors; XGBoost was the reliable generalist. This proved that asymmetric payoffs (being right 20% of the time with 50% wins) matter more than win rate. The framework demonstrated that sector-specific architecture selection beats one-size-fits-all approaches."

---

## 📚 References & Resources

**Financial Theory**:
- Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion
- Risk-Adjusted Returns: Sharpe ratio, Calmar ratio
- Position Sizing: https://www.investopedia.com/terms/p/positionsizing.asp

**ML in Finance**:
- XGBoost for price prediction
- CNNs for time-series: https://arxiv.org/abs/1810.01257
- LLMs for financial analysis: https://arxiv.org/abs/2309.17466

**Backtesting Best Practices**:
- Walk-forward validation
- Out-of-sample testing
- Survivorship bias

---

## 📋 Next Steps

1. **Review this architecture** with domain knowledge
2. **Validate data availability** (confirm 400-stock universe available, 3+ years history)
3. **Prototype feature engineering** (start with technical features)
4. **Create Phase 1 plan** with detailed milestones
5. **Begin implementation** (Phase 1: Foundation)

---

**Status**: Ready for implementation ✅
**Estimated Duration**: 8-10 weeks of focused development
**Complexity**: Advanced (multiple models, financial domain, custom backtest)
**Impact**: Portfolio-ready project demonstrating advanced ML + finance + system design

# Scoring Specification: The Foundation of Portfolio Simulation

## Overview

The **scoring system** is the core engine. Each model produces a **numerical score** (0-100) for each stock on each date. This score encapsulates:

1. **P_win**: Probability the stock moves favorably (0-1)
2. **avg_win**: Average return if P_win occurs (%, as decimal)
3. **avg_loss**: Average loss if P_win doesn't occur (%, as decimal)

From these three components, we calculate **Expected Value (EV)**:
```
EV = (P_win × avg_win) - ((1 - P_win) × avg_loss)
```

The **0-100 score** is then the percentile rank of EV across all 400 stocks.

---

## Why This Foundation Matters

### The Problem with Traditional Scoring
- Most models optimize for **accuracy** (% correct)
- Portfolio optimization wants **Expected Value** (asymmetric payoffs)
- Example: Model A is 55% accurate with +2% avg win, -5% avg loss → EV = -1.3%
- Example: Model B is 35% accurate with +20% avg win, -2% avg loss → EV = +6.8%
- **Model B is better despite lower accuracy** ← This is what Kelly Criterion exploits

### Our Approach
Each scorer extracts P_win and avg win/loss from its training data, then lets Kelly Criterion size positions based on that asymmetry.

---

## Detailed Scoring Per Model

### 1. Linear Regression Scorer

**Concept**: Assume past patterns repeat. Find stocks with reliable mean-reversion.

**Training Phase** (on historical data 2022-2024):
```
For each stock s:
    - Calculate daily returns over lookback window (252 days = 1 year)
    - Fit linear regression: Next_return = β₀ + β₁ × Feature_t + ε
    - Features: momentum (5d, 20d, 60d returns), mean reversion (price/SMA)
    
    - Backtest on training period:
        For each date in training window:
            if predicted_return > 0: predicted_win = 1, else predicted_win = 0
            actual_return = actual next-day return
            
    - Calculate empirical metrics:
        P_win[s] = % of times prediction was correct
        avg_win[s] = mean of returns when P_win=1
        avg_loss[s] = abs(mean of returns when P_win=0)
```

**Scoring Phase** (daily, on new data):
```
def linear_score(stock, date):
    # Load trained regression coefficients
    beta = load_linear_model(stock)
    
    # Extract features as of date
    momentum_5d = (price[date] - price[date-5]) / price[date-5]
    momentum_20d = (price[date] - price[date-20]) / price[date-20]
    momentum_60d = (price[date] - price[date-60]) / price[date-60]
    mean_reversion = price[date] / SMA_50[date]
    
    features = [momentum_5d, momentum_20d, momentum_60d, mean_reversion]
    
    # Predict next return
    predicted_return = beta[0] + sum(beta[i] * features[i-1])
    
    # Use trained P_win/avg_win/avg_loss from training phase
    P_win = P_win_trained[stock]
    avg_win = avg_win_trained[stock]
    avg_loss = avg_loss_trained[stock]
    
    # Calculate EV
    EV = (P_win * avg_win) - ((1 - P_win) * avg_loss)
    
    # Normalize to 0-100 (percentile of all 400 stocks)
    score = percentile_rank(EV, all_stocks_EV) * 100
    
    return score
```

**Characteristics**:
- Fast to compute (linear algebra)
- Works in **stable, mean-reverting sectors** (commodities, gold)
- High win rate, low payoff (many small wins, few small losses)
- Expected profile: Win rate 55-60%, Sharpe 0.8-1.2

---

### 2. CNN Scorer

**Concept**: Learn temporal patterns from 60-day price sequences. Capture momentum in trending regimes.

**Training Phase**:
```
For each stock s:
    - Create 60-day sliding windows of OHLCV data
    - Normalize: (value - min) / (max - min) over the 60-day window
    - Create labels: next_30_day_return > 0 = 1, else = 0
    
    - Train CNN architecture:
        Input: (60, 5) shape → [60 days, 5 OHLCV features]
        Conv1D layers: Learn local temporal patterns
        Output: P(upward_move)
    
    - Evaluate on validation set:
        P_win[s] = accuracy of CNN prediction
        avg_win[s] = mean return when CNN predicted 1
        avg_loss[s] = abs(mean return when CNN predicted 0)
```

**Scoring Phase**:
```
def cnn_score(stock, date):
    # Get 60-day price sequence ending at date
    sequence = get_ohlcv_sequence(stock, date, window=60)
    
    # Normalize
    sequence_norm = normalize_minmax(sequence)
    
    # Predict with CNN
    P_win = cnn_model.predict(sequence_norm.reshape(1, 60, 5))[0][0]
    
    # Calculate volatility as risk measure
    volatility = calculate_volatility(sequence[-20:])  # Last 20 days
    
    # Use trained magnitudes
    avg_win = avg_win_trained[stock] * volatility_factor
    avg_loss = avg_loss_trained[stock] * volatility_factor
    
    # EV with confidence multiplier
    confidence = abs(P_win - 0.5) * 2  # How confident is the prediction?
    EV = (P_win * avg_win - (1 - P_win) * avg_loss) * confidence
    
    score = percentile_rank(EV, all_stocks_EV) * 100
    
    return score
```

**Characteristics**:
- Adapts to volatility regime
- Works in **trending, momentum-driven sectors** (tech, volatile growth)
- Medium win rate, medium payoff
- Expected profile: Win rate 45-52%, Sharpe 1.0-1.5
- Can chase trends (weakness in reversals)

---

### 3. XGBoost Scorer

**Concept**: Synthesize all features (technical, fundamental, sentiment). Be the balanced generalist.

**Training Phase**:
```
For each stock s:
    - Create feature matrix from 3 years of daily data:
        Technical: momentum (5d, 20d, 60d), volatility, RSI, MACD, BB width
        Fundamental: P/E ratio, debt/equity, margin, growth rate
        Sentiment: risk keywords count, insider transactions
        Sector: relative strength vs sector
        
    - Label: next_30_day_return > 0 = 1, else = 0
    
    - Train XGBoost classifier:
        n_estimators=500, max_depth=6, learning_rate=0.05
        objective='binary:logistic' (classification)
    
    - Evaluate:
        P_win[s] = predicted probability for class=1
        Analyze SHAP values to understand what drives wins/losses
        
        avg_win[s] = mean return when model predicts >0.6 probability
        avg_loss[s] = mean return when model predicts <0.4 probability
```

**Scoring Phase**:
```
def xgboost_score(stock, date):
    # Extract all features as of date
    features = extract_all_features(stock, date)  # 50+ features
    
    # Get probability prediction
    P_win = xgb_model.predict_proba(features)[0][1]
    
    # Use SHAP to understand magnitude
    shap_values = explainer.shap_values(features)
    feature_strength = sum(abs(sv) for sv in shap_values)
    
    # Scale magnitude by feature strength
    avg_win = avg_win_trained[stock] * (1 + feature_strength)
    avg_loss = avg_loss_trained[stock] * (1 + feature_strength)
    
    # EV with calibration
    EV = (P_win * avg_win) - ((1 - P_win) * avg_loss)
    
    score = percentile_rank(EV, all_stocks_EV) * 100
    
    return score
```

**Characteristics**:
- Balanced across all features
- Works **consistently across all sectors** (generalist)
- Medium-high win rate, medium payoff
- Expected profile: Win rate 52-55%, Sharpe 1.1-1.4
- Good at feature interactions

---

### 4. LLM+RAG Scorer

**Concept**: Find information asymmetries. Exploit catalysts the market hasn't priced in.

**Training Phase**:
```
For each stock s and relevant SEC filing f:
    1. Extract key sections from filing:
        - Management discussion & analysis (MD&A)
        - Risk factors
        - Business description
        - Liquidity discussion
    
    2. Analyze with LLM:
        Query: "What are the key catalysts that could drive stock price?"
        Extract: catalyst descriptions, probability estimates
        
        Query: "What are major risks to valuation?"
        Extract: risk descriptions, severity estimates
    
    3. Backtest on post-filing returns:
        If catalyst detected + not priced in → label = 1
        If risk detected + market still optimistic → label = 0
        
    4. Calculate:
        P_win[s] = % of times LLM-identified catalysts materialized
        avg_win[s] = return in 30 days post-catalyst (if materialized)
        avg_loss[s] = return if catalyst failed / risk materialized
```

**Scoring Phase**:
```
def llm_score(stock, date):
    # Check cache (catalysts don't change daily)
    cached = check_catalyst_cache(stock, current_month)
    if cached:
        return cached
    
    # Query RAG system
    catalyst_query = f"What are the key growth catalysts for {stock}?"
    risk_query = f"What are the main risks for {stock}?"
    
    catalyst_context = rag_search(catalyst_query, stock)
    risk_context = rag_search(risk_query, stock)
    
    # Parse LLM responses
    catalyst_analysis = llm_parse_catalyst(catalyst_context)
    # Returns: {
    #   'probability': 0.6,  # How likely is this catalyst?
    #   'magnitude': 0.25,   # 25% upside if it happens
    #   'timeline': 'Q1-Q2', # When will it happen?
    #   'pricing_gap': 0.8   # 80% already priced in?
    # }
    
    risk_analysis = llm_parse_risk(risk_context)
    # Returns: {
    #   'severity': 0.4,     # 40% downside if risk materializes
    #   'probability': 0.3   # 30% chance it happens
    # }
    
    # Calculate P_win
    # Catalyst valuable only if market doesn't know
    P_catalyst = catalyst_analysis['probability'] * (1 - catalyst_analysis['pricing_gap'])
    
    # Risk reduces upside
    P_loss = risk_analysis['probability']
    
    P_win = P_catalyst * (1 - P_loss)
    
    # Calculate expected moves
    avg_win = catalyst_analysis['magnitude']
    avg_loss = risk_analysis['severity']
    
    # EV from information edge
    EV = (P_win * avg_win) - ((1 - P_win) * avg_loss)
    
    # Cache result
    save_catalyst_cache(stock, current_month, EV)
    
    score = percentile_rank(EV, all_stocks_EV) * 100
    
    return score
```

**Characteristics**:
- Exploits information asymmetry
- Works in **narrative-driven sectors** (biotech, chemicals, finance)
- **Low win rate, high payoff** (Power Law!)
- Expected profile: Win rate 25-40%, Sharpe 1.5-2.5 (when it works)
- High variance, occasional spectacular wins
- Monthly caching makes it computationally feasible

---

## Scoring Output Format

Each scorer outputs a **ScoreResult** object:

```python
@dataclass
class ScoreResult:
    ticker: str
    date: date
    model_name: str  # "linear", "cnn", "xgboost", "llm"
    
    # Core EV components
    score: float  # 0-100 normalized percentile
    p_win: float  # 0-1 probability of favorable move
    avg_win: float  # % return if right
    avg_loss: float  # % return if wrong
    expected_value: float  # (p_win * avg_win) - ((1-p_win) * avg_loss)
    
    # Metadata for analysis
    confidence: float  # How confident is this score?
    catalyst: str  # (LLM only) What's the catalyst?
    data_points: int  # How many observations support this?
    last_updated: datetime
```

---

## Validation: Score Quality Checks

Before using a score, validate:

```python
def validate_score(result: ScoreResult) -> bool:
    checks = {
        'p_win_range': 0 <= result.p_win <= 1,
        'payoff_reasonable': abs(result.avg_win) < 1.0,  # Max 100% move
        'payoff_ratio': result.avg_win > 0 and result.avg_loss > 0,
        'ev_makes_sense': -1.0 < result.expected_value < 1.0,
        'has_data': result.data_points > 0,
    }
    
    if not all(checks.values()):
        logger.warning(f"Score validation failed for {result.ticker}: {checks}")
        return False
    
    return True
```

---

## Scoring Schedule

### Daily Scoring
```
6:00 AM: Fetch market data from previous close
6:15 AM: Calculate technical features for all 400 stocks
6:30 AM: Run Linear scorer (0.5s per stock) = ~200s total
6:34 AM: Run CNN scorer (1s per stock) = ~400s total
6:40 AM: Run XGBoost scorer (0.3s per stock) = ~120s total
6:42 AM: Check rebalance day

If rebalance day (e.g., every Monday):
    6:45 AM: Run LLM scorer on top 50 candidates (5s each) = ~250s
    7:00 AM: Build portfolios with Kelly sizing
    7:05 AM: Execute trades at market open
```

### Monthly Tasks
```
End of month:
    1. Retrain Linear/CNN/XGBoost (historical data updated)
    2. Update LLM catalyst cache for all holdings
    3. Validate score correlations with actual returns
```

---

## Model-Specific Tuning Parameters

### Linear Scorer
```python
LOOKBACK_PERIOD = 252  # 1 year of data
MOMENTUM_WINDOWS = [5, 20, 60]
MIN_DATA_POINTS = 100  # Need at least 100 days of history
```

### CNN Scorer
```python
SEQUENCE_LENGTH = 60  # 60-day windows
VOLATILITY_MULTIPLIER = 1.5  # Scale payoffs by volatility
MIN_CONFIDENCE = 0.52  # Ignore weak signals
```

### XGBoost Scorer
```python
FEATURE_COUNT = 50+
MIN_IMPORTANT_FEATURES = 5  # How many features matter
PREDICTION_THRESHOLD = 0.55  # Score stocks with >55% win probability
```

### LLM Scorer
```python
CACHE_FREQUENCY = 'monthly'  # Update catalysts monthly
TOP_K_RAG_RESULTS = 5  # Use top 5 RAG chunks for context
LLM_PARSING_TEMPLATES = {...}  # Extract structured data from LLM response
```

---

## Expected Score Distributions

After training, we should see:

### Linear Scorer
- Mean score: 50 (by construction, percentile)
- Std dev: 15
- Skew: Slight negative (mean reversion favors most stocks slightly)
- Examples: Commodity stocks 65-75, Tech 35-45

### CNN Scorer
- Mean score: 50
- Std dev: 18 (higher variance)
- Skew: Depends on regime (trending → positive skew)
- Examples: Trending stocks 70+, choppy stocks 30-40

### XGBoost Scorer
- Mean score: 50
- Std dev: 16
- Skew: Slightly positive (features better signal in up markets)
- More consistent across sectors than Linear/CNN

### LLM Scorer
- Mean score: 50
- Std dev: 20 (highest variance!)
- Skew: Highly positive (rare high-conviction scores)
- Examples: Biotech catalyst stocks 80-95, no-news stocks 20-30

---

## Scoring Examples

### Example 1: Apple (AAPL) on Date 2025-01-15

**Linear Scorer**:
- P_win: 0.57 (historically mean-reverts 57% of the time)
- avg_win: 0.8% (small but reliable)
- avg_loss: 0.6% (limited downside)
- EV = (0.57 × 0.008) - (0.43 × 0.006) = 0.00456 - 0.00258 = 0.00198
- Percentile: 62/100 → **Score: 62**

**CNN Scorer**:
- Sequence shows: Downtrend over 20 days, volatility increasing
- P_win: 0.48 (momentum negative)
- avg_win: 1.5% (if reverses)
- avg_loss: 1.2% (if continues down)
- EV = (0.48 × 0.015) - (0.52 × 0.012) = 0.0072 - 0.00624 = 0.00096
- Percentile: 45/100 → **Score: 45**

**XGBoost Scorer**:
- Features: P/E 28, volatility up, growth +5%, sector weak
- P_win: 0.52 (mixed signal)
- avg_win: 2.1% (feature-driven magnitude)
- avg_loss: 2.0% (balanced downside)
- EV = (0.52 × 0.021) - (0.48 × 0.020) = 0.01092 - 0.0096 = 0.00132
- Percentile: 52/100 → **Score: 52**

**LLM Scorer**:
- Catalyst: "Vision Pro sales ramping Q1 2025"
- Pricing gap: 0.7 (mostly priced in already)
- P_catalyst: 0.65 × (1 - 0.7) = 0.195
- avg_win: 0.15 (15% upside if catalyst exceeds)
- avg_loss: 0.05 (5% downside if disappoints)
- EV = (0.195 × 0.15) - (0.805 × 0.05) = 0.02925 - 0.04025 = -0.011
- Percentile: 35/100 → **Score: 35**

**Portfolio Impact**:
- Linear says: Marginal buy (62/100)
- CNN says: Mild avoid (45/100)
- XGBoost says: Neutral (52/100)
- LLM says: Avoid (35/100)
- Consensus: Slightly negative, probably small/no position

---

## Next: Validation Against Actuals

After scoring, we track:

```python
# Did our score predict next-month returns?
actual_return = (price[date+30] - price[date]) / price[date]

# For each model, correlate score vs actual return
correlation_linear = corr(linear_scores, actual_returns)
# Should be positive (~0.05-0.15 for good signal)

# LLM should have higher correlation in narrative sectors
# Linear should have higher correlation in commodity sectors
```

This validation becomes **the empirical proof** of sector specialization.

---

## Summary

The **scoring system** is everything:
1. Defines P_win and expected payoffs for each stock
2. Feeds into Kelly Criterion for position sizing
3. Creates the power differential between models
4. Validates the Power Law hypothesis

A model with **low win rate + high payoff** beats one with **high win rate + low payoff**.

The scores are the foundation. Everything downstream depends on them being accurate.

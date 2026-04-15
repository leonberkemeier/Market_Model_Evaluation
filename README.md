# Sentinel: AI-Driven Robo-Advisory and Portfolio Management

This project is a hybrid AI-driven Robo-Advisory and Portfolio Management system. It strictly separates high-frequency numerical data from unstructured qualitative intelligence to prevent LLM hallucinations. This module currently houses the core algorithmic brains and execution layers for the broader Sentinel architecture.

## Quick Start
### Prerequisites
- Python 3.9+
- `financial_data_aggregator` project (with populated PostgreSQL/SQLite database)
- Ollama (for LLM Conviction features)

### Installation
```bash
cd /home/archy/Desktop/Server/FinancialData/model_regime_comparison
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## System Architecture

The architecture is divided into 6 core pillars, 4 of which are housed primarily within this module:

1. **Nervous System (financial_data_aggregator)**: Data aggregation via PostgreSQL, RAG, and an MCP layer.
2. **Regime Brain (Housed Here)**: 
   - Uses a Hidden Markov Model (HMM) to infer hidden market states (e.g., Quiet Growth, Volatile Bear) with 70% 3-day hysteresis smoothing to prevent whipsaws.
   - Forward Risk Modeling via Monte Carlo simulations computing hybrid VaR over 60-day and 10-year windows.
3. **Conviction Synthesis (Housed Here)**: 
   - LLM integration acting as a senior analyst.
   - Evaluates "Narrative" vs "Numbers".
   - Assigns a modifier conviction score (`p_final = p_HMM + Δp_LLM`).
4. **Risk-Factor Envelopes (Housed Here)**: 
   - Strict Strategic Asset Allocation (SAA) matrices governing risk profiles (Conservative to Aggressive).
   - Top 50 asset filtering by profile characteristics and Intra-Bucket Sizing via Fractional Kelly Optimization.
   - Tactical retreats triggered dynamically by the Regime Brain.
5. **Gap-Filler Engine (Housed Here)**: 
   - Priority queue system handling recurring deposits (e.g., DCA) and fee-efficient natural rebalancing.
6. **Mirror Ledger (Trading_Simulator)**: High-fidelity trading simulation app for post-mortems and accounting.

## Project Structure

```
model_regime_comparison/
├── config/                          # Configuration & Settings
├── src/                             # Core Logic
│   ├── regime/                      # HMM Regime Brain & Monte Carlo VaR
│   ├── nlp/                         # Conviction Synthesis & LLM Integration
│   ├── portfolio/                   # Risk-Factor Envelopes & SAA Construction
│   ├── execution/                   # Gap-Filler Engine & Priority Queues
│   ├── feature_engineering/         # Core technical/fundamental calculators
│   └── data/                        # Connectors to aggregator MCP/DB
├── tests/                           # Unit & integration tests
├── docs/                            # Deep-dive architecture documentation
└── notebooks/                       # Exploratory system design
```

## Key Concepts

### Regime-Aware Risk Modeling
The system adapts to macro conditions rather than maintaining static assumptions. A Hybrid Monte Carlo approach ensures the model doesn't become "blind" to tail-risk crashes during long periods of calm.

### Fractional Kelly Position Sizing
Allocations inside each Risk-Factor Envelope are optimized using a fractional Kelly approach, bound strictly within the SAA total bucket limit. `f* = ((p * b) - q) / b`, using a customized confidence score `p_final` augmented by LLM analysis.

### Fee-Efficient Execution
Using a Priority Queue Algorithm, the Gap-Filler Engine inherently buys the dip for underweight assets using fresh incoming capital, thus avoiding tax drag and costly turnover friction.

## TODO: Next Steps

- **Phase 3 (Regime Brain)**: Operationalize the HMM state logic and VaR simulation engine.
- **Phase 4 (Risk-Factor Envelopes)**: Finalize SAA matrix boundaries and the top-50 two-step fractional Kelly constructor.
- **Phase 5 (Conviction Synthesis)**: Integrate LLM personas (Venture Scout vs Cynical Auditor).
- **Phase 6 (Execution Engine)**: Build the Gap-Filler priority queue routines for simulated trading.

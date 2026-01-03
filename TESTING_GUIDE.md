# Testing & Dashboard Guide

## ✅ Database Connection Test - PASSED!

The database is **fully functional and ready to use**. Here's what we found:

### Test Results Summary
```
✅ All tests PASSED! Database is ready to use.

Tables Found: 18
├── Stock Prices: 93,680 records ✅
├── Companies: 58 records ✅
├── SEC Filings: 16 records ✅
├── Commodities: 130 records
├── Crypto: 385 records
├── Bonds: 76 records
├── Economic Indicators: 38 records
└── ... and more
```

### Running the Database Test

To verify the database connection at any time:

```bash
cd /home/archy/Desktop/Server/FinancialData/model_regime_comparison

# Using the venv from financial_data_aggregator
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/python test_database.py
```

Expected output: **✅ All tests PASSED!**

## 🎯 Financial Data Aggregator Dashboard

The financial_data_aggregator project includes a **Flask dashboard** for viewing the collected financial data.

### Location
```
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/dashboard/app.py
```

### Features
- **Real-time data viewer** - View latest stock prices, commodities, crypto, bonds
- **Top movers** - Gainers and losers  
- **SEC filings browser** - View collected SEC documents
- **Market summary** - Economic indicators, sector performance
- **Company details** - Search and analyze individual companies

### Running the Dashboard

1. **Make sure financial_data_aggregator is set up**:
   ```bash
   cd /home/archy/Desktop/Server/FinancialData/financial_data_aggregator
   source venv/bin/activate  # Activate venv if needed
   ```

2. **Start the dashboard**:
   ```bash
   python dashboard/app.py
   ```

3. **Open in browser**:
   ```
   http://localhost:5000
   ```

### Data Shown in Dashboard

The dashboard queries the same SQLite database that our project uses:
- **Stock Prices**: Latest daily OHLCV for 58 companies
- **Company Info**: Sector, industry, market cap
- **SEC Filings**: Recent 10-K and other filings (16 total)
- **Crypto Data**: Bitcoin, Ethereum, etc. (15 assets)
- **Commodities**: Gold, oil, etc. (5 commodities)
- **Economic Indicators**: Macro data (4 indicators)

## 🧪 Running Project Tests

### Kelly Optimizer Tests (Unit Tests)

```bash
cd /home/archy/Desktop/Server/FinancialData/model_regime_comparison

# Using aggregator venv
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/python -m pytest tests/test_kelly.py -v
```

**Expected**: 8/8 tests pass

### Data Loader Tests (When Implemented)

Once you implement `src/data/data_loader.py`:

```bash
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/python -m pytest tests/test_data_loader.py -v
```

### Full Test Suite

```bash
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/python -m pytest tests/ -v
```

## 📊 Quick Data Inspection

### Option 1: Use Dashboard
Best for visual exploration:
```bash
cd /home/archy/Desktop/Server/FinancialData/financial_data_aggregator
python dashboard/app.py
```
Then visit `http://localhost:5000`

### Option 2: Use Database Test Script
Quick programmatic check:
```bash
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/python test_database.py
```

### Option 3: Direct SQL Query
Use SQLite CLI:
```bash
sqlite3 /home/archy/Desktop/Server/FinancialData/financial_data.db

# In the SQLite prompt:
SELECT COUNT(*) FROM fact_stock_price;  -- Should return 93,680
SELECT DISTINCT ticker FROM dim_company LIMIT 10;  -- View companies
```

## 🔗 Creating a Python Alias (Optional)

For easier testing, add this to your `.zshrc`:

```bash
alias python_fda='/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/python'
```

Then reload shell:
```bash
source ~/.zshrc
```

Now you can run:
```bash
python_fda test_database.py
python_fda -m pytest tests/ -v
```

## 📝 Sample Test Script

Quick Python script to test data loading:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/archy/Desktop/Server/FinancialData/model_regime_comparison')

from sqlalchemy import create_engine, text
from config import DATABASE_URL
import pandas as pd

engine = create_engine(DATABASE_URL)

# Test 1: Load stock prices
with engine.connect() as conn:
    prices = pd.read_sql(
        text("SELECT * FROM fact_stock_price LIMIT 100"),
        conn
    )
    print(f"✓ Loaded {len(prices)} price records")
    print(prices.groupby('ticker').size())

# Test 2: Load companies
with engine.connect() as conn:
    companies = pd.read_sql(
        text("SELECT * FROM dim_company"),
        conn
    )
    print(f"\n✓ Found {len(companies)} companies")
    print(companies['sector'].value_counts())
```

## 🚀 Next Steps

Now that database connectivity is verified:

1. **Implement `src/data/data_loader.py`**
   - Use the template in `src/data/data_loader_template.py`
   - Write unit tests in `tests/test_data_loader.py`

2. **Implement scorers in `src/scorers/`**
   - Start with `linear_scorer.py`
   - Follow the template pattern in `placeholder_scorers.py`

3. **Test the full pipeline**
   - `python test_database.py` - Verify data
   - `pytest tests/` - Run all tests
   - `python main.py` - Run orchestration

## 🐛 Troubleshooting

### "No module named sqlalchemy"
**Solution**: Use the venv from financial_data_aggregator:
```bash
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/python script.py
```

### "Database not found"
**Solution**: Ensure `financial_data.db` exists in FinancialData folder:
```bash
ls -la /home/archy/Desktop/Server/FinancialData/financial_data.db
```

### Dashboard won't start
**Solution**: Ensure Flask is installed in the venv:
```bash
/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/venv/bin/pip install flask
```

## ✅ Checklist

- [x] SQLite database exists with 93,680 stock price records
- [x] Database connection test passes
- [x] Can query companies, SEC filings, and other data
- [x] Flask dashboard available for data browsing
- [x] venv has all required packages
- [x] Ready to implement data_loader.py and scorers

**Status**: ✅ Ready to begin Phase 1 (Data Pipeline)

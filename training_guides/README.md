# Training Guides - Project Sentinel

This folder contains everything you need to train your Sentinel models.

## 📁 What's Here

### Documentation
- **`00_TRAINING_OVERVIEW.md`** (501 lines) - Complete training guide with all details
- **`01_DATA_LOADER_GUIDE.md`** (567 lines) - How to build data loader from SQLite database

### Jupyter Notebooks
- **`01_data_exploration.ipynb`** - Explore your database and available data
- (Create more notebooks as needed for model training)

### Data
Your database has been copied here for convenience:
```
../data/financial_data.db  (154 MB)
```

Original location: `/home/archy/Desktop/Server/FinancialData/financial_data_aggregator/financial_data.db`

---

## 🚀 Quick Start

### 1. Install Jupyter (if not already installed)
```bash
# In your venv
source ../venv/bin/activate
pip install jupyter matplotlib seaborn
```

### 2. Start Jupyter
```bash
# From this directory (training_guides/)
jupyter notebook
```

This will open Jupyter in your browser.

### 3. Open First Notebook
Open `01_data_exploration.ipynb` to:
- Check your database
- See available tickers
- Verify data for your training period (2019-2024)
- Visualize sample stock prices

### 4. Follow the Guides
1. Read `00_TRAINING_OVERVIEW.md` for the big picture
2. Use `01_DATA_LOADER_GUIDE.md` to build data loader
3. Create your own notebooks for model training

---

## 📊 Your Data

### Database Contents
- **93,710 stock price records** (OHLCV data)
- **Date range**: 1972 to present
- **Tables**: Stock prices, economic indicators, commodities, crypto, bonds

### Sector Tickers (for training)
- **Tech** (10): AAPL, MSFT, NVDA, GOOGL, META, TSLA, AMD, INTC, CRM, ADBE
- **Finance** (10): JPM, BAC, GS, MS, WFC, C, BLK, AXP, USB, PNC
- **Crypto** (5): BTC-USD, ETH-USD, SOL-USD, BNB-USD, ADA-USD
- **Commodities** (4): GLD, SLV, USO, DBC
- **Cyclicals** (10): CAT, DE, BA, HON, MMM, GE, UPS, FDX, DAL, UAL

### Training Period
- **Training**: 2019-01-01 to 2024-06-30 (5.5 years)
- **Hold-out**: 2024-07-01 to 2024-12-31 (6 months)

---

## 🎯 Recommended Workflow

### Phase 1: Data Setup
1. ✅ Run `01_data_exploration.ipynb`
2. ⏳ Create `src/data/data_loader.py` (use guide in `01_DATA_LOADER_GUIDE.md`)
3. ⏳ Test data loader works

### Phase 2: Model Training
4. ⏳ Create `02_linear_model_training.ipynb`
5. ⏳ Train Linear model on Finance sector
6. ⏳ Evaluate performance

### Phase 3: Add More Models
7. ⏳ Create `03_xgboost_model_training.ipynb`
8. ⏳ Train XGBoost on Tech sector
9. ⏳ Compare models

### Phase 4: Full Pipeline
10. ⏳ Run all 20 model-sector combinations
11. ⏳ Select winning models per sector
12. ⏳ Deploy to Sentinel system

---

## 💡 Notebook Tips

### Creating New Notebooks
Create notebooks for each step of your workflow:
- `02_data_loader_implementation.ipynb` - Build and test data loader
- `03_linear_model_training.ipynb` - Train Linear model
- `04_xgboost_model_training.ipynb` - Train XGBoost model
- `05_model_comparison.ipynb` - Compare all models
- etc.

### Useful Imports
```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Now you can import from src/
from src.data.data_loader import SentinelDataLoader
from src.experts.linear_model import LinearModel
```

### Database Connection
```python
import sqlite3
import pandas as pd

db_path = Path.cwd().parent / 'data' / 'financial_data.db'
conn = sqlite3.connect(db_path)

# Your queries here...

conn.close()
```

---

## 📝 Example Notebook Structure

Here's a template for your training notebooks:

```python
# 1. Setup
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# 2. Load Data
from src.data.data_loader import SentinelDataLoader

loader = SentinelDataLoader()
df = loader.load_sector_data('finance', '2019-01-01', '2024-06-30')

# 3. Prepare Features
# (Use your feature engineering modules)

# 4. Train Model
from src.experts.linear_model import LinearModel

model = LinearModel(sector='finance')
model.train(features, targets)

# 5. Evaluate
results = model.score(test_features, test_targets)
print(results)

# 6. Save Model
import pickle
with open('../models/linear_finance_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

## 🔧 Troubleshooting

### Jupyter not starting?
```bash
# Make sure you're in venv
which python  # Should show venv/bin/python

# Install jupyter if missing
pip install jupyter
```

### Can't import from src/?
```python
# Add this at the top of your notebook
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

### Database not found?
```python
# Check database path
from pathlib import Path
db_path = Path.cwd().parent / 'data' / 'financial_data.db'
print(f"Database exists: {db_path.exists()}")
print(f"Database path: {db_path}")
```

---

## 📚 Resources

### Documentation
- `00_TRAINING_OVERVIEW.md` - Full training guide
- `01_DATA_LOADER_GUIDE.md` - Data loading guide
- `../docs/SENTINEL_LAYERS_OVERVIEW_2026-02-22.md` - Architecture overview
- `../TODO_SENTINEL_PROJECT_2026-02-22.md` - Project checklist

### Code Examples
- `../src/experts/linear_model.py` - Linear model implementation
- `../src/experts/xgboost_model.py` - XGBoost implementation
- `../src/regime/hmm_detector.py` - HMM regime detector
- `../src/risk/monte_carlo.py` - Monte Carlo simulator

---

## 🎓 Learning Path

If you're new to Jupyter notebooks:
1. Start with `01_data_exploration.ipynb` 
2. Run each cell (Shift+Enter)
3. Modify code and re-run to experiment
4. Create new notebooks as you progress

**Pro tip**: Save your notebooks frequently (Ctrl+S)

---

## ✅ Checklist

- [ ] Run `01_data_exploration.ipynb`
- [ ] Verify database has your sector tickers
- [ ] Check data for training period (2019-2024)
- [ ] Create `src/data/data_loader.py`
- [ ] Test data loader in a notebook
- [ ] Begin model training

---

**Ready to start?** Open `01_data_exploration.ipynb` in Jupyter!

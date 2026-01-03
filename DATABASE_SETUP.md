# Database Setup for Model Regime Comparison

## ✅ Database Type: SQLite (NOT PostgreSQL)

The project uses **SQLite** to connect with the existing `financial_data_aggregator` project.

### Key Points

1. **Database Location**: `financial_data.db` (created by financial_data_aggregator)
2. **Connection String**: 
   ```
   sqlite:///financial_data.db
   ```
3. **No External Dependencies**: SQLite is built into Python - no separate database server needed

### Configuration

The database URL is set in `config/config.py`:

```python
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///financial_data.db"
)
```

**Using the default location**: The SQLite database file should be at:
```
/home/archy/Desktop/Server/FinancialData/financial_data.db
```

### How to Override

If your database is in a different location:

```bash
# Option 1: Set environment variable
export DATABASE_URL="sqlite:////absolute/path/to/financial_data.db"

# Option 2: Create .env file
echo 'DATABASE_URL=sqlite:////absolute/path/to/financial_data.db' > .env
```

**Note the 4 slashes**: `sqlite:////` = absolute path, `sqlite:///` = relative path

## Database Schema

The SQLite database contains these tables (created by financial_data_aggregator):

### fact_stock_price
```sql
CREATE TABLE fact_stock_price (
    ticker TEXT,
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
)
```

### dim_company
```sql
CREATE TABLE dim_company (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap FLOAT
)
```

### fact_sec_filing
```sql
CREATE TABLE fact_sec_filing (
    ticker TEXT,
    filing_date DATE,
    filing_type TEXT,
    text TEXT
)
```

## Data Loading Template

A template has been provided at `src/data/data_loader_template.py` showing how to:
- Connect to the SQLite database
- Load stock price data
- Load company metadata
- Load SEC filings
- Validate data

### Basic Usage

```python
from src.data.data_loader_template import load_stock_prices
from datetime import date

# Load prices for 2024
prices = load_stock_prices(
    tickers=['AAPL', 'MSFT'],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

print(prices.head())
```

## Verifying the Setup

To verify the SQLite database is accessible:

```bash
cd /home/archy/Desktop/Server/FinancialData/model_regime_comparison

# Option 1: Python test
python3 << 'EOF'
from sqlalchemy import create_engine
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    result = conn.execute("SELECT COUNT(*) FROM fact_stock_price")
    count = result.scalar()
    print(f"✓ Database connected! Found {count} price records")
EOF

# Option 2: SQLite CLI (if installed)
sqlite3 /path/to/financial_data.db ".tables"
```

## Requirements

✅ No additional packages needed - SQLAlchemy handles SQLite automatically

See `requirements.txt` for full list (note: no psycopg2, no postgresql drivers)

## Next Steps

1. **Verify financial_data_aggregator** has populated `financial_data.db`
2. **Test the connection** using the Python snippet above
3. **Implement `src/data/data_loader.py`** based on the template
4. **Write unit tests** to verify data loading works

## Troubleshooting

**Error: "No such table: fact_stock_price"**
- financial_data_aggregator hasn't run yet
- Run the data collection pipeline in financial_data_aggregator

**Error: "unable to open database file"**
- DATABASE_URL path is incorrect
- Check the file exists: `ls -la /path/to/financial_data.db`
- Fix the path in `config/config.py` or `.env`

**Error: "table fact_stock_price has X records but I need Y"**
- Not enough historical data loaded
- Run the full data collection pipeline for sufficient history

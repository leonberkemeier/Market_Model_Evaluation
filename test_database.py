#!/usr/bin/env python3
"""
Quick test script to verify SQLite database connection and available data.
Run this before starting development to ensure everything is set up correctly.
"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATABASE_URL

def test_connection():
    """Test that we can connect to the database."""
    print("=" * 80)
    print("Testing Database Connection")
    print("=" * 80)
    
    print(f"\n📊 Database URL: {DATABASE_URL}")
    
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            print("✅ Successfully connected to SQLite database!")
            return engine
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)


def test_tables(engine):
    """Check what tables exist in the database."""
    print("\n" + "=" * 80)
    print("Available Tables")
    print("=" * 80)
    
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    if not tables:
        print("❌ No tables found in database!")
        print("   Make sure financial_data_aggregator has populated the database.")
        return False
    
    print(f"\n✅ Found {len(tables)} tables:\n")
    for table in sorted(tables):
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  • {table:30s} ({count:,} records)")
    
    return True


def test_stock_prices(engine):
    """Test loading stock price data."""
    print("\n" + "=" * 80)
    print("Testing Stock Price Data")
    print("=" * 80)
    
    try:
        with engine.connect() as conn:
            # Check if fact_stock_price table exists
            result = conn.execute(text("""
                SELECT COUNT(*) as count FROM fact_stock_price
            """))
            count = result.scalar()
            
            if count == 0:
                print("❌ No stock price data found!")
                return False
            
            print(f"✅ Found {count:,} stock price records")
            
            # Get sample data
            df = pd.read_sql(text("""
                SELECT * FROM fact_stock_price LIMIT 5
            """), conn)
            
            print(f"\n📋 Sample data (first 5 records):")
            print(df.to_string(index=False))
            
            return True
    except Exception as e:
        print(f"❌ Error loading stock prices: {e}")
        return False


def test_companies(engine):
    """Test loading company metadata."""
    print("\n" + "=" * 80)
    print("Testing Company Metadata")
    print("=" * 80)
    
    try:
        with engine.connect() as conn:
            # Check if dim_company table exists
            result = conn.execute(text("""
                SELECT COUNT(*) as count FROM dim_company
            """))
            count = result.scalar()
            
            if count == 0:
                print("❌ No company data found!")
                return False
            
            print(f"✅ Found {count:,} companies")
            
            # Get sample data
            df = pd.read_sql(text("""
                SELECT * FROM dim_company LIMIT 5
            """), conn)
            
            print(f"\n📋 Sample data (first 5 records):")
            print(df.to_string(index=False))
            
            return True
    except Exception as e:
        print(f"❌ Error loading companies: {e}")
        return False


def test_sec_filings(engine):
    """Test SEC filings data."""
    print("\n" + "=" * 80)
    print("Testing SEC Filings Data")
    print("=" * 80)
    
    try:
        with engine.connect() as conn:
            # Check if fact_sec_filing table exists
            result = conn.execute(text("""
                SELECT COUNT(*) as count FROM fact_sec_filing
            """))
            count = result.scalar()
            
            if count == 0:
                print("⚠️  No SEC filing data found (optional for MVP)")
                return True
            
            print(f"✅ Found {count:,} SEC filing records")
            
            # Get sample data
            df = pd.read_sql(text("""
                SELECT * FROM fact_sec_filing LIMIT 3
            """), conn)
            
            print(f"\n📋 Sample data (first 3 records):")
            for idx, row in df.iterrows():
                print(f"\n  Record {idx + 1}:")
                for col in df.columns:
                    val = str(row[col])[:100]  # Truncate long text
                    print(f"    {col:20s}: {val}")
            
            return True
    except Exception as e:
        print(f"⚠️  Error loading filings (not critical): {e}")
        return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Model Regime Comparison - Database Connection Test".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Test connection
    engine = test_connection()
    
    # Test tables
    if not test_tables(engine):
        print("\n❌ Database test FAILED - no tables found")
        sys.exit(1)
    
    # Test specific tables
    results = []
    results.append(("Stock Prices", test_stock_prices(engine)))
    results.append(("Companies", test_companies(engine)))
    results.append(("SEC Filings", test_sec_filings(engine)))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8s} {name}")
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("✅ All tests PASSED! Database is ready to use.")
        print("\n📖 Next steps:")
        print("  1. Implement src/data/data_loader.py")
        print("  2. Implement model scorers in src/scorers/")
        print("  3. Test with: pytest tests/test_kelly.py -v")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED. Please fix the issues above.")
        print("\n💡 Troubleshooting:")
        print("  1. Ensure financial_data_aggregator is running")
        print("  2. Check DATABASE_URL in config/config.py")
        print("  3. Verify financial_data.db exists and is populated")
        sys.exit(1)


if __name__ == "__main__":
    main()

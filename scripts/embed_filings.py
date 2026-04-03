#!/usr/bin/env python3
"""
Filing Embedding Backfill Script

Processes SEC filings through the two-stage embedding pipeline:
  Stage 1: Deterministic cleaning (always runs, no LLM needed)
  Stage 2: LLM normalization (optional, requires Ollama server)

Stores normalized chunks into ChromaDB 'sec_filings_normalized' collection.

Usage:
    python scripts/embed_filings.py                    # All filings, with LLM
    python scripts/embed_filings.py --no-llm           # Deterministic only
    python scripts/embed_filings.py --ticker AAPL      # Single ticker
    python scripts/embed_filings.py --dry-run          # Extract & clean, show stats, don't embed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import create_engine, text

from src.config.config import DATABASE_URL
from src.nlp.filing_vectorstore import FilingVectorStore


def get_filings(engine, ticker: str = None):
    """Query all filings with text content."""
    conditions = ["f.filing_text IS NOT NULL", "LENGTH(f.filing_text) > 500"]
    if ticker:
        conditions.append("c.ticker = :ticker")

    where = " AND ".join(conditions)
    query = f"""
        SELECT
            f.filing_id,
            c.ticker,
            ft.filing_type,
            d.date as filing_date,
            f.filing_text
        FROM fact_sec_filing f
        JOIN dim_company c ON f.company_id = c.company_id
        JOIN dim_filing_type ft ON f.filing_type_id = ft.filing_type_id
        JOIN dim_date d ON f.date_id = d.date_id
        WHERE {where}
        ORDER BY c.ticker, d.date
    """

    params = {"ticker": ticker} if ticker else {}
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


def main():
    parser = argparse.ArgumentParser(description="Embed SEC filings into ChromaDB")
    parser.add_argument("--ticker", type=str, help="Process only this ticker")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM normalization (deterministic cleaning only)")
    parser.add_argument("--dry-run", action="store_true", help="Extract and clean, show stats, don't embed")
    args = parser.parse_args()

    logger.add("logs/embed_filings.log", rotation="10 MB")

    store = FilingVectorStore()
    engine = create_engine(DATABASE_URL)

    # Query filings
    filings = get_filings(engine, ticker=args.ticker)
    if not filings:
        logger.info("No filings with text content found.")
        return

    logger.info(f"Found {len(filings)} filing(s) to process")

    total_chunks = 0
    total_stored = 0

    for i, filing in enumerate(filings, 1):
        ticker = filing["ticker"]
        filing_type = filing["filing_type"]
        filing_date = filing["filing_date"]
        filing_id = filing["filing_id"]

        logger.info(f"[{i}/{len(filings)}] {ticker} {filing_type} {filing_date}")

        if args.dry_run:
            # Stage 1 only — show what we'd embed
            chunks = store.extract_and_clean(
                filing["filing_text"], ticker, filing_type,
                str(filing_date), filing_id,
            )
            total_chunks += len(chunks)
            for c in chunks:
                logger.info(
                    f"  {c.metadata['section']} chunk {c.metadata['chunk_index']}: "
                    f"{len(c.text)} chars"
                )
            continue

        try:
            stored = store.process_filing(
                filing_text=filing["filing_text"],
                ticker=ticker,
                filing_type=filing_type,
                filing_date=str(filing_date),
                filing_id=filing_id,
                use_llm=not args.no_llm,
            )
            total_stored += stored
            total_chunks += stored
            logger.info(f"  ✓ {stored} chunks embedded")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")

    if args.dry_run:
        logger.info(f"\nDry run: {total_chunks} chunks would be embedded from {len(filings)} filings")
    else:
        logger.info(f"\nDone: {total_stored} chunks embedded from {len(filings)} filings")


if __name__ == "__main__":
    main()

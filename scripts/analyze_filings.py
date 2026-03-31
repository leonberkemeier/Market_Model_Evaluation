#!/usr/bin/env python3
"""
SEC Filing NLP Analysis Script

Queries filings from fact_sec_filing that have text but haven't been
analyzed by the LLM yet, runs FilingNLPAnalyzer on each, and stores
the results back into fact_filing_analysis.

Usage:
    python scripts/analyze_filings.py                    # Analyze all pending
    python scripts/analyze_filings.py --ticker AAPL      # Only AAPL
    python scripts/analyze_filings.py --force             # Re-analyze all
    python scripts/analyze_filings.py --model llama3.1:8b # Override model
    python scripts/analyze_filings.py --test              # Connectivity test only
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import create_engine, text

from src.config.config import DATABASE_URL, OLLAMA_MODEL
from src.nlp.filing_analyzer import FilingNLPAnalyzer


def get_pending_filings(engine, ticker: str = None, force: bool = False):
    """Query filings that need LLM analysis."""
    conditions = ["f.filing_text IS NOT NULL", "LENGTH(f.filing_text) > 500"]

    if not force:
        conditions.append("fa.llm_analyzed_at IS NULL")

    if ticker:
        conditions.append("c.ticker = :ticker")

    where = " AND ".join(conditions)

    query = f"""
        SELECT
            f.filing_id,
            c.ticker,
            ft.filing_type,
            d.date as filing_date,
            f.filing_text,
            fa.analysis_id
        FROM fact_sec_filing f
        JOIN dim_company c ON f.company_id = c.company_id
        JOIN dim_filing_type ft ON f.filing_type_id = ft.filing_type_id
        JOIN dim_date d ON f.date_id = d.date_id
        LEFT JOIN fact_filing_analysis fa ON f.filing_id = fa.filing_id
        WHERE {where}
        ORDER BY c.ticker, d.date
    """

    params = {"ticker": ticker} if ticker else {}

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


def update_analysis(engine, filing_id: int, analysis_id: int, result_dict: dict):
    """Update fact_filing_analysis with LLM results."""
    with engine.connect() as conn:
        if analysis_id:
            # Update existing row
            conn.execute(
                text("""
                    UPDATE fact_filing_analysis SET
                        mgmt_sentiment = :mgmt_sentiment,
                        risk_count = :risk_count,
                        risk_summary = :risk_summary,
                        revenue_extracted = :revenue_extracted,
                        operating_income_extracted = :operating_income_extracted,
                        guidance_tone = :guidance_tone,
                        llm_model = :llm_model,
                        llm_analyzed_at = :llm_analyzed_at,
                        updated_at = :llm_analyzed_at
                    WHERE filing_id = :filing_id
                """),
                {**result_dict, "filing_id": filing_id, "llm_analyzed_at": datetime.now(timezone.utc)},
            )
        else:
            # Need to create a row first — get company_id and date_id
            row = conn.execute(
                text("""
                    SELECT f.company_id, f.date_id
                    FROM fact_sec_filing f WHERE f.filing_id = :fid
                """),
                {"fid": filing_id},
            ).fetchone()

            if row:
                conn.execute(
                    text("""
                        INSERT INTO fact_filing_analysis
                            (filing_id, company_id, date_id,
                             mgmt_sentiment, risk_count, risk_summary,
                             revenue_extracted, operating_income_extracted,
                             guidance_tone, llm_model, llm_analyzed_at)
                        VALUES
                            (:filing_id, :company_id, :date_id,
                             :mgmt_sentiment, :risk_count, :risk_summary,
                             :revenue_extracted, :operating_income_extracted,
                             :guidance_tone, :llm_model, :llm_analyzed_at)
                    """),
                    {
                        **result_dict,
                        "filing_id": filing_id,
                        "company_id": row[0],
                        "date_id": row[1],
                        "llm_analyzed_at": datetime.now(timezone.utc),
                    },
                )
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Analyze SEC filings with LLM")
    parser.add_argument("--ticker", type=str, help="Analyze only this ticker")
    parser.add_argument("--force", action="store_true", help="Re-analyze already processed filings")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--test", action="store_true", help="Run connectivity test only")
    args = parser.parse_args()

    logger.add("logs/analyze_filings.log", rotation="10 MB")

    # Initialize analyzer
    analyzer = FilingNLPAnalyzer(model=args.model)

    # ── Connectivity test ──
    logger.info(f"Testing connection to Ollama ({analyzer.base_url}, model={analyzer.model})...")
    if not analyzer.test_connectivity():
        logger.error("Ollama connectivity test FAILED. Exiting.")
        sys.exit(1)

    if args.test:
        logger.info("Connectivity test passed. Exiting (--test mode).")
        return

    # ── Query pending filings ──
    engine = create_engine(DATABASE_URL)
    filings = get_pending_filings(engine, ticker=args.ticker, force=args.force)

    if not filings:
        logger.info("No filings to analyze.")
        return

    logger.info(f"Found {len(filings)} filing(s) to analyze")

    # ── Process each filing ──
    success = 0
    failed = 0

    for i, filing in enumerate(filings, 1):
        ticker = filing["ticker"]
        filing_type = filing["filing_type"]
        filing_date = filing["filing_date"]
        filing_id = filing["filing_id"]
        analysis_id = filing["analysis_id"]

        logger.info(f"[{i}/{len(filings)}] {ticker} {filing_type} {filing_date}")

        try:
            result = analyzer.analyze_filing(
                filing_text=filing["filing_text"],
                ticker=ticker,
                filing_type=filing_type,
                filing_date=str(filing_date),
            )

            result_dict = result.to_dict()
            update_analysis(engine, filing_id, analysis_id, result_dict)

            logger.info(
                f"  ✓ Stored: sentiment={result.mgmt_sentiment}, "
                f"risks={result.risk_count}, guidance={result.guidance_tone}, "
                f"sections={result.sections_found}"
            )
            success += 1

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed += 1

    logger.info(f"\nDone: {success} succeeded, {failed} failed out of {len(filings)} filings")


if __name__ == "__main__":
    main()

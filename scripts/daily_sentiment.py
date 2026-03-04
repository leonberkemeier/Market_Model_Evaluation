#!/usr/bin/env python3
"""
Daily Sentiment Scorer

Standalone script designed to run once per day (cron / systemd timer).
Fetches headlines from Finnhub, scores them with FinBERT, and stores
the EWMA sentiment score in fact_sentiment.

Usage:
    python scripts/daily_sentiment.py                  # Score today
    python scripts/daily_sentiment.py --date 2025-06-15  # Score specific date
    python scripts/daily_sentiment.py --backfill 30      # Backfill last 30 days

Cron example (run daily at 07:00):
    0 7 * * * cd /path/to/model_regime_comparison && venv/bin/python scripts/daily_sentiment.py
"""

import sys
import argparse
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from src.config.config import ALL_TICKERS, LOGS_DIR
from src.data.data_loader import DataLoader
from src.nlp.sentiment_scorer import SentimentScorer


def setup_logging():
    """Configure logging for the cron job."""
    logger.remove()
    log_fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    logger.add(LOGS_DIR / "daily_sentiment.log", format=log_fmt, level="INFO", rotation="7 days")
    logger.add(sys.stdout, format=log_fmt, level="INFO")


def score_date(
    scorer: SentimentScorer,
    loader: DataLoader,
    tickers: list,
    target_date: date,
) -> dict:
    """
    Score all tickers for a single date and store results.

    Returns summary dict with counts.
    """
    scored = 0
    skipped = 0
    failed = 0

    for ticker in tickers:
        try:
            # Idempotent: skip if already scored
            if loader.has_sentiment(ticker, target_date):
                skipped += 1
                continue

            result = scorer.compute_ewma_score(ticker, target_date)

            if result is None:
                failed += 1
                continue

            ewma_score, n_headlines, raw_scores = result

            success = loader.store_sentiment(
                ticker=ticker,
                score_date=target_date,
                sentiment_score=ewma_score,
                n_headlines=n_headlines,
                raw_scores=raw_scores,
            )

            if success:
                scored += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"{ticker}: unexpected error — {e}")
            failed += 1

    return {"scored": scored, "skipped": skipped, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Daily FinBERT Sentiment Scorer")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Score for a specific date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--backfill", type=int, default=0,
        help="Backfill N days ending at --date (inclusive).",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override ticker universe.",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("Daily Sentiment Scorer")
    logger.info("=" * 60)

    # Resolve target date(s)
    if args.date:
        end_date = date.fromisoformat(args.date)
    else:
        end_date = date.today()

    if args.backfill > 0:
        dates = [end_date - timedelta(days=i) for i in range(args.backfill)]
        dates.reverse()  # Oldest first
    else:
        dates = [end_date]

    tickers = args.tickers or list(set(ALL_TICKERS))  # Deduplicate
    logger.info(f"Tickers: {len(tickers)}, Dates: {len(dates)}")

    # Init
    loader = DataLoader()
    loader.ensure_sentiment_table()
    scorer = SentimentScorer()

    total_scored = 0
    total_skipped = 0
    total_failed = 0

    for target_date in dates:
        logger.info(f"--- Scoring {target_date} ---")
        summary = score_date(scorer, loader, tickers, target_date)
        total_scored += summary["scored"]
        total_skipped += summary["skipped"]
        total_failed += summary["failed"]

        logger.info(
            f"  {target_date}: scored={summary['scored']}, "
            f"skipped={summary['skipped']}, failed={summary['failed']}"
        )

    logger.info("=" * 60)
    logger.info(
        f"TOTAL: scored={total_scored}, skipped={total_skipped}, "
        f"failed={total_failed}"
    )
    logger.info("=" * 60)

    # Exit code: 1 if everything failed, 0 otherwise
    if total_scored == 0 and total_failed > 0 and total_skipped == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

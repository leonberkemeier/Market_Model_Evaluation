"""
News-Decay "Pulse" Engine

Implements the time-decay sentiment aggregation as a SQL materialized view
(PostgreSQL) with a SQLite fallback (regular view).

Formula per ticker:
    pulse_score = SUM( sentiment_score * (1 - 0.02 * days_passed) )

where days_passed = current_date - score_date, aggregated over the last
50 days of fact_sentiment entries.

The view produces a single float per ticker: higher = more positive recent
sentiment with recent headlines weighted more heavily.

Usage:
    from src.advisory.news_pulse import refresh_news_pulse, get_news_pulse

    refresh_news_pulse()                    # create / refresh the view
    pulse = get_news_pulse()                # {ticker: pulse_score}
    pulse_aapl = get_news_pulse("AAPL")     # {"AAPL": 0.42}
"""

from typing import Dict, Optional

from loguru import logger
from sqlalchemy import text

from src.advisory.models import get_advisory_engine


# ── SQL definitions ───────────────────────────────────────────────────────

# PostgreSQL: true MATERIALIZED VIEW (REFRESH CONCURRENTLY safe with
# unique index on ticker).
_PG_DROP = "DROP MATERIALIZED VIEW IF EXISTS view_news_pulse"
_PG_CREATE = """\
CREATE MATERIALIZED VIEW view_news_pulse AS
SELECT
    c.ticker,
    SUM(
        fs.sentiment_score
        * (1.0 - 0.02 * EXTRACT(DAY FROM (CURRENT_TIMESTAMP - d.date::timestamp)))
    ) AS pulse_score
FROM fact_sentiment fs
JOIN dim_company   c ON fs.company_id = c.company_id
JOIN dim_date      d ON fs.date_id    = d.date_id
WHERE d.date >= (CURRENT_DATE - INTERVAL '50 days')
GROUP BY c.ticker
"""
_PG_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_news_pulse_ticker "
    "ON view_news_pulse (ticker)"
)
_PG_REFRESH = "REFRESH MATERIALIZED VIEW CONCURRENTLY view_news_pulse"

# SQLite: regular view (no MATERIALIZED support).  julianday() for date diff.
_SQLITE_DROP = "DROP VIEW IF EXISTS view_news_pulse"
_SQLITE_CREATE = """\
CREATE VIEW view_news_pulse AS
SELECT
    c.ticker,
    SUM(
        fs.sentiment_score
        * (1.0 - 0.02 * (julianday('now') - julianday(d.date)))
    ) AS pulse_score
FROM fact_sentiment fs
JOIN dim_company   c ON fs.company_id = c.company_id
JOIN dim_date      d ON fs.date_id    = d.date_id
WHERE julianday('now') - julianday(d.date) <= 50
GROUP BY c.ticker
"""


# ── Helpers ───────────────────────────────────────────────────────────────

def _is_postgres(engine) -> bool:
    """Return True if the engine dialect is PostgreSQL."""
    return engine.dialect.name in ("postgresql", "postgres")


def refresh_news_pulse(engine=None) -> None:
    """
    Create or refresh the view_news_pulse view.

    * PostgreSQL: creates a MATERIALIZED VIEW on first call, then
      REFRESH MATERIALIZED VIEW CONCURRENTLY on subsequent calls.
    * SQLite: drops and re-creates a regular VIEW.

    Safe to call repeatedly (idempotent).
    """
    if engine is None:
        engine = get_advisory_engine()

    pg = _is_postgres(engine)

    with engine.begin() as conn:
        if pg:
            # Check if view already exists
            exists = conn.execute(text(
                "SELECT 1 FROM pg_matviews WHERE matviewname = 'view_news_pulse'"
            )).fetchone()

            if exists:
                conn.execute(text(_PG_REFRESH))
                logger.info("view_news_pulse: refreshed (PostgreSQL CONCURRENTLY)")
            else:
                conn.execute(text(_PG_DROP))
                conn.execute(text(_PG_CREATE))
                conn.execute(text(_PG_INDEX))
                logger.info("view_news_pulse: created materialized view (PostgreSQL)")
        else:
            # SQLite: recreate
            conn.execute(text(_SQLITE_DROP))
            conn.execute(text(_SQLITE_CREATE))
            logger.info("view_news_pulse: created view (SQLite)")


def get_news_pulse(
    ticker: Optional[str] = None,
    engine=None,
) -> Dict[str, float]:
    """
    Read the current News Pulse scores.

    Args:
        ticker: If provided, return only this ticker.  Otherwise return all.
        engine: SQLAlchemy engine (uses default advisory engine if None).

    Returns:
        Dict mapping ticker → pulse_score.
    """
    if engine is None:
        engine = get_advisory_engine()

    if ticker:
        query = text(
            "SELECT ticker, pulse_score FROM view_news_pulse WHERE ticker = :t"
        )
        params = {"t": ticker}
    else:
        query = text("SELECT ticker, pulse_score FROM view_news_pulse")
        params = {}

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    result = {row[0]: float(row[1]) if row[1] is not None else 0.0 for row in rows}

    if not result and ticker:
        logger.debug(f"No pulse score for {ticker}")

    return result

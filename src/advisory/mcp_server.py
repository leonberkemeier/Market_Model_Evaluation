"""
Local-Optimized MCP Server for Gemma 3 Decision Agent

Provides three tools via the Model Context Protocol (FastMCP):

1. get_candidates(category_id)   — Top 10 stocks whose Expected Shortfall
   fits the user's risk category.
2. get_ticker_details(ticker)    — MCMC stats + News Pulse + top 3 Key
   Risks from ChromaDB.
3. execute_rebalance(user_id)    — Compares current holdings vs target
   HRP weights and returns a buy/sell order plan.

All tool outputs are kept concise (compact JSON, truncated text) to stay
within Gemma 3:12B's context window.

Usage:
    # As a standalone MCP server (stdio transport):
    python -m src.advisory.mcp_server

    # Programmatic access (used by gemma_advisor_logic.py):
    from src.advisory.mcp_server import get_candidates, get_ticker_details, execute_rebalance
"""

import json
from typing import Optional

from fastmcp import FastMCP
from loguru import logger
from sqlalchemy import text

from src.config.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    RAG_CHROMA_PATH,
    EMBEDDING_COLLECTION,
    TRADING_SIMULATOR_URL,
)
from src.advisory.models import (
    get_advisory_engine,
    get_session,
    DimRiskCategory,
    DimUserProfile,
    FactAssetIntelligence,
)
from src.advisory.news_pulse import get_news_pulse

# ── MCP Server instance ──────────────────────────────────────────────────

mcp = FastMCP(
    name="SentinelAdvisory",
)

# HMM state label map (matches HMMRegimeDetector convention)
_HMM_LABELS = {0: "Bull", 1: "Bear", 2: "Sideways"}


# ── Tool 1: get_candidates ───────────────────────────────────────────────

@mcp.tool
def get_candidates(category_id: int) -> str:
    """
    Find the Top 10 stocks whose Expected Shortfall fits the given risk
    category (1 = Conservative, 5 = Aggressive).

    Returns a compact JSON array of candidate objects sorted by
    win_probability descending.
    """
    session = get_session()
    try:
        # Look up ES threshold for this category
        cat = session.query(DimRiskCategory).filter_by(category_id=category_id).first()
        if cat is None:
            return json.dumps({"error": f"Unknown category_id {category_id}. Valid: 1-5."})

        es_threshold = cat.es_threshold

        # Query latest intelligence row per ticker, filter by ES, rank by win_prob
        # Using raw SQL for window-function support across dialects
        engine = get_advisory_engine()
        query = text("""
            WITH latest AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) AS rn
                FROM fact_asset_intelligence
            )
            SELECT ticker, mu_posterior, sigma_posterior,
                   expected_shortfall_5pct AS es5, win_probability, hmm_state
            FROM latest
            WHERE rn = 1
              AND expected_shortfall_5pct >= :es_threshold
            ORDER BY win_probability DESC
            LIMIT 10
        """)

        with engine.connect() as conn:
            rows = conn.execute(query, {"es_threshold": es_threshold}).fetchall()

        candidates = [
            {
                "ticker": r[0],
                "mu": round(r[1], 5),
                "sigma": round(r[2], 5),
                "es5": round(r[3], 4),
                "win_prob": round(r[4], 3),
                "regime": _HMM_LABELS.get(r[5], "?"),
            }
            for r in rows
        ]

        logger.info(
            f"get_candidates(cat={category_id}, ES>{es_threshold}): "
            f"{len(candidates)} results"
        )
        return json.dumps(candidates)

    finally:
        session.close()


# ── Tool 2: get_ticker_details ────────────────────────────────────────────

def _get_chroma_risks(ticker: str, n: int = 3) -> list[str]:
    """Query ChromaDB for top N risk-factor snippets for a ticker."""
    try:
        import chromadb
        import httpx

        client = chromadb.PersistentClient(path=RAG_CHROMA_PATH)
        collection = client.get_or_create_collection(name=EMBEDDING_COLLECTION)

        if collection.count() == 0:
            return []

        # Embed a risk-focused query
        with httpx.Client(timeout=30) as http:
            resp = http.post(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": [f"{ticker} key risks and concerns"]},
            )
            resp.raise_for_status()
            query_embedding = resp.json().get("embeddings", [[]])[0]

        if not query_embedding:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where={"ticker": ticker, "section": "risk_factors"} if collection.count() > 0 else None,
        )

        snippets = []
        for doc in (results.get("documents") or [[]])[0]:
            # Truncate to ~200 chars for Gemma's context budget
            snippet = doc[:200].strip()
            if len(doc) > 200:
                snippet += "…"
            snippets.append(snippet)

        return snippets

    except Exception as e:
        logger.debug(f"ChromaDB risk query failed for {ticker}: {e}")
        return []


@mcp.tool
def get_ticker_details(ticker: str) -> str:
    """
    Get detailed analysis for a single ticker: MCMC posterior stats,
    News Pulse score, and the latest 3 Key Risks from SEC filings.

    Returns a compact JSON object.
    """
    engine = get_advisory_engine()

    # 1. MCMC stats from fact_asset_intelligence (latest row)
    query = text("""
        SELECT ticker, mu_posterior, sigma_posterior,
               expected_shortfall_5pct, win_probability, hmm_state, timestamp
        FROM fact_asset_intelligence
        WHERE ticker = :ticker
        ORDER BY timestamp DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        row = conn.execute(query, {"ticker": ticker}).fetchone()

    if row is None:
        return json.dumps({"error": f"No intelligence data for {ticker}"})

    detail = {
        "ticker": row[0],
        "mu": round(row[1], 5),
        "sigma": round(row[2], 5),
        "es5": round(row[3], 4),
        "win_prob": round(row[4], 3),
        "regime": _HMM_LABELS.get(row[5], "?"),
        "as_of": str(row[6]),
    }

    # 2. News Pulse
    pulse = get_news_pulse(ticker=ticker, engine=engine)
    detail["news_pulse"] = round(pulse.get(ticker, 0.0), 3)

    # 3. Key Risks from ChromaDB
    risks = _get_chroma_risks(ticker, n=3)
    if risks:
        detail["key_risks"] = risks

    logger.info(f"get_ticker_details({ticker}): regime={detail['regime']}, pulse={detail['news_pulse']}")
    return json.dumps(detail)


# ── Tool 3: execute_rebalance ─────────────────────────────────────────────

@mcp.tool
def execute_rebalance(user_id: int) -> str:
    """
    Compare current portfolio holdings vs target weights derived from
    the latest MCMC intelligence, and compute a buy/sell order plan.

    Returns a JSON object with the proposed orders (dry-run; no trades
    are executed until confirmed).
    """
    session = get_session()
    try:
        # Look up user profile
        user = session.query(DimUserProfile).filter_by(user_id=user_id).first()
        if user is None:
            return json.dumps({"error": f"Unknown user_id {user_id}"})

        portfolio_id = user.portfolio_id
        category_id = user.category_id

        # Get ES threshold for category
        cat = session.query(DimRiskCategory).filter_by(category_id=category_id).first()
        es_threshold = cat.es_threshold if cat else -0.125  # moderate default

        # Get latest intelligence rows fitting category
        engine = get_advisory_engine()
        query = text("""
            WITH latest AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY timestamp DESC) AS rn
                FROM fact_asset_intelligence
            )
            SELECT ticker, mu_posterior, win_probability
            FROM latest
            WHERE rn = 1
              AND expected_shortfall_5pct >= :es_threshold
              AND win_probability > 0.5
            ORDER BY win_probability DESC
            LIMIT 20
        """)

        with engine.connect() as conn:
            rows = conn.execute(query, {"es_threshold": es_threshold}).fetchall()

        if not rows:
            return json.dumps({
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "orders": [],
                "note": "No eligible assets found for rebalancing",
            })

        # Compute HRP-style weights (simplified inverse-variance weighting)
        # Full HRP would require a covariance matrix; here we use
        # win_probability as a proxy for quality-weighted allocation
        total_wp = sum(r[2] for r in rows)
        target_weights = {r[0]: round(r[2] / total_wp, 4) for r in rows}

        # Fetch current holdings from Trading Simulator
        from src.execution.api_client import TradingSimulatorClient

        client = TradingSimulatorClient(base_url=TRADING_SIMULATOR_URL)
        try:
            portfolio = client.get_portfolio(portfolio_id)
            holdings = client.get_holdings(portfolio_id)
        except Exception as e:
            return json.dumps({
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "error": f"Trading API unavailable: {e}",
                "target_weights": target_weights,
            })

        current_weights = {}
        for h in holdings:
            if portfolio.total_value > 0:
                current_weights[h.ticker] = round(h.total_value / portfolio.total_value, 4)

        # Compute deltas
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())
        orders = []
        for t in sorted(all_tickers):
            target_w = target_weights.get(t, 0.0)
            current_w = current_weights.get(t, 0.0)
            delta = round(target_w - current_w, 4)

            if abs(delta) < 0.005:
                continue  # within tolerance

            dollar_delta = round(delta * portfolio.total_value, 2)
            orders.append({
                "ticker": t,
                "action": "BUY" if delta > 0 else "SELL",
                "weight_delta": delta,
                "est_dollar_amount": abs(dollar_delta),
            })

        result = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "portfolio_value": round(portfolio.total_value, 2),
            "n_target_assets": len(target_weights),
            "n_orders": len(orders),
            "orders": orders,
        }

        logger.info(
            f"execute_rebalance(user={user_id}): "
            f"{len(orders)} orders, portfolio=${portfolio.total_value:,.0f}"
        )
        return json.dumps(result)

    finally:
        session.close()


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()

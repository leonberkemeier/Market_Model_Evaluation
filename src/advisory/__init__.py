"""
Robo-Advisory System — Gemma 3 Decision Agent

Extends the Bayesian quantitative pipeline into a multi-user advisory
system powered by a locally hosted Gemma 3:12B via Ollama.

Modules:
    models          — SQLAlchemy ORM for fact_asset_intelligence
    news_pulse      — Time-decay News Pulse materialized view
    mcp_server      — FastMCP server bridging Gemma 3 to the data layer
    gemma_advisor_logic — Decision agent orchestration loop
"""

from .models import (
    FactAssetIntelligence,
    DimRiskCategory,
    DimUserProfile,
    refresh_assets,
    get_advisory_engine,
    init_advisory_tables,
)
from .news_pulse import refresh_news_pulse, get_news_pulse

__all__ = [
    "FactAssetIntelligence",
    "DimRiskCategory",
    "DimUserProfile",
    "refresh_assets",
    "get_advisory_engine",
    "init_advisory_tables",
    "refresh_news_pulse",
    "get_news_pulse",
]

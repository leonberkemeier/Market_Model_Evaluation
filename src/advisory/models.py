"""
Asset Intelligence ORM Models

SQLAlchemy declarative models for the advisory system's core tables:

- fact_asset_intelligence : Daily MCMC results deposited by the Bayesian
  math team.  Composite PK on (ticker, timestamp).
- dim_risk_category      : Static lookup mapping category_id (1–5) to
  expected-shortfall thresholds used by get_candidates().
- dim_user_profile       : Per-user risk category and portfolio link.
"""

from datetime import datetime, timedelta, timezone
from typing import List

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    SmallInteger,
    String,
    CheckConstraint,
    create_engine,
    delete,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    sessionmaker,
)

from src.config.config import DATABASE_URL


# ── Declarative base ──────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── fact_asset_intelligence ───────────────────────────────────────────────

class FactAssetIntelligence(Base):
    """
    Handoff table where the Bayesian pipeline deposits daily results.

    One row per (ticker, timestamp).  Consumers: MCP tools, Gemma advisor.
    """

    __tablename__ = "fact_asset_intelligence"

    ticker = Column(String(20), primary_key=True, comment="Ticker symbol")
    timestamp = Column(
        DateTime(timezone=True),
        primary_key=True,
        default=lambda: datetime.now(timezone.utc),
        comment="UTC timestamp of the computation run",
    )

    # Bayesian posterior from Tier 3
    mu_posterior = Column(Float, nullable=False, comment="Posterior mean (expected return)")
    sigma_posterior = Column(Float, nullable=False, comment="Posterior std (uncertainty)")

    # Risk metric from Copula MC
    expected_shortfall_5pct = Column(
        Float,
        nullable=False,
        comment="5 % Expected Shortfall (CVaR) — always negative or zero",
    )

    # Win probability from Kelly pipeline
    win_probability = Column(
        Float,
        nullable=False,
        comment="P(positive return) from posterior predictive",
    )

    # HMM regime state (Tier 2 output)
    hmm_state = Column(
        SmallInteger,
        CheckConstraint("hmm_state IN (0, 1, 2)", name="ck_hmm_state"),
        nullable=False,
        comment="0 = Bull, 1 = Bear, 2 = Sideways",
    )

    def __repr__(self) -> str:
        return (
            f"<AssetIntel {self.ticker} @ {self.timestamp:%Y-%m-%d %H:%M} "
            f"μ={self.mu_posterior:+.4f} σ={self.sigma_posterior:.4f} "
            f"ES5={self.expected_shortfall_5pct:.4f} "
            f"P(win)={self.win_probability:.2f} state={self.hmm_state}>"
        )


# ── dim_risk_category ─────────────────────────────────────────────────────

class DimRiskCategory(Base):
    """
    Static lookup: risk-category → ES threshold for candidate filtering.

    category_id 1 (Conservative) to 5 (Aggressive).
    es_threshold is the *minimum* (most negative) acceptable ES at the 5 %
    level, e.g. −0.05 for category 1.
    """

    __tablename__ = "dim_risk_category"

    category_id = Column(Integer, primary_key=True, autoincrement=False)
    label = Column(String(30), nullable=False)
    es_threshold = Column(
        Float,
        nullable=False,
        comment="Minimum acceptable ES (e.g. -0.05 for Conservative)",
    )

    def __repr__(self) -> str:
        return f"<RiskCat {self.category_id}: {self.label} ES>{self.es_threshold}>"


# ── dim_user_profile ──────────────────────────────────────────────────────

class DimUserProfile(Base):
    """
    Minimal user profile linking a user to a risk category and a
    Trading Simulator portfolio.
    """

    __tablename__ = "dim_user_profile"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False, unique=True)
    category_id = Column(Integer, nullable=False, default=3, comment="FK → dim_risk_category")
    portfolio_id = Column(Integer, nullable=False, comment="Trading Simulator portfolio ID")

    def __repr__(self) -> str:
        return f"<User {self.user_id}: {self.username} cat={self.category_id} port={self.portfolio_id}>"


# ── Engine / Session helpers ──────────────────────────────────────────────

_engine = None
_SessionFactory = None


def get_advisory_engine():
    """Return (or create) the shared SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, echo=False)
    return _engine


def get_session() -> Session:
    """Return a new Session bound to the advisory engine."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_advisory_engine())
    return _SessionFactory()


def init_advisory_tables() -> None:
    """
    Create all advisory tables if they do not exist, and seed
    dim_risk_category with the default 5 tiers.
    """
    engine = get_advisory_engine()
    Base.metadata.create_all(engine)
    logger.info("Advisory tables created / verified")

    # Seed risk categories (idempotent)
    _seed_risk_categories()


# ── Seed data ─────────────────────────────────────────────────────────────

_DEFAULT_CATEGORIES = [
    # Linear interpolation: ES thresholds from −0.05 (cat 1) to −0.20 (cat 5)
    (1, "Conservative",       -0.05),
    (2, "Moderately Conservative", -0.0875),
    (3, "Moderate",           -0.125),
    (4, "Moderately Aggressive",  -0.1625),
    (5, "Aggressive",         -0.20),
]


def _seed_risk_categories() -> None:
    """Insert default risk categories if table is empty."""
    session = get_session()
    try:
        existing = session.query(DimRiskCategory).count()
        if existing > 0:
            return

        for cat_id, label, es_thresh in _DEFAULT_CATEGORIES:
            session.add(DimRiskCategory(
                category_id=cat_id,
                label=label,
                es_threshold=es_thresh,
            ))
        session.commit()
        logger.info(f"Seeded {len(_DEFAULT_CATEGORIES)} risk categories")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to seed risk categories: {e}")
    finally:
        session.close()


# ── refresh_assets ────────────────────────────────────────────────────────

def refresh_assets(session: Session | None = None) -> int:
    """
    Delete rows from fact_asset_intelligence that are older than 24 hours.

    Args:
        session: Optional SQLAlchemy session.  If None a new one is created
                 and committed automatically.

    Returns:
        Number of rows deleted.
    """
    own_session = session is None
    if own_session:
        session = get_session()

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        stmt = (
            delete(FactAssetIntelligence)
            .where(FactAssetIntelligence.timestamp < cutoff)
        )
        result = session.execute(stmt)
        deleted = result.rowcount

        if own_session:
            session.commit()

        logger.info(f"refresh_assets: purged {deleted} rows older than {cutoff:%Y-%m-%d %H:%M} UTC")
        return deleted

    except Exception as e:
        if own_session:
            session.rollback()
        logger.error(f"refresh_assets failed: {e}")
        raise
    finally:
        if own_session:
            session.close()

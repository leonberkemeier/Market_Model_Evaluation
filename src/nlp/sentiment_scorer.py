"""
FinBERT Sentiment Scorer

Fetches financial news headlines from Finnhub and scores them using
ProsusAI/finbert. Produces a single EWMA-smoothed sentiment score
per ticker per day, stored in fact_sentiment for consumption by
the Bayesian evaluator (Tier 3).

Pipeline position: runs offline (daily cron) → writes to DB →
    BayesianEvaluator reads at backtest time.
"""

import json
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config.config import (
    FINBERT_MODEL,
    FINNHUB_API_KEY,
    SENTIMENT_EWMA_SPAN,
    SENTIMENT_HEADLINE_WINDOW,
    SENTIMENT_MIN_HEADLINES,
)


class SentimentScorer:
    """
    Fetches headlines via Finnhub, scores with FinBERT, and produces
    a 3-day EWMA sentiment score per ticker.

    The FinBERT model is loaded lazily on first use to avoid paying
    the import cost when the class is only instantiated for config.
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        finnhub_key: str = FINNHUB_API_KEY,
        ewma_span: int = SENTIMENT_EWMA_SPAN,
        headline_window: int = SENTIMENT_HEADLINE_WINDOW,
        min_headlines: int = SENTIMENT_MIN_HEADLINES,
    ):
        """
        Args:
            model_name: HuggingFace model identifier for FinBERT.
            finnhub_key: Finnhub API key.
            ewma_span: EWMA span in days for smoothing daily scores.
            headline_window: Number of days to look back for headlines.
            min_headlines: Minimum headlines required to produce a score.
        """
        self.model_name = model_name
        self.finnhub_key = finnhub_key
        self.ewma_span = ewma_span
        self.headline_window = headline_window
        self.min_headlines = min_headlines

        self._pipeline = None  # Lazy-loaded
        self._finnhub = None   # Lazy-loaded
        self.logger = logger.bind(module="sentiment_scorer")

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_pipeline(self):
        """Load FinBERT pipeline on first use."""
        if self._pipeline is not None:
            return

        self.logger.info(f"Loading FinBERT model: {self.model_name}")
        from transformers import pipeline as hf_pipeline

        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
            top_k=None,           # Return all 3 class probabilities
            truncation=True,
            max_length=512,
        )
        self.logger.info("FinBERT loaded")

    def _ensure_finnhub(self):
        """Initialise Finnhub client on first use."""
        if self._finnhub is not None:
            return

        if not self.finnhub_key:
            raise ValueError(
                "FINNHUB_API_KEY is not set. "
                "Export it or add to .env before running sentiment scoring."
            )

        import finnhub

        self._finnhub = finnhub.Client(api_key=self.finnhub_key)
        self.logger.info("Finnhub client initialised")

    # ------------------------------------------------------------------
    # Headline fetching
    # ------------------------------------------------------------------

    def fetch_headlines(
        self,
        ticker: str,
        as_of_date: date,
    ) -> List[dict]:
        """
        Fetch company news from Finnhub for the look-back window.

        Args:
            ticker: Stock ticker (e.g. 'AAPL').
            as_of_date: Reference date. Only headlines published
                        strictly before this date are returned
                        (no look-ahead bias).

        Returns:
            List of dicts with keys: headline, summary, datetime, source.
        """
        self._ensure_finnhub()

        # Window: [as_of_date - headline_window, as_of_date)
        # Finnhub 'to' is inclusive, so we use as_of_date - 1 day
        from_date = as_of_date - timedelta(days=self.headline_window)
        to_date = as_of_date - timedelta(days=1)

        try:
            raw = self._finnhub.company_news(
                ticker,
                _from=from_date.isoformat(),
                to=to_date.isoformat(),
            )
        except Exception as e:
            self.logger.warning(f"{ticker}: Finnhub fetch failed — {e}")
            return []

        if not raw:
            return []

        articles = []
        for item in raw:
            headline = item.get("headline", "").strip()
            if not headline:
                continue

            articles.append({
                "headline": headline,
                "summary": item.get("summary", "").strip(),
                "datetime": item.get("datetime", 0),  # UNIX timestamp
                "source": item.get("source", ""),
            })

        self.logger.debug(
            f"{ticker}: {len(articles)} headlines "
            f"[{from_date} → {to_date}]"
        )
        return articles

    # ------------------------------------------------------------------
    # FinBERT scoring
    # ------------------------------------------------------------------

    def score_headlines(self, headlines: List[dict]) -> List[float]:
        """
        Score a batch of headlines with FinBERT.

        For each headline, concatenates headline + summary (if available)
        and feeds it through the model. The score is P(positive) - P(negative),
        giving a value in [-1, 1].

        Args:
            headlines: List of dicts with 'headline' and optional 'summary'.

        Returns:
            List of float scores in [-1, 1], one per headline.
        """
        self._ensure_pipeline()

        texts = []
        for h in headlines:
            text = h["headline"]
            summary = h.get("summary", "")
            if summary and len(summary) > 20:
                # Concatenate for richer context, truncated by tokeniser
                text = f"{text}. {summary}"
            texts.append(text)

        if not texts:
            return []

        # Batch inference
        try:
            results = self._pipeline(texts, batch_size=32)
        except Exception as e:
            self.logger.error(f"FinBERT inference failed: {e}")
            return [0.0] * len(texts)

        scores = []
        for result in results:
            # result is a list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
            prob_map = {r["label"]: r["score"] for r in result}
            p_pos = prob_map.get("positive", 0.0)
            p_neg = prob_map.get("negative", 0.0)
            scores.append(p_pos - p_neg)

        return scores

    # ------------------------------------------------------------------
    # EWMA aggregation
    # ------------------------------------------------------------------

    def compute_ewma_score(
        self,
        ticker: str,
        as_of_date: date,
    ) -> Optional[Tuple[float, int, List[float]]]:
        """
        Compute the EWMA sentiment score for a ticker as of a date.

        Steps:
            1. Fetch headlines for [as_of_date - window, as_of_date)
            2. Score each headline with FinBERT
            3. Group scores by publication date, take daily mean
            4. Apply EWMA over the daily means
            5. Return the final EWMA value

        Args:
            ticker: Stock ticker.
            as_of_date: Reference date (exclusive upper bound).

        Returns:
            (ewma_score, n_headlines, raw_scores) or None if
            fewer than min_headlines available.
        """
        headlines = self.fetch_headlines(ticker, as_of_date)

        if len(headlines) < self.min_headlines:
            self.logger.debug(
                f"{ticker}: Only {len(headlines)} headlines, "
                f"need >= {self.min_headlines}"
            )
            return None

        raw_scores = self.score_headlines(headlines)

        # Build daily score series from UNIX timestamps
        daily_scores: Dict[date, List[float]] = {}
        for h, score in zip(headlines, raw_scores):
            ts = h.get("datetime", 0)
            if ts > 0:
                pub_date = pd.Timestamp(ts, unit="s").date()
            else:
                continue
            daily_scores.setdefault(pub_date, []).append(score)

        if not daily_scores:
            return None

        # Daily mean
        daily_mean = pd.Series({
            d: np.mean(s) for d, s in sorted(daily_scores.items())
        })
        daily_mean.index = pd.DatetimeIndex(daily_mean.index)

        # EWMA — if only 1 day of data, return that day's mean
        if len(daily_mean) == 1:
            ewma_val = float(daily_mean.iloc[0])
        else:
            ewma = daily_mean.ewm(span=self.ewma_span, min_periods=1).mean()
            ewma_val = float(ewma.iloc[-1])

        # Clamp to [-1, 1]
        ewma_val = max(-1.0, min(1.0, ewma_val))

        return ewma_val, len(headlines), raw_scores

    # ------------------------------------------------------------------
    # Universe-level convenience
    # ------------------------------------------------------------------

    def score_universe(
        self,
        tickers: List[str],
        as_of_date: date,
    ) -> Dict[str, Tuple[float, int, List[float]]]:
        """
        Score all tickers in the universe for a given date.

        Args:
            tickers: List of stock tickers.
            as_of_date: Reference date.

        Returns:
            Dict of ticker -> (ewma_score, n_headlines, raw_scores).
            Tickers with insufficient data are omitted.
        """
        results = {}
        failed = 0

        for ticker in tickers:
            try:
                result = self.compute_ewma_score(ticker, as_of_date)
                if result is not None:
                    results[ticker] = result
                else:
                    failed += 1
            except Exception as e:
                self.logger.error(f"{ticker}: scoring failed — {e}")
                failed += 1

        self.logger.info(
            f"Sentiment scored: {len(results)}/{len(tickers)} tickers "
            f"({failed} insufficient/failed)"
        )
        return results

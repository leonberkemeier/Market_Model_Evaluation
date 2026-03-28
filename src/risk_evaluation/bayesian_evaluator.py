"""
Bayesian Inference Engine (Tier 3)

Produces posterior predictive distributions (μ, σ) for each stock using
sklearn BayesianRidge. The posterior uncertainty (credible interval width)
directly feeds the confidence score in Bayesian Kelly sizing.

Features:
    - Lagged frac-diff returns (Tier 1 output)
    - HMM state_id one-hot encoded (Tier 2 output)
    - Macro feature: VIX or FEDFUNDS (configurable for robustness tests)
    - Bond yield spread (10Y - 2Y)

Training uses an expanding window to prevent look-ahead bias.

Pipeline position: Tier 3 → feeds into Copula MC → Bayesian Kelly.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from src.config.config import BAYESIAN_N_LAGS, BAYESIAN_CI_LEVEL, HMM_N_STATES


@dataclass
class BayesianPosterior:
    """
    Output of Tier 3 for a single stock at a single evaluation date.

    Contains the posterior predictive distribution parameters from
    BayesianRidge, plus the credible interval used for Kelly confidence.
    """
    ticker: str
    evaluation_date: date

    # Posterior predictive distribution
    mu: float                       # Posterior mean (expected return)
    sigma: float                    # Posterior std (uncertainty)

    # Credible interval
    ci_lower: float                 # Lower bound of 95% CI
    ci_upper: float                 # Upper bound of 95% CI
    credible_interval_width: float  # ci_upper - ci_lower

    # Confidence score (inversely proportional to CI width)
    # Used as haircut in Bayesian Kelly: higher = more confident
    confidence_score: float

    # Metadata
    macro_feature_name: str = "VIX"
    features_used: List[str] = field(default_factory=list)
    training_end_date: Optional[date] = None
    n_training_samples: int = 0


class BayesianEvaluator:
    """
    Bayesian Ridge evaluator producing posterior predictive distributions.

    For each stock, trains a BayesianRidge model on:
        X = [frac_diff_lag_1, ..., frac_diff_lag_N, state_0, state_1, state_2, macro, spread, sentiment]
        y = next-period frac-diff return

    The model's `predict(X, return_std=True)` gives (μ, σ) for each
    prediction, which defines the posterior predictive distribution.
    """

    def __init__(
        self,
        n_lags: int = BAYESIAN_N_LAGS,
        ci_level: float = BAYESIAN_CI_LEVEL,
        n_hmm_states: int = HMM_N_STATES,
        macro_feature_name: str = "VIX",
    ):
        """
        Args:
            n_lags: Number of lagged frac-diff returns as features.
            ci_level: Credible interval level (default 0.95).
            n_hmm_states: Number of HMM states for one-hot encoding.
            macro_feature_name: Name of the macro feature ('VIX' or 'FEDFUNDS').
        """
        self.n_lags = n_lags
        self.ci_level = ci_level
        self.n_hmm_states = n_hmm_states
        self.macro_feature_name = macro_feature_name

        # z-score for credible interval
        from scipy.stats import norm
        self._ci_z = norm.ppf((1 + ci_level) / 2)

        self.logger = logger.bind(module="bayesian_evaluator")

    def _build_feature_matrix(
        self,
        frac_diff_returns: pd.Series,
        hmm_states: np.ndarray,
        hmm_index: pd.DatetimeIndex,
        macro_feature: pd.Series,
        bond_spread: Optional[pd.Series] = None,
        sentiment_score: Optional[pd.Series] = None,
        filing_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build the feature matrix and target for a single stock.

        Bond spread, sentiment, and filing features are optional — if empty
        or None they are excluded from the feature set rather than causing
        the entire matrix to be dropped via an inner join on sparse data.

        Returns:
            (X, y): Feature DataFrame and target Series, aligned on common dates.
                    y is the next-period frac-diff return (shifted by 1).
        """
        # Create lagged features from frac-diff returns
        lag_data = {}
        for lag in range(1, self.n_lags + 1):
            lag_data[f"fd_lag_{lag}"] = frac_diff_returns.shift(lag)

        lags_df = pd.DataFrame(lag_data, index=frac_diff_returns.index)

        # HMM state one-hot encoding
        hmm_df = pd.DataFrame(index=hmm_index)
        for s in range(self.n_hmm_states):
            hmm_df[f"state_{s}"] = (hmm_states == s).astype(float)

        # Macro feature — use left join + ffill so sparse macro data
        # does not discard return rows
        macro_df = macro_feature.rename("macro").to_frame()

        # Target: next-period return
        target = frac_diff_returns.shift(-1).rename("target")

        # Join core features on common dates
        combined = lags_df.join(hmm_df, how="inner")
        combined = combined.join(macro_df, how="left")
        combined["macro"] = combined["macro"].ffill()

        # Bond spread — only include when the series has meaningful data
        _has_spread = (
            bond_spread is not None
            and isinstance(bond_spread, pd.Series)
            and not bond_spread.empty
            and bond_spread.notna().sum() >= 20
        )
        if _has_spread:
            spread_df = bond_spread.rename("spread").to_frame()
            combined = combined.join(spread_df, how="left")
            combined["spread"] = combined["spread"].ffill()
        else:
            self.logger.debug("Bond spread excluded — insufficient data")

        # Sentiment — same optional pattern
        _has_sentiment = (
            sentiment_score is not None
            and isinstance(sentiment_score, pd.Series)
            and not sentiment_score.empty
            and sentiment_score.notna().sum() >= 10
        )
        if _has_sentiment:
            sent_df = sentiment_score.rename("sentiment").to_frame()
            combined = combined.join(sent_df, how="left")
            combined["sentiment"] = combined["sentiment"].ffill()
        else:
            self.logger.debug("Sentiment excluded — insufficient data")

        # Filing NLP features — quarterly, forward-filled
        _has_filing = (
            filing_features is not None
            and isinstance(filing_features, pd.DataFrame)
            and not filing_features.empty
            and len(filing_features) >= 2
        )
        if _has_filing:
            for col in ["mgmt_sentiment", "risk_count", "guidance_tone"]:
                if col in filing_features.columns:
                    col_series = filing_features[col]
                    col_df = col_series.rename(f"filing_{col}").to_frame()
                    combined = combined.join(col_df, how="left")
                    combined[f"filing_{col}"] = combined[f"filing_{col}"].ffill()
        else:
            self.logger.debug("Filing NLP features excluded — insufficient data")

        combined = combined.join(target, how="inner")

        # Drop rows with any NaN (from lagging and shifting)
        combined = combined.dropna()

        if combined.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        X = combined.drop(columns=["target"])
        y = combined["target"]

        return X, y

    def evaluate(
        self,
        ticker: str,
        frac_diff_returns: pd.Series,
        hmm_states: np.ndarray,
        hmm_index: pd.DatetimeIndex,
        macro_feature: pd.Series,
        bond_spread: Optional[pd.Series] = None,
        sentiment_score: Optional[pd.Series] = None,
        filing_features: Optional[pd.DataFrame] = None,
        train_end: date = date.today(),
    ) -> Optional[BayesianPosterior]:
        """
        Produce posterior predictive distribution for a single stock.

        Uses expanding-window training: trains on all data up to train_end,
        predicts for the next period.

        Args:
            ticker: Stock ticker.
            frac_diff_returns: Frac-diff return series for this stock.
            hmm_states: HMM state ID array (from get_state_sequence).
            hmm_index: DatetimeIndex corresponding to hmm_states.
            macro_feature: VIX or FEDFUNDS series (date-indexed).
            bond_spread: Bond yield spread series (optional, date-indexed).
            sentiment_score: Sentiment EWMA series (optional, date-indexed).
            filing_features: Filing NLP DataFrame (optional, date-indexed).
            train_end: Last date to include in training (expanding window).

        Returns:
            BayesianPosterior or None if insufficient data.
        """
        X, y = self._build_feature_matrix(
            frac_diff_returns, hmm_states, hmm_index, macro_feature,
            bond_spread, sentiment_score, filing_features,
        )

        if X.empty or len(X) < 30:
            self.logger.warning(f"{ticker}: Insufficient data for Bayesian evaluation")
            return None

        # Expanding window: train on data up to train_end
        train_end_ts = pd.Timestamp(train_end)
        train_mask = X.index <= train_end_ts

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        if len(X_train) < 20:
            self.logger.warning(
                f"{ticker}: Only {len(X_train)} training samples, need >= 20"
            )
            return None

        # Scale features for numerical stability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit BayesianRidge
        model = BayesianRidge(
            compute_score=True,
            fit_intercept=True,
        )
        model.fit(X_train_scaled, y_train)

        # Predict for the latest available feature row (most recent date <= train_end)
        X_pred = X_train_scaled[-1:, :]
        y_mean, y_std = model.predict(X_pred, return_std=True)

        mu = float(y_mean[0])
        sigma = float(y_std[0])

        # Credible interval
        ci_lower = mu - self._ci_z * sigma
        ci_upper = mu + self._ci_z * sigma
        ci_width = ci_upper - ci_lower

        # Confidence score: inversely proportional to CI width
        # Bounded to (0, 1] — narrower CI = higher confidence
        confidence_score = 1.0 / (1.0 + ci_width)

        return BayesianPosterior(
            ticker=ticker,
            evaluation_date=train_end,
            mu=mu,
            sigma=sigma,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            credible_interval_width=ci_width,
            confidence_score=confidence_score,
            macro_feature_name=self.macro_feature_name,
            features_used=list(X.columns),
            training_end_date=train_end,
            n_training_samples=len(X_train),
        )

    def evaluate_universe(
        self,
        frac_diff_dict: Dict[str, pd.Series],
        hmm_states_dict: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]],
        macro_feature: pd.Series,
        bond_spread: Optional[pd.Series] = None,
        sentiment_scores: Optional[Dict[str, pd.Series]] = None,
        filing_features: Optional[Dict[str, pd.DataFrame]] = None,
        train_end: date = date.today(),
    ) -> Dict[str, BayesianPosterior]:
        """
        Evaluate all stocks in the universe.

        Args:
            frac_diff_dict: Dict of ticker -> frac-diff return series.
            hmm_states_dict: Dict of ticker -> (state_ids, index).
            macro_feature: VIX or FEDFUNDS series.
            bond_spread: Bond yield spread series (optional).
            sentiment_scores: Dict of ticker -> sentiment series (optional).
            filing_features: Dict of ticker -> filing NLP DataFrame (optional).
            train_end: Expanding window cutoff date.

        Returns:
            Dict of ticker -> BayesianPosterior.
        """
        posteriors = {}
        failed = []

        for ticker, fd_returns in frac_diff_dict.items():
            try:
                if ticker not in hmm_states_dict:
                    failed.append(ticker)
                    continue

                states, idx = hmm_states_dict[ticker]
                ticker_sentiment = (
                    sentiment_scores.get(ticker) if sentiment_scores else None
                )
                ticker_filing = (
                    filing_features.get(ticker) if filing_features else None
                )

                posterior = self.evaluate(
                    ticker=ticker,
                    frac_diff_returns=fd_returns,
                    hmm_states=states,
                    hmm_index=idx,
                    macro_feature=macro_feature,
                    bond_spread=bond_spread,
                    sentiment_score=ticker_sentiment,
                    filing_features=ticker_filing,
                    train_end=train_end,
                )

                if posterior is not None:
                    posteriors[ticker] = posterior
                else:
                    failed.append(ticker)

            except Exception as e:
                self.logger.error(f"{ticker}: Bayesian evaluation failed — {e}")
                failed.append(ticker)

        self.logger.info(
            f"Bayesian evaluation: {len(posteriors)}/{len(frac_diff_dict)} stocks "
            f"({len(failed)} failed)"
        )
        return posteriors

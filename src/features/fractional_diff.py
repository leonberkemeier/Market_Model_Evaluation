"""
Fractional Differentiation (Tier 1)

Transforms price series to achieve stationarity while preserving as much
long-memory as possible. Based on López de Prado, "Advances in Financial
Machine Learning", Chapter 5.

Standard differencing (d=1) makes a series stationary but destroys memory.
Fractional differencing (0 < d < 1) finds the minimum d that achieves
stationarity via the ADF test, preserving predictive signal.

Pipeline position: Tier 1 → feeds into HMM (Tier 2) and Bayesian inference (Tier 3).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from loguru import logger

from statsmodels.tsa.stattools import adfuller


def _get_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute fractional differentiation weights.

    Weights follow the recursion: w_k = -w_{k-1} * (d - k + 1) / k
    Starting with w_0 = 1.

    Args:
        d: Fractional differentiation order (0 < d <= 1)
        threshold: Drop weights below this absolute value

    Returns:
        Array of weights (from w_0 to w_K where |w_K| >= threshold)
    """
    weights = [1.0]
    k = 1

    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        # Safety cap to prevent infinite loops
        if k > 10000:
            break

    return np.array(weights)


def frac_diff_fixed_window(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply fractional differentiation with a fixed-width window.

    For each time step t, computes:
        X_t^(d) = sum_{k=0}^{K} w_k * X_{t-k}

    where weights w_k are computed from the binomial expansion of (1-B)^d
    and truncated when |w_k| < threshold.

    Args:
        series: Price series (date-indexed). Should be log-prices or prices.
        d: Fractional differentiation order (0 < d <= 1).
           d=0 → original series, d=1 → standard first difference.
        threshold: Weight truncation threshold.

    Returns:
        Fractionally differentiated series (shorter due to window loss).
    """
    weights = _get_weights(d, threshold)
    width = len(weights)

    if width > len(series):
        logger.warning(
            f"Weight window ({width}) exceeds series length ({len(series)}). "
            f"Returning empty series."
        )
        return pd.Series(dtype=float)

    # Apply convolution
    result = {}
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1].values
        # Weights are applied in reverse order: w_0 * X_t + w_1 * X_{t-1} + ...
        result[series.index[i]] = np.dot(weights, window[::-1])

    return pd.Series(result, dtype=float)


def find_min_d(
    series: pd.Series,
    max_d: float = 1.0,
    step: float = 0.05,
    p_value: float = 0.05,
    threshold: float = 1e-5,
) -> float:
    """
    Find the minimum fractional differentiation order d that achieves
    stationarity (ADF test p-value < threshold).

    Uses a grid search from d=0 upward in increments of `step`.
    Returns the first d where ADF rejects the unit root null.

    Args:
        series: Price series (log-prices recommended).
        max_d: Maximum d to search (usually 1.0).
        step: Grid step size for d search.
        p_value: ADF p-value threshold for stationarity.
        threshold: Weight truncation for frac_diff.

    Returns:
        Minimum d that achieves stationarity.
        Returns max_d if no d achieves stationarity.
    """
    # Use log prices for numerical stability
    log_series = np.log(series.replace(0, np.nan).dropna())

    if log_series.empty or len(log_series) < 50:
        logger.warning("Series too short for ADF test, returning d=1.0")
        return 1.0

    d_values = np.arange(step, max_d + step, step)

    for d in d_values:
        try:
            diffed = frac_diff_fixed_window(log_series, d, threshold)

            if diffed.empty or len(diffed) < 30:
                continue

            # Drop NaN/inf values
            clean = diffed.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < 30:
                continue

            adf_result = adfuller(clean, maxlag=1, regression="c", autolag=None)
            adf_pvalue = adf_result[1]

            if adf_pvalue < p_value:
                logger.info(
                    f"Minimum d={d:.2f} achieves stationarity "
                    f"(ADF p-value={adf_pvalue:.4f})"
                )
                return round(d, 2)

        except Exception as e:
            logger.debug(f"ADF failed at d={d:.2f}: {e}")
            continue

    logger.warning(f"No d in [0, {max_d}] achieved stationarity, returning {max_d}")
    return max_d


def frac_diff_series(
    series: pd.Series,
    max_d: float = 1.0,
    step: float = 0.05,
    p_value: float = 0.05,
    threshold: float = 1e-5,
) -> Tuple[pd.Series, float]:
    """
    Convenience function: find optimal d and apply fractional differentiation.

    Args:
        series: Price series (date-indexed).
        max_d: Maximum d to search.
        step: Grid step for d search.
        p_value: ADF p-value threshold.
        threshold: Weight truncation threshold.

    Returns:
        Tuple of (fractionally_differentiated_series, optimal_d).
    """
    d = find_min_d(series, max_d=max_d, step=step, p_value=p_value, threshold=threshold)

    # Apply frac-diff using log prices
    log_series = np.log(series.replace(0, np.nan).dropna())
    diffed = frac_diff_fixed_window(log_series, d, threshold)

    logger.info(
        f"Frac-diff applied: d={d:.2f}, "
        f"input={len(series)}, output={len(diffed)}"
    )

    return diffed, d

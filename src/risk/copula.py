"""
Student-t Copula for Tail Dependence

Replaces Pearson correlation with a copula that captures tail dependence —
stocks crash together more than Gaussian correlation implies.

Fitting procedure:
    1. Transform marginals to uniform via empirical CDF (rank-based)
    2. Transform uniform marginals to Student-t quantiles
    3. Estimate correlation matrix R on the transformed data
    4. Estimate degrees of freedom ν via profile log-likelihood

Sampling procedure:
    1. Draw from multivariate Student-t(ν, R)
    2. Transform each marginal through Student-t CDF → uniform
    3. Map uniform → Normal(μ_i, σ_i) using each stock's Bayesian posterior

Pipeline position: after Tier 3 posteriors, before Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from loguru import logger
from scipy import stats
from scipy.stats import rankdata, t as t_dist

from src.config.config import COPULA_DOF_MIN, COPULA_DOF_MAX, COPULA_DOF_STEP


class StudentTCopula:
    """
    Student-t copula for modeling tail-dependent correlations.

    Captures the empirical observation that extreme co-movements
    (crashes) are more frequent than Gaussian correlation implies.
    """

    def __init__(
        self,
        dof_min: int = COPULA_DOF_MIN,
        dof_max: int = COPULA_DOF_MAX,
        dof_step: int = COPULA_DOF_STEP,
    ):
        """
        Args:
            dof_min: Minimum degrees of freedom to search.
            dof_max: Maximum degrees of freedom to search.
            dof_step: Step size for ν grid search.
        """
        self.dof_min = dof_min
        self.dof_max = dof_max
        self.dof_step = dof_step

        # Fitted parameters
        self.nu: Optional[float] = None          # Degrees of freedom
        self.corr_matrix: Optional[np.ndarray] = None  # Correlation matrix R
        self.tickers: list = []                   # Ordered ticker list
        self.fitted = False

        self.logger = logger.bind(module="copula")

    def fit(self, returns_matrix: pd.DataFrame) -> 'StudentTCopula':
        """
        Fit the Student-t copula to a multi-asset return matrix.

        Args:
            returns_matrix: DataFrame with columns = tickers, rows = dates,
                            values = returns. Must have >= 30 rows.

        Returns:
            Self for method chaining.
        """
        returns_matrix = returns_matrix.dropna()

        if len(returns_matrix) < 30:
            raise ValueError(
                f"Need >= 30 observations, got {len(returns_matrix)}"
            )

        self.tickers = list(returns_matrix.columns)
        n_obs, n_assets = returns_matrix.shape

        self.logger.info(
            f"Fitting Student-t copula: {n_assets} assets, {n_obs} observations"
        )

        # Step 1: Transform marginals to uniform via empirical CDF
        # Use rank / (n+1) to avoid 0 and 1 (which blow up quantile transforms)
        uniform_data = np.zeros_like(returns_matrix.values, dtype=float)
        for j in range(n_assets):
            uniform_data[:, j] = rankdata(returns_matrix.values[:, j]) / (n_obs + 1)

        # Step 2 & 3: Profile likelihood over ν to find best dof
        best_nu = self.dof_min
        best_ll = -np.inf

        for nu in range(self.dof_min, self.dof_max + 1, self.dof_step):
            # Transform uniform → Student-t quantiles
            t_data = t_dist.ppf(uniform_data, df=nu)

            # Handle numerical issues (clamp extreme quantiles)
            t_data = np.clip(t_data, -10, 10)

            # Estimate correlation matrix on t-transformed data
            corr = np.corrcoef(t_data, rowvar=False)

            # Regularize: ensure positive definiteness
            corr = self._regularize_corr(corr)

            # Log-likelihood of multivariate t
            ll = self._log_likelihood(t_data, corr, nu)

            if ll > best_ll:
                best_ll = ll
                best_nu = nu

        # Final fit with best ν
        self.nu = best_nu
        t_data = t_dist.ppf(uniform_data, df=self.nu)
        t_data = np.clip(t_data, -10, 10)
        self.corr_matrix = self._regularize_corr(np.corrcoef(t_data, rowvar=False))
        self.fitted = True

        self.logger.info(
            f"Copula fitted: ν={self.nu}, "
            f"mean off-diag corr={self._mean_offdiag(self.corr_matrix):.3f}"
        )

        return self

    def sample(
        self,
        n_samples: int,
        marginal_params: Dict[str, Tuple[float, float]],
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Draw correlated samples using the fitted copula.

        For each sample:
        1. Draw from multivariate Student-t(ν, R)
        2. Transform each dimension through t CDF → Uniform
        3. Map Uniform → Normal(μ_i, σ_i) using each ticker's posterior

        Args:
            n_samples: Number of samples to draw.
            marginal_params: Dict of ticker -> (mu, sigma) from BayesianPosterior.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame (n_samples × n_tickers) of correlated return samples.
        """
        if not self.fitted:
            raise ValueError("Copula not fitted. Call fit() first.")

        # Only use tickers that are in both the copula and the posterior set
        common_tickers = [t for t in self.tickers if t in marginal_params]

        if len(common_tickers) < 2:
            # Fallback: independent normal samples
            self.logger.warning(
                "Fewer than 2 common tickers between copula and posteriors, "
                "using independent normals."
            )
            rng = np.random.default_rng(seed)
            samples = {}
            for ticker, (mu, sigma) in marginal_params.items():
                samples[ticker] = rng.normal(mu, max(sigma, 1e-8), size=n_samples)
            return pd.DataFrame(samples)

        # Get indices of common tickers in copula ordering
        indices = [self.tickers.index(t) for t in common_tickers]
        sub_corr = self.corr_matrix[np.ix_(indices, indices)]

        rng = np.random.default_rng(seed)
        n_dim = len(common_tickers)

        # Draw from multivariate Student-t via the standard construction:
        # X = μ + (Z / sqrt(V/ν)) where Z ~ N(0, R), V ~ χ²(ν)
        z = rng.multivariate_normal(np.zeros(n_dim), sub_corr, size=n_samples)
        chi2_samples = rng.chisquare(self.nu, size=(n_samples, 1))
        t_samples = z / np.sqrt(chi2_samples / self.nu)

        # Transform each dimension: t CDF → Uniform → Normal(μ_i, σ_i)
        result = {}
        for i, ticker in enumerate(common_tickers):
            mu, sigma = marginal_params[ticker]
            sigma = max(sigma, 1e-8)  # Prevent zero division

            # t CDF → uniform
            u = t_dist.cdf(t_samples[:, i], df=self.nu)
            # Uniform → Normal(μ, σ)
            result[ticker] = stats.norm.ppf(u, loc=mu, scale=sigma)

        # Add any tickers not in copula as independent normals
        for ticker, (mu, sigma) in marginal_params.items():
            if ticker not in result:
                sigma = max(sigma, 1e-8)
                result[ticker] = rng.normal(mu, sigma, size=n_samples)

        return pd.DataFrame(result)

    def _log_likelihood(
        self, t_data: np.ndarray, corr: np.ndarray, nu: int
    ) -> float:
        """
        Compute copula log-likelihood for given ν and correlation matrix.

        Uses the density of the multivariate t-distribution evaluated
        at the t-transformed data.
        """
        try:
            n_obs, n_dim = t_data.shape
            # Use scipy multivariate_t log-pdf
            rv = stats.multivariate_t(
                loc=np.zeros(n_dim), shape=corr, df=nu
            )
            ll = rv.logpdf(t_data).sum()
            return float(ll)
        except Exception:
            return -np.inf

    @staticmethod
    def _regularize_corr(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Ensure correlation matrix is positive definite via eigenvalue clipping."""
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, eps)
        corr_reg = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Re-normalize to correlation (diag = 1)
        d = np.sqrt(np.diag(corr_reg))
        corr_reg = corr_reg / np.outer(d, d)
        np.fill_diagonal(corr_reg, 1.0)

        return corr_reg

    @staticmethod
    def _mean_offdiag(corr: np.ndarray) -> float:
        """Mean of off-diagonal elements in correlation matrix."""
        n = corr.shape[0]
        if n < 2:
            return 0.0
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(corr[mask]))

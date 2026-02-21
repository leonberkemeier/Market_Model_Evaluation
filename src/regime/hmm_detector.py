"""
HMM Regime Detection (Layer II)

Uses Hidden Markov Model to classify market state into Bull/Bear/Sideways regimes
based on historical returns and volatility patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from loguru import logger
import pickle
from pathlib import Path

try:
    from hmmlearn import hmm
except ImportError:
    logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
    hmm = None


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.
    
    Detects three market states:
    - Bull: High positive returns, moderate volatility
    - Bear: Negative returns, high volatility  
    - Sideways: Near-zero returns, low volatility
    
    The HMM learns the transition probabilities between states from historical data.
    """
    
    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
        model_path: Optional[Path] = None
    ):
        """
        Initialize HMM Regime Detector.
        
        Args:
            n_states: Number of hidden states (default: 3 for Bull/Bear/Sideways)
            n_iter: Number of EM iterations for training
            random_state: Random seed for reproducibility
            model_path: Path to save/load trained model
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_path = model_path or Path("models/hmm_regime_model.pkl")
        
        self.model = None
        self.state_labels = {}  # Maps state index to regime name
        self.fitted = False
        
        self.logger = logger.bind(module="hmm_detector")
    
    def _prepare_features(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> np.ndarray:
        """
        Prepare features for HMM from price series.
        
        Features:
        1. Returns (normalized)
        2. Volatility (rolling std of returns)
        
        Args:
            prices: Series of closing prices
            window: Rolling window for volatility calculation
            
        Returns:
            Array of shape (n_samples, 2) with [returns, volatility]
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=window).std().dropna()
        
        # Align returns and volatility
        aligned_returns = returns[volatility.index]
        
        # Combine into feature matrix
        features = np.column_stack([
            aligned_returns.values,
            volatility.values
        ])
        
        return features
    
    def fit(
        self,
        prices: pd.Series,
        window: int = 20,
        verbose: bool = True
    ) -> 'HMMRegimeDetector':
        """
        Train HMM on historical price data.
        
        Args:
            prices: Series of historical closing prices
            window: Rolling window for feature calculation
            verbose: Print training progress
            
        Returns:
            Self for method chaining
        """
        if hmm is None:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.logger.info(f"Training HMM with {len(prices)} price points...")
        
        # Prepare features
        features = self._prepare_features(prices, window=window)
        
        if verbose:
            self.logger.info(f"Feature matrix shape: {features.shape}")
        
        # Initialize and train Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(features)
        
        # Predict states for entire history to label them
        states = self.model.predict(features)
        
        # Label states based on mean returns in each state
        state_means = []
        state_vols = []
        
        for state_idx in range(self.n_states):
            state_mask = states == state_idx
            state_returns = features[state_mask, 0]
            state_volatility = features[state_mask, 1]
            
            state_means.append(state_returns.mean())
            state_vols.append(state_volatility.mean())
        
        # Map states to regime labels
        # Bull: Highest mean return
        # Bear: Lowest mean return
        # Sideways: Middle mean return
        sorted_indices = np.argsort(state_means)
        
        self.state_labels = {
            sorted_indices[0]: "bear",      # Lowest returns
            sorted_indices[1]: "sideways",  # Middle returns
            sorted_indices[2]: "bull"       # Highest returns
        }
        
        if verbose:
            self.logger.info("Regime labels assigned:")
            for state_idx, label in self.state_labels.items():
                self.logger.info(
                    f"  State {state_idx} = {label.upper()}: "
                    f"mean_return={state_means[state_idx]:.4f}, "
                    f"mean_vol={state_vols[state_idx]:.4f}"
                )
        
        self.fitted = True
        
        # Save model
        if self.model_path:
            self._save_model()
        
        return self
    
    def predict_regime(
        self,
        prices: pd.Series,
        window: int = 20,
        return_probabilities: bool = False
    ) -> Dict[str, any]:
        """
        Predict current market regime.
        
        Args:
            prices: Recent price history (at least window + 20 bars)
            window: Rolling window for feature calculation
            return_probabilities: If True, return probabilities for each state
            
        Returns:
            Dict with:
                - regime: Current regime name ("bull", "bear", "sideways")
                - confidence: Probability of current regime (0-1)
                - probabilities: Dict of probabilities for all regimes (optional)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first or load_model()")
        
        # Prepare features
        features = self._prepare_features(prices, window=window)
        
        # Predict state
        state_idx = self.model.predict(features[-1:].reshape(1, -1))[0]
        regime = self.state_labels[state_idx]
        
        # Get probabilities
        state_probs = self.model.predict_proba(features[-1:].reshape(1, -1))[0]
        confidence = float(state_probs[state_idx])
        
        result = {
            "regime": regime,
            "confidence": confidence,
            "state_index": int(state_idx)
        }
        
        if return_probabilities:
            result["probabilities"] = {
                self.state_labels[i]: float(state_probs[i])
                for i in range(self.n_states)
            }
        
        return result
    
    def predict_regime_history(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Predict regime for entire price history.
        
        Useful for backtesting and validation.
        
        Args:
            prices: Historical price series
            window: Rolling window for feature calculation
            
        Returns:
            DataFrame with columns: date, regime, confidence, state_index
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first or load_model()")
        
        # Prepare features
        features = self._prepare_features(prices, window=window)
        
        # Get dates (aligned with features)
        dates = prices.index[window:]
        
        # Predict states
        states = self.model.predict(features)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features)
        
        # Create result DataFrame
        results = []
        for i, (date, state_idx) in enumerate(zip(dates, states)):
            regime = self.state_labels[state_idx]
            confidence = probabilities[i, state_idx]
            
            results.append({
                "date": date,
                "regime": regime,
                "confidence": confidence,
                "state_index": state_idx
            })
        
        return pd.DataFrame(results).set_index("date")
    
    def get_regime_statistics(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each regime from historical data.
        
        Args:
            prices: Historical price series
            window: Rolling window for feature calculation
            
        Returns:
            Dict mapping regime name to statistics (mean_return, volatility, duration, frequency)
        """
        regime_history = self.predict_regime_history(prices, window=window)
        
        # Calculate returns
        returns = prices.pct_change()
        aligned_returns = returns[regime_history.index]
        
        stats = {}
        
        for regime_name in ["bull", "bear", "sideways"]:
            regime_mask = regime_history["regime"] == regime_name
            regime_returns = aligned_returns[regime_mask]
            
            # Calculate duration (consecutive days in regime)
            regime_series = (regime_history["regime"] == regime_name).astype(int)
            regime_changes = regime_series.diff().fillna(0)
            regime_starts = regime_changes == 1
            
            durations = []
            current_duration = 0
            
            for in_regime in regime_series:
                if in_regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            stats[regime_name] = {
                "mean_return": float(regime_returns.mean()),
                "volatility": float(regime_returns.std()),
                "frequency": float(regime_mask.sum() / len(regime_history)),
                "avg_duration_days": float(np.mean(durations)) if durations else 0.0,
                "max_duration_days": float(np.max(durations)) if durations else 0.0
            }
        
        return stats
    
    def _save_model(self):
        """Save trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "state_labels": self.state_labels,
            "n_states": self.n_states,
            "fitted": self.fitted
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self, model_path: Optional[Path] = None):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to model file (uses self.model_path if not provided)
        """
        path = model_path or self.model_path
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.state_labels = model_data["state_labels"]
        self.n_states = model_data["n_states"]
        self.fitted = model_data["fitted"]
        
        self.logger.info(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Generate sample data (trending upward with periods of decline)
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    
    # Simulate regime changes
    prices = [100]
    regime_changes = [0, 300, 600, 800]  # Indices where regime changes
    
    for i in range(1, 1000):
        if i < 300:  # Bull
            drift = 0.001
            vol = 0.01
        elif i < 600:  # Sideways
            drift = 0.0
            vol = 0.008
        elif i < 800:  # Bear
            drift = -0.002
            vol = 0.02
        else:  # Bull again
            drift = 0.0015
            vol = 0.012
        
        prices.append(prices[-1] * (1 + np.random.normal(drift, vol)))
    
    price_series = pd.Series(prices, index=dates)
    
    print("=== Training HMM Regime Detector ===")
    detector = HMMRegimeDetector(n_states=3)
    detector.fit(price_series)
    
    print("\n=== Current Regime Prediction ===")
    current_regime = detector.predict_regime(price_series, return_probabilities=True)
    print(f"Regime: {current_regime['regime'].upper()}")
    print(f"Confidence: {current_regime['confidence']:.2%}")
    print(f"Probabilities: {current_regime['probabilities']}")
    
    print("\n=== Regime Statistics ===")
    stats = detector.get_regime_statistics(price_series)
    for regime, regime_stats in stats.items():
        print(f"\n{regime.upper()}:")
        print(f"  Mean Return: {regime_stats['mean_return']:.4f}")
        print(f"  Volatility: {regime_stats['volatility']:.4f}")
        print(f"  Frequency: {regime_stats['frequency']:.2%}")
        print(f"  Avg Duration: {regime_stats['avg_duration_days']:.1f} days")
        print(f"  Max Duration: {regime_stats['max_duration_days']:.0f} days")
    
    print("\n=== Regime History (last 10 days) ===")
    history = detector.predict_regime_history(price_series)
    print(history.tail(10))

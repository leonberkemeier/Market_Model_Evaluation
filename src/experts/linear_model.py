"""
Linear Regression Model for Phase 1 Model-Sector Comparison

This model uses Ridge Regression (L2 regularization) to predict stock returns.
Expected to perform well on mean-reverting sectors (Finance, Commodities).

Architecture:
- Linear regression with L2 penalty
- Standardized features (zero mean, unit variance)
- Cross-validation for hyperparameter tuning
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
from pathlib import Path

from .base_expert import ComparisonModel, ScorerOutput, ScoreResult

logger = logging.getLogger(__name__)


class LinearModel(ComparisonModel):
    """
    Linear Regression model with Ridge regularization.
    
    Strengths:
    - Fast training and prediction
    - Interpretable (feature coefficients)
    - Stable predictions
    - Good for mean-reverting patterns
    
    Expected Performance:
    - Finance: Sharpe ~1.5 (mean-reverting, rate-sensitive)
    - Commodities: Sharpe ~1.4 (supply/demand equilibrium)
    - Tech: Sharpe ~1.2 (growth-driven, less linear)
    - Crypto: Sharpe ~0.9 (momentum-driven, non-linear)
    - Cyclicals: Sharpe ~1.1 (complex multi-factor)
    """
    
    def __init__(self, sector: Optional[str] = None):
        """
        Initialize Linear model.
        
        Args:
            sector: Optional sector specialization (None = generalist)
        """
        super().__init__(name="linear", sector=sector)
        
        # Model components
        self.model: Optional[Ridge] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Hyperparameters (to be tuned)
        self.alpha = 1.0  # L2 regularization strength
        
        # Training metadata
        self.feature_names: List[str] = []
        self.feature_importance: Optional[np.ndarray] = None
    
    def train(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        hyperparameter_tune: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Linear model with optional hyperparameter tuning.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target returns (n_samples,)
            hyperparameter_tune: Whether to tune alpha via CV
            **kwargs: Additional training parameters
                - cv_splits: Number of CV splits (default: 5)
                - alpha_range: Range of alpha values to try
        
        Returns:
            dict with training metrics
        """
        logger.info(f"Training Linear model on {len(features)} samples with {features.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)
        y = targets.values
        
        # Hyperparameter tuning
        if hyperparameter_tune:
            best_alpha = self._tune_hyperparameters(
                X_scaled, y,
                cv_splits=kwargs.get('cv_splits', 5),
                alpha_range=kwargs.get('alpha_range', [0.01, 0.1, 1.0, 10.0, 100.0])
            )
            self.alpha = best_alpha
        
        # Train final model
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Store feature importance (absolute coefficients)
        self.feature_importance = np.abs(self.model.coef_)
        
        # Calculate training metrics
        train_preds = self.model.predict(X_scaled)
        train_mse = np.mean((y - train_preds) ** 2)
        train_r2 = self.model.score(X_scaled, y)
        
        logger.info(f"Linear model trained. Alpha={self.alpha:.3f}, R2={train_r2:.3f}, MSE={train_mse:.6f}")
        
        return {
            "alpha": self.alpha,
            "train_r2": train_r2,
            "train_mse": train_mse,
            "n_features": len(self.feature_names),
            "top_features": self._get_top_features(n=10)
        }
    
    def _tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: int = 5,
        alpha_range: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
    ) -> float:
        """
        Tune alpha using time series cross-validation.
        
        Args:
            X: Scaled feature matrix
            y: Target values
            cv_splits: Number of CV splits
            alpha_range: Alpha values to try
        
        Returns:
            best_alpha: Best alpha value
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        best_alpha = 1.0
        best_score = -np.inf
        
        for alpha in alpha_range:
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)  # R^2
                scores.append(score)
            
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
        
        logger.info(f"Best alpha: {best_alpha:.3f} (CV R2: {best_score:.3f})")
        return best_alpha
    
    def predict(
        self,
        features: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Generate predictions for given features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            **kwargs: Additional prediction parameters (unused)
        
        Returns:
            predictions: Predicted returns (n_samples,)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features match training
        if list(features.columns) != self.feature_names:
            logger.warning("Feature names mismatch. Reordering to match training.")
            features = features[self.feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def score(
        self,
        tickers: List[str],
        features_dict: Dict[str, pd.DataFrame],
        regime: Optional[str] = None,
        **kwargs
    ) -> ScorerOutput:
        """
        Score multiple tickers.
        
        Args:
            tickers: List of tickers to score
            features_dict: Dict mapping ticker -> features DataFrame
            regime: Current market regime (for score adjustment)
            **kwargs: Additional scoring parameters
        
        Returns:
            ScorerOutput with scores for all tickers
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scores = []
        for ticker in tickers:
            if ticker not in features_dict:
                logger.warning(f"Ticker {ticker} not in features_dict. Skipping.")
                continue
            
            features = features_dict[ticker]
            
            # Predict
            predictions = self.predict(features)
            
            # Use most recent prediction
            latest_prediction = float(predictions[-1])
            
            # Calculate confidence (based on prediction magnitude)
            confidence = min(abs(latest_prediction) / 0.02, 1.0)  # Normalize to [0, 1]
            
            # Create score result
            score_result = self._calculate_score_result(
                ticker=ticker,
                prediction=latest_prediction,
                confidence=confidence,
                regime=regime
            )
            
            scores.append(score_result)
        
        return ScorerOutput(scores=scores)
    
    def _get_top_features(self, n: int = 10) -> List[tuple]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
        
        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importance is None:
            return []
        
        # Sort by importance
        indices = np.argsort(self.feature_importance)[::-1][:n]
        top_features = [
            (self.feature_names[i], float(self.feature_importance[i]))
            for i in indices
        ]
        
        return top_features
    
    def save_model(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model files
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_path = path / "linear_model.pkl"
        scaler_path = path / "linear_scaler.pkl"
        metadata_path = path / "linear_metadata.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            "sector": self.sector,
            "alpha": self.alpha,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance
        }, metadata_path)
        
        logger.info(f"Linear model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory containing model files
        """
        model_path = path / "linear_model.pkl"
        scaler_path = path / "linear_scaler.pkl"
        metadata_path = path / "linear_metadata.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model components
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        metadata = joblib.load(metadata_path)
        self.sector = metadata["sector"]
        self.alpha = metadata["alpha"]
        self.feature_names = metadata["feature_names"]
        self.feature_importance = metadata["feature_importance"]
        
        logger.info(f"Linear model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and hyperparameters.
        
        Returns:
            dict with model info
        """
        info = {
            "name": self.name,
            "sector": self.sector,
            "architecture": "Ridge Regression",
            "alpha": self.alpha,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "trained": self.model is not None
        }
        
        if self.feature_importance is not None:
            info["top_features"] = self._get_top_features(n=5)
        
        return info

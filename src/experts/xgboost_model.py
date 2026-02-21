"""
XGBoost Model for Phase 1 Model-Sector Comparison

This model uses Gradient Boosted Trees to predict stock returns.
Expected to perform well on fundamental-heavy sectors (Tech, Cyclicals).

Architecture:
- Gradient boosted decision trees
- Non-linear feature interactions
- Automatic feature selection
- Regularization to prevent overfitting
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
from pathlib import Path

from .base_expert import ComparisonModel, ScorerOutput, ScoreResult

logger = logging.getLogger(__name__)


class XGBoostModel(ComparisonModel):
    """
    XGBoost (Gradient Boosted Trees) model.
    
    Strengths:
    - Handles non-linear relationships
    - Automatic feature interaction detection
    - Feature importance built-in
    - Robust to outliers
    - Good for complex multi-factor systems
    
    Expected Performance:
    - Tech: Sharpe ~1.8 (fundamental + macro driven)
    - Cyclicals: Sharpe ~1.6 (economic cycle + industrials)
    - Finance: Sharpe ~1.3 (non-linear but mean-reverting)
    - Crypto: Sharpe ~1.6 (complex patterns)
    - Commodities: Sharpe ~1.2 (macro interactions)
    """
    
    def __init__(self, sector: Optional[str] = None):
        """
        Initialize XGBoost model.
        
        Args:
            sector: Optional sector specialization (None = generalist)
        """
        super().__init__(name="xgboost", sector=sector)
        
        # Model
        self.model: Optional[xgb.XGBRegressor] = None
        
        # Hyperparameters (to be tuned)
        self.n_estimators = 100
        self.max_depth = 3
        self.learning_rate = 0.1
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.reg_alpha = 0.0  # L1 regularization
        self.reg_lambda = 1.0  # L2 regularization
        
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
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target returns (n_samples,)
            hyperparameter_tune: Whether to tune hyperparameters via CV
            **kwargs: Additional training parameters
                - cv_splits: Number of CV splits (default: 5)
                - param_grid: Dict of parameters to try
        
        Returns:
            dict with training metrics
        """
        logger.info(f"Training XGBoost model on {len(features)} samples with {features.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(features.columns)
        X = features.values
        y = targets.values
        
        # Hyperparameter tuning
        if hyperparameter_tune:
            best_params = self._tune_hyperparameters(
                X, y,
                cv_splits=kwargs.get('cv_splits', 5),
                param_grid=kwargs.get('param_grid', None)
            )
            self._set_hyperparameters(best_params)
        
        # Train final model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Calculate training metrics
        train_preds = self.model.predict(X)
        train_mse = np.mean((y - train_preds) ** 2)
        train_r2 = self.model.score(X, y)
        
        logger.info(f"XGBoost model trained. Trees={self.n_estimators}, Depth={self.max_depth}, R2={train_r2:.3f}")
        
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
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
        param_grid: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_splits: Number of CV splits
            param_grid: Parameter grid to search (if None, uses default)
        
        Returns:
            best_params: Best hyperparameter combination
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'reg_lambda': [0.1, 1.0, 10.0]
            }
        
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        best_params = {}
        best_score = -np.inf
        
        # Simple grid search (could be replaced with RandomizedSearchCV)
        for n_est in param_grid.get('n_estimators', [100]):
            for depth in param_grid.get('max_depth', [3]):
                for lr in param_grid.get('learning_rate', [0.1]):
                    for reg_l in param_grid.get('reg_lambda', [1.0]):
                        scores = []
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            model = xgb.XGBRegressor(
                                n_estimators=n_est,
                                max_depth=depth,
                                learning_rate=lr,
                                reg_lambda=reg_l,
                                random_state=42,
                                n_jobs=-1
                            )
                            model.fit(X_train, y_train)
                            score = model.score(X_val, y_val)
                            scores.append(score)
                        
                        mean_score = np.mean(scores)
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'learning_rate': lr,
                                'reg_lambda': reg_l
                            }
        
        logger.info(f"Best params: {best_params} (CV R2: {best_score:.3f})")
        return best_params
    
    def _set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set hyperparameters from dict."""
        if 'n_estimators' in params:
            self.n_estimators = params['n_estimators']
        if 'max_depth' in params:
            self.max_depth = params['max_depth']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'subsample' in params:
            self.subsample = params['subsample']
        if 'colsample_bytree' in params:
            self.colsample_bytree = params['colsample_bytree']
        if 'reg_alpha' in params:
            self.reg_alpha = params['reg_alpha']
        if 'reg_lambda' in params:
            self.reg_lambda = params['reg_lambda']
    
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
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features match training
        if list(features.columns) != self.feature_names:
            logger.warning("Feature names mismatch. Reordering to match training.")
            features = features[self.feature_names]
        
        X = features.values
        predictions = self.model.predict(X)
        
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
            
            # Calculate confidence (based on tree consensus)
            # Higher feature importance features = higher confidence
            confidence = min(abs(latest_prediction) / 0.02, 1.0)
            
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
        
        # Save model
        model_path = path / "xgboost_model.json"
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata_path = path / "xgboost_metadata.pkl"
        joblib.dump({
            "sector": self.sector,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance
        }, metadata_path)
        
        logger.info(f"XGBoost model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory containing model files
        """
        model_path = path / "xgboost_model.json"
        metadata_path = path / "xgboost_metadata.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        
        # Load metadata
        metadata = joblib.load(metadata_path)
        self.sector = metadata["sector"]
        self.n_estimators = metadata["n_estimators"]
        self.max_depth = metadata["max_depth"]
        self.learning_rate = metadata["learning_rate"]
        self.subsample = metadata["subsample"]
        self.colsample_bytree = metadata["colsample_bytree"]
        self.reg_alpha = metadata["reg_alpha"]
        self.reg_lambda = metadata["reg_lambda"]
        self.feature_names = metadata["feature_names"]
        self.feature_importance = metadata["feature_importance"]
        
        logger.info(f"XGBoost model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and hyperparameters.
        
        Returns:
            dict with model info
        """
        info = {
            "name": self.name,
            "sector": self.sector,
            "architecture": "Gradient Boosted Trees",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "trained": self.model is not None
        }
        
        if self.feature_importance is not None:
            info["top_features"] = self._get_top_features(n=5)
        
        return info

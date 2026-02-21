"""
Base Expert Class (Layer III)

Abstract base class for all expert models (comparison models and sector experts).
All experts output standardized predictions for downstream risk management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.data_structures import ScoreResult, ScorerOutput


class BaseExpert(ABC):
    """
    Abstract base class for expert models.
    
    All experts (whether comparison models or sector-specific) must implement:
    - train(): Train the model on historical data
    - predict(): Generate predictions for stocks
    - score(): Convert predictions to ScoreResult format
    
    This ensures consistent interface across all model types.
    """
    
    def __init__(
        self,
        name: str,
        model_path: Optional[Path] = None,
        **kwargs
    ):
        """
        Initialize expert model.
        
        Args:
            name: Expert name (e.g., "xgboost", "tech_expert")
            model_path: Path to save/load trained model
            **kwargs: Additional model-specific parameters
        """
        self.name = name
        self.model_path = model_path or Path(f"models/{name}_model.pkl")
        self.model = None
        self.trained = False
        
        self.logger = logger.bind(module=f"expert_{name}")
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the expert model.
        
        Args:
            X: Training features
            y: Training labels (e.g., forward returns)
            validation_data: Optional (X_val, y_val) for validation
            **kwargs: Model-specific training parameters
            
        Returns:
            Dict of training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        regime: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate raw predictions.
        
        Args:
            X: Feature matrix for prediction
            regime: Current market regime ("bull", "bear", "sideways")
            **kwargs: Model-specific prediction parameters
            
        Returns:
            Array of predictions (probabilities or scores)
        """
        pass
    
    @abstractmethod
    def score(
        self,
        tickers: List[str],
        features: Dict[str, pd.DataFrame],
        regime: Optional[str] = None,
        **kwargs
    ) -> ScorerOutput:
        """
        Score stocks and return standardized output.
        
        This is the main method called by the pipeline.
        
        Args:
            tickers: List of tickers to score
            features: Dict mapping ticker to feature DataFrame
            regime: Current market regime
            **kwargs: Additional parameters
            
        Returns:
            ScorerOutput with ScoreResult for each ticker
        """
        pass
    
    def save_model(self, path: Optional[Path] = None):
        """
        Save trained model to disk.
        
        Args:
            path: Save path (uses self.model_path if not provided)
        """
        import pickle
        
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "name": self.name,
            "trained": self.trained,
            "metadata": self._get_model_metadata()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Optional[Path] = None):
        """
        Load trained model from disk.
        
        Args:
            path: Load path (uses self.model_path if not provided)
        """
        import pickle
        
        path = path or self.model_path
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.trained = model_data["trained"]
        
        self.logger.info(f"Model loaded from {path}")
    
    def _get_model_metadata(self) -> Dict:
        """
        Get model metadata for saving.
        
        Returns:
            Dict with model information
        """
        return {
            "name": self.name,
            "trained": self.trained,
            "model_type": self.__class__.__name__
        }
    
    def _calculate_score_result(
        self,
        prediction: float,
        historical_returns: Optional[pd.Series] = None,
        regime: Optional[str] = None
    ) -> ScoreResult:
        """
        Convert raw prediction to ScoreResult.
        
        Args:
            prediction: Model prediction (probability or score)
            historical_returns: Historical returns for calculating statistics
            regime: Current market regime
            
        Returns:
            ScoreResult with EV, probabilities, payoffs
        """
        # Default implementation - subclasses should override for better estimates
        
        if historical_returns is not None and len(historical_returns) > 20:
            # Calculate from historical data
            positive_returns = historical_returns[historical_returns > 0]
            negative_returns = historical_returns[historical_returns < 0]
            
            p_win = len(positive_returns) / len(historical_returns)
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.05
            avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.03
        else:
            # Use defaults based on prediction
            p_win = float(prediction) if 0 <= prediction <= 1 else 0.5
            avg_win = 0.05  # 5% default win
            avg_loss = 0.03  # 3% default loss
        
        # Adjust for regime
        if regime == "bear":
            p_win *= 0.8  # Reduce win probability in bear market
            avg_loss *= 1.5  # Increase expected loss
        elif regime == "bull":
            p_win *= 1.1  # Increase win probability (capped at 1.0)
            avg_win *= 1.2  # Increase expected win
        
        p_win = min(max(p_win, 0.0), 1.0)  # Clip to [0, 1]
        
        # Calculate EV
        ev = (p_win * avg_win) - ((1 - p_win) * avg_loss)
        
        # Confidence based on prediction strength
        confidence = abs(prediction - 0.5) * 2 if 0 <= prediction <= 1 else 0.5
        
        return ScoreResult(
            ev=ev,
            p_win=p_win,
            avg_win=avg_win,
            avg_loss=avg_loss,
            score=prediction * 100 if 0 <= prediction <= 1 else 50.0,
            confidence=confidence,
            data_points=len(historical_returns) if historical_returns is not None else 0
        )
    
    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"


class ComparisonModel(BaseExpert):
    """
    Base class for comparison models (Linear, CNN, XGBoost, LLM).
    
    These models are generalists that can score any stock in any sector.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_type = "comparison"
    
    def can_score_ticker(self, ticker: str) -> bool:
        """
        Check if model can score this ticker.
        
        Comparison models can score any ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            True (comparison models are generalists)
        """
        return True


class SectorExpert(BaseExpert):
    """
    Base class for sector-specific experts (Tech, Crypto, Finance, etc.).
    
    These models specialize in one sector and only score stocks in that sector.
    """
    
    def __init__(
        self,
        name: str,
        sector: str,
        sector_tickers: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.model_type = "sector_expert"
        self.sector = sector
        self.sector_tickers = sector_tickers or []
    
    def can_score_ticker(self, ticker: str) -> bool:
        """
        Check if model can score this ticker.
        
        Sector experts only score tickers in their sector.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            True if ticker is in this expert's sector
        """
        return ticker in self.sector_tickers
    
    def add_sector_ticker(self, ticker: str):
        """Add ticker to sector."""
        if ticker not in self.sector_tickers:
            self.sector_tickers.append(ticker)
    
    def _get_model_metadata(self) -> Dict:
        """Get model metadata including sector info."""
        metadata = super()._get_model_metadata()
        metadata.update({
            "sector": self.sector,
            "sector_tickers": self.sector_tickers
        })
        return metadata


# Example placeholder implementation
class PlaceholderExpert(ComparisonModel):
    """
    Placeholder expert for testing.
    
    Returns random predictions until real models are implemented.
    """
    
    def train(self, X, y, validation_data=None, **kwargs):
        """Placeholder training."""
        self.trained = True
        return {"loss": 0.5, "accuracy": 0.6}
    
    def predict(self, X, regime=None, **kwargs):
        """Random predictions."""
        np.random.seed(42)
        return np.random.uniform(0.4, 0.6, size=len(X))
    
    def score(self, tickers, features, regime=None, **kwargs):
        """Score tickers with random predictions."""
        scores = {}
        
        for ticker in tickers:
            if ticker in features:
                # Random prediction
                prediction = np.random.uniform(0.4, 0.6)
                
                # Convert to ScoreResult
                scores[ticker] = self._calculate_score_result(
                    prediction=prediction,
                    regime=regime
                )
        
        return ScorerOutput(
            scores=scores,
            model_name=self.name,
            timestamp=pd.Timestamp.now()
        )

"""Placeholder implementations of scorers for testing."""

from datetime import date
from typing import Dict, Optional
import pandas as pd
from pathlib import Path

from .base_scorer import BaseScorer
from ..data_structures import ScoreResult


class LinearScorer(BaseScorer):
    """Placeholder linear regression scorer."""
    
    def __init__(self):
        super().__init__("linear")
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> bool:
        """Placeholder training."""
        self.is_trained = True
        return True
    
    def score(self, ticker: str, current_date: date, features: Dict[str, float]) -> ScoreResult:
        """Placeholder scoring."""
        # TODO: Implement actual linear regression scoring
        return ScoreResult(
            ticker=ticker,
            date=current_date,
            model_name="linear",
            score=50.0,
            p_win=0.55,
            avg_win=0.01,
            avg_loss=0.01,
            ev=0.0,
            confidence=0.5,
        )
    
    def save(self, path: Path) -> bool:
        """Placeholder save."""
        return True
    
    def load(self, path: Path) -> bool:
        """Placeholder load."""
        return True


class CNNScorer(BaseScorer):
    """Placeholder CNN scorer."""
    
    def __init__(self):
        super().__init__("cnn")
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> bool:
        """Placeholder training."""
        self.is_trained = True
        return True
    
    def score(self, ticker: str, current_date: date, features: Dict[str, float]) -> ScoreResult:
        """Placeholder scoring."""
        # TODO: Implement actual CNN scoring
        return ScoreResult(
            ticker=ticker,
            date=current_date,
            model_name="cnn",
            score=50.0,
            p_win=0.50,
            avg_win=0.02,
            avg_loss=0.02,
            ev=0.0,
            confidence=0.5,
        )
    
    def save(self, path: Path) -> bool:
        """Placeholder save."""
        return True
    
    def load(self, path: Path) -> bool:
        """Placeholder load."""
        return True


class XGBoostScorer(BaseScorer):
    """Placeholder XGBoost scorer."""
    
    def __init__(self):
        super().__init__("xgboost")
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> bool:
        """Placeholder training."""
        self.is_trained = True
        return True
    
    def score(self, ticker: str, current_date: date, features: Dict[str, float]) -> ScoreResult:
        """Placeholder scoring."""
        # TODO: Implement actual XGBoost scoring
        return ScoreResult(
            ticker=ticker,
            date=current_date,
            model_name="xgboost",
            score=50.0,
            p_win=0.52,
            avg_win=0.015,
            avg_loss=0.015,
            ev=0.0,
            confidence=0.5,
        )
    
    def save(self, path: Path) -> bool:
        """Placeholder save."""
        return True
    
    def load(self, path: Path) -> bool:
        """Placeholder load."""
        return True


class LLMScorer(BaseScorer):
    """Placeholder LLM scorer."""
    
    def __init__(self):
        super().__init__("llm")
    
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> bool:
        """Placeholder training."""
        self.is_trained = True
        return True
    
    def score(self, ticker: str, current_date: date, features: Dict[str, float]) -> ScoreResult:
        """Placeholder scoring."""
        # TODO: Implement actual LLM scoring
        return ScoreResult(
            ticker=ticker,
            date=current_date,
            model_name="llm",
            score=50.0,
            p_win=0.35,
            avg_win=0.05,
            avg_loss=0.03,
            ev=0.007,
            confidence=0.5,
        )
    
    def save(self, path: Path) -> bool:
        """Placeholder save."""
        return True
    
    def load(self, path: Path) -> bool:
        """Placeholder load."""
        return True

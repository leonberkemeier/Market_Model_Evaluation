"""Base class for model scorers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import date
import pandas as pd
from pathlib import Path

from ..data_structures import ScoreResult, ScorerOutput


class BaseScorer(ABC):
    """
    Abstract base class for stock scoring models.
    Each scorer produces a ScoreResult containing P_win, avg_win, avg_loss, and EV.
    """
    
    def __init__(self, model_name: str, model_path: Optional[Path] = None):
        """
        Initialize scorer.
        
        Args:
            model_name: Name of the model ("linear", "cnn", "xgboost", "llm")
            model_path: Path to saved model file (if applicable)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.is_trained = False
    
    @abstractmethod
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Train the scorer on historical data.
        
        Args:
            training_data: Training dataset with features and labels
            validation_data: Validation dataset for tuning (optional)
            
        Returns:
            True if training successful, False otherwise
        """
        pass
    
    @abstractmethod
    def score(self, ticker: str, current_date: date, features: Dict[str, float]) -> ScoreResult:
        """
        Score a single stock on a specific date.
        
        Args:
            ticker: Stock ticker
            current_date: Date to score for
            features: Dictionary of computed features
            
        Returns:
            ScoreResult with score, P_win, avg_win, avg_loss, EV
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> bool:
        """
        Save trained model to disk.
        
        Args:
            path: Directory to save model
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> bool:
        """
        Load trained model from disk.
        
        Args:
            path: Directory containing saved model
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def score_batch(self, tickers: List[str], current_date: date, features_dict: Dict[str, Dict[str, float]]) -> ScorerOutput:
        """
        Score multiple stocks at once.
        
        Args:
            tickers: List of stock tickers
            current_date: Date to score for
            features_dict: Dictionary of ticker -> features
            
        Returns:
            ScorerOutput containing all scores
        """
        scores = []
        for ticker in tickers:
            if ticker in features_dict:
                score_result = self.score(ticker, current_date, features_dict[ticker])
                if score_result.validate():
                    scores.append(score_result)
        
        return ScorerOutput(
            date=current_date,
            model_name=self.model_name,
            scores=scores
        )
    
    def normalize_scores(self, scores: List[ScoreResult]) -> List[ScoreResult]:
        """
        Normalize raw scores to 0-100 percentile scale.
        
        Args:
            scores: List of ScoreResult objects
            
        Returns:
            List with normalized scores (0-100 scale)
        """
        if not scores:
            return scores
        
        # Extract raw EV values
        ev_values = [s.ev for s in scores]
        min_ev = min(ev_values)
        max_ev = max(ev_values)
        
        # Normalize to 0-100
        normalized_scores = []
        for score in scores:
            if max_ev == min_ev:
                normalized_score = 50.0  # Default to middle if all same
            else:
                normalized_score = 100.0 * (score.ev - min_ev) / (max_ev - min_ev)
            
            score.score = normalized_score
            normalized_scores.append(score)
        
        return normalized_scores
    
    def validate_output(self, result: ScoreResult) -> bool:
        """
        Validate score output before returning.
        
        Args:
            result: ScoreResult to validate
            
        Returns:
            True if valid, False otherwise
        """
        return result.validate()

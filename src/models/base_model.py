from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class NFLPredictionModel(ABC):
    """Base class for all NFL prediction models"""

    def __init__(self):
        self.confidence_threshold = 0.8
        self.high_confidence_threshold = 0.85
        self.is_trained = False
        self.model_name = "Base NFL Model"

    @abstractmethod
    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single game

        Args:
            game_data: Dictionary containing game information including:
                - game_time: datetime
                - home_team: str
                - away_team: str
                - weather_conditions: str
                - additional model-specific data

        Returns:
            Dictionary containing:
                - predicted_winner: str
                - win_probability: float
                - confidence_score: float
                - model_specific_factors: dict
        """
        pass

    @abstractmethod
    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the model with historical data

        Args:
            training_data: Dictionary containing:
                - games: List of historical games
                - features: Training features
                - labels: Training labels
                - additional model-specific data
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model performance on test data

        Args:
            test_data: Dictionary containing test data similar to training_data

        Returns:
            Dictionary containing performance metrics:
                - accuracy: float
                - precision: float
                - recall: float
                - f1_score: float
        """
        pass

    def is_high_confidence(self, prediction: Dict[str, Any]) -> bool:
        """Check if prediction meets high confidence criteria"""
        return prediction["confidence_score"] >= self.high_confidence_threshold

    def save_model(self, model_path: str) -> None:
        """Save model to disk"""
        raise NotImplementedError("Save method not implemented")

    def load_model(self, model_path: str) -> None:
        """Load model from disk"""
        raise NotImplementedError("Load method not implemented")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status"""
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "confidence_threshold": self.confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
        }

from abc import ABC, abstractmethod
from typing import Dict, Any


class NFLTotalPredictionModel(ABC):
    """Abstract base class for NFL total prediction models"""

    @abstractmethod
    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict game outcomes

        Args:
            game_data: Dictionary containing game features

        Returns:
            Dictionary containing:
            - total: Dict with points prediction and over/under
            - spread: Dict with spread prediction
            - moneyline: Dict with win probabilities
            - confidence: Overall confidence in predictions
            - explanation: Explanation of predictions
        """
        pass

    @abstractmethod
    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train model on historical data

        Args:
            training_data: Dictionary containing:
            - total_points: Actual total points
            - spread: Actual spread
            - home_win: Whether home team won
            - features: Game statistics and conditions
        """
        pass

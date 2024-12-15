"""
Base total prediction model.
"""
from typing import Dict, Any
import numpy as np


class TotalPredictionModel:
    def __init__(self):
        self.name = "Base Total Prediction Model"

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions for total points in a game.

        Args:
            game_data: Dictionary containing game information

        Returns:
            Dictionary containing predictions
        """
        return {"predicted_total": 0.0, "confidence": 0.0, "recommendation": "PASS"}


class NFLTotalPredictionModel(TotalPredictionModel):
    def __init__(self):
        super().__init__()
        self.feature_columns = [
            "home_points_per_game",
            "away_points_per_game",
            "home_points_allowed",
            "away_points_allowed",
            "home_yards_per_play",
            "away_yards_per_play",
            "home_turnover_diff",
            "away_turnover_diff",
            "home_win_pct",
            "away_win_pct",
            "spread",
            "over_under",
        ]

        self.total_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
        )

        self.confidence_thresholds = {"high": 0.75, "medium": 0.50, "low": 0.25}

    def calculate_confidence(self, prediction, features):
        """Calculate prediction confidence based on multiple factors"""
        base_confidence = 0.25

        # Feature reliability boost
        if all(features.get(col) is not None for col in self.feature_columns):
            base_confidence += 0.15

        # Historical accuracy boost
        if hasattr(self, "historical_accuracy"):
            base_confidence += self.historical_accuracy * 0.25

        # Prediction strength boost
        prediction_strength = (
            abs(prediction["total_points"] - features.get("vegas_total", 0)) / 10
        )
        base_confidence += min(prediction_strength * 0.1, 0.35)

        return min(base_confidence, 1.0)

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            features = pd.DataFrame([self._extract_features(game_data)])

            total_points = float(
                self.total_model.predict(features[self.feature_columns])[0]
            )
            confidence = self.calculate_confidence(
                {"total_points": total_points}, game_data
            )

            vegas_total = game_data.get("vegas_total", total_points)
            point_diff = abs(total_points - vegas_total)

            recommendation = "PASS"
            if confidence > self.confidence_thresholds["high"] and point_diff > 3:
                recommendation = "STRONG " + (
                    "OVER" if total_points > vegas_total else "UNDER"
                )
            elif confidence > self.confidence_thresholds["medium"]:
                recommendation = "OVER" if total_points > vegas_total else "UNDER"

            return {
                "total_points": total_points,
                "prediction": recommendation,
                "confidence": confidence * 100,
                "point_difference": point_diff,
                "explanation": f"Predicted total: {total_points:.1f} (Vegas: {vegas_total})\n"
                f"Confidence: {confidence*100:.1f}%\n"
                f"Recommendation: {recommendation}",
            }

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise


from typing import Dict, Any
import numpy as np


class TotalPredictionModel:
    def __init__(self):
        self.name = "Base Total Prediction Model"

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions for total points in a game.

        Args:
            game_data: Dictionary containing game information

        Returns:
            Dictionary containing predictions
        """
        return {"predicted_total": 0.0, "confidence": 0.0, "recommendation": "PASS"}


from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging


class NFLTotalPredictionModel(TotalPredictionModel):
    """Model specifically designed for predicting game totals (over/under)"""

    def __init__(self):
        super().__init__()
        self.feature_columns = [
            "home_points_per_game",
            "away_points_per_game",
            "home_points_allowed",
            "away_points_allowed",
        ]

        # Initialize models
        self.total_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.spread_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.win_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def _extract_features(self, game):
        return {
            "home_points_per_game": game.get("home_points_per_game", 0),
            "away_points_per_game": game.get("away_points_per_game", 0),
            "home_points_allowed": game.get("home_points_allowed", 0),
            "away_points_allowed": game.get("away_points_allowed", 0),
        }

    def train(self, training_data) -> None:
        """Train ensemble models"""
        try:
            # Extract features
            X = training_data[self.feature_columns].values

            # Extract target variables
            total_points = training_data["total_points"].values
            spreads = training_data["spread"].values
            home_wins = training_data["home_win"].values

            # Train models
            self.total_model.fit(X, total_points)
            self.spread_model.fit(X, spreads)
            self.win_model.fit(X, home_wins)

            logging.info(f"Successfully trained {self.__class__.__name__}")

        except Exception as e:
            logging.error(f"Error training {self.__class__.__name__}: {str(e)}")
            raise

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict game outcomes using ensemble models"""
        try:
            # Extract features
            features = pd.DataFrame([self._extract_features(game_data)])

            # Make predictions
            total_points = float(
                self.total_model.predict(features[self.feature_columns])[0]
            )
            spread_pred = float(
                self.spread_model.predict(features[self.feature_columns])[0]
            )
            win_prob = float(
                self.win_model.predict_proba(features[self.feature_columns])[0][1]
            )

            # Calculate confidence based on model metrics
            confidence = 0.25  # Base confidence level

            return {
                "total_points": total_points,
                "total_prediction": "OVER"
                if total_points > game_data.get("vegas_total", total_points)
                else "UNDER",
                "spread": spread_pred,
                "spread_prediction": "HOME" if spread_pred > 0 else "AWAY",
                "home_win_probability": win_prob * 100,
                "confidence": confidence * 100,
                "explanation": f"Ensemble model predicts {total_points:.1f} total points, "
                f"{abs(spread_pred):.1f} point {'win' if spread_pred > 0 else 'loss'} for home team, "
                f"with {win_prob * 100:.1f}% home win probability",
            }

        except Exception as e:
            logging.error(
                f"Error making predictions with {self.__class__.__name__}: {str(e)}"
            )
            raise

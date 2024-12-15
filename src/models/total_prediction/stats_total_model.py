from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ..base_total_model import NFLTotalPredictionModel
import logging


class StatsTotalModel(NFLTotalPredictionModel):
    """Statistical model for total points prediction"""

    def __init__(self):
        # Initialize models for each prediction type
        self.total_model = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=42
        )
        self.spread_model = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=42
        )
        self.win_model = RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=42
        )

        self.scaler = StandardScaler()

        self.feature_columns = [
            "home_points_per_game",
            "away_points_per_game",
            "home_points_allowed",
            "away_points_allowed",
            "home_yards_per_game",
            "away_yards_per_game",
            "home_yards_allowed",
            "away_yards_allowed",
            "home_pass_yards_per_game",
            "away_pass_yards_per_game",
            "home_rush_yards_per_game",
            "away_rush_yards_per_game",
            "home_third_down_conv",
            "away_third_down_conv",
            "home_fourth_down_conv",
            "away_fourth_down_conv",
            "home_time_of_possession",
            "away_time_of_possession",
            "home_turnover_margin",
            "away_turnover_margin",
        ]

    def _extract_features(self, game_data):
        """Extract relevant features from game data"""
        features = {}

        # Basic stats
        features["home_points_per_game"] = game_data.get("home_team_stats", {}).get(
            "points_per_game", 0
        )
        features["away_points_per_game"] = game_data.get("away_team_stats", {}).get(
            "points_per_game", 0
        )
        features["home_points_allowed"] = game_data.get("home_team_stats", {}).get(
            "points_allowed_per_game", 0
        )
        features["away_points_allowed"] = game_data.get("away_team_stats", {}).get(
            "points_allowed_per_game", 0
        )

        # Yardage stats
        features["home_yards_per_game"] = game_data.get("home_team_stats", {}).get(
            "total_yards_per_game", 0
        )
        features["away_yards_per_game"] = game_data.get("away_team_stats", {}).get(
            "total_yards_per_game", 0
        )
        features["home_yards_allowed"] = game_data.get("home_team_stats", {}).get(
            "yards_allowed_per_game", 0
        )
        features["away_yards_allowed"] = game_data.get("away_team_stats", {}).get(
            "yards_allowed_per_game", 0
        )

        # Passing stats
        features["home_pass_yards_per_game"] = game_data.get("home_team_stats", {}).get(
            "pass_yards_per_game", 0
        )
        features["away_pass_yards_per_game"] = game_data.get("away_team_stats", {}).get(
            "pass_yards_per_game", 0
        )

        # Rushing stats
        features["home_rush_yards_per_game"] = game_data.get("home_team_stats", {}).get(
            "rush_yards_per_game", 0
        )
        features["away_rush_yards_per_game"] = game_data.get("away_team_stats", {}).get(
            "rush_yards_per_game", 0
        )

        # Conversion stats
        features["home_third_down_conv"] = game_data.get("home_team_stats", {}).get(
            "third_down_conversion", 0
        )
        features["away_third_down_conv"] = game_data.get("away_team_stats", {}).get(
            "third_down_conversion", 0
        )
        features["home_fourth_down_conv"] = game_data.get("home_team_stats", {}).get(
            "fourth_down_conversion", 0
        )
        features["away_fourth_down_conv"] = game_data.get("away_team_stats", {}).get(
            "fourth_down_conversion", 0
        )

        # Time of possession
        features["home_time_of_possession"] = game_data.get("home_team_stats", {}).get(
            "time_of_possession", 0
        )
        features["away_time_of_possession"] = game_data.get("away_team_stats", {}).get(
            "time_of_possession", 0
        )

        # Turnover stats
        features["home_turnover_margin"] = game_data.get("home_team_stats", {}).get(
            "turnover_margin", 0
        )
        features["away_turnover_margin"] = game_data.get("away_team_stats", {}).get(
            "turnover_margin", 0
        )

        return features

    def train(self, training_data) -> None:
        """Train statistical models"""
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
        """Predict game outcomes using statistical models"""
        try:
            # Extract features using the same method as in training
            features = self._extract_features(game_data)
            features_df = pd.DataFrame([features])

            # Fill missing values with defaults
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0

            # Ensure features are in the same order as during training
            features_array = features_df[self.feature_columns].values

            # Make predictions
            total_points = float(self.total_model.predict(features_array)[0])
            spread_pred = float(self.spread_model.predict(features_array)[0])
            win_prob = float(self.win_model.predict_proba(features_array)[0][1])

            # Calculate confidence based on model metrics
            confidence = 0.05  # Base confidence level

            return {
                "total_points": total_points,
                "total_prediction": "OVER"
                if total_points > game_data.get("vegas_total", total_points)
                else "UNDER",
                "spread": spread_pred,
                "spread_prediction": "HOME" if spread_pred > 0 else "AWAY",
                "home_win_probability": win_prob * 100,
                "confidence": confidence * 100,
                "explanation": f"Statistical model predicts {total_points:.1f} total points, "
                f"{abs(spread_pred):.1f} point {'win' if spread_pred > 0 else 'loss'} for home team, "
                f"with {win_prob * 100:.1f}% home win probability",
            }

        except Exception as e:
            logging.error(
                f"Error making predictions with {self.__class__.__name__}: {str(e)}"
            )
            raise

    def _get_default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model fails"""
        return {
            "total_points": 47.5,
            "total_prediction": "UNDER",
            "spread": 0.0,
            "spread_prediction": "HOME",
            "home_win_probability": 55.0,
            "confidence": 5.0,
            "explanation": "Using league averages due to model error",
        }

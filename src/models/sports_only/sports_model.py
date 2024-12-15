from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from ..base_model import NFLPredictionModel
import logging

logger = logging.getLogger(__name__)


class SportsModel(NFLPredictionModel):
    """Traditional sports analytics model enhanced with advanced metrics"""

    def __init__(self):
        """Initialize the model"""
        self.model = GradientBoostingClassifier(
            n_estimators=500,  # Increased for better convergence
            learning_rate=0.1,  # Increased to prevent underfitting
            max_depth=4,  # Reduced to prevent overfitting
            min_samples_split=10,  # Increased for better generalization
            min_samples_leaf=5,  # Increased for stability
            subsample=0.8,  # Slight decrease for better generalization
            max_features=0.7,  # Use 70% of features instead of sqrt
            random_state=123,
        )
        self.scaler = StandardScaler()
        self.feature_means = {}  # Store mean values for each feature
        self.feature_columns = [
            # Core performance metrics (most reliable)
            "home_points_per_game",
            "away_points_per_game",
            "home_points_allowed_per_game",
            "away_points_allowed_per_game",
            "home_win_pct",
            "away_win_pct",
            "home_last_5_wins",
            "away_last_5_wins",
            # Key offensive metrics
            "home_total_yards",
            "away_total_yards",
            "home_yards_per_play",
            "away_yards_per_play",
            "home_third_down_pct",
            "away_third_down_pct",
            "home_red_zone_pct",
            "away_red_zone_pct",
            # Key defensive metrics
            "home_defensive_efficiency",
            "away_defensive_efficiency",
            "home_sacks_per_game",
            "away_sacks_per_game",
            "home_turnover_diff",
            "away_turnover_diff",
            # Game context
            "is_division_game",
            "days_rest",
            "home_field_advantage",
            # Feature interactions
            "points_diff",
            "yards_diff",
            "turnover_diff_ratio",
            "win_streak_diff",
        ]
        self.trained = False

    def _extract_features(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features from game data"""
        features = {}

        try:
            # Get team data (check both root level and game_info)
            home_team = game_data.get("home_team") or game_data.get(
                "game_info", {}
            ).get("home_team")
            away_team = game_data.get("away_team") or game_data.get(
                "game_info", {}
            ).get("away_team")

            if not home_team or not away_team:
                logger.warning("Missing team information")
                return {col: 0.0 for col in self.feature_columns}

            # Get team stats
            home_stats = game_data.get("home_stats", {})
            away_stats = game_data.get("away_stats", {})
            game_info = game_data.get("game_info", {})
            weather = game_data.get("weather", {})

            # Core performance metrics
            features["home_points_per_game"] = float(
                home_stats.get("points_per_game", 0)
            )
            features["away_points_per_game"] = float(
                away_stats.get("points_per_game", 0)
            )
            features["home_points_allowed_per_game"] = float(
                home_stats.get("points_allowed", 0)
            )
            features["away_points_allowed_per_game"] = float(
                away_stats.get("points_allowed", 0)
            )
            features["home_win_pct"] = float(game_info.get("home_win_percentage", 0))
            features["away_win_pct"] = float(game_info.get("away_win_percentage", 0))
            features["home_last_5_wins"] = float(home_stats.get("last_5_wins", 0)) / 5.0
            features["away_last_5_wins"] = float(away_stats.get("last_5_wins", 0)) / 5.0

            # Key offensive metrics
            features["home_total_yards"] = float(home_stats.get("total_yards", 0))
            features["away_total_yards"] = float(away_stats.get("total_yards", 0))
            features["home_yards_per_play"] = float(home_stats.get("yards_per_play", 0))
            features["away_yards_per_play"] = float(away_stats.get("yards_per_play", 0))
            features["home_third_down_pct"] = float(home_stats.get("third_down_pct", 0))
            features["away_third_down_pct"] = float(away_stats.get("third_down_pct", 0))
            features["home_red_zone_pct"] = float(home_stats.get("red_zone_pct", 0))
            features["away_red_zone_pct"] = float(away_stats.get("red_zone_pct", 0))

            # Key defensive metrics
            features["home_defensive_efficiency"] = float(
                home_stats.get("defensive_efficiency", 0)
            )
            features["away_defensive_efficiency"] = float(
                away_stats.get("defensive_efficiency", 0)
            )
            features["home_sacks_per_game"] = float(home_stats.get("sacks_per_game", 0))
            features["away_sacks_per_game"] = float(away_stats.get("sacks_per_game", 0))
            features["home_turnover_diff"] = float(
                home_stats.get("turnovers_forced", 0)
            )
            features["away_turnover_diff"] = float(
                away_stats.get("turnovers_forced", 0)
            )

            # Game context
            features["is_division_game"] = float(game_info.get("is_division_game", 0))
            features["days_rest"] = float(game_info.get("days_rest", 7))
            features["home_field_advantage"] = 1.0  # Always 1 for home team

            # Feature interactions
            features["points_diff"] = (
                features["home_points_per_game"]
                - features["home_points_allowed_per_game"]
            ) - (
                features["away_points_per_game"]
                - features["away_points_allowed_per_game"]
            )
            features["yards_diff"] = (
                features["home_total_yards"] - features["away_total_yards"]
            ) / 100.0
            features["turnover_diff_ratio"] = (
                features["home_turnover_diff"] - features["away_turnover_diff"]
            ) / (features["home_turnover_diff"] + features["away_turnover_diff"] + 1)
            features["win_streak_diff"] = (
                features["home_last_5_wins"] - features["away_last_5_wins"]
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {col: 0.0 for col in self.feature_columns}

    def train(self, training_data: Dict[str, Any]) -> None:
        """Train model on historical data"""
        try:
            games = training_data["games"]
            labels = training_data["labels"]

            # Extract features for all games
            features_list = []
            for game in games:
                features = self._extract_features(game)
                features_list.append(features)

            # Convert to DataFrame
            X = pd.DataFrame(features_list)

            # Calculate and store means for each feature
            self.feature_means = X.mean().to_dict()

            # Fill missing values with feature means
            for col in X.columns:
                X[col] = X[col].fillna(self.feature_means.get(col, 0))

            # Scale features
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

            # Train model
            self.model.fit(X, labels)
            self.trained = True

            # Log feature importance
            importance = self.model.feature_importances_
            top_features = sorted(
                zip(self.feature_columns, importance), key=lambda x: x[1], reverse=True
            )[:5]
            logger.info("Top 5 important features:")
            for feature, importance in top_features:
                logger.info(f"{feature}: {importance:.4f}")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            self.trained = False

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single game"""
        try:
            # Extract features
            features = self._extract_features(game_data)

            # Convert to DataFrame
            X = pd.DataFrame([features])

            # Fill missing values with stored means
            for col in X.columns:
                X[col] = X[col].fillna(self.feature_means.get(col, 0))

            # Scale features if model is trained
            if hasattr(self.scaler, "mean_"):
                X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

            if not self.trained:
                logger.warning("Model not trained yet")
                return {
                    "predicted_winner": None,
                    "win_probability": 0.5,
                    "confidence_score": 0.0,
                }

            # Make prediction
            prob = self.model.predict_proba(X)[0]
            home_win_prob = prob[1]

            # Calculate data completeness (excluding derived features)
            core_features = [
                f
                for f in features.keys()
                if not any(x in f for x in ["diff", "advantage", "impact"])
            ]
            non_zero_features = sum(
                1 for k, v in features.items() if k in core_features and v != 0
            )
            data_completeness = non_zero_features / len(core_features)

            # Get team names
            home_team = game_data.get("home_team") or game_data.get(
                "game_info", {}
            ).get("home_team")
            away_team = game_data.get("away_team") or game_data.get(
                "game_info", {}
            ).get("away_team")

            # Calculate confidence based on probability margin and data quality
            prob_margin = abs(home_win_prob - 0.5)
            confidence = min(prob_margin * 2, 1.0)  # Scale up confidence but cap at 1.0

            return {
                "predicted_winner": home_team if home_win_prob > 0.5 else away_team,
                "win_probability": max(home_win_prob, 1 - home_win_prob),
                "confidence_score": confidence,
                "model_specific_factors": {
                    "home_win_probability": home_win_prob,
                    "data_completeness": data_completeness,
                },
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                "predicted_winner": None,
                "win_probability": 0.5,
                "confidence_score": 0.0,
            }

    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        try:
            games = test_data["games"]
            labels = test_data["labels"]

            # Extract features for all games
            features_list = []
            for game in games:
                features = self._extract_features(game)
                features_list.append(features)

            # Convert to DataFrame
            X = pd.DataFrame(features_list)

            # Fill missing values with stored means
            for col in X.columns:
                X[col] = X[col].fillna(self.feature_means.get(col, 0))

            # Scale features
            if hasattr(self.scaler, "mean_"):
                X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            # Calculate metrics
            accuracy = np.mean(predictions == labels)
            confidence_scores = np.max(probabilities, axis=1)
            avg_confidence = np.mean(confidence_scores)

            # Calculate feature importance
            feature_importance = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )

            return {
                "accuracy": float(accuracy),
                "avg_confidence": float(avg_confidence),
                "feature_importance": feature_importance,
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"accuracy": 0.0, "avg_confidence": 0.0, "feature_importance": {}}

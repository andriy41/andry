"""NFL Advanced Predictions Module."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLAdvancedPredictor:
    """Advanced NFL game prediction system using multiple models and features."""

    def __init__(self):
        """Initialize the advanced predictor."""
        self.models = {
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
        }

        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.performance_metrics = {}

    def prepare_features(self, game_data):
        """Prepare advanced features for prediction."""
        try:
            features = pd.DataFrame()

            # Team Performance Metrics
            features["off_dvoa_diff"] = (
                game_data["home_off_dvoa"] - game_data["away_off_dvoa"]
            )
            features["def_dvoa_diff"] = (
                game_data["home_def_dvoa"] - game_data["away_def_dvoa"]
            )
            features["st_dvoa_diff"] = (
                game_data["home_st_dvoa"] - game_data["away_st_dvoa"]
            )

            # QB Performance
            features["qb_rating_diff"] = (
                game_data["home_qb_rating"] - game_data["away_qb_rating"]
            )
            features["pass_yards_diff"] = (
                game_data["home_pass_yards"] - game_data["away_pass_yards"]
            )
            features["rush_yards_diff"] = (
                game_data["home_rush_yards"] - game_data["away_rush_yards"]
            )

            # Defensive Metrics
            features["sacks_diff"] = game_data["home_sacks"] - game_data["away_sacks"]
            features["turnovers_diff"] = (
                game_data["home_turnovers"] - game_data["away_turnovers"]
            )
            features["def_tds_diff"] = (
                game_data["home_def_tds"] - game_data["away_def_tds"]
            )

            # Situational Factors
            features["rest_days_diff"] = (
                game_data["home_rest_days"] - game_data["away_rest_days"]
            )
            features["travel_distance"] = game_data["away_travel_distance"]
            features["is_division_game"] = game_data["is_division_game"].astype(int)
            features["is_primetime"] = game_data["is_primetime"].astype(int)

            # Weather Impact
            if "temperature" in game_data.columns:
                features["temperature"] = game_data["temperature"]
                features["wind_speed"] = game_data["wind_speed"]
                features["is_precipitation"] = game_data["is_precipitation"].astype(int)

            # Recent Form
            features["home_form"] = self._calculate_form(
                game_data["home_recent_results"]
            )
            features["away_form"] = self._calculate_form(
                game_data["away_recent_results"]
            )

            # Injury Impact
            features["home_injury_impact"] = self._calculate_injury_impact(
                game_data["home_injuries"]
            )
            features["away_injury_impact"] = self._calculate_injury_impact(
                game_data["away_injuries"]
            )

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _calculate_form(self, recent_results):
        """Calculate team form based on recent results."""
        try:
            # Weight more recent games higher
            weights = [0.35, 0.25, 0.20, 0.12, 0.08]  # Last 5 games
            return sum(
                result * weight for result, weight in zip(recent_results, weights)
            )
        except Exception as e:
            logger.error(f"Error calculating form: {str(e)}")
            return 0.5

    def _calculate_injury_impact(self, injuries):
        """Calculate impact of injuries on team performance."""
        try:
            position_weights = {
                "QB": 0.30,
                "WR": 0.15,
                "RB": 0.15,
                "OL": 0.10,
                "DL": 0.10,
                "LB": 0.08,
                "DB": 0.07,
                "TE": 0.05,
            }

            impact = 0
            for injury in injuries:
                if injury["position"] in position_weights:
                    impact += position_weights[injury["position"]] * injury["severity"]

            return min(impact, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating injury impact: {str(e)}")
            return 0.0

    def train_models(self, X_train, y_train):
        """Train all prediction models."""
        try:
            X_scaled = self.scaler.fit_transform(X_train)

            for name, model in self.models.items():
                model.fit(X_scaled, y_train)
                self.feature_importance[name] = dict(
                    zip(X_train.columns, model.feature_importances_)
                )
                logger.info(f"Trained {name} model successfully")

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")

    def predict_game(self, game_data):
        """Generate prediction for a single game."""
        try:
            features = self.prepare_features(game_data)
            X_scaled = self.scaler.transform(features)

            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(X_scaled)[0]
                predictions[name] = {
                    "win_probability": pred_proba[1],
                    "confidence": max(pred_proba),
                }

            # Combine predictions with weights based on recent performance
            final_prediction = self._combine_predictions(predictions)

            # Adjust for special circumstances
            final_prediction = self._adjust_prediction(final_prediction, game_data)

            return final_prediction

        except Exception as e:
            logger.error(f"Error predicting game: {str(e)}")
            return {"win_probability": 0.5, "confidence": 0.0}

    def _combine_predictions(self, predictions):
        """Combine predictions from different models."""
        try:
            # Get model weights based on recent performance
            weights = self._get_model_weights()

            weighted_prob = 0
            total_weight = 0

            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 1.0)
                weighted_prob += pred["win_probability"] * weight
                total_weight += weight

            combined_prob = weighted_prob / total_weight if total_weight > 0 else 0.5

            # Calculate overall confidence
            confidence = (
                sum(
                    pred["confidence"] * weights.get(model_name, 1.0)
                    for model_name, pred in predictions.items()
                )
                / total_weight
            )

            return {"win_probability": combined_prob, "confidence": confidence}

        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return {"win_probability": 0.5, "confidence": 0.0}

    def _get_model_weights(self):
        """Get weights for each model based on recent performance."""
        try:
            if not self.performance_metrics:
                return {name: 1.0 for name in self.models.keys()}

            # Calculate weights based on recent accuracy
            weights = {}
            for name, metrics in self.performance_metrics.items():
                recent_accuracy = metrics.get("recent_accuracy", 0.5)
                weights[name] = 0.5 + recent_accuracy / 2  # Scale between 0.5 and 1.0

            return weights

        except Exception as e:
            logger.error(f"Error getting model weights: {str(e)}")
            return {name: 1.0 for name in self.models.keys()}

    def _adjust_prediction(self, prediction, game_data):
        """Adjust prediction based on special circumstances."""
        try:
            prob = prediction["win_probability"]
            confidence = prediction["confidence"]

            # Adjust for home field advantage
            if game_data.get("is_neutral_site", False):
                prob = (prob + 0.5) / 2  # Reduce home field impact
                confidence *= 0.9

            # Adjust for weather
            if game_data.get("extreme_weather", False):
                prob = (prob + 0.5) / 2  # Extreme weather makes game more unpredictable
                confidence *= 0.8

            # Adjust for key injuries
            if game_data.get("home_qb_out", False):
                prob *= 0.8
                confidence *= 0.9
            if game_data.get("away_qb_out", False):
                prob *= 1.2
                confidence *= 0.9

            # Adjust for playoff implications
            if game_data.get("playoff_implications", False):
                if game_data.get("home_needs_win", False):
                    prob *= 1.1
                if game_data.get("away_needs_win", False):
                    prob *= 0.9

            # Ensure probability stays between 0 and 1
            prob = max(min(prob, 1.0), 0.0)
            confidence = max(min(confidence, 1.0), 0.0)

            return {"win_probability": prob, "confidence": confidence}

        except Exception as e:
            logger.error(f"Error adjusting prediction: {str(e)}")
            return prediction

    def update_performance_metrics(self, predictions, actual_outcomes):
        """Update performance metrics for each model."""
        try:
            for name in self.models.keys():
                correct = sum(
                    1
                    for pred, actual in zip(predictions[name], actual_outcomes)
                    if (pred > 0.5) == actual
                )
                accuracy = correct / len(actual_outcomes)

                if name not in self.performance_metrics:
                    self.performance_metrics[name] = {}

                self.performance_metrics[name]["recent_accuracy"] = accuracy
                self.performance_metrics[name]["predictions_count"] = len(
                    actual_outcomes
                )

            logger.info("Updated performance metrics successfully")

        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

    def save_model(self, path="models/nfl_advanced_model.joblib"):
        """Save the model to disk."""
        try:
            model_data = {
                "models": self.models,
                "scaler": self.scaler,
                "feature_importance": self.feature_importance,
                "performance_metrics": self.performance_metrics,
            }

            joblib.dump(model_data, path)
            logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path="models/nfl_advanced_model.joblib"):
        """Load the model from disk."""
        try:
            model_data = joblib.load(path)

            self.models = model_data["models"]
            self.scaler = model_data["scaler"]
            self.feature_importance = model_data["feature_importance"]
            self.performance_metrics = model_data["performance_metrics"]

            logger.info(f"Model loaded successfully from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def get_feature_importance(self):
        """Get feature importance for all models."""
        try:
            combined_importance = {}
            for name, importance in self.feature_importance.items():
                for feature, value in importance.items():
                    if feature not in combined_importance:
                        combined_importance[feature] = 0
                    combined_importance[feature] += value

            # Average the importance values
            for feature in combined_importance:
                combined_importance[feature] /= len(self.feature_importance)

            return dict(
                sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
            )

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}

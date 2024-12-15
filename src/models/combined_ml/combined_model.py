from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ..advanced_system.advanced_model import AdvancedModel
from ..vedic_basic.vedic_model import VedicModel
from ..base_model import NFLPredictionModel
from typing import List


class CombinedModel(NFLPredictionModel):
    """Combined ML system integrating traditional stats with astrological factors"""

    def __init__(self):
        super().__init__()
        # Initialize sub-models
        self.advanced_model = AdvancedModel()
        self.vedic_model = VedicModel()

        # Initialize ensemble models
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
        )

        self.scaler = StandardScaler()
        self.trained = False

        # Model weights for ensemble
        self.model_weights = {"rf": 0.4, "gb": 0.4, "nn": 0.2}

    def train(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """Train all models using historical game data"""
        # Train sub-models first
        self.advanced_model.train(training_data)
        self.vedic_model.train(training_data)

        # Get predictions from sub-models for training data
        features_list = []
        labels = []
        for game in training_data["games"]:
            try:
                features = self._get_combined_features(game)
                features_list.append(features)
                labels.append(1 if game["winner"] == "home" else 0)
            except Exception as e:
                logger.warning(f"Error getting features for game: {str(e)}")
                continue

        if not features_list:
            raise ValueError("No valid games to train on")

        # Convert to numpy array and scale
        X = np.array(features_list)
        y = np.array(labels)

        # Fill NaN values with 0 before scaling
        X = np.nan_to_num(X, 0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train ensemble models
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        self.nn_model.fit(X_scaled, y)

        self.trained = True

        # Calculate training metrics
        rf_score = self.rf_model.score(X_scaled, y)
        gb_score = self.gb_model.score(X_scaled, y)
        nn_score = self.nn_model.score(X_scaled, y)

        # Get feature importance (only available for RF and GB)
        feature_importance = {}
        for name, importance in zip(
            self._get_feature_names(),
            self.rf_model.feature_importances_ * 0.5
            + self.gb_model.feature_importances_ * 0.5,
        ):
            feature_importance[name] = importance

        return {
            "rf_accuracy": rf_score,
            "gb_accuracy": gb_score,
            "nn_accuracy": nn_score,
            "weighted_accuracy": (
                rf_score * self.model_weights["rf"]
                + gb_score * self.model_weights["gb"]
                + nn_score * self.model_weights["nn"]
            ),
            "feature_importance": feature_importance,
        }

    def _get_combined_features(self, game_data: Dict[str, Any]) -> np.ndarray:
        """Get combined features from both sub-models"""
        try:
            # Get predictions from sub-models
            advanced_pred = self.advanced_model.predict(game_data)
            vedic_pred = self.vedic_model.predict(game_data)

            # Extract features from advanced model
            advanced_features = []
            advanced_contributions = advanced_pred.get(
                "model_specific_factors", {}
            ).get("feature_contributions", {})
            for feature in self.advanced_model.feature_columns:
                advanced_features.append(
                    float(advanced_contributions.get(feature, 0.0))
                )

            # Extract features from vedic model
            vedic_features = []
            vedic_contributions = vedic_pred.get("model_specific_factors", {}).get(
                "feature_contributions", {}
            )
            for feature in self.vedic_model.feature_names:
                vedic_features.append(float(vedic_contributions.get(feature, 0.0)))

            # Add model confidences and probabilities
            meta_features = [
                float(advanced_pred.get("confidence_score", 0.5)),
                float(vedic_pred.get("confidence_score", 0.5)),
                float(
                    advanced_pred.get("model_specific_factors", {}).get(
                        "home_win_probability", 0.5
                    )
                ),
                float(
                    vedic_pred.get("model_specific_factors", {}).get(
                        "home_win_probability", 0.5
                    )
                ),
            ]

            # Combine all features and ensure they are numpy arrays
            features = np.concatenate(
                [
                    np.array(advanced_features, dtype=np.float64),
                    np.array(vedic_features, dtype=np.float64),
                    np.array(meta_features, dtype=np.float64),
                ]
            )

            # Fill any remaining NaN values
            features = np.nan_to_num(features, 0.0)

            return features

        except Exception as e:
            logger.error(f"Error getting combined features: {str(e)}")
            # Return default features if extraction fails
            total_features = (
                len(self.advanced_model.feature_columns)
                + len(self.vedic_model.feature_names)
                + 4  # Number of meta features
            )
            return np.zeros(total_features, dtype=np.float64)

    def _get_feature_names(self) -> List[str]:
        """Get names of all features used in the combined model"""
        advanced_features = self.advanced_model.feature_columns
        vedic_features = self.vedic_model.feature_names
        meta_features = [
            "advanced_confidence",
            "vedic_confidence",
            "advanced_home_win_prob",
            "vedic_home_win_prob",
        ]
        return advanced_features + vedic_features + meta_features

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using ensemble of models"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Get predictions from sub-models
            advanced_pred = self.advanced_model.predict(game_data)
            vedic_pred = self.vedic_model.predict(game_data)

            # Get combined features
            features = self._get_combined_features(game_data)

            # Scale features
            X = np.array([features])
            X_scaled = self.scaler.transform(X)

            # Get predictions from each model
            rf_pred = self.rf_model.predict(X_scaled)[0]
            gb_pred = self.gb_model.predict(X_scaled)[0]
            nn_pred = self.nn_model.predict(X_scaled)[0]

            rf_prob = self.rf_model.predict_proba(X_scaled)[0]
            gb_prob = self.gb_model.predict_proba(X_scaled)[0]
            nn_prob = self.nn_model.predict_proba(X_scaled)[0]

            # Calculate weighted probability
            home_win_prob = (
                rf_prob[1] * self.model_weights["rf"]
                + gb_prob[1] * self.model_weights["gb"]
                + nn_prob[1] * self.model_weights["nn"]
            )

            # Calculate confidence based on:
            # 1. Model agreement
            # 2. Probability margin
            # 3. Sub-model confidences
            model_predictions = [rf_pred, gb_pred, nn_pred]
            agreement_ratio = sum(
                p == model_predictions[0] for p in model_predictions
            ) / len(model_predictions)
            prob_margin = abs(home_win_prob - 0.5) * 2

            confidence_factors = [
                agreement_ratio,
                prob_margin,
                advanced_pred.get("confidence_score", 0.5),
                vedic_pred.get("confidence_score", 0.5),
            ]
            confidence_score = float(np.mean(confidence_factors))

            # Get feature importance
            feature_names = self._get_feature_names()
            feature_importance = {}
            rf_importance = self.rf_model.feature_importances_
            gb_importance = self.gb_model.feature_importances_

            for i, name in enumerate(feature_names):
                feature_importance[name] = float(
                    rf_importance[i] * self.model_weights["rf"]
                    + gb_importance[i] * self.model_weights["gb"]
                ) / (self.model_weights["rf"] + self.model_weights["gb"])

            return {
                "home_win_probability": float(home_win_prob),
                "confidence_score": float(confidence_score),
                "model_specific_factors": {
                    "rf_prediction": bool(rf_pred),
                    "gb_prediction": bool(gb_pred),
                    "nn_prediction": bool(nn_pred),
                    "rf_probability": float(rf_prob[1]),
                    "gb_probability": float(gb_prob[1]),
                    "nn_probability": float(nn_prob[1]),
                    "agreement_ratio": float(agreement_ratio),
                    "probability_margin": float(prob_margin),
                    "advanced_model_prediction": advanced_pred,
                    "vedic_model_prediction": vedic_pred,
                    "feature_importance": feature_importance,
                },
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                "home_win_probability": 0.5,
                "confidence_score": 0.0,
                "model_specific_factors": {"error": str(e)},
            }

    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")

        correct = 0
        high_confidence_correct = 0
        high_confidence_total = 0
        total_games = len(test_data["games"])

        for i, game in enumerate(test_data["games"]):
            prediction = self.predict(game)
            actual_winner = game["winner"]

            if prediction["model_specific_factors"]["rf_prediction"] == actual_winner:
                correct += 1

            if prediction["confidence_score"] >= 0.7:  # High confidence threshold
                high_confidence_total += 1
                if (
                    prediction["model_specific_factors"]["rf_prediction"]
                    == actual_winner
                ):
                    high_confidence_correct += 1

        return {
            "overall_accuracy": correct / total_games if total_games > 0 else 0,
            "high_confidence_accuracy": high_confidence_correct / high_confidence_total
            if high_confidence_total > 0
            else 0,
            "high_confidence_rate": high_confidence_total / total_games
            if total_games > 0
            else 0,
        }

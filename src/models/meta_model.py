"""Meta-model for optimizing NFL predictions and identifying favorable situations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, mean_squared_error
import joblib

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for a single prediction."""

    actual_outcome: float
    predicted_prob: float
    model_confidence: float
    feature_values: Dict[str, float]
    risk_factors: List[str]
    was_correct: bool
    prediction_error: float
    betting_outcome: Optional[float] = None


class MetaModel:
    """Meta-model for optimizing predictions and identifying favorable situations."""

    def __init__(self):
        """Initialize the meta-model components."""
        # Model accuracy prediction
        self.accuracy_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.01, subsample=0.8
        )

        # Optimal feature weights
        self.feature_weight_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.01, subsample=0.8
        )

        # Confidence calibration
        self.confidence_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.01, subsample=0.8
        )

        # Value betting identification
        self.value_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.01, subsample=0.8
        )

        self.feature_scaler = StandardScaler()
        self.prediction_history: List[PredictionMetrics] = []
        self.confidence_thresholds: Dict[str, float] = {
            "accuracy": 0.6,
            "value": 0.1,  # Minimum edge required
        }

        # Feature importance tracking
        self.feature_importance: Dict[str, float] = {}
        self.optimal_weights: Dict[str, float] = {}

    def update(self, prediction_metrics: List[PredictionMetrics]):
        """Update meta-model with new prediction results."""
        self.prediction_history.extend(prediction_metrics)

        # Prepare training data
        X_accuracy, y_accuracy = self._prepare_accuracy_data()
        X_weights, y_weights = self._prepare_weight_data()
        X_confidence, y_confidence = self._prepare_confidence_data()
        X_value, y_value = self._prepare_value_data()

        # Train models if enough data
        if len(self.prediction_history) >= 50:
            # Train accuracy prediction model
            self.accuracy_model.fit(X_accuracy, y_accuracy)

            # Train feature weight optimization model
            self.feature_weight_model.fit(X_weights, y_weights)

            # Train confidence calibration model
            self.confidence_model.fit(X_confidence, y_confidence)

            # Train value betting model
            self.value_model.fit(X_value, y_value)

            # Update feature importance
            self._update_feature_importance()

            # Update optimal weights
            self._update_optimal_weights()

            # Update confidence thresholds
            self._update_confidence_thresholds()

    def predict_accuracy(self, features: Dict[str, float], confidence: float) -> float:
        """Predict likelihood of model being accurate."""
        X = self._prepare_meta_features(features, confidence)
        try:
            return self.accuracy_model.predict_proba(X)[0][1]
        except:
            return 0.5

    def get_optimal_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get optimal feature weights for current situation."""
        X = self._prepare_meta_features(
            features, 0
        )  # Confidence not needed for weights
        try:
            weights = self.feature_weight_model.predict(X)[0]
            return {name: weight for name, weight in zip(features.keys(), weights)}
        except:
            return {name: 1.0 for name in features.keys()}

    def calibrate_confidence(
        self, raw_confidence: float, features: Dict[str, float]
    ) -> float:
        """Calibrate raw confidence score."""
        X = self._prepare_meta_features(features, raw_confidence)
        try:
            return self.confidence_model.predict(X)[0]
        except:
            return raw_confidence

    def assess_betting_value(
        self, prediction: Dict[str, any], odds: float
    ) -> Dict[str, float]:
        """Assess if current situation has betting value."""
        features = prediction["feature_values"]
        confidence = prediction["confidence"]
        win_prob = prediction["win_probability"]

        X = self._prepare_meta_features(features, confidence)
        try:
            # Predict expected value
            implied_prob = 1 / odds
            edge = win_prob - implied_prob
            model_ev = self.value_model.predict(X)[0]

            return {
                "edge": edge,
                "model_ev": model_ev,
                "recommended_bet": edge > self.confidence_thresholds["value"]
                and model_ev > 0,
                "bet_size_multiplier": min(edge * 2, 1.0) if model_ev > 0 else 0,
            }
        except:
            return {
                "edge": 0,
                "model_ev": 0,
                "recommended_bet": False,
                "bet_size_multiplier": 0,
            }

    def _prepare_accuracy_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for accuracy prediction model."""
        features = []
        targets = []

        for metric in self.prediction_history:
            features.append(
                list(metric.feature_values.values()) + [metric.model_confidence]
            )
            targets.append(metric.was_correct)

        X = self.feature_scaler.fit_transform(features)
        y = np.array(targets)

        return X, y

    def _prepare_weight_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for feature weight optimization."""
        features = []
        weights = []

        for metric in self.prediction_history:
            features.append(list(metric.feature_values.values()))
            # Calculate optimal weights based on prediction error
            weight = 1 / (1 + metric.prediction_error)
            weights.append([weight] * len(metric.feature_values))

        X = self.feature_scaler.transform(features)
        y = np.array(weights)

        return X, y

    def _prepare_confidence_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for confidence calibration."""
        features = []
        true_confidences = []

        for metric in self.prediction_history:
            features.append(
                list(metric.feature_values.values()) + [metric.model_confidence]
            )
            # True confidence is inverse of prediction error
            true_confidence = 1 / (1 + metric.prediction_error)
            true_confidences.append(true_confidence)

        X = self.feature_scaler.transform(features)
        y = np.array(true_confidences)

        return X, y

    def _prepare_value_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for value betting model."""
        features = []
        values = []

        for metric in self.prediction_history:
            if metric.betting_outcome is not None:
                features.append(
                    list(metric.feature_values.values()) + [metric.model_confidence]
                )
                values.append(metric.betting_outcome)

        if not features:  # No betting data yet
            return np.array([]), np.array([])

        X = self.feature_scaler.transform(features)
        y = np.array(values)

        return X, y

    def _prepare_meta_features(
        self, features: Dict[str, float], confidence: float
    ) -> np.ndarray:
        """Prepare features for meta-model prediction."""
        X = np.array(list(features.values()) + [confidence]).reshape(1, -1)
        return self.feature_scaler.transform(X)

    def _update_feature_importance(self):
        """Update feature importance tracking."""
        try:
            importance = self.accuracy_model.feature_importances_
            feature_names = list(self.prediction_history[0].feature_values.keys()) + [
                "confidence"
            ]
            self.feature_importance = {
                name: imp for name, imp in zip(feature_names, importance)
            }
        except:
            pass

    def _update_optimal_weights(self):
        """Update optimal feature weights."""
        try:
            weights = self.feature_weight_model.feature_importances_
            feature_names = list(self.prediction_history[0].feature_values.keys())
            self.optimal_weights = {
                name: weight for name, weight in zip(feature_names, weights)
            }
        except:
            pass

    def _update_confidence_thresholds(self):
        """Dynamically update confidence thresholds."""
        if len(self.prediction_history) < 50:
            return

        # Calculate accuracy at different confidence levels
        confidence_bins = np.linspace(0, 1, 20)
        accuracies = []

        for threshold in confidence_bins:
            high_conf_preds = [
                m.was_correct
                for m in self.prediction_history
                if m.model_confidence >= threshold
            ]
            if high_conf_preds:
                accuracies.append(np.mean(high_conf_preds))
            else:
                accuracies.append(0)

        # Find threshold where accuracy significantly improves
        for i in range(len(confidence_bins)):
            if i > 0 and accuracies[i] > accuracies[i - 1] * 1.1:  # 10% improvement
                self.confidence_thresholds["accuracy"] = confidence_bins[i]
                break

        # Update value threshold based on betting outcomes
        if any(m.betting_outcome is not None for m in self.prediction_history):
            value_bins = np.linspace(0, 0.3, 30)  # Edge up to 30%
            values = []

            for threshold in value_bins:
                value_bets = [
                    m.betting_outcome
                    for m in self.prediction_history
                    if m.betting_outcome is not None
                    and m.predicted_prob - 0.5 >= threshold
                ]
                if value_bets:
                    values.append(np.mean(value_bets))
                else:
                    values.append(0)

            # Find threshold where value becomes positive
            for i in range(len(value_bins)):
                if values[i] > 0:
                    self.confidence_thresholds["value"] = value_bins[i]
                    break

    def save(self, path: str):
        """Save meta-model to disk."""
        joblib.dump(
            {
                "accuracy_model": self.accuracy_model,
                "feature_weight_model": self.feature_weight_model,
                "confidence_model": self.confidence_model,
                "value_model": self.value_model,
                "feature_scaler": self.feature_scaler,
                "confidence_thresholds": self.confidence_thresholds,
                "feature_importance": self.feature_importance,
                "optimal_weights": self.optimal_weights,
            },
            path,
        )

    def load(self, path: str):
        """Load meta-model from disk."""
        models = joblib.load(path)
        self.accuracy_model = models["accuracy_model"]
        self.feature_weight_model = models["feature_weight_model"]
        self.confidence_model = models["confidence_model"]
        self.value_model = models["value_model"]
        self.feature_scaler = models["feature_scaler"]
        self.confidence_thresholds = models["confidence_thresholds"]
        self.feature_importance = models["feature_importance"]
        self.optimal_weights = models["optimal_weights"]

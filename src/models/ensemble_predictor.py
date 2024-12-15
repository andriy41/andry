from typing import Dict, Any, List
import numpy as np
from .base_model import NFLPredictionModel
from .advanced_system.advanced_model import AdvancedSystemModel
from .vedic_basic.vedic_model import VedicBasicModel
from .combined_ml.combined_model import CombinedMLModel
from .sports_only.sports_model import SportsOnlyModel


class NFLEnsemblePredictor:
    """Ensemble predictor that combines all four prediction models"""

    def __init__(self):
        self.models: List[NFLPredictionModel] = [
            AdvancedSystemModel(),
            VedicBasicModel(),
            CombinedMLModel(),
            SportsOnlyModel(),
        ]

        # Define model weights
        self.weights = {
            "AdvancedSystemModel": 0.25,
            "VedicBasicModel": 0.25,
            "CombinedMLModel": 0.25,
            "SportsOnlyModel": 0.25,
        }

        # Confidence thresholds
        self.min_confidence = 0.80  # Minimum confidence for any model
        self.high_confidence = 0.85  # Threshold for high confidence predictions
        self.agreement_threshold = 0.75  # Minimum agreement ratio among models

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble prediction for a game"""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(game_data)
                predictions.append(
                    {"model_name": model.__class__.__name__, "prediction": pred}
                )
            except Exception as e:
                print(
                    f"Warning: {model.__class__.__name__} failed with error: {str(e)}"
                )

        if not predictions:
            raise ValueError("No models were able to make predictions")

        # Calculate weighted probabilities
        home_win_probs = []
        confidence_scores = []

        for pred in predictions:
            model_name = pred["model_name"]
            prediction = pred["prediction"]
            weight = self.weights[model_name]

            # Convert prediction to home win probability
            is_home_winner = prediction["predicted_winner"] == game_data["home_team"]
            home_prob = (
                prediction["win_probability"]
                if is_home_winner
                else 1 - prediction["win_probability"]
            )

            home_win_probs.append(home_prob * weight)
            confidence_scores.append(prediction["confidence_score"])

        # Calculate final home win probability
        final_home_prob = sum(home_win_probs) / sum(self.weights.values())

        # Determine if this is a high confidence prediction
        min_confidence = min(confidence_scores)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # Check model agreement
        home_win_votes = sum(1 for prob in home_win_probs if prob > 0.5)
        agreement_ratio = max(home_win_votes, len(predictions) - home_win_votes) / len(
            predictions
        )

        # Calculate final confidence score
        confidence_factors = [
            avg_confidence,
            agreement_ratio,
            abs(final_home_prob - 0.5) * 2,  # Probability margin
        ]
        final_confidence = min(confidence_factors)

        # Determine if this is a valid prediction
        is_high_confidence = (
            min_confidence >= self.min_confidence
            and agreement_ratio >= self.agreement_threshold
            and final_confidence >= self.high_confidence
        )

        return {
            "predicted_winner": game_data["home_team"]
            if final_home_prob > 0.5
            else game_data["away_team"],
            "win_probability": max(final_home_prob, 1 - final_home_prob),
            "confidence_score": final_confidence,
            "is_high_confidence": is_high_confidence,
            "model_specific_factors": {
                "model_predictions": [
                    {
                        "model": pred["model_name"],
                        "predicted_winner": pred["prediction"]["predicted_winner"],
                        "confidence": pred["prediction"]["confidence_score"],
                        "factors": pred["prediction"]["model_specific_factors"],
                    }
                    for pred in predictions
                ],
                "agreement_ratio": agreement_ratio,
                "min_model_confidence": min_confidence,
                "avg_model_confidence": avg_confidence,
            },
        }

    def train(self, training_data: Dict[str, Any]) -> None:
        """Train all models in the ensemble"""
        for model in self.models:
            try:
                model.train(training_data)
            except Exception as e:
                print(f"Warning: Failed to train {model.__class__.__name__}: {str(e)}")

    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        correct = 0
        high_confidence_correct = 0
        high_confidence_total = 0

        for game in test_data["games"]:
            prediction = self.predict(game)
            actual_winner = game["winner"]

            if prediction["predicted_winner"] == actual_winner:
                correct += 1

            if prediction["is_high_confidence"]:
                high_confidence_total += 1
                if prediction["predicted_winner"] == actual_winner:
                    high_confidence_correct += 1

        total_games = len(test_data["games"])

        return {
            "overall_accuracy": correct / total_games,
            "high_confidence_accuracy": high_confidence_correct / high_confidence_total
            if high_confidence_total > 0
            else 0,
            "high_confidence_rate": high_confidence_total / total_games,
            "model_evaluations": {
                model.__class__.__name__: model.evaluate(test_data)
                for model in self.models
            },
        }


class NFLEnsemblePredictor:
    def __init__(self):
        self.models = [
            AdvancedSystemModel(),
            VedicBasicModel(),
            CombinedMLModel(),
            SportsOnlyModel(),
        ]

        # Dynamic weights based on recent performance
        self.weights = {
            "AdvancedSystemModel": 0.30,  # Increased weight for advanced analytics
            "VedicBasicModel": 0.20,  # Adjusted based on historical performance
            "CombinedMLModel": 0.30,  # Increased for ML component
            "SportsOnlyModel": 0.20,  # Pure sports metrics
        }

        # Enhanced confidence thresholds
        self.confidence_thresholds = {
            "min": 0.80,
            "high": 0.85,
            "agreement": 0.75,
            "strong_prediction": 0.90,
        }

    def update_weights(self, recent_performance: Dict[str, float]):
        """Dynamically adjust model weights based on performance"""
        total_performance = sum(recent_performance.values())
        if total_performance > 0:
            for model_name, performance in recent_performance.items():
                self.weights[model_name] = (
                    performance / total_performance
                ) * 0.8 + self.weights[model_name] * 0.2

    def calculate_confidence(self, predictions, agreement_ratio):
        """Enhanced confidence calculation"""
        confidence_metrics = {
            "model_agreement": agreement_ratio * 0.4,
            "prediction_strength": np.mean(
                [p["prediction"]["confidence_score"] for p in predictions]
            )
            * 0.3,
            "consistency": (
                1 - np.std([p["prediction"]["win_probability"] for p in predictions])
            )
            * 0.3,
        }
        return sum(confidence_metrics.values())

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(game_data)
                predictions.append(
                    {"model_name": model.__class__.__name__, "prediction": pred}
                )
            except Exception as e:
                print(f"Model prediction error - {model.__class__.__name__}: {str(e)}")

        if not predictions:
            raise ValueError("No valid predictions available")

        # Calculate weighted ensemble prediction
        weighted_predictions = self._calculate_weighted_predictions(predictions)
        agreement_ratio = self._calculate_agreement_ratio(predictions)
        confidence = self.calculate_confidence(predictions, agreement_ratio)

        final_prediction = {
            "predicted_winner": weighted_predictions["winner"],
            "win_probability": weighted_predictions["probability"],
            "confidence_score": confidence,
            "is_high_confidence": confidence >= self.confidence_thresholds["high"],
            "recommendation": self._generate_recommendation(
                confidence, weighted_predictions
            ),
            "model_insights": self._generate_model_insights(predictions),
        }

        return final_prediction

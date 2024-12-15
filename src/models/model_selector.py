"""Dynamic model selection and weighting system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
import logging
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance metrics."""

    model_name: str
    accuracy: float
    confidence_calibration: float  # Brier score
    prediction_error: float  # MSE
    sample_count: int
    last_updated: datetime

    # Specific situation performance
    division_accuracy: float
    playoff_accuracy: float
    early_season_accuracy: float
    late_season_accuracy: float
    high_data_accuracy: float
    low_data_accuracy: float


@dataclass
class GameContext:
    """Game context for model selection."""

    is_division: bool
    is_playoff: bool
    week: int
    data_completeness: float  # 0-1 scale
    home_team: str
    away_team: str
    season: int
    similar_games: List[Tuple[str, str, float]]  # (home, away, similarity)


class ModelSelector:
    """Dynamic model selection and weighting system."""

    def __init__(self):
        """Initialize the model selector."""
        self.performance_history: Dict[str, ModelPerformance] = {}
        self.situation_weights = {
            "division": 1.2,
            "playoff": 1.5,
            "early_season": 0.8,
            "late_season": 1.1,
            "high_data": 1.2,
            "low_data": 0.8,
        }

        # Initialize weight adjustment model
        self.weight_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.01, subsample=0.8
        )

        # Performance tracking
        self.recent_performance: Dict[str, List[float]] = {}  # Last 10 games
        self.seasonal_performance: Dict[str, Dict[int, float]] = {}  # By season
        self.situation_performance: Dict[str, Dict[str, float]] = {}  # By situation

    def select_models(self, context: GameContext) -> Dict[str, float]:
        """Select and weight models based on context."""
        weights = {}

        for model_name, perf in self.performance_history.items():
            # Base weight from overall performance
            base_weight = perf.accuracy

            # Adjust for game situation
            situation_weight = self._calculate_situation_weight(perf, context)

            # Adjust for data availability
            data_weight = self._calculate_data_weight(perf, context)

            # Adjust for recent performance
            recent_weight = self._calculate_recent_weight(model_name, context)

            # Adjust for similar historical games
            historical_weight = self._calculate_historical_weight(model_name, context)

            # Combine weights
            weights[model_name] = (
                0.4 * base_weight
                + 0.2 * situation_weight
                + 0.15 * data_weight
                + 0.15 * recent_weight
                + 0.1 * historical_weight
            )

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Fallback to equal weights
            count = len(weights)
            weights = {k: 1.0 / count for k in weights.keys()}

        return weights

    def update_performance(
        self, model_name: str, prediction: float, actual: float, context: GameContext
    ):
        """Update model performance metrics."""
        error = abs(prediction - actual)
        correct = (prediction > 0.5) == (actual > 0.5)

        # Update recent performance
        if model_name not in self.recent_performance:
            self.recent_performance[model_name] = []
        self.recent_performance[model_name].append(correct)
        if len(self.recent_performance[model_name]) > 10:
            self.recent_performance[model_name].pop(0)

        # Update seasonal performance
        if model_name not in self.seasonal_performance:
            self.seasonal_performance[model_name] = {}
        if context.season not in self.seasonal_performance[model_name]:
            self.seasonal_performance[model_name][context.season] = []
        self.seasonal_performance[model_name][context.season].append(correct)

        # Update situation performance
        if model_name not in self.situation_performance:
            self.situation_performance[model_name] = {
                "division": [],
                "playoff": [],
                "early_season": [],
                "late_season": [],
                "high_data": [],
                "low_data": [],
            }

        situations = self._get_situations(context)
        for situation in situations:
            self.situation_performance[model_name][situation].append(correct)

        # Update overall performance metrics
        if model_name not in self.performance_history:
            self.performance_history[model_name] = ModelPerformance(
                model_name=model_name,
                accuracy=correct,
                confidence_calibration=error,
                prediction_error=error**2,
                sample_count=1,
                last_updated=datetime.now(),
                division_accuracy=correct if context.is_division else 0.0,
                playoff_accuracy=correct if context.is_playoff else 0.0,
                early_season_accuracy=correct if context.week <= 4 else 0.0,
                late_season_accuracy=correct if context.week >= 13 else 0.0,
                high_data_accuracy=correct if context.data_completeness >= 0.8 else 0.0,
                low_data_accuracy=correct if context.data_completeness < 0.8 else 0.0,
            )
        else:
            perf = self.performance_history[model_name]
            n = perf.sample_count

            # Update metrics using moving average
            perf.accuracy = (perf.accuracy * n + correct) / (n + 1)
            perf.confidence_calibration = (perf.confidence_calibration * n + error) / (
                n + 1
            )
            perf.prediction_error = (perf.prediction_error * n + error**2) / (n + 1)
            perf.sample_count += 1
            perf.last_updated = datetime.now()

            # Update situation-specific accuracies
            if context.is_division:
                perf.division_accuracy = (perf.division_accuracy * n + correct) / (
                    n + 1
                )
            if context.is_playoff:
                perf.playoff_accuracy = (perf.playoff_accuracy * n + correct) / (n + 1)
            if context.week <= 4:
                perf.early_season_accuracy = (
                    perf.early_season_accuracy * n + correct
                ) / (n + 1)
            if context.week >= 13:
                perf.late_season_accuracy = (
                    perf.late_season_accuracy * n + correct
                ) / (n + 1)
            if context.data_completeness >= 0.8:
                perf.high_data_accuracy = (perf.high_data_accuracy * n + correct) / (
                    n + 1
                )
            else:
                perf.low_data_accuracy = (perf.low_data_accuracy * n + correct) / (
                    n + 1
                )

    def _calculate_situation_weight(
        self, perf: ModelPerformance, context: GameContext
    ) -> float:
        """Calculate weight based on game situation."""
        if context.is_playoff:
            return perf.playoff_accuracy * self.situation_weights["playoff"]
        elif context.is_division:
            return perf.division_accuracy * self.situation_weights["division"]
        elif context.week <= 4:
            return perf.early_season_accuracy * self.situation_weights["early_season"]
        elif context.week >= 13:
            return perf.late_season_accuracy * self.situation_weights["late_season"]
        else:
            return perf.accuracy

    def _calculate_data_weight(
        self, perf: ModelPerformance, context: GameContext
    ) -> float:
        """Calculate weight based on data availability."""
        if context.data_completeness >= 0.8:
            return perf.high_data_accuracy * self.situation_weights["high_data"]
        else:
            return perf.low_data_accuracy * self.situation_weights["low_data"]

    def _calculate_recent_weight(self, model_name: str, context: GameContext) -> float:
        """Calculate weight based on recent performance."""
        if model_name not in self.recent_performance:
            return 0.5

        recent_games = self.recent_performance[model_name]
        if not recent_games:
            return 0.5

        # Exponentially weighted recent performance
        weights = np.exp(np.linspace(-1, 0, len(recent_games)))
        weights /= weights.sum()

        return np.average(recent_games, weights=weights)

    def _calculate_historical_weight(
        self, model_name: str, context: GameContext
    ) -> float:
        """Calculate weight based on similar historical games."""
        if not context.similar_games:
            return 0.5

        if model_name not in self.situation_performance:
            return 0.5

        # Get performance in similar games
        similar_performances = []
        for home, away, similarity in context.similar_games:
            # Find matching historical games
            for situation, results in self.situation_performance[model_name].items():
                if len(results) > 0:
                    similar_performances.append((np.mean(results), similarity))

        if not similar_performances:
            return 0.5

        # Weight by similarity
        total_similarity = sum(sim for _, sim in similar_performances)
        if total_similarity == 0:
            return 0.5

        weighted_perf = sum(perf * sim for perf, sim in similar_performances)
        return weighted_perf / total_similarity

    def _get_situations(self, context: GameContext) -> List[str]:
        """Get list of applicable situations for a game context."""
        situations = []
        if context.is_division:
            situations.append("division")
        if context.is_playoff:
            situations.append("playoff")
        if context.week <= 4:
            situations.append("early_season")
        if context.week >= 13:
            situations.append("late_season")
        if context.data_completeness >= 0.8:
            situations.append("high_data")
        else:
            situations.append("low_data")
        return situations

    def get_performance_stats(self) -> Dict[str, any]:
        """Get detailed performance statistics."""
        stats = {
            "overall_performance": {},
            "situation_performance": {},
            "recent_trends": {},
            "seasonal_trends": {},
        }

        # Overall performance
        for model_name, perf in self.performance_history.items():
            stats["overall_performance"][model_name] = {
                "accuracy": perf.accuracy,
                "confidence_calibration": perf.confidence_calibration,
                "prediction_error": perf.prediction_error,
                "sample_count": perf.sample_count,
            }

        # Situation performance
        for model_name, situations in self.situation_performance.items():
            stats["situation_performance"][model_name] = {
                situation: np.mean(results) if results else 0.0
                for situation, results in situations.items()
            }

        # Recent trends
        for model_name, recent in self.recent_performance.items():
            if recent:
                stats["recent_trends"][model_name] = np.mean(recent)
            else:
                stats["recent_trends"][model_name] = 0.0

        # Seasonal trends
        for model_name, seasons in self.seasonal_performance.items():
            stats["seasonal_trends"][model_name] = {
                season: np.mean(results) for season, results in seasons.items()
            }

        return stats

    def save(self, path: str):
        """Save model selector state."""
        state = {
            "performance_history": self.performance_history,
            "recent_performance": self.recent_performance,
            "seasonal_performance": self.seasonal_performance,
            "situation_performance": self.situation_performance,
            "situation_weights": self.situation_weights,
            "weight_model": self.weight_model,
        }
        joblib.dump(state, path)

    def load(self, path: str):
        """Load model selector state."""
        state = joblib.load(path)
        self.performance_history = state["performance_history"]
        self.recent_performance = state["recent_performance"]
        self.seasonal_performance = state["seasonal_performance"]
        self.situation_performance = state["situation_performance"]
        self.situation_weights = state["situation_weights"]
        self.weight_model = state["weight_model"]

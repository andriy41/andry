"""Multi-stage prediction pipeline with specialized models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
from .meta_model import MetaModel, PredictionMetrics
from .model_selector import ModelSelector, GameContext
from .specialized_models import EnsemblePredictor
import os

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    """Risk factor information."""

    name: str
    severity: float  # 0 to 1
    description: str


@dataclass
class StageResult:
    """Result from a prediction stage."""

    stage_name: str
    prediction: Dict[str, float]
    confidence: float
    risk_factors: List[RiskFactor]
    metadata: Dict[str, any]


class PredictionPipeline:
    """Four-stage prediction pipeline with specialized models."""

    def __init__(self, base_predictor):
        """Initialize the pipeline with predictors."""
        self.base_predictor = base_predictor
        self.ensemble_predictor = EnsemblePredictor()
        self.ensemble_predictor_path = os.path.join(
            os.path.dirname(__file__), "saved_models", "ensemble_predictor.joblib"
        )

        # Load ensemble predictor if exists
        if os.path.exists(self.ensemble_predictor_path):
            self.ensemble_predictor.load(self.ensemble_predictor_path)

        self.confidence_thresholds = {
            "stage1": 0.6,  # Basic outcome
            "stage2": 0.65,  # Confidence assessment
            "stage3": 0.7,  # Risk analysis
            "stage4": 0.75,  # Final validation
        }
        self.min_model_agreement = 0.7
        self.min_historical_similarity = 0.6
        self.max_risk_severity = 0.3
        self.meta_model = MetaModel()
        self.model_selector = ModelSelector()
        self.model_selector_path = os.path.join(
            os.path.dirname(__file__), "saved_models", "model_selector.joblib"
        )

        # Load model selector if exists
        if os.path.exists(self.model_selector_path):
            self.model_selector.load(self.model_selector_path)

    def predict(
        self, game_data: pd.DataFrame, odds: Optional[float] = None
    ) -> Optional[Dict[str, any]]:
        """Run the full prediction pipeline with meta-model optimization."""
        try:
            # Stage 1: Basic outcome prediction
            stage1_result = self._run_stage1(game_data)
            if not self._check_stage_confidence(stage1_result):
                logger.info(
                    f"Stage 1 confidence too low: {stage1_result.confidence:.3f}"
                )
                return self._create_rejection_response(stage1_result)

            # Get meta-model predictions
            features = self._extract_features(game_data, stage1_result)
            accuracy_prob = self.meta_model.predict_accuracy(
                features, stage1_result.confidence
            )
            optimal_weights = self.meta_model.get_optimal_weights(features)

            # Adjust prediction based on optimal weights
            adjusted_pred = self._adjust_prediction(
                stage1_result.prediction, optimal_weights
            )
            stage1_result.prediction.update(adjusted_pred)

            # Stage 2: Confidence assessment with meta-model calibration
            stage2_result = self._run_stage2(game_data, stage1_result)
            calibrated_confidence = self.meta_model.calibrate_confidence(
                stage2_result.confidence, features
            )
            stage2_result.confidence = calibrated_confidence

            if not self._check_stage_confidence(stage2_result):
                logger.info(
                    f"Stage 2 confidence too low: {stage2_result.confidence:.3f}"
                )
                return self._create_rejection_response(stage2_result)

            # Stage 3: Risk factor analysis
            stage3_result = self._run_stage3(game_data, stage1_result, stage2_result)
            if not self._check_stage_confidence(stage3_result):
                logger.info(
                    f"Stage 3 confidence too low: {stage3_result.confidence:.3f}"
                )
                return self._create_rejection_response(stage3_result)

            # Stage 4: Final validation
            stage4_result = self._run_stage4(
                game_data, stage1_result, stage2_result, stage3_result
            )
            if not self._check_stage_confidence(stage4_result):
                logger.info(
                    f"Stage 4 confidence too low: {stage4_result.confidence:.3f}"
                )
                return self._create_rejection_response(stage4_result)

            # Get betting value assessment if odds provided
            betting_value = None
            if odds is not None:
                betting_value = self.meta_model.assess_betting_value(
                    stage4_result.prediction, odds
                )

            # Create final response
            response = self._create_final_response(
                stage1_result, stage2_result, stage3_result, stage4_result
            )

            # Add meta-model insights
            response.update(
                {
                    "meta_model_insights": {
                        "predicted_accuracy": accuracy_prob,
                        "optimal_weights": optimal_weights,
                        "calibrated_confidence": calibrated_confidence,
                        "betting_value": betting_value,
                    }
                }
            )

            return response

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            return None

    def update_meta_model(
        self,
        game_result: Dict[str, any],
        prediction: Dict[str, any],
        betting_outcome: Optional[float] = None,
    ):
        """Update meta-model with game results."""
        try:
            actual_outcome = (
                1 if game_result["home_score"] > game_result["away_score"] else 0
            )
            predicted_prob = prediction["win_probability"]
            model_confidence = prediction["confidence"]
            feature_values = prediction["meta_model_insights"]["optimal_weights"]
            risk_factors = [r["name"] for r in prediction["risk_factors"]]
            was_correct = (actual_outcome == 1 and predicted_prob > 0.5) or (
                actual_outcome == 0 and predicted_prob < 0.5
            )
            prediction_error = abs(actual_outcome - predicted_prob)

            metrics = PredictionMetrics(
                actual_outcome=actual_outcome,
                predicted_prob=predicted_prob,
                model_confidence=model_confidence,
                feature_values=feature_values,
                risk_factors=risk_factors,
                was_correct=was_correct,
                prediction_error=prediction_error,
                betting_outcome=betting_outcome,
            )

            self.meta_model.update([metrics])

        except Exception as e:
            logger.error(f"Error updating meta-model: {str(e)}")

    def _run_stage1(self, game_data: pd.DataFrame) -> StageResult:
        """Stage 1: Basic outcome prediction with specialized models."""
        try:
            # Get predictions from specialized models
            ensemble_pred = self.ensemble_predictor.predict(game_data)

            # Get base model predictions with dynamic weights
            base_predictions = self.base_predictor.predict(game_data)

            # Create game context for model selection
            context = self._create_game_context(game_data)

            # Get model weights from selector
            model_weights = self.model_selector.select_models(context)

            # Combine predictions
            if ensemble_pred["is_high_confidence"]:
                # Use ensemble prediction with higher weight for high confidence picks
                win_prob = (
                    0.7 * ensemble_pred["prediction"]
                    + 0.3 * base_predictions["win_prob"]
                )
                confidence = (
                    0.7 * ensemble_pred["confidence"]
                    + 0.3 * base_predictions["confidence"]
                )
            else:
                # Use weighted combination
                win_prob = (
                    0.4 * ensemble_pred["prediction"]
                    + 0.6 * base_predictions["win_prob"]
                )
                confidence = (
                    0.4 * ensemble_pred["confidence"]
                    + 0.6 * base_predictions["confidence"]
                )

            # Adjust spread and total based on model agreement
            base_spread = base_predictions["spread"]
            spread_adjustment = (win_prob - 0.5) * 7  # Convert probability to points
            adjusted_spread = (
                base_spread + spread_adjustment * ensemble_pred["agreement_factor"]
            )

            prediction = {
                "win_prob": win_prob,
                "spread": adjusted_spread,
                "total_points": base_predictions["total_points"],
                "model_predictions": {
                    **base_predictions["model_predictions"],
                    **ensemble_pred["model_predictions"],
                },
                "model_weights": {
                    **model_weights,
                    "ensemble": 0.7 if ensemble_pred["is_high_confidence"] else 0.4,
                },
                "specialized_insights": {
                    "ensemble_confidence": ensemble_pred["confidence"],
                    "is_high_confidence": ensemble_pred["is_high_confidence"],
                    "model_confidences": ensemble_pred["model_confidences"],
                    "agreement_factor": ensemble_pred["agreement_factor"],
                },
                "context": context,
            }

            return StageResult(
                stage_name="Basic Prediction",
                prediction=prediction,
                confidence=confidence,
                risk_factors=[],
                metadata={"context": context},
            )

        except Exception as e:
            logger.error(f"Error in stage 1: {str(e)}")
            raise

    def _run_stage2(
        self, game_data: pd.DataFrame, stage1_result: StageResult
    ) -> StageResult:
        """Stage 2: Confidence assessment."""
        base_pred = stage1_result.prediction

        # Check temporal patterns
        temporal_features = base_pred.get("temporal_features", {})
        historical_similarity = temporal_features.get("historical_similarity", 0)
        recent_form_diff = abs(
            temporal_features.get("home_recent_form", 0.5)
            - temporal_features.get("away_recent_form", 0.5)
        )

        # Check statistical features
        stats_features = base_pred.get("simulation_results", {})
        sim_uncertainty = stats_features.get("win_prob_std", 0.5)

        # Calculate overall confidence
        confidence = np.mean(
            [
                stage1_result.confidence,
                historical_similarity,
                1 - sim_uncertainty,
                recent_form_diff,
            ]
        )

        # Identify risk factors
        risk_factors = []
        if historical_similarity < self.min_historical_similarity:
            risk_factors.append(
                RiskFactor(
                    name="Low Historical Similarity",
                    severity=1 - historical_similarity,
                    description="Few similar historical games found",
                )
            )
        if sim_uncertainty > 0.3:
            risk_factors.append(
                RiskFactor(
                    name="High Simulation Uncertainty",
                    severity=sim_uncertainty,
                    description="High variance in simulation results",
                )
            )

        return StageResult(
            stage_name="Confidence Assessment",
            prediction=base_pred,
            confidence=confidence,
            risk_factors=risk_factors,
            metadata={
                "historical_similarity": historical_similarity,
                "simulation_uncertainty": sim_uncertainty,
                "recent_form_difference": recent_form_diff,
            },
        )

    def _run_stage3(
        self,
        game_data: pd.DataFrame,
        stage1_result: StageResult,
        stage2_result: StageResult,
    ) -> StageResult:
        """Stage 3: Risk factor analysis."""
        # Collect all risk factors
        all_risks = stage1_result.risk_factors + stage2_result.risk_factors

        # Check game-specific risks
        game_risks = self._analyze_game_risks(game_data)
        all_risks.extend(game_risks)

        # Calculate risk-adjusted confidence
        base_confidence = (stage1_result.confidence + stage2_result.confidence) / 2
        risk_penalty = (
            sum(risk.severity for risk in all_risks) / len(all_risks)
            if all_risks
            else 0
        )
        confidence = base_confidence * (1 - risk_penalty)

        return StageResult(
            stage_name="Risk Analysis",
            prediction=stage1_result.prediction,
            confidence=confidence,
            risk_factors=all_risks,
            metadata={"risk_penalty": risk_penalty, "total_risks": len(all_risks)},
        )

    def _run_stage4(
        self,
        game_data: pd.DataFrame,
        stage1_result: StageResult,
        stage2_result: StageResult,
        stage3_result: StageResult,
    ) -> StageResult:
        """Stage 4: Final validation."""
        # Check if all previous stages are confident enough
        stage_confidences = [
            stage1_result.confidence,
            stage2_result.confidence,
            stage3_result.confidence,
        ]

        # Check if any stage has too many risks
        total_risks = len(stage3_result.risk_factors)
        high_severity_risks = sum(
            1
            for risk in stage3_result.risk_factors
            if risk.severity > self.max_risk_severity
        )

        # Calculate final confidence
        base_confidence = np.mean(stage_confidences)
        risk_penalty = (total_risks * 0.1) + (high_severity_risks * 0.2)
        confidence = max(0, base_confidence - risk_penalty)

        # Add any final risk factors
        risk_factors = stage3_result.risk_factors.copy()
        if high_severity_risks > 0:
            risk_factors.append(
                RiskFactor(
                    name="Multiple High Severity Risks",
                    severity=min(1.0, high_severity_risks * 0.3),
                    description=f"Found {high_severity_risks} high severity risk factors",
                )
            )

        return StageResult(
            stage_name="Final Validation",
            prediction=stage1_result.prediction,
            confidence=confidence,
            risk_factors=risk_factors,
            metadata={
                "stage_confidences": stage_confidences,
                "total_risks": total_risks,
                "high_severity_risks": high_severity_risks,
            },
        )

    def _analyze_game_risks(self, game_data: pd.DataFrame) -> List[RiskFactor]:
        """Analyze game-specific risk factors."""
        risks = []

        # Check if it's early in the season
        if game_data["week"].iloc[0] <= 4:
            risks.append(
                RiskFactor(
                    name="Early Season Game",
                    severity=0.3,
                    description="Limited data available for current season",
                )
            )

        # Check for key player availability
        home_starters = game_data["home_starters_available"].iloc[0]
        away_starters = game_data["away_starters_available"].iloc[0]
        if home_starters < 0.9 or away_starters < 0.9:
            risks.append(
                RiskFactor(
                    name="Key Player Unavailable",
                    severity=max(1 - home_starters, 1 - away_starters),
                    description="One or more key players may be missing",
                )
            )

        # Check for extreme weather (if available)
        if "weather_severity" in game_data.columns:
            weather_severity = game_data["weather_severity"].iloc[0]
            if weather_severity > 0.5:
                risks.append(
                    RiskFactor(
                        name="Severe Weather",
                        severity=weather_severity,
                        description="Weather conditions may impact game",
                    )
                )

        return risks

    def _check_stage_confidence(self, result: StageResult) -> bool:
        """Check if stage confidence meets threshold."""
        threshold = self.confidence_thresholds.get(f"stage{result.stage_name[0]}", 0.6)
        return result.confidence >= threshold

    def _create_rejection_response(self, failed_stage: StageResult) -> Dict[str, any]:
        """Create response for rejected prediction."""
        return {
            "prediction_made": False,
            "stage_failed": failed_stage.stage_name,
            "confidence": failed_stage.confidence,
            "risk_factors": [
                {"name": r.name, "severity": r.severity, "description": r.description}
                for r in failed_stage.risk_factors
            ],
            "metadata": failed_stage.metadata,
        }

    def _create_final_response(self, *stages: StageResult) -> Dict[str, any]:
        """Create response for successful prediction."""
        final_stage = stages[-1]
        return {
            "prediction_made": True,
            "win_probability": final_stage.prediction["win_prob"],
            "spread": final_stage.prediction["spread"],
            "total_points": final_stage.prediction["total_points"],
            "confidence": final_stage.confidence,
            "risk_factors": [
                {"name": r.name, "severity": r.severity, "description": r.description}
                for r in final_stage.risk_factors
            ],
            "stage_results": [
                {
                    "stage_name": s.stage_name,
                    "confidence": s.confidence,
                    "metadata": s.metadata,
                }
                for s in stages
            ],
            "model_predictions": final_stage.prediction["model_predictions"],
            "model_weights": final_stage.prediction["model_weights"],
        }

    def _extract_features(
        self, game_data: pd.DataFrame, stage_result: StageResult
    ) -> Dict[str, float]:
        """Extract features for meta-model."""
        features = {}

        # Basic game features
        features["week"] = game_data["week"].iloc[0] / 17  # Normalize
        features["is_division"] = float(game_data["is_division_game"].iloc[0])
        features["is_conference"] = float(game_data["is_conference_game"].iloc[0])

        # Model agreement features
        predictions = stage_result.prediction["model_predictions"]
        features["model_agreement"] = 1 - np.std(list(predictions.values()))
        features["prediction_strength"] = (
            abs(stage_result.prediction["win_prob"] - 0.5) * 2
        )

        # Statistical features
        features["elo_diff"] = stage_result.prediction.get(
            "home_elo", 1500
        ) - stage_result.prediction.get("away_elo", 1500)
        features["power_ranking_diff"] = stage_result.prediction.get(
            "home_power_ranking", 0.5
        ) - stage_result.prediction.get("away_power_ranking", 0.5)

        # Risk features
        features["risk_count"] = len(stage_result.risk_factors)
        features["max_risk_severity"] = (
            max(risk.severity for risk in stage_result.risk_factors)
            if stage_result.risk_factors
            else 0
        )

        return features

    def _adjust_prediction(
        self, prediction: Dict[str, float], weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust prediction based on optimal feature weights."""
        # Adjust win probability
        base_prob = prediction["win_prob"]
        weighted_features = sum(
            weights.get(k, 1.0) * v
            for k, v in prediction.items()
            if k in weights and isinstance(v, (int, float))
        )

        # Combine base probability with weighted features
        adjusted_prob = 0.7 * base_prob + 0.3 * (0.5 + weighted_features / len(weights))

        # Adjust spread and total based on probability change
        prob_change = adjusted_prob - base_prob
        spread_adjustment = prob_change * 7  # Rough conversion of probability to points

        return {
            "win_prob": adjusted_prob,
            "spread": prediction["spread"] + spread_adjustment,
            "total_points": prediction["total_points"],  # Keep total unchanged
        }

    def _create_game_context(self, game_data: pd.DataFrame) -> GameContext:
        """Create game context for model selection."""
        # Extract game information
        is_division = game_data["is_division_game"].iloc[0]
        is_playoff = game_data.get("is_playoff", False).iloc[0]
        week = game_data["week"].iloc[0]
        home_team = game_data["home_team"].iloc[0]
        away_team = game_data["away_team"].iloc[0]
        season = game_data["season"].iloc[0]

        # Calculate data completeness
        required_columns = [
            "home_wins",
            "home_losses",
            "away_wins",
            "away_losses",
            "home_points_for",
            "home_points_against",
            "away_points_for",
            "away_points_against",
        ]
        data_completeness = sum(
            1 for col in required_columns if col in game_data.columns
        ) / len(required_columns)

        # Get similar historical games from temporal analyzer
        similar_games = self.base_predictor.temporal_analyzer.get_similar_games(
            self.base_predictor._create_game_context(game_data)
        )
        similar_games = [(g.home_team, g.away_team, s) for g, s in similar_games[:5]]

        return GameContext(
            is_division=is_division,
            is_playoff=is_playoff,
            week=week,
            data_completeness=data_completeness,
            home_team=home_team,
            away_team=away_team,
            season=season,
            similar_games=similar_games,
        )

    def update_model_selector(
        self, game_result: Dict[str, any], prediction: Dict[str, any]
    ):
        """Update model selector with game results."""
        try:
            actual = 1 if game_result["home_score"] > game_result["away_score"] else 0

            # Update each model's performance
            for model_name, model_pred in prediction["model_predictions"].items():
                self.model_selector.update_performance(
                    model_name=model_name,
                    prediction=model_pred,
                    actual=actual,
                    context=prediction["context"],
                )

            # Save updated model selector
            os.makedirs(os.path.dirname(self.model_selector_path), exist_ok=True)
            self.model_selector.save(self.model_selector_path)

        except Exception as e:
            logger.error(f"Error updating model selector: {str(e)}")

    def get_model_performance_stats(self) -> Dict[str, any]:
        """Get detailed model performance statistics."""
        return self.model_selector.get_performance_stats()

    def save_meta_model(self, path: str):
        """Save meta-model to disk."""
        self.meta_model.save(path)

    def load_meta_model(self, path: str):
        """Load meta-model from disk."""
        self.meta_model.load(path)

    def update_ensemble_predictor(
        self, game_result: Dict[str, any], prediction: Dict[str, any]
    ):
        """Update ensemble predictor with game results."""
        try:
            # Save updated ensemble predictor
            os.makedirs(os.path.dirname(self.ensemble_predictor_path), exist_ok=True)
            self.ensemble_predictor.save(self.ensemble_predictor_path)

        except Exception as e:
            logger.error(f"Error updating ensemble predictor: {str(e)}")

    def get_ensemble_stats(self) -> Dict[str, any]:
        """Get statistics about ensemble predictor performance."""
        try:
            specialized_insights = prediction.get("specialized_insights", {})
            return {
                "ensemble_confidence": specialized_insights.get("ensemble_confidence"),
                "high_confidence_rate": specialized_insights.get("is_high_confidence"),
                "model_confidences": specialized_insights.get("model_confidences"),
                "agreement_factor": specialized_insights.get("agreement_factor"),
            }
        except Exception as e:
            logger.error(f"Error getting ensemble stats: {str(e)}")
            return {}

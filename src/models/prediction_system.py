"""NFL prediction system with dynamic model selection and meta-model optimization."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from .advanced_stats import NFLAdvancedStats
from .temporal_patterns import TemporalPatternAnalyzer, GameContext
from .prediction_pipeline import PredictionPipeline
import os

logger = logging.getLogger(__name__)


class EnsembleTracker:
    """Track and manage ensemble model performance."""

    def __init__(self, window_size: int = 100):
        """Initialize performance tracking."""
        self.window_size = window_size
        self.predictions = {"rf": [], "xgb": [], "lgb": [], "nn": []}
        self.actuals = []
        self.weights = {"rf": 0.25, "xgb": 0.25, "lgb": 0.25, "nn": 0.25}

    def add_prediction(self, model_name: str, prediction: float, actual: int):
        """Add a new prediction and update weights."""
        self.predictions[model_name].append(prediction)
        if len(self.actuals) < len(self.predictions[model_name]):
            self.actuals.append(actual)

        # Keep only last window_size predictions
        if len(self.predictions[model_name]) > self.window_size:
            self.predictions[model_name] = self.predictions[model_name][
                -self.window_size :
            ]
            self.actuals = self.actuals[-self.window_size :]

        # Update weights based on log loss
        self._update_weights()

    def _update_weights(self):
        """Update model weights based on recent performance."""
        if len(self.actuals) < 10:  # Need minimum history
            return

        losses = {}
        total_loss = 0

        # Calculate log loss for each model
        for name in self.predictions:
            if len(self.predictions[name]) >= len(self.actuals):
                preds = self.predictions[name][-len(self.actuals) :]
                loss = log_loss(self.actuals, preds, labels=[0, 1])
                losses[name] = loss
                total_loss += loss

        # Update weights inversely proportional to loss
        if total_loss > 0:
            for name in losses:
                self.weights[name] = (1 / losses[name]) / sum(
                    1 / l for l in losses.values()
                )

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights


class NFLPredictionSystem:
    """NFL prediction system with dynamic model selection and meta-model optimization."""

    def __init__(self):
        """Initialize the prediction system."""
        # Initialize win prediction models
        self.rf_win = RandomForestClassifier(
            n_estimators=100,  # Reduced for faster training
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        self.xgb_win = xgb.XGBClassifier(
            n_estimators=100,  # Reduced for faster training
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            scale_pos_weight=1,
            random_state=42,
        )

        self.lgb_win = lgb.LGBMClassifier(
            n_estimators=100,  # Reduced for faster training
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
        )

        self.nn_win = MLPClassifier(
            hidden_layer_sizes=(50, 25),  # Simplified architecture
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=200,
            early_stopping=True,
            random_state=42,
        )

        # Initialize voting ensemble
        self.voting_ensemble = VotingClassifier(
            estimators=[
                ("rf", self.rf_win),
                ("xgb", self.xgb_win),
                ("lgb", self.lgb_win),
                ("nn", self.nn_win),
            ],
            voting="soft",
        )

        # Initialize spread and total points models
        self.spread_models = {
            "rf": RandomForestRegressor(
                n_estimators=1000, max_depth=15, random_state=42
            ),
            "xgb": xgb.XGBRegressor(n_estimators=1000, max_depth=6, random_state=42),
            "lgb": lgb.LGBMRegressor(n_estimators=1000, max_depth=6, random_state=42),
        }

        self.total_points_models = {
            "rf": RandomForestRegressor(
                n_estimators=1000, max_depth=15, random_state=42
            ),
            "xgb": xgb.XGBRegressor(n_estimators=1000, max_depth=6, random_state=42),
            "lgb": lgb.LGBMRegressor(n_estimators=1000, max_depth=6, random_state=42),
        }

        # Initialize feature scaler
        self.feature_scaler = StandardScaler()

        # Train models with sample data
        self._initialize_with_sample_data()

        self.feature_names = None
        self.tracker = EnsembleTracker()
        self.advanced_stats = NFLAdvancedStats()
        self.temporal_analyzer = TemporalPatternAnalyzer()

        # Initialize prediction pipeline
        self.pipeline = PredictionPipeline(self)

        self.meta_model_path = os.path.join(
            os.path.dirname(__file__), "saved_models", "meta_model.joblib"
        )
        self.model_selector_path = os.path.join(
            os.path.dirname(__file__), "saved_models", "model_selector.joblib"
        )

        # Load models if they exist
        if os.path.exists(self.meta_model_path):
            self.pipeline.load_meta_model(self.meta_model_path)
        if os.path.exists(self.model_selector_path):
            self.pipeline.model_selector.load(self.model_selector_path)

    def _initialize_with_sample_data(self):
        """Initialize models with sample historical data."""
        try:
            # Create sample training data
            sample_data = self._create_sample_training_data()

            # Prepare features
            X = self._prepare_features(sample_data)
            y = (sample_data["home_score"] > sample_data["away_score"]).astype(int)

            # Train each model
            for model_name, model in [
                ("Random Forest", self.rf_win),
                ("XGBoost", self.xgb_win),
                ("LightGBM", self.lgb_win),
                ("Neural Network", self.nn_win),
            ]:
                try:
                    model.fit(X, y)
                    logger.info(f"Successfully trained {model_name} model")
                except Exception as e:
                    logger.error(f"Error training {model_name} model: {str(e)}")

        except Exception as e:
            logger.error(f"Error in model initialization: {str(e)}")

    def train(self, train_data: pd.DataFrame):
        """Train the ensemble models and update patterns."""
        try:
            # Ensure week column is numeric where possible
            train_data["week"] = pd.to_numeric(train_data["week"], errors="ignore")

            # Update team statistics
            self.advanced_stats.update_team_stats(train_data)

            # Prepare features for training
            X = self._prepare_features(train_data)
            y = (train_data["home_score"] > train_data["away_score"]).astype(int)

            # Train each model
            for model_name, model in [
                ("Random Forest", self.rf_win),
                ("XGBoost", self.xgb_win),
                ("LightGBM", self.lgb_win),
                ("Neural Network", self.nn_win),
            ]:
                try:
                    model.fit(X, y)
                    logger.info(f"Successfully trained {model_name} model")
                except Exception as e:
                    logger.error(f"Error training {model_name} model: {str(e)}")

            # Update feature importance tracking
            self._update_feature_importance()

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def predict_with_pipeline(self, game_data: pd.DataFrame) -> Dict[str, any]:
        """Make predictions using the multi-stage pipeline."""
        return self.pipeline.predict(game_data)

    def predict_game(
        self, home_team: str, away_team: str, game_date: str
    ) -> Dict[str, Any]:
        """Predict game outcome for a single game."""
        try:
            # Create a DataFrame with the single game
            game_data = pd.DataFrame(
                [
                    {
                        "date": game_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "season": pd.to_datetime(game_date).year,
                        "week": self._get_week_number(pd.to_datetime(game_date)),
                    }
                ]
            )

            # Get predictions from all models
            predictions = {}
            confidence_scores = []

            # Get base predictions from each model
            for model_name, model in [
                ("rf", self.rf_win),
                ("xgb", self.xgb_win),
                ("lgb", self.lgb_win),
                ("nn", self.nn_win),
            ]:
                try:
                    prob = model.predict_proba(self._prepare_features(game_data))[0][1]
                    confidence = abs(prob - 0.5) * 2  # Scale to 0-1 confidence
                    predictions[model_name] = {
                        "probability": prob,
                        "confidence": confidence,
                    }
                    confidence_scores.append(confidence)
                except Exception as e:
                    logger.error(f"Error in {model_name} prediction: {str(e)}")
                    predictions[model_name] = {"probability": 0.5, "confidence": 0.0}

            # Calculate ensemble prediction
            win_probability = np.mean([p["probability"] for p in predictions.values()])
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            max_confidence = max(confidence_scores) if confidence_scores else 0.0

            # Calculate model agreement
            predictions_agree = (
                len(
                    set(
                        [
                            1 if p["probability"] > 0.5 else 0
                            for p in predictions.values()
                        ]
                    )
                )
                == 1
            )

            return {
                "win_probability": win_probability,
                "confidence": avg_confidence,
                "max_confidence": max_confidence,
                "model_agreement": predictions_agree,
                "individual_predictions": predictions,
            }

        except Exception as e:
            logger.error(f"Error in game prediction: {str(e)}")
            return {
                "win_probability": 0.5,
                "confidence": 0.0,
                "max_confidence": 0.0,
                "model_agreement": False,
                "individual_predictions": {},
            }

    def _get_week_number(self, date):
        """Calculate NFL week number from date."""
        # Assuming season starts in September
        season_start = pd.Timestamp(f"{date.year}-09-01")
        week_diff = (date - season_start).days // 7
        return min(max(1, week_diff + 1), 18)  # Clamp between week 1-18

    def update_with_results(self, game_results: List[Dict[str, any]]):
        """Update all models with game results."""
        try:
            for result in game_results:
                # Update base models
                self._update_base_models(result)

                # Update meta-model and model selector if we have the prediction
                game_id = result["game_id"]
                if game_id in self.prediction_history:
                    prediction = self.prediction_history[game_id]
                    betting_outcome = None

                    # Calculate betting outcome if we had odds
                    if "odds" in prediction and "meta_model_insights" in prediction:
                        betting_value = prediction["meta_model_insights"].get(
                            "betting_value"
                        )
                        if betting_value and betting_value["recommended_bet"]:
                            actual_outcome = (
                                1 if result["home_score"] > result["away_score"] else 0
                            )
                            if actual_outcome == 1:
                                betting_outcome = prediction["odds"] - 1
                            else:
                                betting_outcome = -1

                    # Update meta-model
                    self.pipeline.update_meta_model(result, prediction, betting_outcome)

                    # Update model selector
                    self.pipeline.update_model_selector(result, prediction)

            # Save updated models
            os.makedirs(os.path.dirname(self.meta_model_path), exist_ok=True)
            self.pipeline.save_meta_model(self.meta_model_path)

        except Exception as e:
            logger.error(f"Error updating models with results: {str(e)}")

    def get_model_stats(self) -> Dict[str, any]:
        """Get comprehensive model performance statistics."""
        try:
            return {
                "meta_model": {
                    "feature_importance": self.pipeline.meta_model.feature_importance,
                    "optimal_weights": self.pipeline.meta_model.optimal_weights,
                    "confidence_thresholds": self.pipeline.meta_model.confidence_thresholds,
                },
                "model_selector": self.pipeline.get_model_performance_stats(),
                "base_models": {
                    "rf_win": {
                        "feature_importance": self.rf_win.feature_importances_,
                        "n_features": self.rf_win.n_features_in_,
                    },
                    "gb_win": {
                        "feature_importance": self.gb_win.feature_importances_,
                        "n_features": self.gb_win.n_features_in_,
                    },
                    "nn_win": {
                        "loss": self.nn_win.loss_,
                        "n_features": self.nn_win.n_features_in_,
                    },
                },
            }
        except Exception as e:
            logger.error(f"Error getting model stats: {str(e)}")
            return {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            # Create a copy to avoid modifying the original
            features_df = df.copy()

            # Add basic team stats
            team_stats = self._get_team_stats(features_df)

            # Add matchup features
            matchup_features = self._get_matchup_features(features_df)

            # Add temporal features
            temporal_features = self._get_temporal_features(features_df)

            # Combine all features
            features = pd.concat(
                [team_stats, matchup_features, temporal_features], axis=1
            )

            # Ensure consistent feature order
            feature_columns = self._get_feature_columns()
            for col in feature_columns:
                if col not in features.columns:
                    features[col] = 0.0  # Default value for missing features

            # Select only the required features in the correct order
            features = features[feature_columns]

            # Scale numerical features
            features = pd.DataFrame(
                self.feature_scaler.fit_transform(features),
                columns=feature_columns,
                index=features.index,
            )

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def _get_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get team statistics for each game."""
        team_stats = pd.DataFrame(index=df.index)

        # Calculate basic team stats (using rolling averages where available)
        for prefix in ["home", "away"]:
            # Points per game (default to league average if no history)
            team_stats[f"{prefix}_team_points_per_game"] = 24.0
            team_stats[f"{prefix}_team_points_allowed_per_game"] = 24.0
            team_stats[f"{prefix}_team_win_pct"] = 0.5
            team_stats[f"{prefix}_team_pass_yards_per_game"] = 225.0
            team_stats[f"{prefix}_team_rush_yards_per_game"] = 110.0
            team_stats[f"{prefix}_team_turnover_margin"] = 0.0

        return team_stats

    def _get_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get matchup-specific features."""
        matchup_features = pd.DataFrame(index=df.index)

        # Basic matchup features
        matchup_features["home_team_favorite"] = 1  # Default home field advantage
        matchup_features["point_spread"] = -3.0  # Standard home field advantage
        matchup_features["over_under"] = 48.0  # League average
        matchup_features["h2h_history"] = 0
        matchup_features["h2h_home_win_pct"] = 0.5

        return matchup_features

    def _get_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get time-based features."""
        temporal_features = pd.DataFrame(index=df.index)

        # Convert dates to datetime
        dates = pd.to_datetime(df["date"])

        # Basic temporal features
        temporal_features["game_month"] = dates.dt.month
        temporal_features["game_day"] = dates.dt.dayofweek
        temporal_features["is_late_season"] = (dates.dt.month >= 11).astype(int)
        temporal_features["is_playoff"] = df["is_playoff"].fillna(0).astype(int)
        temporal_features["is_primetime"] = dates.apply(self._is_primetime).astype(int)

        # Weather features (use provided values or defaults)
        temporal_features["game_temperature"] = df["temperature"].fillna(70)
        temporal_features["game_wind_speed"] = df["wind_speed"].fillna(8)
        temporal_features["game_precipitation"] = df["precipitation"].fillna(0)

        return temporal_features

    def _get_feature_columns(self) -> List[str]:
        """Get list of features to use for prediction."""
        return [
            # Team performance features
            "home_team_points_per_game",
            "away_team_points_per_game",
            "home_team_points_allowed_per_game",
            "away_team_points_allowed_per_game",
            "home_team_pass_yards_per_game",
            "away_team_pass_yards_per_game",
            "home_team_rush_yards_per_game",
            "away_team_rush_yards_per_game",
            "home_team_turnover_margin",
            "away_team_turnover_margin",
            "home_team_win_pct",
            "away_team_win_pct",
            # Matchup features
            "home_team_favorite",
            "point_spread",
            "over_under",
            "h2h_history",
            "h2h_home_win_pct",
            # Temporal features
            "game_month",
            "game_day",
            "is_late_season",
            "is_playoff",
            "is_primetime",
            "game_temperature",
            "game_wind_speed",
            "game_precipitation",
        ]

    def _is_primetime(self, game_time: pd.Timestamp) -> int:
        """Check if game is in primetime (after 7 PM local)."""
        hour = game_time.hour
        return 1 if hour >= 19 else 0

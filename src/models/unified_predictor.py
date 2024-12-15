import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
import os
from sklearn.preprocessing import StandardScaler
from models.enhanced_features import enhance_training_data
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedPredictor:
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize the predictor with multiple models.

        Args:
            models: Dictionary of model name to model object mappings
        """
        self.models = models
        self.historical_data = pd.DataFrame()
        self.scalers = {name: StandardScaler() for name in models.keys()}
        self._load_historical_data()

    def prepare_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare features for each model according to their expected feature sets.

        Args:
            data: DataFrame containing the features

        Returns:
            Dictionary of prepared features for each model
        """
        try:
            # Convert datetime columns in input data
            if "game_datetime" in data.columns:
                data = data.copy()
                data["game_datetime"] = pd.to_datetime(data["game_datetime"])

            # Combine historical and new data for feature calculation
            if not self.historical_data.empty:
                combined_data = pd.concat(
                    [self.historical_data, data], ignore_index=True
                )
            else:
                combined_data = data

            # Get enhanced features
            enhanced_data = enhance_training_data(combined_data)
            prediction_features = enhanced_data.tail(len(data)).copy()

            # Define feature sets for each model
            model_features = {
                "statistical": [
                    "win_pct",
                    "loss_pct",
                    "points_per_game",
                    "points_allowed_per_game",
                    "home_win_pct",
                    "away_win_pct",
                    "div_win_pct",
                    "conf_win_pct",
                    "q1_points_per_game",
                    "q2_points_per_game",
                    "q3_points_per_game",
                    "q4_points_per_game",
                    "avg_margin",
                    "avg_margin_home",
                    "avg_margin_away",
                    "scoring_consistency",
                    "defense_consistency",
                    "early_season_win_pct",
                    "mid_season_win_pct",
                    "late_season_win_pct",
                    "points_per_drive",
                    "yards_per_play",
                    "momentum",
                    "off_efficiency",
                    "third_down_conv",
                ],
                "basic": [
                    "win_pct",
                    "points_per_game",
                    "points_allowed_per_game",
                    "home_win_pct",
                    "away_win_pct",
                    "avg_margin",
                    "early_season_win_pct",
                    "mid_season_win_pct",
                    "late_season_win_pct",
                    "momentum",
                    "off_efficiency",
                    "third_down_conv",
                ],
                "vedic": [
                    "win_pct",
                    "points_per_game",
                    "points_allowed_per_game",
                    "home_win_pct",
                    "away_win_pct",
                    "early_season_win_pct",
                    "mid_season_win_pct",
                    "momentum",
                    "third_down_conv",
                ],
                "all": [
                    "win_pct",
                    "loss_pct",
                    "points_per_game",
                    "points_allowed_per_game",
                    "home_win_pct",
                    "away_win_pct",
                    "div_win_pct",
                    "conf_win_pct",
                    "q1_points_per_game",
                    "q2_points_per_game",
                    "q3_points_per_game",
                    "q4_points_per_game",
                    "avg_margin",
                    "avg_margin_home",
                    "avg_margin_away",
                    "scoring_consistency",
                    "defense_consistency",
                    "early_season_win_pct",
                    "mid_season_win_pct",
                    "late_season_win_pct",
                    "points_per_drive",
                    "yards_per_play",
                    "strength_of_schedule",
                    "momentum",
                    "off_efficiency",
                    "third_down_conv",
                ],
            }

            # Prepare features for each model
            features = {}
            for model_name, feature_list in model_features.items():
                if model_name not in self.models:
                    continue

                # Create home and away versions of each feature
                model_cols = []
                for feat in feature_list:
                    model_cols.extend([f"home_{feat}", f"away_{feat}"])
                    if not any(x in feat for x in ["home_", "away_", "diff"]):
                        model_cols.append(f"{feat}_diff")

                # Add game-specific features
                model_cols.extend(
                    ["is_home_game", "is_division_game", "is_conference_game"]
                )

                # Select features and fill missing values
                X = prediction_features[model_cols].fillna(0)

                # Scale features
                X = self.scalers[model_name].fit_transform(X)
                features[model_name] = X

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def predict(self, game_data: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
        """
        Make predictions using all available models.

        Args:
            game_data: DataFrame containing game information

        Returns:
            Dictionary mapping model names to lists of prediction dictionaries
        """
        # Prepare features
        X = self.prepare_features(game_data)

        predictions = {}
        for model_name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X[model_name])
                    home_win_probs = probs[:, 1]
                else:
                    # For models that don't support probabilities
                    preds = model.predict(X[model_name])
                    home_win_probs = (preds > 0).astype(float)

                predictions[model_name] = [
                    {"home_win_prob": float(prob), "away_win_prob": float(1 - prob)}
                    for prob in home_win_probs
                ]
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                predictions[model_name] = []

        return predictions

    def _load_historical_data(self):
        """Load historical data for feature calculation"""
        try:
            # Load historical data
            historical_file = "data/nfl_games.csv"
            if os.path.exists(historical_file):
                self.historical_data = pd.read_csv(historical_file)
                # Convert datetime columns
                self.historical_data["game_datetime"] = pd.to_datetime(
                    self.historical_data["game_datetime"]
                )
                logger.info(f"Loaded {len(self.historical_data)} historical games")
            else:
                logger.warning(f"Historical data file {historical_file} not found")
                self.historical_data = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            self.historical_data = pd.DataFrame()

    def predict_games(
        self, future_games_path: str
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Predict outcomes for multiple games.

        Args:
            future_games_path: Path to CSV file containing future games

        Returns:
            Dictionary of predictions from each model
        """
        try:
            # Load and preprocess future games
            future_games = pd.read_csv(future_games_path)

            # Convert game_date to datetime (timezone naive)
            future_games["game_date"] = pd.to_datetime(
                future_games["game_date"]
            ).dt.tz_localize(None)

            # Prepare features for each model
            model_features = self.prepare_features(future_games)

            # Initialize predictions dictionary
            predictions = {name: [] for name in self.models.keys()}

            # Make predictions with each model
            for game_idx in range(len(future_games)):
                game = future_games.iloc[game_idx]

                try:
                    # Add small delay between predictions to avoid rate limiting
                    if game_idx > 0:
                        time.sleep(1)  # 1 second delay between predictions

                    for model_name, model in self.models.items():
                        try:
                            # Get features for this model
                            game_features = model_features[model_name][
                                game_idx : game_idx + 1
                            ]

                            # Get prediction probability
                            pred_prob = float(model.predict_proba(game_features)[0][1])

                            # Store prediction
                            predictions[model_name].append(
                                {
                                    "home_win_prob": pred_prob,
                                    "home_team": game["home_team"],
                                    "away_team": game["away_team"],
                                    "game_date": game["game_date"].strftime("%Y-%m-%d"),
                                }
                            )

                        except Exception as model_error:
                            logger.error(
                                f"Error with model {model_name} for game {game_idx}: {str(model_error)}"
                            )
                            # Add default prediction
                            predictions[model_name].append(
                                {
                                    "home_win_prob": 0.5,
                                    "home_team": game["home_team"],
                                    "away_team": game["away_team"],
                                    "game_date": game["game_date"].strftime("%Y-%m-%d"),
                                }
                            )

                except Exception as game_error:
                    logger.error(
                        f"Error predicting {game['home_team']} vs {game['away_team']}: {str(game_error)}"
                    )
                    # Add default predictions for all models
                    for model_name in self.models.keys():
                        predictions[model_name].append(
                            {
                                "home_win_prob": 0.5,
                                "home_team": game["home_team"],
                                "away_team": game["away_team"],
                                "game_date": game["game_date"].strftime("%Y-%m-%d"),
                            }
                        )

            return predictions

        except Exception as e:
            logger.error(f"Error in predict_games: {str(e)}")
            raise


def predict_games(games_file: str) -> Dict[str, List[Dict[str, float]]]:
    """
    Predict outcomes for games in the input file.

    Args:
        games_file: Path to CSV file containing games to predict
    """
    # Load games data
    games = pd.read_csv(games_file)

    # Initialize predictor
    models = {
        "basic": joblib.load("models/nfl_predictor_basic.joblib"),
        "statistical": joblib.load("models/nfl_predictor_statistical.joblib"),
        "sports": joblib.load("models/nfl_sports_only_model.joblib"),
        "ensemble": joblib.load("models/ensemble/nfl_ensemble.joblib"),
    }
    predictor = UnifiedPredictor(models)

    # Make predictions
    predictions = predictor.predict(games)

    # Print predictions
    for model_name, preds in predictions.items():
        print(f"\n{model_name.upper()} MODEL PREDICTIONS:")
        print("-" * 50)
        for i, pred in enumerate(preds):
            print(f"Game {i+1}:")
            print(f"Home Win Probability: {pred['home_win_prob']:.1%}")
            print(f"Away Win Probability: {pred['away_win_prob']:.1%}")
            print("-" * 30)

    return predictions


if __name__ == "__main__":
    # Predict games from the processed games file
    predict_games("data/processed/nfl_games_processed.csv")

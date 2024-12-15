"""NFL model training module for building and optimizing prediction models."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nfl_advanced_system import NFLAdvancedSystem
from data_processing.nfl_data_processor import NFLDataProcessor
from evaluation.nfl_model_evaluator import NFLModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLModelTrainer:
    def __init__(self):
        """Initialize NFL model trainer."""
        self.processor = NFLDataProcessor()
        self.evaluator = NFLModelEvaluator()
        self.model_system = NFLAdvancedSystem()

    def prepare_training_data(
        self,
        game_data: pd.DataFrame,
        team_stats: pd.DataFrame,
        weather_data: pd.DataFrame = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training.

        Args:
            game_data: DataFrame containing game information
            team_stats: DataFrame containing team statistics
            weather_data: Optional DataFrame containing weather information

        Returns:
            Tuple of (features array, target array)
        """
        try:
            # Process and combine data
            feature_data = self.processor.prepare_model_features(
                game_data, team_stats, weather_data
            )

            # Separate features and target
            X = feature_data.drop("home_win", axis=1)
            y = feature_data["home_win"]

            return X, y

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def train_models(
        self, X: np.ndarray, y: np.ndarray, model_params: Dict = None, cv_folds: int = 5
    ) -> Dict:
        """Train and optimize models.

        Args:
            X: Feature array
            y: Target array
            model_params: Optional dictionary of model parameters for grid search
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing trained models
        """
        try:
            # Default model parameters if none provided
            if model_params is None:
                model_params = {
                    "random_forest": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [10, 15, 20],
                        "min_samples_split": [2, 5, 10],
                    },
                    "gradient_boosting": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [3, 4, 5],
                        "learning_rate": [0.01, 0.1],
                    },
                }

            trained_models = {}

            # Train each model with grid search
            for name, model in self.model_system.models.items():
                logger.info(f"Training {name} model...")

                # Create pipeline with scaling
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

                # Perform grid search
                grid_search = GridSearchCV(
                    pipeline,
                    {f"model__{k}": v for k, v in model_params[name].items()},
                    cv=cv_folds,
                    scoring="accuracy",
                    n_jobs=-1,
                )

                grid_search.fit(X, y)

                # Store best model
                trained_models[name] = grid_search.best_estimator_

                logger.info(f"{name} best parameters: {grid_search.best_params_}")
                logger.info(f"{name} best score: {grid_search.best_score_:.3f}")

            return trained_models

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def evaluate_models(
        self,
        models: Dict,
        X: np.ndarray,
        y: np.ndarray,
        game_data: pd.DataFrame,
        save_dir: str = None,
    ) -> Dict:
        """Evaluate trained models.

        Args:
            models: Dictionary of trained models
            X: Feature array
            y: Target array
            game_data: Original game data for situation-specific evaluation
            save_dir: Optional directory to save evaluation results

        Returns:
            Dictionary containing evaluation results
        """
        try:
            evaluation_results = {}

            for name, model in models.items():
                logger.info(f"Evaluating {name} model...")

                # Make predictions
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]

                # Generate evaluation report
                evaluation_results[name] = self.evaluator.generate_evaluation_report(
                    y,
                    y_pred,
                    y_prob,
                    game_data,
                    save_dir=os.path.join(save_dir, name) if save_dir else None,
                )

            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            raise

    def save_models(self, models: Dict, save_dir: str):
        """Save trained models to disk.

        Args:
            models: Dictionary of trained models
            save_dir: Directory to save models
        """
        try:
            os.makedirs(save_dir, exist_ok=True)

            for name, model in models.items():
                model_path = os.path.join(save_dir, f"nfl_{name}_model.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def load_models(self, load_dir: str) -> Dict:
        """Load trained models from disk.

        Args:
            load_dir: Directory containing saved models

        Returns:
            Dictionary containing loaded models
        """
        try:
            loaded_models = {}

            for name in self.model_system.models.keys():
                model_path = os.path.join(load_dir, f"nfl_{name}_model.joblib")
                if os.path.exists(model_path):
                    loaded_models[name] = joblib.load(model_path)
                    logger.info(f"Loaded {name} model from {model_path}")

            return loaded_models

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def train_and_evaluate(
        self,
        game_data: pd.DataFrame,
        team_stats: pd.DataFrame,
        weather_data: pd.DataFrame = None,
        model_params: Dict = None,
        save_dir: str = None,
    ) -> Tuple[Dict, Dict]:
        """Complete training and evaluation pipeline.

        Args:
            game_data: DataFrame containing game information
            team_stats: DataFrame containing team statistics
            weather_data: Optional DataFrame containing weather information
            model_params: Optional dictionary of model parameters for grid search
            save_dir: Optional directory to save models and evaluation results

        Returns:
            Tuple of (trained models dictionary, evaluation results dictionary)
        """
        try:
            # Prepare data
            X, y = self.prepare_training_data(game_data, team_stats, weather_data)

            # Split data
            (
                X_train,
                X_test,
                y_train,
                y_test,
                game_data_train,
                game_data_test,
            ) = train_test_split(X, y, game_data, test_size=0.2, random_state=42)

            # Train models
            trained_models = self.train_models(X_train, y_train, model_params)

            # Evaluate models
            evaluation_results = self.evaluate_models(
                trained_models,
                X_test,
                y_test,
                game_data_test,
                save_dir=os.path.join(save_dir, "evaluation") if save_dir else None,
            )

            # Save models if directory provided
            if save_dir:
                self.save_models(trained_models, os.path.join(save_dir, "models"))

            return trained_models, evaluation_results

        except Exception as e:
            logger.error(f"Error in training and evaluation pipeline: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    trainer = NFLModelTrainer()

    # Load sample data (replace with actual data loading)
    sample_game_data = pd.DataFrame(
        {
            "game_date": ["2023-01-01", "2023-01-08"],
            "home_team": ["NE", "GB"],
            "away_team": ["BUF", "DET"],
            "home_score": [21, 28],
            "away_score": [24, 17],
            "home_win": [0, 1],
        }
    )

    sample_team_stats = pd.DataFrame(
        {
            "team": ["NE", "GB", "BUF", "DET"],
            "points_scored": [300, 350, 400, 280],
            "points_allowed": [250, 300, 280, 320],
            "total_yards": [4000, 4500, 4800, 3800],
            "games_played": [16, 16, 16, 16],
        }
    )

    # Train and evaluate models
    trained_models, evaluation_results = trainer.train_and_evaluate(
        sample_game_data, sample_team_stats, save_dir="trained_models"
    )

    print("\nTraining and evaluation complete!")
    print("\nEvaluation results summary:")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name} model:")
        print(f"Overall accuracy: {results['overall_metrics']['accuracy']:.3f}")

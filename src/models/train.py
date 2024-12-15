"""
NFL Prediction Model Training Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from pathlib import Path
import joblib
from typing import Dict, List, Tuple
import json
import os

from models.trainers import (
    LSTMTrainer,
    XGBoostTrainer,
    RandomForestTrainer,
    SVMTrainer,
    MLPTrainer,
    AdaBoostTrainer,
)
from models.ensemble import NFLEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = None):
        """
        Initialize the model training pipeline

        Args:
            config_path (str): Path to model configuration file
        """
        self.config = (
            self._load_config(config_path) if config_path else self._default_config()
        )
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def _default_config(self) -> Dict:
        """Default model configuration"""
        return {
            "test_size": 0.2,
            "random_state": 42,
            "model_weights": {
                "lstm": 1.0,
                "xgboost": 2.0,
                "random_forest": 1.0,
                "svm": 0.5,
                "mlp": 1.0,
                "adaboost": 0.5,
            },
            "feature_groups": {
                "basic": [
                    "point_differential",
                    "home_ppg",
                    "away_ppg",
                    "matchup_strength",
                    "home_win_pct",
                    "away_win_pct",
                    "win_pct_diff",
                    "is_favorite",
                ],
                "momentum": [
                    "home_momentum",
                    "away_momentum",
                    "home_form",
                    "away_form",
                    "rolling_score_recent",
                    "rolling_score_season",
                    "rolling_score_recent_away",
                    "rolling_score_season_away",
                ],
                "historical": [
                    "h2h_games",
                    "h2h_home_wins",
                    "h2h_away_wins",
                    "h2h_avg_point_diff",
                ],
                "context": [
                    "home_rest_days",
                    "away_rest_days",
                    "is_winter",
                    "is_warm_weather",
                    "weather_impact",
                    "is_division_game",
                    "is_conference_game",
                    "is_primetime",
                    "is_afternoon",
                    "is_morning",
                    "is_weekend",
                ],
            },
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from file"""
        with open(config_path, "r") as f:
            return json.load(f)

    def prepare_data(
        self, data_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        logger.info("Preparing data for training")

        # Load data
        data = pd.read_csv(data_path)

        # Sort by date for time-based split
        data["game_date"] = pd.to_datetime(data["game_date"])
        data = data.sort_values("game_date")

        # Extract features from dates
        date_features = ["game_date"]
        for col in date_features:
            data[f"{col}_month"] = data[col].dt.month
            data[f"{col}_dayofweek"] = data[col].dt.dayofweek
            data[f"{col}_is_weekend"] = data[col].dt.dayofweek.isin([5, 6])

        # Drop non-feature columns and categorical columns we won't use
        drop_cols = date_features + [
            "game_id",
            "home_team",
            "away_team",
            "stadium",
            "home_division",
            "away_division",
        ]
        features = data.drop(columns=drop_cols)

        # Get target variable
        target = features.pop("home_win")

        # Handle missing values
        features = features.fillna(0)

        # Make sure all columns are numeric
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]

        # Log the features we're using
        logger.info(
            f"Using {len(features.columns)} features: {features.columns.tolist()}"
        )

        # Convert to numpy arrays
        X = features.values
        y = target.values

        # Use time-based split instead of random split
        split_idx = int(len(X) * (1 - self.config["test_size"]))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Save feature names for later use
        self.feature_names = features.columns.tolist()

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """Train all models and collect their metrics"""
        logger.info("Training individual models")

        results = {}

        # Initialize and train each model
        trainers = {
            "lstm": LSTMTrainer(),
            "xgboost": XGBoostTrainer(),
            "random_forest": RandomForestTrainer(),
            "svm": SVMTrainer(),
            "mlp": MLPTrainer(),
            "adaboost": AdaBoostTrainer(),
        }

        for name, trainer in trainers.items():
            try:
                logger.info(f"Training {name} model...")
                trainer.train(X_train, y_train)

                # Evaluate on both train and test sets
                train_metrics = trainer.evaluate(X_train, y_train)
                test_metrics = trainer.evaluate(X_test, y_test)

                results[name] = {
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "best_params": trainer.best_params,
                }

                # Save individual model
                model_path = f"models/trained/{name}_model.joblib"
                trainer.save(model_path)
                logger.info(f"Saved {name} model to {model_path}")

                # Store model for ensemble
                self.models[name] = trainer

            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
                continue

        return results

    def create_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Create and optimize ensemble model"""
        logger.info("Creating ensemble model")

        try:
            # Initialize ensemble with base weights from config
            self.ensemble = NFLEnsemble(
                models={name: trainer.model for name, trainer in self.models.items()},
                weights=self.config["model_weights"],
            )

            # Calibrate models for better probability estimates
            logger.info("Calibrating models")
            self.ensemble.calibrate_models(X_test, y_test)

            # Optimize ensemble weights using advanced optimization
            logger.info("Optimizing ensemble weights")
            optimized_weights = self.ensemble.optimize_weights(X_test, y_test)
            logger.info("Optimized weights:")
            for name, weight in optimized_weights.items():
                logger.info(f"{name}: {weight:.3f}")

            # Get optimization history
            history = self.ensemble.get_optimization_history()
            logger.info("Weight optimization history:")
            for i, step in enumerate(history[-5:]):  # Show last 5 steps
                logger.info(f"Step {len(history)-5+i}:")
                logger.info(f"  Log Loss: {step['log_loss']:.4f}")
                logger.info(f"  AUC: {step['auc']:.4f}")
                logger.info(f"  Brier Score: {step['brier_score']:.4f}")

            # Save ensemble model with calibrated models and optimization history
            ensemble_path = "models/trained/ensemble_model.joblib"
            self.ensemble.save(ensemble_path)
            logger.info(f"Saved ensemble model to {ensemble_path}")

        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            raise

    def evaluate_ensemble(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate ensemble model performance"""
        logger.info("Evaluating ensemble model")

        try:
            # Get comprehensive metrics
            metrics = self.ensemble.evaluate(X_test, y_test)

            logger.info("Ensemble model metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")

            # Compare with base models
            logger.info("\nComparison with base models:")
            for name, trainer in self.models.items():
                base_metrics = trainer.evaluate(X_test, y_test)
                logger.info(f"\n{name.upper()} model metrics:")
                for metric, value in base_metrics.items():
                    logger.info(f"{metric}: {value:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating ensemble model: {e}")
            raise

    def analyze_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze feature importance across models"""
        logger.info("Analyzing feature importance")

        importance_analysis = {}

        # Get feature importance from tree-based models
        for name, model in self.models.items():
            if hasattr(model, "feature_importance"):
                importance = model.feature_importance.get("gain", {})
                if importance:
                    # Sort features by importance
                    sorted_features = sorted(
                        importance.items(), key=lambda x: x[1], reverse=True
                    )
                    importance_analysis[name] = sorted_features

        # Log top features for each model
        for model_name, features in importance_analysis.items():
            logger.info(f"\nTop 10 important features for {model_name}:")
            for feature, importance in features[:10]:
                logger.info(f"{feature}: {importance:.4f}")

        return importance_analysis

    def save_training_report(
        self,
        results: Dict[str, Dict],
        ensemble_metrics: Dict[str, float],
        feature_importance: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """Save comprehensive training report"""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": self.config,
            "individual_models": results,
            "ensemble_model": {
                "metrics": ensemble_metrics,
                "weights": self.ensemble.weights,
            },
            "feature_importance": feature_importance,
        }

        report_path = "models/reports/training_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved training report to {report_path}")

    def train_pipeline(self, data_path: str, output_dir: str) -> Dict:
        """
        Run complete training pipeline

        Args:
            data_path (str): Path to processed data file
            output_dir (str): Directory to save trained models

        Returns:
            Dict containing evaluation results
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(data_path)

        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test)

        # Create and optimize ensemble
        self.create_ensemble(X_test, y_test)

        # Evaluate ensemble
        ensemble_metrics = self.evaluate_ensemble(X_test, y_test)

        # Analyze feature importance
        feature_importance = self.analyze_feature_importance()

        # Save training report
        self.save_training_report(results, ensemble_metrics, feature_importance)

        return results


def main():
    # Initialize trainer
    trainer = ModelTrainer()

    # Run training pipeline
    results = trainer.train_pipeline(
        data_path="data/processed/nfl_games_processed.csv", output_dir="models/trained"
    )

    # Log results
    logger.info("\nTraining Results:")
    for model, metrics in results.items():
        logger.info(f"{model}: {metrics['test_metrics']['accuracy']:.3f} accuracy")


if __name__ == "__main__":
    main()

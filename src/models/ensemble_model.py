import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from typing import Tuple, List, Dict
import joblib
import logging
from models.enhanced_features import enhance_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLEnsembleModel:
    def __init__(self, n_splits: int = 5):
        """
        Initialize ensemble model with multiple base models.

        Args:
            n_splits: Number of folds for cross-validation
        """
        self.n_splits = n_splits
        self.models = {
            "rf": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=42
            ),
        }
        self.scalers = {}
        self.trained_models = {}

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for different models.

        Args:
            X: Input features
            y: Target labels
        """
        prepared_data = {}

        # Standard features for RF and GB
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        prepared_data["rf"] = X_scaled
        prepared_data["gb"] = X_scaled

        return prepared_data, y

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """
        Train ensemble using k-fold cross validation.

        Args:
            X: Input features
            y: Target labels
        """
        metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            logger.info(f"Training fold {fold}/{self.n_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Prepare data for each model type
            train_data, y_train = self.prepare_data(X_train, y_train)
            val_data, y_val = self.prepare_data(X_val, y_val)

            fold_preds = []

            # Train each model
            for name, model in self.models.items():
                model.fit(train_data[name], y_train)
                preds = model.predict_proba(val_data[name])[:, 1]
                fold_preds.append(preds)

            # Ensemble predictions (average probabilities)
            ensemble_preds = np.mean(fold_preds, axis=0)
            binary_preds = (ensemble_preds > 0.5).astype(int)

            # Calculate metrics
            metrics["accuracy"].append(accuracy_score(y_val, binary_preds))
            metrics["precision"].append(precision_score(y_val, binary_preds))
            metrics["recall"].append(recall_score(y_val, binary_preds))
            metrics["f1"].append(f1_score(y_val, binary_preds))

            # Save models for this fold
            self.trained_models[f"fold_{fold}"] = {
                name: model for name, model in self.models.items()
            }

        # Log average metrics
        for metric, values in metrics.items():
            logger.info(
                f"Average {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}"
            )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Input features
        """
        all_preds = []

        # Get predictions from each fold's models
        for fold_models in self.trained_models.values():
            fold_preds = []

            # Prepare data
            test_data, _ = self.prepare_data(X, np.zeros(len(X)))

            # Get predictions from each model
            for name, model in fold_models.items():
                preds = model.predict_proba(test_data[name])[:, 1]
                fold_preds.append(preds)

            # Average predictions for this fold
            fold_ensemble = np.mean(fold_preds, axis=0)
            all_preds.append(fold_ensemble)

        # Average predictions across all folds
        final_probs = np.mean(all_preds, axis=0)
        return (final_probs > 0.5).astype(int)

    def save(self, path: str):
        """Save the ensemble model."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "NFLEnsembleModel":
        """Load a saved ensemble model."""
        return joblib.load(path)


def train_ensemble():
    """Train and evaluate the ensemble model."""
    # Load and prepare data
    data = pd.read_csv("data/processed/nfl_games_processed.csv")

    # Enhance features
    enhanced_data = enhance_training_data(data)

    # Use existing features from the processed data
    feature_cols = [
        "home_win_percentage",
        "away_win_percentage",  # Team performance
        "home_rest_days",
        "away_rest_days",  # Rest advantage
        "h2h_home_wins",
        "h2h_away_wins",  # Head-to-head history
        "home_momentum",
        "away_momentum",  # Recent form
        "home_form",
        "away_form",  # Season form
        "win_pct_diff",  # Relative strength
        "is_division_game",
        "is_conference_game",  # Game context
        "is_favorite",  # Vegas odds
        "rolling_score_recent",
        "rolling_score_season",  # Scoring trends
        "streak",  # Win/loss streaks
    ]

    # Convert boolean columns to int
    for col in ["is_division_game", "is_conference_game", "is_favorite"]:
        enhanced_data[col] = enhanced_data[col].astype(int)

    # Select features that exist in the data
    available_features = [col for col in feature_cols if col in enhanced_data.columns]
    logger.info(f"Using features: {available_features}")

    # Handle missing values
    X_data = enhanced_data[available_features].copy()

    # Fill missing values with appropriate defaults
    X_data["home_win_percentage"] = X_data["home_win_percentage"].fillna(0.5)
    X_data["away_win_percentage"] = X_data["away_win_percentage"].fillna(0.5)
    X_data["home_rest_days"] = X_data["home_rest_days"].fillna(7)  # Default to a week
    X_data["away_rest_days"] = X_data["away_rest_days"].fillna(7)
    X_data["h2h_home_wins"] = X_data["h2h_home_wins"].fillna(0)
    X_data["h2h_away_wins"] = X_data["h2h_away_wins"].fillna(0)
    X_data["home_momentum"] = X_data["home_momentum"].fillna(0)
    X_data["away_momentum"] = X_data["away_momentum"].fillna(0)
    X_data["win_pct_diff"] = X_data["win_pct_diff"].fillna(0)
    X_data["rolling_score_recent"] = X_data["rolling_score_recent"].fillna(
        X_data["rolling_score_recent"].mean()
    )
    X_data["rolling_score_season"] = X_data["rolling_score_season"].fillna(
        X_data["rolling_score_season"].mean()
    )
    X_data["streak"] = X_data["streak"].fillna(0)

    # Convert to numpy array
    X = X_data.values
    y = enhanced_data["home_win"].astype(int).values

    # Train ensemble
    ensemble = NFLEnsembleModel()
    metrics = ensemble.train(X, y)

    # Save the model
    ensemble.save("models/ensemble/nfl_ensemble.joblib")

    return metrics


if __name__ == "__main__":
    train_ensemble()

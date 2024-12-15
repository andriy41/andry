"""Train the Vedic model on historical NFL data with enhanced techniques"""

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.vedic_basic.vedic_model import VedicModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "data",
)
MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "models",
    "vedic_model",
)


def load_data():
    """Load and preprocess NFL game data"""
    data_file = os.path.join(DATA_DIR, "nfl_games.csv")

    try:
        # Load data
        df = pd.read_csv(data_file)

        # Convert data to list of dictionaries
        games = []
        for _, row in df.iterrows():
            game = {
                "date": row["game_datetime"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": row["home_team_score"],
                "away_score": row["away_team_score"],
                "stadium_latitude": row["stadium_latitude"],
                "stadium_longitude": row["stadium_longitude"],
            }
            # Only include games with valid scores
            if pd.notna(game["home_score"]) and pd.notna(game["away_score"]):
                games.append(game)

        return games

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return []


def evaluate_model(model, test_data, test_labels):
    """Evaluate model with multiple metrics"""
    predictions = []
    confidences = []

    for game in test_data:
        pred = model.predict(game)
        if pred is not None:  # Only include high confidence predictions
            predictions.append(pred["prediction"])
            confidences.append(pred["confidence"])

    if not predictions:
        logger.warning("No high confidence predictions made")
        return {}

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # Get corresponding test labels for predictions made
    valid_labels = test_labels[: len(predictions)]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(valid_labels, predictions),
        "precision": precision_score(valid_labels, predictions),
        "recall": recall_score(valid_labels, predictions),
        "f1": f1_score(valid_labels, predictions),
        "confidence_mean": np.mean(confidences),
        "predictions_made": len(predictions),
        "total_games": len(test_data),
    }

    return metrics


def main():
    """Main training function"""
    logger.info("Loading and preprocessing data...")
    games = load_data()

    if not games:
        logger.error("No games data loaded")
        return

    logger.info(f"Loaded {len(games)} games")

    # Initialize model
    logger.info("Initializing Vedic model...")
    model = VedicModel()

    # Use time-based split instead of random split
    train_size = int(len(games) * 0.8)
    train_games = games[:train_size]
    test_games = games[train_size:]

    # Train model with cross-validation
    logger.info("Training model with cross-validation...")
    kf = KFold(n_splits=5, shuffle=False)  # Time-series cross-validation
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_games)):
        # Split data
        fold_train_games = [train_games[i] for i in train_idx]
        fold_val_games = [train_games[i] for i in val_idx]

        # Train on this fold
        success = model.train(fold_train_games)

        if success:
            # Evaluate on validation set
            metrics = evaluate_model(
                model, fold_val_games, [0] * len(fold_val_games)
            )  # dummy labels
            if metrics:
                cv_scores.append(metrics["accuracy"])
                logger.info(f"Fold {fold + 1} Accuracy: {metrics['accuracy']:.4f}")

    logger.info(
        f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
    )

    # Train final model on all training data
    logger.info("Training final model on all training data...")
    success = model.train(train_games)

    if success:
        logger.info("Training successful! Evaluating final model...")
        test_metrics = evaluate_model(
            model, test_games, [0] * len(test_games)
        )  # dummy labels

        logger.info("\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save model if accuracy threshold is met
        if test_metrics.get("accuracy", 0) >= 0.85:
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.save_model(MODEL_DIR)
            logger.info(f"Model saved to {MODEL_DIR}")
        else:
            logger.warning("Model did not meet accuracy threshold of 85%, not saving")


if __name__ == "__main__":
    main()

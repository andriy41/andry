import sys
import os
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from models.total_prediction.stats_total_model import StatsTotalModel
from models.total_prediction.ml_total_model import MLTotalModel
from models.total_prediction.neuro_total_model import NeuroTotalModel
from models.total_prediction.vedic_total_model import VedicTotalModel


def load_and_prepare_data():
    """Load and prepare data for validation"""
    # Load the processed data
    df = pd.read_csv(
        "/Users/space/Downloads/NFL_Project/data/processed/nfl_processed_data.csv"
    )

    # Convert game_date to datetime
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Parse JSON team stats
    df["home_team_stats"] = df["home_team_stats"].apply(json.loads)
    df["away_team_stats"] = df["away_team_stats"].apply(json.loads)

    # Calculate total points for each game
    df["total_points"] = df["home_score"] + df["away_score"]

    # Sort by date
    df = df.sort_values("game_date")

    # Get the last 150 games for validation
    validation_set = df.tail(150)
    training_set = df.iloc[:-150]

    # Convert to dictionary format expected by models
    training_dict = {"games": training_set.to_dict("records")}
    validation_dict = {"games": validation_set.to_dict("records")}

    return training_dict, validation_dict


def combine_predictions(stats_pred, ml_pred, neural_pred, vedic_pred):
    """Combine predictions from all models and determine confidence level"""

    # Get individual predictions
    predictions = {"OVER": 0, "UNDER": 0}

    # Count votes for each prediction
    for pred in [stats_pred, ml_pred, neural_pred, vedic_pred]:
        if pred["predicted_total"] > pred["line"]:
            predictions["OVER"] += 1
        else:
            predictions["UNDER"] += 1

    # Calculate margins
    margins = []
    for pred in [stats_pred, ml_pred, neural_pred, vedic_pred]:
        margin = abs(pred["predicted_total"] - pred["line"])
        margins.append(margin)

    avg_margin = sum(margins) / len(margins)

    # Determine final prediction
    final_pred = "OVER" if predictions["OVER"] > predictions["UNDER"] else "UNDER"

    # Determine confidence level
    if predictions[final_pred] >= 3 and avg_margin > 5:
        confidence = "HIGH"
    elif predictions[final_pred] >= 3 or avg_margin > 5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "prediction": final_pred,
        "confidence": confidence,
        "margin": avg_margin,
        "model_agreement": predictions[final_pred],
    }


def evaluate_predictions(predictions, actuals, lines):
    results = {
        "total_games": len(predictions),
        "correct_predictions": 0,
        "accuracy": 0,
        "high_confidence_correct": 0,
        "high_confidence_total": 0,
        "medium_confidence_correct": 0,
        "medium_confidence_total": 0,
        "low_confidence_correct": 0,
        "low_confidence_total": 0,
    }

    for i in range(len(predictions)):
        pred = predictions[i]
        actual = actuals[i]
        line = lines[i]

        # Calculate confidence based on model consensus and margin
        confidence = pred["confidence"]

        # Check if prediction was correct
        if (pred["prediction"] == "OVER" and actual > line) or (
            pred["prediction"] == "UNDER" and actual < line
        ):
            results["correct_predictions"] += 1

            if confidence == "HIGH":
                results["high_confidence_correct"] += 1
            elif confidence == "MEDIUM":
                results["medium_confidence_correct"] += 1
            else:
                results["low_confidence_correct"] += 1

        # Track totals by confidence
        if confidence == "HIGH":
            results["high_confidence_total"] += 1
        elif confidence == "MEDIUM":
            results["medium_confidence_total"] += 1
        else:
            results["low_confidence_total"] += 1

    # Calculate accuracy metrics
    results["accuracy"] = results["correct_predictions"] / results["total_games"]
    results["high_confidence_accuracy"] = (
        results["high_confidence_correct"] / results["high_confidence_total"]
        if results["high_confidence_total"] > 0
        else 0
    )
    results["medium_confidence_accuracy"] = (
        results["medium_confidence_correct"] / results["medium_confidence_total"]
        if results["medium_confidence_total"] > 0
        else 0
    )
    results["low_confidence_accuracy"] = (
        results["low_confidence_correct"] / results["low_confidence_total"]
        if results["low_confidence_total"] > 0
        else 0
    )

    return results


def run_validation():
    # Load and prepare data
    training_set, validation_set = load_and_prepare_data()

    # Initialize models
    stats_total_model = StatsTotalModel()
    ml_total_model = MLTotalModel()
    neural_total_model = NeuroTotalModel()
    vedic_total_model = VedicTotalModel()

    # Train models
    stats_total_model.train(training_set)
    ml_total_model.train(training_set)
    neural_total_model.train(training_set)
    vedic_total_model.train(training_set)

    predictions = []
    actuals = []
    lines = []

    # Make predictions on validation set
    for game in validation_set["games"]:
        # Get predictions from each model
        stats_pred = stats_total_model.predict_total(game)
        ml_pred = ml_total_model.predict_total(game)
        neural_pred = neural_total_model.predict_total(game)
        vedic_pred = vedic_total_model.predict_total(game)

        # Combine predictions
        combined_pred = combine_predictions(
            stats_pred, ml_pred, neural_pred, vedic_pred
        )

        predictions.append(combined_pred)
        actuals.append(game["total_points"])
        lines.append(game.get("line", 0))

    # Evaluate results
    results = evaluate_predictions(predictions, actuals, lines)

    # Print results
    print("\nValidation Results (150 Games):")
    print(f"Overall Accuracy: {results['accuracy']:.2%}")
    print("\nBy Confidence Level:")
    print(
        f"High Confidence ({results['high_confidence_total']} games): {results['high_confidence_accuracy']:.2%}"
    )
    print(
        f"Medium Confidence ({results['medium_confidence_total']} games): {results['medium_confidence_accuracy']:.2%}"
    )
    print(
        f"Low Confidence ({results['low_confidence_total']} games): {results['low_confidence_accuracy']:.2%}"
    )

    return results


if __name__ == "__main__":
    run_validation()

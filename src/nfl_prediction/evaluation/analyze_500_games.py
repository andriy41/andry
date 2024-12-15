"""
Analyze and compare NFL predictions from all models with confidence levels
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.models.vedic_basic.vedic_model import VedicModel
from src.models.advanced_system.advanced_model import AdvancedModel
from src.models.combined_ml.combined_model import CombinedModel
from src.models.sports_only.sports_model import SportsModel
from src.models.total_prediction.neuro_total_model import NeuroTotalModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    """Load and prepare the historical games data"""
    data_dir = project_root / "data" / "historical"
    data = pd.read_csv(data_dir / "nfl_games_training.csv")
    data["gameday"] = pd.to_datetime(data["gameday"])

    # Sort by date and split into train/test
    data = data.sort_values("gameday")
    train_data = data.iloc[:-500].copy()
    test_data = data.iloc[-500:].copy()

    logger.info(
        f"Loaded {len(test_data)} games for analysis and {len(train_data)} for training"
    )
    logger.info(
        f"Date range for test data: {test_data['gameday'].min()} to {test_data['gameday'].max()}"
    )

    return train_data, test_data


def train_models(train_data):
    """Initialize and train all models"""
    models = {
        "Vedic": VedicModel(),
        "Advanced": AdvancedModel(),
        "Combined": CombinedModel(),
        "Sports": SportsModel(),
        "Neuro": NeuroTotalModel(),
    }

    trained_models = {}
    logger.info("Training models...")

    for name, model in models.items():
        try:
            model.train(train_data)
            logger.info(f"Successfully trained {name} model")
            trained_models[name] = model
        except Exception as e:
            logger.error(f"Error training {name} model: {str(e)}")
            continue

    return trained_models


def get_prediction_confidence(model, game_data):
    """Get prediction confidence from model"""
    try:
        pred = model.predict(game_data)
        confidence = pred.get("confidence", None)
        if confidence is None:
            # If no confidence provided, calculate from probabilities
            probs = pred.get("probabilities", [0.5, 0.5])
            confidence = max(probs)
        return confidence
    except Exception as e:
        logger.error(f"Error getting prediction confidence: {str(e)}")
        return 0.5


def categorize_predictions(confidence):
    """Categorize predictions by confidence level"""
    return pd.cut(
        confidence,
        bins=[0, 0.6, 0.75, 0.85, 1.0],
        labels=["Low", "Good", "Great", "Elite"],
    )


def analyze_predictions(models, test_data):
    """Analyze predictions from all models"""
    all_predictions = []

    for name, model in models.items():
        logger.info(f"\nAnalyzing {name} model...")
        model_predictions = []

        for idx, game in test_data.iterrows():
            try:
                pred = model.predict(game.to_dict())
                confidence = get_prediction_confidence(model, game.to_dict())

                model_predictions.append(
                    {
                        "Model": name,
                        "Date": game["gameday"],
                        "Prediction": pred.get("prediction", 0),
                        "Confidence": confidence,
                        "Actual": 1 if game["home_score"] > game["away_score"] else 0,
                        "Home Team": game["home_team"],
                        "Away Team": game["away_team"],
                        "Home Score": game["home_score"],
                        "Away Score": game["away_score"],
                    }
                )

            except Exception as e:
                logger.error(
                    f"Error predicting with {name} model on game {idx}: {str(e)}"
                )
                continue

        if model_predictions:
            all_predictions.extend(model_predictions)
            logger.info(f"Made {len(model_predictions)} predictions with {name} model")

    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    if predictions_df.empty:
        logger.error("No predictions were made by any model")
        return None, None

    # Add confidence category
    predictions_df["Confidence Level"] = categorize_predictions(
        predictions_df["Confidence"]
    )

    # Calculate accuracy metrics
    results = {}
    for model_name in predictions_df["Model"].unique():
        model_data = predictions_df[predictions_df["Model"] == model_name]

        results[model_name] = {
            "overall": {
                "total": len(model_data),
                "correct": (model_data["Prediction"] == model_data["Actual"]).sum(),
                "accuracy": (model_data["Prediction"] == model_data["Actual"]).mean(),
                "avg_confidence": model_data["Confidence"].mean(),
            },
            "by_category": {},
        }

        for category in ["Elite", "Great", "Good", "Low"]:
            cat_data = model_data[model_data["Confidence Level"] == category]
            if not cat_data.empty:
                results[model_name]["by_category"][category] = {
                    "total": len(cat_data),
                    "correct": (cat_data["Prediction"] == cat_data["Actual"]).sum(),
                    "accuracy": (cat_data["Prediction"] == cat_data["Actual"]).mean(),
                    "avg_confidence": cat_data["Confidence"].mean(),
                }

    return results, predictions_df


def plot_confidence_comparison(predictions_df, save_dir):
    """Create visualization plots for model comparison"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 1. Confidence Distribution by Model
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [group["Confidence"].values for name, group in predictions_df.groupby("Model")],
        labels=predictions_df["Model"].unique(),
    )
    plt.title("Confidence Distribution by Model")
    plt.ylabel("Confidence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "confidence_distribution.png")
    plt.close()

    # 2. Accuracy vs Confidence
    plt.figure(figsize=(12, 6))
    for model in predictions_df["Model"].unique():
        model_data = predictions_df[predictions_df["Model"] == model]
        conf_bins = np.linspace(0, 1, 11)
        accuracies = []
        conf_centers = []

        for i in range(len(conf_bins) - 1):
            mask = (model_data["Confidence"] >= conf_bins[i]) & (
                model_data["Confidence"] < conf_bins[i + 1]
            )
            if mask.any():
                acc = (
                    model_data[mask]["Prediction"] == model_data[mask]["Actual"]
                ).mean()
                accuracies.append(acc)
                conf_centers.append((conf_bins[i] + conf_bins[i + 1]) / 2)

        plt.plot(conf_centers, accuracies, marker="o", label=model)

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Confidence by Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_vs_confidence.png")
    plt.close()

    # 3. Prediction Distribution by Confidence Level
    plt.figure(figsize=(12, 6))
    conf_level_counts = (
        predictions_df.groupby(["Model", "Confidence Level"])
        .size()
        .unstack(fill_value=0)
    )
    conf_level_counts.plot(kind="bar", stacked=True)
    plt.title("Prediction Distribution by Confidence Level")
    plt.xlabel("Model")
    plt.ylabel("Number of Predictions")
    plt.legend(title="Confidence Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "prediction_distribution.png")
    plt.close()


def print_comparative_results(results):
    """Print comparative analysis of all models"""
    logger.info("\n=== NFL Model Comparison Analysis ===")

    # Print overall comparison
    logger.info("\nOverall Model Performance:")
    headers = ["Model", "Accuracy", "Avg Confidence", "Total Games"]
    row_format = "{:<15} {:<12} {:<15} {:<12}"
    logger.info(row_format.format(*headers))
    logger.info("-" * 54)

    for model_name, model_results in results.items():
        overall = model_results["overall"]
        logger.info(
            row_format.format(
                model_name,
                f"{overall['accuracy']:.1%}",
                f"{overall['avg_confidence']:.2f}",
                overall["total"],
            )
        )

    # Print confidence level breakdown
    logger.info("\nAccuracy by Confidence Level:")
    headers = ["Model", "Elite", "Great", "Good", "Low"]
    row_format = "{:<15} {:<12} {:<12} {:<12} {:<12}"
    logger.info(row_format.format(*headers))
    logger.info("-" * 63)

    for model_name, model_results in results.items():
        accuracies = []
        for level in ["Elite", "Great", "Good", "Low"]:
            if level in model_results["by_category"]:
                acc = f"{model_results['by_category'][level]['accuracy']:.1%}"
            else:
                acc = "N/A"
            accuracies.append(acc)

        logger.info(row_format.format(model_name, *accuracies))

    # Print prediction counts by confidence level
    logger.info("\nPrediction Counts by Confidence Level:")
    headers = ["Model", "Elite", "Great", "Good", "Low"]
    logger.info(row_format.format(*headers))
    logger.info("-" * 63)

    for model_name, model_results in results.items():
        counts = []
        for level in ["Elite", "Great", "Good", "Low"]:
            if level in model_results["by_category"]:
                count = str(model_results["by_category"][level]["total"])
            else:
                count = "0"
            counts.append(count)

        logger.info(row_format.format(model_name, *counts))


def main():
    """Main function to run the analysis"""
    # Load data
    train_data, test_data = load_data()

    # Train models
    models = train_models(train_data)

    if not models:
        logger.error("No models were successfully trained. Exiting.")
        return

    # Analyze predictions
    results, predictions_df = analyze_predictions(models, test_data)

    if results is None:
        logger.error("No predictions were made. Exiting.")
        return

    # Print comparative results
    print_comparative_results(results)

    # Create visualization plots
    plot_confidence_comparison(predictions_df, "model_comparison_plots")

    logger.info(
        "\nVisualization plots have been saved to the 'model_comparison_plots' directory"
    )


if __name__ == "__main__":
    main()

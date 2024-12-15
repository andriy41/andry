import joblib
import pandas as pd
import numpy as np
from models.unified_predictor import UnifiedPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from datetime import datetime


def analyze_model_agreement(
    predictions: Dict[str, List[Dict[str, float]]]
) -> pd.DataFrame:
    """
    Analyze how different models agree or disagree on predictions.

    Args:
        predictions: Dictionary of model predictions

    Returns:
        DataFrame with agreement analysis
    """
    agreement_data = []

    for game_idx in range(len(predictions["all"])):
        game_preds = {}
        for model_name, model_preds in predictions.items():
            game_preds[model_name] = model_preds[game_idx]["home_win_prob"]

        # Calculate agreement metrics
        pred_values = list(game_preds.values())
        avg_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        max_diff = max(pred_values) - min(pred_values)

        agreement_data.append(
            {
                "game_idx": game_idx,
                "avg_home_win_prob": avg_pred,
                "prediction_std": std_pred,
                "max_model_disagreement": max_diff,
                "high_confidence": avg_pred > 0.7 or avg_pred < 0.3,
                "model_agreement": std_pred < 0.1,
            }
        )

    return pd.DataFrame(agreement_data)


def visualize_predictions(
    predictions: Dict[str, List[Dict[str, float]]],
    future_games: pd.DataFrame,
    output_dir: str = "outputs",
):
    """
    Create visualizations of predictions and save them.

    Args:
        predictions: Dictionary of model predictions
        future_games: DataFrame of games to predict
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    plot_data = []
    for game_idx in range(len(predictions["all"])):
        game = future_games.iloc[game_idx]
        for model_name, model_preds in predictions.items():
            plot_data.append(
                {
                    "model": model_name,
                    "game": f"{game['home_team']} vs {game['away_team']}",
                    "home_win_prob": model_preds[game_idx]["home_win_prob"],
                }
            )

    plot_df = pd.DataFrame(plot_data)

    # Create model comparison plot
    plt.figure(figsize=(15, 8))
    sns.barplot(data=plot_df, x="game", y="home_win_prob", hue="model")
    plt.xticks(rotation=45, ha="right")
    plt.title("Model Predictions Comparison")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    plt.close()

    # Create confidence distribution plot
    plt.figure(figsize=(10, 6))
    for model in predictions.keys():
        model_probs = [p["home_win_prob"] for p in predictions[model]]
        sns.kdeplot(model_probs, label=model)
    plt.title("Distribution of Prediction Confidences")
    plt.xlabel("Home Win Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_distribution.png")
    plt.close()


def generate_prediction_report(
    predictions: Dict[str, List[Dict[str, float]]],
    future_games: pd.DataFrame,
    agreement_analysis: pd.DataFrame,
) -> str:
    """
    Generate a detailed prediction report.

    Args:
        predictions: Dictionary of model predictions
        future_games: DataFrame of games to predict
        agreement_analysis: DataFrame with model agreement analysis

    Returns:
        Formatted report string
    """
    report = ["NFL Game Predictions Report", "=" * 30, ""]

    for game_idx in range(len(predictions["all"])):
        game = future_games.iloc[game_idx]
        agreement = agreement_analysis.iloc[game_idx]

        report.append(
            f"\nGame {game_idx + 1}: {game['home_team']} vs {game['away_team']}"
        )
        report.append("-" * 50)

        # Add prediction from each model
        for model_name, model_preds in predictions.items():
            home_win_prob = model_preds[game_idx]["home_win_prob"]
            home_team_pred = "WIN" if home_win_prob > 0.5 else "LOSS"
            report.append(
                f"{model_name.upper()} Model: {home_team_pred} ({home_win_prob:.1%} home win probability)"
            )

        # Add agreement analysis
        report.append(f"\nModel Agreement Analysis:")
        report.append(
            f"Average Home Win Probability: {agreement['avg_home_win_prob']:.1%}"
        )
        report.append(
            f"Model Consensus: {'High' if agreement['model_agreement'] else 'Low'}"
        )
        report.append(
            f"Prediction Confidence: {'High' if agreement['high_confidence'] else 'Low'}"
        )

        report.append("\n" + "=" * 50)

    return "\n".join(report)


def main():
    try:
        # Load the models
        models = {
            "statistical": joblib.load("models/nfl_predictor_statistical.joblib"),
            "basic": joblib.load("models/nfl_predictor_basic.joblib"),
            "vedic": joblib.load("models/nfl_predictor_vedic.joblib"),
            "all": joblib.load("models/nfl_predictor_all.joblib"),
        }

        # Initialize predictor
        predictor = UnifiedPredictor(models)

        # Load future games
        future_games = pd.read_csv("data/future_games.csv")

        # Make predictions
        predictions = predictor.predict_games("data/future_games.csv")

        # Analyze model agreement
        agreement_analysis = analyze_model_agreement(predictions)

        # Generate visualizations
        visualize_predictions(predictions, future_games)

        # Generate and save report
        report = generate_prediction_report(
            predictions, future_games, agreement_analysis
        )
        with open("outputs/prediction_report.txt", "w") as f:
            f.write(report)

        print(
            "Predictions complete! Check the outputs directory for detailed analysis."
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()

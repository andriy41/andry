import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from models.prediction_system import NFLPredictionSystem


def analyze_patterns(df):
    """Analyze patterns in the historical data."""
    print("\n=== Historical Data Analysis ===")

    # Overall home field advantage
    home_wins = (df["home_score"] > df["away_score"]).mean()
    print(f"\nHome team win percentage: {home_wins:.3f}")

    # Scoring patterns
    print("\nScoring Statistics:")
    print(f"Average home score: {df['home_score'].mean():.1f}")
    print(f"Average away score: {df['away_score'].mean():.1f}")
    print(f"Average total score: {(df['home_score'] + df['away_score']).mean():.1f}")

    # Season timing effects
    df["month"] = pd.to_datetime(df["date"]).dt.month
    monthly_home_wins = df.groupby("month").apply(
        lambda x: (x["home_score"] > x["away_score"]).mean()
    )
    print("\nHome Win % by Month:")
    print(monthly_home_wins)

    # Weather effects
    print("\nWeather Effects on Scoring:")
    cold_games = df[df["temperature"] < 40]
    warm_games = df[df["temperature"] > 75]
    print(
        f"Cold games (<40°F) avg total score: {(cold_games['home_score'] + cold_games['away_score']).mean():.1f}"
    )
    print(
        f"Warm games (>75°F) avg total score: {(warm_games['home_score'] + warm_games['away_score']).mean():.1f}"
    )

    # Playoff vs Regular Season
    playoff_games = df[df["is_playoff"] == 1]
    regular_games = df[df["is_playoff"] == 0]
    print("\nPlayoff vs Regular Season:")
    print(
        f"Playoff games avg margin: {abs(playoff_games['home_score'] - playoff_games['away_score']).mean():.1f}"
    )
    print(
        f"Regular games avg margin: {abs(regular_games['home_score'] - regular_games['away_score']).mean():.1f}"
    )

    return {
        "home_advantage": home_wins,
        "monthly_patterns": monthly_home_wins,
        "weather_effects": {
            "cold_games_score": (
                cold_games["home_score"] + cold_games["away_score"]
            ).mean(),
            "warm_games_score": (
                warm_games["home_score"] + warm_games["away_score"]
            ).mean(),
        },
    }


def evaluate_predictions(prediction_system, test_data):
    """Evaluate prediction accuracy on test data."""
    print("\n=== Model Evaluation ===")

    predictions = []
    actuals = []

    for _, game in test_data.iterrows():
        try:
            # Prepare game data
            game_data = pd.DataFrame(
                [
                    {
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "date": game["date"],
                        "season": game["season"],
                        "week": game["week"],
                        "temperature": game["temperature"],
                        "wind_speed": game["wind_speed"],
                        "precipitation": game["precipitation"],
                        "is_playoff": game["is_playoff"],
                    }
                ]
            )

            # Get model predictions
            X = prediction_system._prepare_features(game_data)
            probs = prediction_system.rf_win.predict_proba(X)[0]

            # Get predicted winner
            predicted_winner = (
                game["home_team"] if probs[1] > 0.5 else game["away_team"]
            )
            actual_winner = (
                game["home_team"]
                if game["home_score"] > game["away_score"]
                else game["away_team"]
            )

            predictions.append(predicted_winner)
            actuals.append(actual_winner)

        except Exception as e:
            print(f"Error predicting game: {str(e)}")
            continue

    if predictions and actuals:
        # Calculate accuracy
        accuracy = accuracy_score(actuals, predictions)
        print(f"\nOverall Prediction Accuracy: {accuracy:.3f}")

        # Detailed classification report
        print("\nDetailed Classification Report:")
        unique_teams = sorted(list(set(predictions + actuals)))
        print(classification_report(actuals, predictions, target_names=unique_teams))

        return accuracy, predictions, actuals
    else:
        print("\nNo valid predictions were made.")
        return 0.0, [], []


def main():
    # Load historical data
    from data.historical_games import generate_historical_games

    historical_data = generate_historical_games(2018, 2023)

    # Analyze patterns
    patterns = analyze_patterns(historical_data)

    # Split data for training and testing
    train_data, test_data = train_test_split(
        historical_data, test_size=0.2, shuffle=True, random_state=42
    )

    # Initialize and train prediction system
    prediction_system = NFLPredictionSystem()
    prediction_system.train(train_data)

    # Evaluate predictions
    accuracy, predictions, actuals = evaluate_predictions(prediction_system, test_data)

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot 1: Monthly Home Win Percentage
    plt.subplot(1, 2, 1)
    patterns["monthly_patterns"].plot(kind="bar")
    plt.title("Home Win % by Month")
    plt.xlabel("Month")
    plt.ylabel("Win Percentage")

    # Plot 2: Prediction Accuracy by Team
    plt.subplot(1, 2, 2)
    team_accuracy = {}
    for pred, actual in zip(predictions, actuals):
        if actual not in team_accuracy:
            team_accuracy[actual] = {"correct": 0, "total": 0}
        team_accuracy[actual]["total"] += 1
        if pred == actual:
            team_accuracy[actual]["correct"] += 1

    team_acc_values = {
        team: data["correct"] / data["total"] for team, data in team_accuracy.items()
    }
    plt.bar(team_acc_values.keys(), team_acc_values.values())
    plt.title("Prediction Accuracy by Team")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("prediction_analysis.png")

    # Save results
    results = pd.DataFrame({"Actual": actuals, "Predicted": predictions})
    results.to_csv("prediction_results.csv", index=False)

    print(
        "\nAnalysis complete! Check prediction_analysis.png and prediction_results.csv for detailed results."
    )


if __name__ == "__main__":
    main()

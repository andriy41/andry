"""
Evaluate the Enhanced Total Model on past NFL games.
"""
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.total_prediction.enhanced_total_model import EnhancedTotalModel
from sklearn.preprocessing import LabelEncoder


def analyze_season_performance(test_data, predictions, confidence):
    """Analyze model performance by season."""
    seasons = test_data["season"].unique()
    season_metrics = []

    for season in sorted(seasons):
        mask = test_data["season"] == season
        season_preds = predictions[mask]
        season_actuals = test_data.loc[mask, "total_points"]
        season_conf = confidence[mask]

        # Calculate metrics
        mae = mean_absolute_error(season_actuals, season_preds)
        mse = mean_squared_error(season_actuals, season_preds)
        r2 = r2_score(season_actuals, season_preds)

        # High confidence predictions
        high_conf_mask = season_conf > np.percentile(confidence, 75)
        high_conf_mae = (
            mean_absolute_error(
                season_actuals[high_conf_mask], season_preds[high_conf_mask]
            )
            if any(high_conf_mask)
            else None
        )

        # Average confidence
        avg_conf = np.mean(season_conf)

        # Average total points
        avg_total = np.mean(season_actuals)

        season_metrics.append(
            {
                "season": season,
                "games": len(season_actuals),
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "high_conf_mae": high_conf_mae,
                "avg_confidence": avg_conf,
                "avg_total_points": avg_total,
            }
        )

    return pd.DataFrame(season_metrics)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling categorical variables and missing values."""
    data = data.copy()

    # Convert boolean columns to int
    bool_cols = ["is_dome", "is_primetime", "is_division_game", "div_game"]
    for col in bool_cols:
        if col in data.columns:
            data[col] = data[col].astype(int)

    # Fill missing numeric values with mean
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].mean())

    return data


def evaluate_model():
    # Load and process data
    data = pd.read_csv("data/enhanced_historical_games.csv")

    # Convert gameday to datetime
    data["gameday"] = pd.to_datetime(data["gameday"])

    # Sort by date
    data = data.sort_values("gameday")

    # Print dataset information
    print("\nDataset Information:")
    print(f"Total number of games: {len(data)}")
    print(
        f"Date range: {data['gameday'].min().strftime('%Y-%m-%d')} to {data['gameday'].max().strftime('%Y-%m-%d')}"
    )
    print(f"Number of seasons: {data['season'].nunique()}")

    # Use the last 20% of games as test set
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"\nTraining set size: {len(train_data)} games")
    print(f"Test set size: {len(test_data)} games")

    # Get feature columns (excluding target and metadata columns)
    exclude_cols = [
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "total_points",
        "home_win",
        "season",
        "week",
        "game_id",
        "game_type",
        "weekday",
        "gametime",
        "location",
        "result",
        "total",
        "overtime",
        "old_game_id",
        "gsis",
        "nfl_detail_id",
        "pfr",
        "pff",
        "espn",
        "ftn",
        "away_qb_name",
        "home_qb_name",
        "away_coach",
        "home_coach",
        "referee",
        "stadium",
        "surface",
        "roof",
        "away_qb_id",
        "home_qb_id",
        "stadium_id",
    ]

    # Get only numeric columns
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"\nNumber of features used: {len(feature_cols)}")

    # Preprocess data
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Train the model
    model = EnhancedTotalModel()
    model.fit(train_data[feature_cols], train_data["total_points"])

    # Make predictions on test set
    predictions, confidence = model.predict_with_confidence(test_data[feature_cols])

    # Calculate overall metrics
    mse = mean_squared_error(test_data["total_points"], predictions)
    mae = mean_absolute_error(test_data["total_points"], predictions)
    r2 = r2_score(test_data["total_points"], predictions)

    print("\nOverall Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")

    # Analyze performance by season
    season_metrics = analyze_season_performance(test_data, predictions, confidence)

    print("\nPerformance by Season:")
    print("-" * 100)
    print("Season  Games  MAE    MSE     R²      High Conf MAE  Avg Conf  Avg Total")
    print("-" * 100)

    for _, row in season_metrics.iterrows():
        high_conf_mae = (
            f"{row['high_conf_mae']:.2f}" if pd.notnull(row["high_conf_mae"]) else "N/A"
        )
        print(
            f"{int(row['season'])}  {int(row['games']):5d}  {row['mae']:.2f}  {row['mse']:.2f}  {row['r2']:.3f}  {high_conf_mae:12s}  {row['avg_confidence']:.2f}  {row['avg_total_points']:.1f}"
        )

    # Print last 10 predictions with actual values
    print("\nLast 10 Games Predictions:")
    print("Date\t\tTeams\t\t\tPredicted\tActual\tDiff\tConfidence")
    print("-" * 90)

    for i in range(-10, 0):
        game = test_data.iloc[i]
        pred = predictions[i]
        conf = confidence[i]
        diff = abs(pred - game["total_points"])
        date_str = game["gameday"].strftime("%Y-%m-%d")
        print(
            f"{date_str}\t{game['away_team']} @ {game['home_team']}\t\t{pred:.1f}\t\t{game['total_points']}\t{diff:.1f}\t{conf:.2f}"
        )

    # Analyze high confidence predictions
    high_conf_mask = confidence > np.percentile(confidence, 75)
    high_conf_predictions = predictions[high_conf_mask]
    high_conf_actuals = test_data["total_points"].iloc[high_conf_mask]

    high_conf_mae = mean_absolute_error(high_conf_actuals, high_conf_predictions)
    print(f"\nHigh Confidence Predictions MAE: {high_conf_mae:.2f}")
    print(f"Number of high confidence predictions: {len(high_conf_predictions)}")

    # Print feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = []
    for name, model in model.models.items():
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            for feat, imp in zip(feature_cols, importance):
                feature_importance.append((feat, imp))

    if feature_importance:
        avg_importance = (
            pd.DataFrame(feature_importance, columns=["feature", "importance"])
            .groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )

        for feat, imp in avg_importance[:10].items():
            print(f"{feat}: {imp:.4f}")


if __name__ == "__main__":
    evaluate_model()

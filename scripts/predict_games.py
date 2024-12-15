import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/nfl_prediction_model.joblib")

# Sample upcoming games data
upcoming_games = pd.DataFrame(
    {
        "home_team": ["Bills", "Cowboys", "Raiders"],
        "away_team": ["Chiefs", "Eagles", "Chargers"],
        "home_points_per_game": [24.5, 26.3, 21.8],
        "away_points_per_game": [23.1, 25.8, 22.4],
        "home_yards_per_play": [5.8, 5.9, 5.2],
        "away_yards_per_play": [5.6, 5.7, 5.4],
        "home_turnover_diff": [3, 5, -1],
        "away_turnover_diff": [2, 4, 1],
        "home_win_pct": [0.65, 0.72, 0.45],
        "away_win_pct": [0.68, 0.75, 0.55],
        "spread": [-2.5, -1.5, 3.0],
        "over_under": [47.5, 51.5, 44.5],
    }
)

# Make predictions
predictions = model.predict(upcoming_games[model.feature_names_in_])

# Display results
upcoming_games["predicted_total"] = predictions.round(1)
print("\nPredicted Game Totals:")
print(upcoming_games[["home_team", "away_team", "predicted_total"]])

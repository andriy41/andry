import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load historical NFL data using the correct path
data = pd.read_csv("nfl_data.csv")  # Using the file we created earlier

# Define features
features = [
    "home_points_per_game",
    "away_points_per_game",
    "home_yards_per_play",
    "away_yards_per_play",
    "home_turnover_diff",
    "away_turnover_diff",
    "home_win_pct",
    "away_win_pct",
    "spread",
    "over_under",
]

# Prepare data
X = data[features]
y = data["total_points"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
)

model.fit(X_train, y_train)

# Create models directory if it doesn't exist
import os

os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/nfl_prediction_model.joblib")

print("Model trained and saved successfully!")

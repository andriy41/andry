import pandas as pd
import numpy as np

# Create sample NFL data
n_samples = 1000
data = {
    "home_points_per_game": np.random.normal(24, 5, n_samples),
    "away_points_per_game": np.random.normal(21, 5, n_samples),
    "home_yards_per_play": np.random.normal(5.5, 0.8, n_samples),
    "away_yards_per_play": np.random.normal(5.2, 0.8, n_samples),
    "home_turnover_diff": np.random.randint(-3, 4, n_samples),
    "away_turnover_diff": np.random.randint(-3, 4, n_samples),
    "home_win_pct": np.random.uniform(0.2, 0.8, n_samples),
    "away_win_pct": np.random.uniform(0.2, 0.8, n_samples),
    "spread": np.random.normal(0, 7, n_samples),
    "over_under": np.random.normal(47, 4, n_samples),
    "total_points": np.random.normal(45, 10, n_samples),
}

df = pd.DataFrame(data)
df.to_csv("nfl_data.csv", index=False)

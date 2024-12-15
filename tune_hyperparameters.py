import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def tune_model_parameters(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestRegressor()  # Changed to Regressor to match the prediction task
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_absolute_error")
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


def main():
    # Load data
    data = pd.read_csv("nfl_data.csv")

    # Key predictive features
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

    X = data[features]
    y = data["total_points"]

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get optimized parameters
    best_params = tune_model_parameters(X_train, y_train)
    print("Best parameters found:", best_params)

    # Train with optimized parameters
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"New Total MAE: {mae:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)


if __name__ == "__main__":
    main()

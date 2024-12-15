import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime

from models.total_prediction.stats_total_model import StatsTotalModel
from models.total_prediction.ml_total_model import MLTotalModel
from models.total_prediction.astro_total_model import AstroTotalModel
from models.total_prediction.neuro_total_model import NeuroTotalModel
from models.total_prediction.vedic_total_model import VedicTotalModel
from models.total_prediction.total_model import TotalPredictionModel


def generate_synthetic_data(n_samples=1000):
    """Generate synthetic NFL game data for training"""
    np.random.seed(42)

    data = []
    for _ in range(n_samples):
        # Generate base game stats
        home_off = np.random.normal(24, 5)  # Home offense rating
        away_off = np.random.normal(21, 5)  # Away offense rating
        home_def = np.random.normal(20, 4)  # Home defense rating
        away_def = np.random.normal(23, 4)  # Away defense rating

        game = {
            # Offensive stats
            "home_points_per_game": home_off,
            "away_points_per_game": away_off,
            "home_points_allowed": home_def,
            "away_points_allowed": away_def,
            "home_yards_per_game": np.random.normal(350, 50),
            "away_yards_per_game": np.random.normal(330, 50),
            "home_yards_allowed": np.random.normal(330, 40),
            "away_yards_allowed": np.random.normal(340, 40),
            "home_pass_yards_per_game": np.random.normal(240, 40),
            "away_pass_yards_per_game": np.random.normal(230, 40),
            "home_rush_yards_per_game": np.random.normal(110, 30),
            "away_rush_yards_per_game": np.random.normal(100, 30),
            # Efficiency stats
            "home_third_down_conv": np.random.normal(0.4, 0.05),
            "away_third_down_conv": np.random.normal(0.38, 0.05),
            "home_fourth_down_conv": np.random.normal(0.5, 0.1),
            "away_fourth_down_conv": np.random.normal(0.48, 0.1),
            "home_time_of_possession": np.random.normal(30, 2),
            "away_time_of_possession": np.random.normal(30, 2),
            "home_turnover_margin": np.random.normal(0, 1),
            "away_turnover_margin": np.random.normal(0, 1),
            # Game conditions
            "is_dome": np.random.choice([True, False]),
            "temperature": np.random.normal(65, 15),
            "wind_speed": np.random.normal(8, 4),
            "is_playoff": np.random.choice([True, False], p=[0.1, 0.9]),
            # Target variables (to be calculated)
            "total_points": None,
            "spread": None,
            "home_win": None,
        }

        # Calculate realistic outcomes based on team ratings
        home_score = (
            home_off * 0.6 + (100 - away_def) * 0.4 + np.random.normal(0, 3)
        )  # Add some randomness
        away_score = away_off * 0.6 + (100 - home_def) * 0.4 + np.random.normal(0, 3)

        # Add home field advantage
        home_score += 3 if not game["is_playoff"] else 2

        # Weather effects
        if not game["is_dome"]:
            if game["temperature"] < 32:  # Cold weather
                home_score -= 1
                away_score -= 2  # Away team affected more
            if game["wind_speed"] > 15:  # High wind
                home_score -= 1
                away_score -= 1

        # Set target variables
        game["total_points"] = home_score + away_score
        game["spread"] = home_score - away_score
        game["home_win"] = int(home_score > away_score)

        data.append(game)

    return pd.DataFrame(data)


def train_models():
    """Train total prediction models"""
    print("Generating synthetic training data...")
    data = generate_synthetic_data(n_samples=1000)

    # Initialize models
    stats_model = StatsTotalModel()
    ml_model = MLTotalModel()
    astro_model = AstroTotalModel()
    neuro_model = NeuroTotalModel()
    vedic_model = VedicTotalModel()
    total_model = TotalPredictionModel()

    # Train StatsTotalModel
    print("\nTraining StatsTotalModel...")
    stats_model.train(data)

    # Train MLTotalModel
    print("\nTraining MLTotalModel...")
    ml_model.train(data)

    # Train AstroTotalModel
    print("\nTraining AstroTotalModel...")
    astro_model.train(data)

    # Train NeuroTotalModel
    print("\nTraining NeuroTotalModel...")
    neuro_model.train(data)

    # Train VedicTotalModel
    print("\nTraining VedicTotalModel...")
    vedic_model.train(data)

    # Train TotalPredictionModel (ensemble)
    print("\nTraining TotalPredictionModel (ensemble)...")
    total_model.train(data)

    # Save models
    os.makedirs("models/trained", exist_ok=True)
    joblib.dump(stats_model, "models/trained/stats_total_model.joblib")
    joblib.dump(ml_model, "models/trained/ml_total_model.joblib")
    joblib.dump(astro_model, "models/trained/astro_total_model.joblib")
    joblib.dump(neuro_model, "models/trained/neuro_total_model.joblib")
    joblib.dump(vedic_model, "models/trained/vedic_total_model.joblib")
    joblib.dump(total_model, "models/trained/total_model.joblib")

    print("\nTraining complete! Models saved to models/trained/")

    # Test predictions
    test_game = generate_synthetic_data(n_samples=1).iloc[0].to_dict()

    stats_pred = stats_model.predict(test_game)
    ml_pred = ml_model.predict(test_game)
    astro_pred = astro_model.predict(test_game)
    neuro_pred = neuro_model.predict(test_game)
    vedic_pred = vedic_model.predict(test_game)
    total_pred = total_model.predict(test_game)

    print(f"\nTest Predictions:")

    print(f"StatsTotalModel:")
    print(
        f"  Total Points: {stats_pred['total_points']:.1f} ({stats_pred['total_prediction']})"
    )
    print(f"  Spread: {stats_pred['spread']:.1f} ({stats_pred['spread_prediction']})")
    print(f"  Home Win Probability: {stats_pred['home_win_probability']:.1f}%")
    print(f"  Overall Confidence: {stats_pred['confidence']:.1f}%")
    print(f"  Explanation: {stats_pred['explanation']}")

    print(f"\nMLTotalModel:")
    print(
        f"  Total Points: {ml_pred['total_points']:.1f} ({ml_pred['total_prediction']})"
    )
    print(f"  Spread: {ml_pred['spread']:.1f} ({ml_pred['spread_prediction']})")
    print(f"  Home Win Probability: {ml_pred['home_win_probability']:.1f}%")
    print(f"  Overall Confidence: {ml_pred['confidence']:.1f}%")
    print(f"  Explanation: {ml_pred['explanation']}")

    print(f"\nAstroTotalModel:")
    print(
        f"  Total Points: {astro_pred['total_points']:.1f} ({astro_pred['total_prediction']})"
    )
    print(f"  Spread: {astro_pred['spread']:.1f} ({astro_pred['spread_prediction']})")
    print(f"  Home Win Probability: {astro_pred['home_win_probability']:.1f}%")
    print(f"  Overall Confidence: {astro_pred['confidence']:.1f}%")
    print(f"  Explanation: {astro_pred['explanation']}")

    print(f"\nNeuroTotalModel:")
    print(
        f"  Total Points: {neuro_pred['total_points']:.1f} ({neuro_pred['total_prediction']})"
    )
    print(f"  Spread: {neuro_pred['spread']:.1f} ({neuro_pred['spread_prediction']})")
    print(f"  Home Win Probability: {neuro_pred['home_win_probability']:.1f}%")
    print(f"  Overall Confidence: {neuro_pred['confidence']:.1f}%")
    print(f"  Explanation: {neuro_pred['explanation']}")

    print(f"\nVedicTotalModel:")
    print(
        f"  Total Points: {vedic_pred['total_points']:.1f} ({vedic_pred['total_prediction']})"
    )
    print(f"  Spread: {vedic_pred['spread']:.1f} ({vedic_pred['spread_prediction']})")
    print(f"  Home Win Probability: {vedic_pred['home_win_probability']:.1f}%")
    print(f"  Overall Confidence: {vedic_pred['confidence']:.1f}%")
    print(f"  Explanation: {vedic_pred['explanation']}")

    print(f"\nTotalPredictionModel (Ensemble):")
    print(
        f"  Total Points: {total_pred['total_points']:.1f} ({total_pred['total_prediction']})"
    )
    print(f"  Spread: {total_pred['spread']:.1f} ({total_pred['spread_prediction']})")
    print(f"  Home Win Probability: {total_pred['home_win_probability']:.1f}%")
    print(f"  Overall Confidence: {total_pred['confidence']:.1f}%")
    print(f"  Explanation: {total_pred['explanation']}")

    print(f"\nActual Values:")
    print(f"  Total Points: {test_game['total_points']:.1f}")
    print(f"  Spread: {test_game['spread']:.1f}")
    print(f"  Home Win: {'Yes' if test_game['home_win'] else 'No'}")


if __name__ == "__main__":
    train_models()

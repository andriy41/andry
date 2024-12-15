import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
import os


def load_model():
    """Load the trained enhanced model"""
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "enhanced_total_model.joblib"
    )
    return joblib.load(model_path)


def predict_game(
    home_team,
    away_team,
    is_primetime=False,
    is_division_game=False,
    is_dome=False,
    temperature=None,
    wind_speed=None,
):
    """Make a prediction for a specific game"""
    # Load the model
    model = load_model()

    # Create feature dictionary
    game_features = {
        "home_score": model.home_avg_score if hasattr(model, "home_avg_score") else 24,
        "away_score": model.away_avg_score if hasattr(model, "away_avg_score") else 21,
        "is_primetime": is_primetime,
        "is_division_game": is_division_game,
        "is_dome": is_dome,
        "temperature": temperature,
        "wind_speed": wind_speed,
    }

    # Convert to DataFrame
    X = pd.DataFrame([game_features])

    # Make prediction
    prediction, confidence = model.predict(X)

    return {
        "predicted_total": round(prediction[0], 1),
        "confidence": round(confidence[0] * 100, 1),
        "home_team": home_team,
        "away_team": away_team,
        "features_used": game_features,
    }


def main():
    """Test predictions on some upcoming games"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Example predictions
    games = [
        {
            "home_team": "DAL",
            "away_team": "PHI",
            "is_primetime": True,
            "is_division_game": True,
            "is_dome": True,
        },
        {
            "home_team": "GB",
            "away_team": "CHI",
            "is_primetime": False,
            "is_division_game": True,
            "is_dome": False,
            "temperature": 35,
            "wind_speed": 12,
        },
    ]

    for game in games:
        result = predict_game(
            home_team=game["home_team"],
            away_team=game["away_team"],
            is_primetime=game["is_primetime"],
            is_division_game=game["is_division_game"],
            is_dome=game["is_dome"],
            temperature=game.get("temperature"),
            wind_speed=game.get("wind_speed"),
        )

        print(f"\nPrediction for {result['home_team']} vs {result['away_team']}:")
        print(f"Predicted Total Points: {result['predicted_total']}")
        print(f"Confidence: {result['confidence']}%")
        print("Features Used:")
        for k, v in result["features_used"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

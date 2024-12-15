#!/usr/bin/env python3

import pandas as pd
from datetime import datetime, timedelta
import pytz
from models.ensemble_predictor import NFLEnsemblePredictor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_historical_games():
    """Load historical NFL games for training"""
    df = pd.read_csv("data/nfl_games.csv")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize("UTC")

    # Filter for completed games (those with scores)
    historical = df.dropna(subset=["home_score", "away_score"]).copy()

    # Add win/loss labels
    historical["home_win"] = (
        historical["home_score"] > historical["away_score"]
    ).astype(int)

    return historical


def load_upcoming_games():
    """Load upcoming NFL games from CSV"""
    df = pd.read_csv("data/nfl_games.csv")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize("UTC")

    # Filter for upcoming games (next 7 days)
    now = pd.Timestamp.now(tz="UTC")
    upcoming = df[
        (df["game_date"] > now) & (df["game_date"] <= now + pd.Timedelta(days=7))
    ].copy()

    return upcoming


def get_stadium_location(team):
    """Get stadium coordinates for a team"""
    # Stadium coordinates
    coordinates = {
        "SF": {"latitude": 37.7749, "longitude": -122.4194},
        "SEA": {"latitude": 47.6062, "longitude": -122.3321},
        "KC": {"latitude": 39.0997, "longitude": -94.5786},
        "LAR": {"latitude": 34.0522, "longitude": -118.2437},
        "DEN": {"latitude": 39.7392, "longitude": -104.9903},
        "LV": {"latitude": 36.1699, "longitude": -115.1398},
        "ARI": {"latitude": 33.4484, "longitude": -112.0740},
        "LAC": {"latitude": 34.0522, "longitude": -118.2437},
        "DAL": {"latitude": 32.7767, "longitude": -96.7970},
        "NYG": {"latitude": 40.7128, "longitude": -74.0060},
        "PHI": {"latitude": 39.9526, "longitude": -75.1652},
        "WAS": {"latitude": 38.9072, "longitude": -77.0369},
        "GB": {"latitude": 44.5133, "longitude": -88.0133},
        "MIN": {"latitude": 44.9778, "longitude": -93.2650},
        "CHI": {"latitude": 41.8781, "longitude": -87.6298},
        "DET": {"latitude": 42.3314, "longitude": -83.0458},
        "NO": {"latitude": 29.9511, "longitude": -90.0715},
        "TB": {"latitude": 27.9506, "longitude": -82.4572},
        "ATL": {"latitude": 33.7490, "longitude": -84.3880},
        "CAR": {"latitude": 35.2271, "longitude": -80.8431},
        "NE": {"latitude": 42.3601, "longitude": -71.0589},
        "NYJ": {"latitude": 40.7128, "longitude": -74.0060},
        "BUF": {"latitude": 42.8864, "longitude": -78.8784},
        "MIA": {"latitude": 25.7617, "longitude": -80.1918},
        "BAL": {"latitude": 39.2904, "longitude": -76.6122},
        "PIT": {"latitude": 40.4406, "longitude": -79.9959},
        "CLE": {"latitude": 41.4993, "longitude": -81.6944},
        "CIN": {"latitude": 39.1031, "longitude": -84.5120},
        "TEN": {"latitude": 36.1627, "longitude": -86.7816},
        "IND": {"latitude": 39.7684, "longitude": -86.1581},
        "HOU": {"latitude": 29.7604, "longitude": -95.3698},
        "JAX": {"latitude": 30.3322, "longitude": -81.6557},
    }
    return coordinates.get(team, {"latitude": 0, "longitude": 0})


def prepare_game_data(game):
    """Prepare game data dictionary for prediction"""
    return {
        "game_time": game["game_date"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "stadium_location": get_stadium_location(game["home_team"]),
        # Add historical stats
        "home_stats": get_team_stats(game["home_team"]),
        "away_stats": get_team_stats(game["away_team"]),
    }


def get_team_stats(team):
    """Get team statistics for prediction"""
    return {
        "win_pct": 0.5,  # Placeholder
        "points_scored_avg": 24.0,
        "points_allowed_avg": 24.0,
        "yards_per_game": 350.0,
        "yards_allowed_per_game": 350.0,
        "pass_yards_per_game": 250.0,
        "rush_yards_per_game": 100.0,
        "third_down_pct": 0.4,
        "red_zone_pct": 0.6,
        "turnover_diff": 0,
        "sacks": 25,
        "injuries_impact": 0.0,
    }


def main():
    # Initialize predictor
    predictor = NFLEnsemblePredictor()

    # Load and prepare historical data for training
    historical_games = load_historical_games()
    logger.info(f"Loaded {len(historical_games)} historical games for training")

    # Train each model
    training_data = [prepare_game_data(game) for _, game in historical_games.iterrows()]
    labels = historical_games["home_win"].values

    for model in predictor.models:
        try:
            logger.info(f"Training {model.__class__.__name__}...")
            model.train({"games": training_data, "labels": labels})
        except Exception as e:
            logger.error(f"Error training {model.__class__.__name__}: {str(e)}")

    # Load upcoming games
    upcoming_games = load_upcoming_games()
    logger.info(f"Found {len(upcoming_games)} upcoming games")

    # Make predictions for each game
    for _, game in upcoming_games.iterrows():
        game_data = prepare_game_data(game)

        try:
            prediction = predictor.predict(game_data)

            # Format confidence as percentage
            confidence = f"{prediction['confidence_score']*100:.1f}%"

            # Get individual model predictions
            model_predictions = prediction["model_specific_factors"][
                "model_predictions"
            ]

            logger.info(f"\n{game['away_team']} @ {game['home_team']}")
            logger.info(
                f"Predicted Winner: {prediction['predicted_winner']} ({confidence} confidence)"
            )

            if prediction["is_high_confidence"]:
                logger.info("*** HIGH CONFIDENCE PICK ***")

            logger.info("\nModel Breakdown:")
            for model_pred in model_predictions:
                model_conf = f"{model_pred['confidence']*100:.1f}%"
                logger.info(
                    f"{model_pred['model']}: {model_pred['predicted_winner']} ({model_conf})"
                )

        except Exception as e:
            logger.error(
                f"Error predicting {game['away_team']} @ {game['home_team']}: {str(e)}"
            )


if __name__ == "__main__":
    main()

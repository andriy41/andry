import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from predict_total import TotalPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("synthetic_games_predictions.log"),
        logging.StreamHandler(),
    ],
)


def generate_synthetic_games(n_samples=1000):
    """Generate synthetic NFL games with realistic features"""
    try:
        teams = [
            "ARI",
            "ATL",
            "BAL",
            "BUF",
            "CAR",
            "CHI",
            "CIN",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GB",
            "HOU",
            "IND",
            "JAX",
            "KC",
            "LAC",
            "LAR",
            "LV",
            "MIA",
            "MIN",
            "NE",
            "NO",
            "NYG",
            "NYJ",
            "PHI",
            "PIT",
            "SEA",
            "SF",
            "TB",
            "TEN",
            "WAS",
        ]

        venues = {
            "ARI": {"name": "State Farm Stadium", "indoor": True, "altitude": 1150},
            "ATL": {"name": "Mercedes-Benz Stadium", "indoor": True, "altitude": 1050},
            "DAL": {"name": "AT&T Stadium", "indoor": True, "altitude": 595},
            "DET": {"name": "Ford Field", "indoor": True, "altitude": 600},
            "HOU": {"name": "NRG Stadium", "indoor": True, "altitude": 50},
            "IND": {"name": "Lucas Oil Stadium", "indoor": True, "altitude": 715},
            "LV": {"name": "Allegiant Stadium", "indoor": True, "altitude": 2030},
            "MIN": {"name": "U.S. Bank Stadium", "indoor": True, "altitude": 840},
            "NO": {"name": "Caesars Superdome", "indoor": True, "altitude": 3},
            "LAR": {"name": "SoFi Stadium", "indoor": True, "altitude": 60},
        }

        # Default outdoor venue template
        outdoor_template = {"indoor": False, "altitude": 0}

        # Add outdoor venues
        for team in teams:
            if team not in venues:
                venues[team] = {"name": f"{team} Stadium", **outdoor_template}

        games = []
        start_date = datetime(2023, 9, 7)  # NFL 2023 season start

        for i in range(n_samples):
            # Select random teams
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])

            # Generate game date
            game_date = start_date + timedelta(days=np.random.randint(0, 120))

            # Generate basic stats
            home_score = max(0, int(np.random.normal(24, 7)))
            away_score = max(0, int(np.random.normal(24, 7)))

            game = {
                "game_id": f"2023_SYNTH_{i+1}",
                "season": 2023,
                "week": min(17, (game_date - start_date).days // 7 + 1),
                "game_date": game_date.strftime("%Y-%m-%d"),
                "venue_name": venues[home_team]["name"],
                "venue_indoor": venues[home_team]["indoor"],
                "venue_altitude": venues[home_team]["altitude"],
                # Teams and scores
                "home_team": home_team,
                "home_score": home_score,
                "away_team": away_team,
                "away_score": away_score,
                # Generate other features
                "temperature": np.random.normal(65, 15),
                "wind_speed": np.random.exponential(8),
                "precipitation": np.random.beta(2, 8),
                # Team stats
                "home_points_per_game": np.random.normal(24, 4),
                "away_points_per_game": np.random.normal(24, 4),
                "home_points_allowed": np.random.normal(24, 4),
                "away_points_allowed": np.random.normal(24, 4),
                "home_yards_per_game": np.random.normal(350, 50),
                "away_yards_per_game": np.random.normal(350, 50),
                "home_yards_allowed": np.random.normal(350, 50),
                "away_yards_allowed": np.random.normal(350, 50),
                # Advanced stats
                "home_pass_yards_per_game": np.random.normal(240, 40),
                "away_pass_yards_per_game": np.random.normal(240, 40),
                "home_rush_yards_per_game": np.random.normal(110, 30),
                "away_rush_yards_per_game": np.random.normal(110, 30),
                "home_third_down_conv": np.random.beta(4, 6),
                "away_third_down_conv": np.random.beta(4, 6),
                # Game context
                "is_division_game": np.random.binomial(1, 0.3),
                "is_primetime": np.random.binomial(1, 0.2),
                "days_rest": np.random.choice([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            }

            games.append(game)

        return pd.DataFrame(games)

    except Exception as e:
        logging.error(f"Error generating synthetic games: {str(e)}")
        raise


def predict_synthetic_games():
    """Make predictions for synthetic NFL games"""
    try:
        # Generate synthetic games
        games_df = generate_synthetic_games()

        # Initialize predictor
        predictor = TotalPredictor()

        # Make predictions for each game
        results = []
        for _, game in games_df.iterrows():
            try:
                prediction = predictor.predict_total(game.to_dict())

                actual_total = game["home_score"] + game["away_score"]

                results.append(
                    {
                        "game_id": game["game_id"],
                        "date": game["game_date"],
                        "matchup": f"{game['away_team']} @ {game['home_team']}",
                        "predicted_total": prediction["total"],
                        "actual_total": actual_total,
                        "confidence": prediction["confidence"],
                        "model_predictions": prediction["model_predictions"],
                        "explanation": prediction["explanation"],
                    }
                )

                logging.info(
                    f"Made prediction for {game['away_team']} @ {game['home_team']}"
                )

            except Exception as e:
                logging.error(f"Error predicting game {game.get('game_id')}: {str(e)}")
                continue

        # Save results
        output_file = f"predictions/synthetic_games_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"Saved predictions to {output_file}")

        # Calculate accuracy metrics
        predictions_df = pd.DataFrame(results)
        mae = abs(
            predictions_df["predicted_total"] - predictions_df["actual_total"]
        ).mean()
        rmse = (
            (predictions_df["predicted_total"] - predictions_df["actual_total"]) ** 2
        ).mean() ** 0.5

        logging.info(f"Prediction Metrics:")
        logging.info(f"Mean Absolute Error: {mae:.2f}")
        logging.info(f"Root Mean Square Error: {rmse:.2f}")

        return results

    except Exception as e:
        logging.error(f"Error in predict_synthetic_games: {str(e)}")
        raise


if __name__ == "__main__":
    predict_synthetic_games()

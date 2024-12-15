import json
import pandas as pd
import logging
import numpy as np
from predict_total import TotalPredictor
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("real_games_predictions.log"),
        logging.StreamHandler(),
    ],
)


def load_real_games():
    """Load real NFL game data"""
    try:
        with open("data/processed/nfl_games_processed_summary.json", "r") as f:
            data = json.load(f)

            # Create sample games using the feature list
            games = []
            feature_list = data["feature_list"]

            # Generate sample games using the features
            for i in range(100):  # Generate 100 sample games
                game = {}
                for feature in feature_list:
                    if feature in ["game_id", "season", "week"]:
                        game[feature] = i
                    elif feature in ["home_score", "away_score"]:
                        game[feature] = np.random.randint(0, 45)
                    elif feature in ["home_team", "away_team"]:
                        game[feature] = np.random.choice(["NE", "BUF", "MIA", "NYJ"])
                    elif "percentage" in feature.lower():
                        game[feature] = np.random.random()
                    elif "yards" in feature.lower():
                        game[feature] = np.random.randint(0, 500)
                    else:
                        game[feature] = 0  # Default value for other features

                games.append(game)

            games_df = pd.DataFrame(games)
            logging.info(f"Generated {len(games_df)} sample NFL games")
            return games_df

    except Exception as e:
        logging.error(f"Error loading real games: {str(e)}")
        raise


def predict_real_games():
    """Make predictions for real NFL games"""
    try:
        # Load real game data
        games_df = load_real_games()

        # Initialize predictor
        predictor = TotalPredictor()

        # Make predictions for each game
        results = []
        for _, game in games_df.iterrows():
            try:
                prediction = predictor.predict_total(
                    {
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "venue_name": game.get("venue_name", ""),
                        "venue_indoor": game.get("venue_indoor", False),
                        "game_date": game.get("game_date", ""),
                        "season": game["season"],
                        "week": game["week"],
                        # Add all available features from the game data
                        **{
                            k: v
                            for k, v in game.items()
                            if k not in ["game_id", "total_points"]
                        },
                    }
                )

                actual_total = game.get("home_score", 0) + game.get("away_score", 0)

                results.append(
                    {
                        "game_id": game["game_id"],
                        "date": game.get("game_date", ""),
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
        output_file = (
            f"predictions/real_games_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
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
        logging.error(f"Error in predict_real_games: {str(e)}")
        raise


if __name__ == "__main__":
    predict_real_games()

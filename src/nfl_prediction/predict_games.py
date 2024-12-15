"""
Make NFL game predictions using trained ensemble model
"""
import logging
from datetime import datetime
from data_collection.data_fetcher import NFLDataFetcher
from models.ensemble import NFLEnsemble
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLPredictor:
    """Makes predictions for NFL games using trained ensemble model"""

    def __init__(self, model_path: str = "saved_models/ensemble"):
        self.ensemble = NFLEnsemble(use_lstm=True)
        self.ensemble.load_models(model_path)
        self.data_fetcher = NFLDataFetcher()

    def predict_upcoming_games(self) -> pd.DataFrame:
        """Predict all upcoming games in the current season"""
        # Get today's date
        today = datetime.now()

        # Fetch games for the next 30 days
        end_date = today + pd.Timedelta(days=30)
        games = self.data_fetcher.fetch_games_by_dates(
            today.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        )

        predictions = []

        for game in games:
            try:
                # Get game details
                game_id = game["id"]
                details = self.data_fetcher.fetch_game_details(game_id)

                # Prepare game data
                game_data = {
                    "game_time": game["date"],
                    "stadium_location": {
                        "latitude": float(details["gameInfo"]["venue"]["latitude"]),
                        "longitude": float(details["gameInfo"]["venue"]["longitude"]),
                    },
                    "home_team": game["competitions"][0]["competitors"][0]["team"][
                        "abbreviation"
                    ],
                    "away_team": game["competitions"][0]["competitors"][1]["team"][
                        "abbreviation"
                    ],
                }

                # Get prediction
                pred = self.ensemble.predict([game_data])

                # Add to results
                predictions.append(
                    {
                        "game_id": game_id,
                        "date": game["date"],
                        "home_team": game_data["home_team"],
                        "away_team": game_data["away_team"],
                        "predicted_winner": pred["predicted_winner"],
                        "win_probability": pred["win_probability"],
                        "model_predictions": pred["model_predictions"],
                    }
                )

            except Exception as e:
                logger.error(f"Error predicting game {game_id}: {e}")
                continue

        return pd.DataFrame(predictions)

    def predict_specific_game(self, game_id: str) -> dict:
        """Predict outcome of a specific game"""
        try:
            # Fetch game details
            game = self.data_fetcher.fetch_game_details(game_id)

            # Prepare game data
            game_data = {
                "game_time": game["date"],
                "stadium_location": {
                    "latitude": float(game["gameInfo"]["venue"]["latitude"]),
                    "longitude": float(game["gameInfo"]["venue"]["longitude"]),
                },
                "home_team": game["competitions"][0]["competitors"][0]["team"][
                    "abbreviation"
                ],
                "away_team": game["competitions"][0]["competitors"][1]["team"][
                    "abbreviation"
                ],
            }

            # Get prediction
            return self.ensemble.predict([game_data])

        except Exception as e:
            logger.error(f"Error predicting game {game_id}: {e}")
            raise


def main():
    """Main prediction function"""
    try:
        predictor = NFLPredictor()
        predictions = predictor.predict_upcoming_games()

        # Display predictions
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print("\nUpcoming Game Predictions:")
        print(
            predictions[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "predicted_winner",
                    "win_probability",
                ]
            ]
        )

        return predictions

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


if __name__ == "__main__":
    main()

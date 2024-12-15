"""Script to predict all NFL games using our prediction system."""

import pandas as pd
import logging
import numpy as np
from models.prediction_system import NFLPredictionSystem
from data.team_metadata import TEAM_INFO
from models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_team_metadata(df):
    """Add team metadata like climate and dome information."""
    # Add home team metadata
    df["home_climate"] = df["home_team"].map(lambda x: TEAM_INFO[x]["climate"])
    df["home_dome"] = df["home_team"].map(lambda x: int(TEAM_INFO[x]["dome"]))

    # Add away team metadata
    df["away_climate"] = df["away_team"].map(lambda x: TEAM_INFO[x]["climate"])
    df["away_dome"] = df["away_team"].map(lambda x: int(TEAM_INFO[x]["dome"]))

    # Add division/conference info
    df["home_division"] = df["home_team"].map(lambda x: TEAM_INFO[x]["division"])
    df["away_division"] = df["away_team"].map(lambda x: TEAM_INFO[x]["division"])
    df["home_conference"] = df["home_team"].map(lambda x: TEAM_INFO[x]["conference"])
    df["away_conference"] = df["away_team"].map(lambda x: TEAM_INFO[x]["conference"])

    # Calculate division and conference game flags
    df["is_division_game"] = (df["home_division"] == df["away_division"]).astype(int)
    df["is_conference_game"] = (df["home_conference"] == df["away_conference"]).astype(
        int
    )

    return df


def add_vedic_features(df):
    """Add Vedic astrological features for each game."""
    vedic_calc = NFLVedicCalculator()

    for idx, row in df.iterrows():
        try:
            # Convert game_date to string in YYYY-MM-DD format
            game_date = pd.to_datetime(row["game_date"]).strftime("%Y-%m-%d")

            # Calculate Vedic features for the game
            vedic_features = vedic_calc.calculate_game_features(
                game_date, row["home_team"], row["away_team"]
            )

            # Add features to dataframe
            for feature, value in vedic_features.items():
                df.at[idx, feature] = value
        except Exception as e:
            logger.error(
                f"Error calculating Vedic features for game {row['home_team']} vs {row['away_team']}: {e}"
            )
            # Set default values for missing features
            default_features = {
                "mars_strength": 0.5,
                "jupiter_strength": 0.5,
                "saturn_strength": 0.5,
                "home_team_yoga": 0.5,
                "away_team_yoga": 0.5,
                "home_nakshatra_score": 0.5,
                "away_nakshatra_score": 0.5,
                "planetary_alignment": 0.5,
                "moon_phase_score": 0.5,
            }
            for feature, value in default_features.items():
                df.at[idx, feature] = value

    return df


def evaluate_predictions(df, probabilities):
    """Evaluate prediction accuracy."""
    # Actual results (1 if home team won, 0 if away team won)
    actual_results = (df["home_score"] > df["away_score"]).astype(int)

    # Predicted results (1 if home team predicted to win, 0 if away team)
    predicted_results = (probabilities > 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(predicted_results == actual_results)

    # Calculate correct predictions by week
    df["week"] = pd.to_datetime(df["game_date"]).dt.isocalendar().week
    df["predicted_winner"] = predicted_results
    df["actual_winner"] = actual_results
    df["correct"] = (df["predicted_winner"] == df["actual_winner"]).astype(int)

    weekly_accuracy = df.groupby("week").agg({"correct": ["count", "sum", "mean"]})

    return accuracy, weekly_accuracy


def main():
    # Load both historical and future games
    logger.info("Loading games data...")
    historical_games = pd.read_csv("data/nfl_games.csv")
    future_games = pd.read_csv("data/future_games.csv")

    # Combine the dataframes
    games_df = pd.concat([historical_games, future_games], ignore_index=True)

    # Add team metadata
    logger.info("Adding team metadata...")
    games_df = add_team_metadata(games_df)

    # Add Vedic features
    logger.info("Calculating Vedic features...")
    games_df = add_vedic_features(games_df)

    # Split into training (completed games) and prediction sets
    training_df = games_df.dropna(subset=["home_score", "away_score"])
    prediction_df = games_df[games_df["home_score"].isna()]

    # Initialize and train the model
    logger.info("Training model...")
    prediction_system = NFLPredictionSystem()
    prediction_system.train(training_df)

    # Evaluate model on historical games
    logger.info("\nEvaluating historical predictions...")
    historical_probs = prediction_system.predict(training_df)
    accuracy, weekly_accuracy = evaluate_predictions(training_df, historical_probs)

    logger.info(f"\nOverall Prediction Accuracy: {accuracy*100:.1f}%")
    logger.info("\nWeekly Prediction Accuracy:")
    logger.info(weekly_accuracy.to_string())

    # Make predictions for future games
    logger.info("\nPredictions for upcoming games:")
    probabilities = prediction_system.predict(prediction_df)

    # Add predictions to the dataframe
    prediction_df["home_win_probability"] = probabilities

    # Display predictions
    for _, game in prediction_df.iterrows():
        home_prob = game["home_win_probability"]
        away_prob = 1 - home_prob

        # Format probabilities as percentages
        home_pct = f"{home_prob*100:.1f}%"
        away_pct = f"{away_prob*100:.1f}%"

        # Determine favorite
        if home_prob > 0.5:
            favorite = f"{game['home_team']} ({home_pct})"
            underdog = f"{game['away_team']} ({away_pct})"
        else:
            favorite = f"{game['away_team']} ({away_pct})"
            underdog = f"{game['home_team']} ({home_pct})"

        logger.info(
            f"{game['away_team']} @ {game['home_team']}: {favorite} over {underdog}"
        )


if __name__ == "__main__":
    main()

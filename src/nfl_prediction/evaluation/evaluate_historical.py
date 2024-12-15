"""Evaluate NFL prediction models on historical data."""

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm

from src.models.nfl_model_integrator import NFLModelIntegrator
from src.models.vedic_basic.vedic_model import VedicModel
from src.models.advanced_system.advanced_model import AdvancedModel
from src.models.combined_ml.combined_model import CombinedModel
from src.models.sports_only.sports_model import SportsModel
from src.models.total_prediction.neuro_total_model import NeuroTotalModel
from src.app.astrology.stadium_data import get_stadium_coordinates, load_stadium_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_record(record_str):
    """Convert record string (e.g. '4-11') to win percentage."""
    try:
        if not record_str or pd.isna(record_str):
            return 0.0
        wins, losses = map(int, record_str.split("-"))
        total_games = wins + losses
        return wins / total_games if total_games > 0 else 0.0
    except:
        return 0.0


def parse_time_of_possession(time_str):
    """Convert time of possession string (e.g. '31:45') to total seconds."""
    try:
        if not time_str or pd.isna(time_str):
            return 0.0
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    except:
        return 0.0


def parse_sacks(sacks_str):
    """Convert sacks string (e.g. '2-10' or '2.0') to sacks per game."""
    try:
        if not sacks_str or pd.isna(sacks_str):
            return 0.0
        if isinstance(sacks_str, (int, float)):
            return float(sacks_str)
        sacks = sacks_str.split("-")[0]
        return float(sacks)
    except:
        return 0.0


def prepare_game_data(row):
    """Convert DataFrame row to game data dictionary."""
    try:
        game_date = row.get("game_date", "")
        if not game_date:
            logger.warning("Missing game date")
            return None

        try:
            game_time = datetime.strptime(game_date, "%Y-%m-%dT%H:%MZ")
        except:
            logger.warning(f"Invalid game date format: {game_date}")
            return None

        # Get stadium coordinates
        stadium_coords = {
            "latitude": float(row.get("venue_latitude", 0)),
            "longitude": float(row.get("venue_longitude", 0)),
        }

        # Get team names and ensure they exist
        home_team = str(row.get("home_team", "")).strip()
        away_team = str(row.get("away_team", "")).strip()
        if (
            not home_team
            or not away_team
            or home_team.lower() == "nan"
            or away_team.lower() == "nan"
        ):
            logger.warning("Missing or invalid team names")
            return None

        # Get scores
        try:
            home_score = int(row.get("home_score", 0))
            away_score = int(row.get("away_score", 0))
        except:
            logger.warning("Invalid score format")
            return None

        game_data = {
            "game_id": str(row.get("game_id", "")),
            "game_time": game_time,
            "game_date": game_date,
            "stadium_location": stadium_coords,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_id": str(row.get("home_team_id", "")),
            "away_team_id": str(row.get("away_team_id", "")),
            "season": int(row.get("season", game_time.year)),
            "game_info": {
                "home_team": home_team,
                "away_team": away_team,
                "venue": row.get("venue_name", ""),
                "home_record": row.get("home_record", "0-0"),
                "away_record": row.get("away_record", "0-0"),
                "home_win_percentage": float(row.get("home_win_percentage", 0)),
                "away_win_percentage": float(row.get("away_win_percentage", 0)),
                "is_division_game": bool(row.get("is_division_game", False)),
                "is_conference_game": bool(row.get("is_conference_game", False)),
                "playoff_implications": bool(row.get("playoff_implications", False)),
            },
            "home_stats": {
                # Basic stats
                "points_per_game": float(row.get("home_totalpointspergame", 0)),
                "points_allowed": float(row.get("home_totalpointspergameallowed", 0)),
                "passing_yards": float(row.get("home_passingyardspergame", 0)),
                "rushing_yards": float(row.get("home_rushingyardspergame", 0)),
                "qb_rating": float(row.get("home_qb_rating", 0)),
                "defensive_efficiency": float(row.get("home_defense_efficiency", 0)),
                "sacks_per_game": parse_sacks(row.get("home_sacksyardslost", "0-0")),
                "turnovers_forced": float(row.get("home_turnovers", 0)),
                "third_down_pct": parse_record(row.get("home_thirddowneff", "0-0")),
                "red_zone_pct": parse_record(row.get("home_redzoneattempts", "0-0")),
                "plays_per_game": float(row.get("home_totaloffensiveplays", 0)),
                "time_of_possession": parse_time_of_possession(
                    row.get("home_possessiontime", "0:00")
                ),
                "yards_per_play": float(row.get("home_yardsperplay", 0)),
                # Advanced stats
                "success_rate": float(row.get("home_success_rate", 0)),
                "explosive_play_rate": float(row.get("home_explosive_play_rate", 0)),
                "air_yards_per_pass": float(row.get("home_air_yards_per_pass", 0)),
                "yac_per_completion": float(row.get("home_yac_per_completion", 0)),
                "pressure_rate": float(row.get("home_pressure_rate", 0)),
                "stuff_rate": float(row.get("home_stuff_rate", 0)),
                "expected_points_offense": float(
                    row.get("home_expected_points_offense", 0)
                ),
                "expected_points_defense": float(
                    row.get("home_expected_points_defense", 0)
                ),
                # Drive efficiency
                "points_per_drive": float(row.get("home_points_per_drive", 0)),
                "yards_per_drive": float(row.get("home_yards_per_drive", 0)),
                "plays_per_drive": float(row.get("home_plays_per_drive", 0)),
                "drive_success_rate": float(row.get("home_drive_success_rate", 0)),
                # Situational
                "goal_line_success": float(row.get("home_goal_line_success", 0)),
                "two_minute_success": float(row.get("home_two_minute_success", 0)),
                # Advanced metrics
                "dvoa_offense": float(row.get("home_dvoa_offense", 0)),
                "dvoa_defense": float(row.get("home_dvoa_defense", 0)),
                "dvoa_special_teams": float(row.get("home_dvoa_special_teams", 0)),
                "power_index": float(row.get("home_power_index", 0)),
                "playoff_probability": float(row.get("home_playoff_probability", 0)),
            },
            "away_stats": {
                # Basic stats
                "points_per_game": float(row.get("away_totalpointspergame", 0)),
                "points_allowed": float(row.get("away_totalpointspergameallowed", 0)),
                "passing_yards": float(row.get("away_passingyardspergame", 0)),
                "rushing_yards": float(row.get("away_rushingyardspergame", 0)),
                "qb_rating": float(row.get("away_qb_rating", 0)),
                "defensive_efficiency": float(row.get("away_defense_efficiency", 0)),
                "sacks_per_game": parse_sacks(row.get("away_sacksyardslost", "0-0")),
                "turnovers_forced": float(row.get("away_turnovers", 0)),
                "third_down_pct": parse_record(row.get("away_thirddowneff", "0-0")),
                "red_zone_pct": parse_record(row.get("away_redzoneattempts", "0-0")),
                "plays_per_game": float(row.get("away_totaloffensiveplays", 0)),
                "time_of_possession": parse_time_of_possession(
                    row.get("away_possessiontime", "0:00")
                ),
                "yards_per_play": float(row.get("away_yardsperplay", 0)),
                # Advanced stats
                "success_rate": float(row.get("away_success_rate", 0)),
                "explosive_play_rate": float(row.get("away_explosive_play_rate", 0)),
                "air_yards_per_pass": float(row.get("away_air_yards_per_pass", 0)),
                "yac_per_completion": float(row.get("away_yac_per_completion", 0)),
                "pressure_rate": float(row.get("away_pressure_rate", 0)),
                "stuff_rate": float(row.get("away_stuff_rate", 0)),
                "expected_points_offense": float(
                    row.get("away_expected_points_offense", 0)
                ),
                "expected_points_defense": float(
                    row.get("away_expected_points_defense", 0)
                ),
                # Drive efficiency
                "points_per_drive": float(row.get("away_points_per_drive", 0)),
                "yards_per_drive": float(row.get("away_yards_per_drive", 0)),
                "plays_per_drive": float(row.get("away_plays_per_drive", 0)),
                "drive_success_rate": float(row.get("away_drive_success_rate", 0)),
                # Situational
                "goal_line_success": float(row.get("away_goal_line_success", 0)),
                "two_minute_success": float(row.get("away_two_minute_success", 0)),
                # Advanced metrics
                "dvoa_offense": float(row.get("away_dvoa_offense", 0)),
                "dvoa_defense": float(row.get("away_dvoa_defense", 0)),
                "dvoa_special_teams": float(row.get("away_dvoa_special_teams", 0)),
                "power_index": float(row.get("away_power_index", 0)),
                "playoff_probability": float(row.get("away_playoff_probability", 0)),
            },
            "weather": {
                "temperature": float(row.get("temperature", 70)),
                "wind_speed": float(row.get("wind_speed", 0)),
                "precipitation": float(row.get("precipitation", 0)),
                "weather_advantage": float(row.get("weather_advantage", 0)),
            },
            "stadium": {
                "indoor": bool(row.get("venue_indoor", False)),
                "altitude": float(row.get("venue_altitude", 0)),
                "field_condition": float(row.get("field_condition", 0)),
            },
            "market": {
                "spread": float(row.get("spread", 0)),
                "over_under": float(row.get("over_under", 0)),
                "home_public_betting_percentage": float(
                    row.get("home_public_betting_percentage", 50)
                ),
                "line_movement": float(row.get("line_movement", 0)),
                "sharp_money_indicators": float(row.get("sharp_money_indicators", 0)),
            },
            "winner": "home" if home_score > away_score else "away",
        }

        return game_data

    except Exception as e:
        logger.error(f"Error preparing game data: {str(e)}")
        return None


def prepare_training_data(games_df):
    """Prepare training data in both formats needed by different models"""
    # Format for VedicModel and others
    games_data = {
        "games": [
            prepare_game_data(row)
            for _, row in games_df.iterrows()
            if prepare_game_data(row)
        ],
        "labels": [
            (int(row["home_score"]) > int(row["away_score"]))
            for _, row in games_df.iterrows()
            if prepare_game_data(row)
        ],
    }

    # Format specifically for SportsModel
    sports_data = {"games": [], "labels": []}

    for _, row in games_df.iterrows():
        game_data = prepare_game_data(row)
        if not game_data:
            continue

        # Extract sports-specific features
        sports_game = {
            "home_team": game_data["home_team"],
            "away_team": game_data["away_team"],
            "home_stats": game_data["home_stats"],
            "away_stats": game_data["away_stats"],
            "game_info": game_data["game_info"],
            "stadium_location": game_data["stadium_location"],
        }

        sports_data["games"].append(sports_game)
        sports_data["labels"].append(int(row["home_score"]) > int(row["away_score"]))

    # Format for NeuroTotalModel
    features_list = []
    targets_list = []

    for _, row in games_df.iterrows():
        try:
            game_data = prepare_game_data(row)
            if not game_data:
                continue

            # Extract features for total prediction
            feature_dict = {
                "home_points_per_game": float(
                    game_data["home_stats"]["points_per_game"]
                ),
                "away_points_per_game": float(
                    game_data["away_stats"]["points_per_game"]
                ),
                "home_points_allowed": float(game_data["home_stats"]["points_allowed"]),
                "away_points_allowed": float(game_data["away_stats"]["points_allowed"]),
                "home_plays_per_game": float(game_data["home_stats"]["plays_per_game"]),
                "away_plays_per_game": float(game_data["away_stats"]["plays_per_game"]),
                "home_time_of_possession": float(
                    game_data["home_stats"]["time_of_possession"]
                ),
                "away_time_of_possession": float(
                    game_data["away_stats"]["time_of_possession"]
                ),
                "home_yards_per_play": float(game_data["home_stats"]["yards_per_play"]),
                "away_yards_per_play": float(game_data["away_stats"]["yards_per_play"]),
                "home_third_down_pct": float(game_data["home_stats"]["third_down_pct"]),
                "away_third_down_pct": float(game_data["away_stats"]["third_down_pct"]),
                "home_red_zone_pct": float(game_data["home_stats"]["red_zone_pct"]),
                "away_red_zone_pct": float(game_data["away_stats"]["red_zone_pct"]),
                "home_qb_rating": float(game_data["home_stats"]["qb_rating"]),
                "away_qb_rating": float(game_data["away_stats"]["qb_rating"]),
                "home_defensive_efficiency": float(
                    game_data["home_stats"]["defensive_efficiency"]
                ),
                "away_defensive_efficiency": float(
                    game_data["away_stats"]["defensive_efficiency"]
                ),
                "home_sacks_per_game": float(game_data["home_stats"]["sacks_per_game"]),
                "away_sacks_per_game": float(game_data["away_stats"]["sacks_per_game"]),
                "home_turnovers_forced": float(
                    game_data["home_stats"]["turnovers_forced"]
                ),
                "away_turnovers_forced": float(
                    game_data["away_stats"]["turnovers_forced"]
                ),
                "is_division_game": 0,  # TODO: Add division game logic
                "is_primetime": 0,  # TODO: Add primetime game logic
                "days_rest": 7,  # Default to 7 days rest
                "playoff_implications": 0,  # TODO: Add playoff implications logic
                "temperature": 70,  # Default values for environmental factors
                "wind_speed": 5,
                "is_dome": 0,
                "precipitation_chance": 0,
                "altitude": 0,
            }

            features_list.append(feature_dict)

            # Add target (total points)
            total_points = int(row["home_score"]) + int(row["away_score"])
            targets_list.append(total_points)

        except Exception as e:
            logger.warning(f"Error preparing features for game: {str(e)}")

    return {
        "vedic_format": games_data,
        "sports_format": sports_data,
        "neuro_format": {"features": features_list, "targets": targets_list},
    }


def evaluate_models():
    """Evaluate all models on historical data."""
    logger.info("Loading historical game data...")
    games_df = pd.read_csv("data/nfl_games_2019_2024.csv")

    # Split data into training and testing sets (80-20 split)
    train_df = games_df.sample(frac=0.8, random_state=42)
    test_df = games_df.drop(train_df.index)

    # Prepare training data in both formats
    logger.info("Preparing training data...")
    train_data = prepare_training_data(train_df)
    test_data = prepare_training_data(test_df)

    # Initialize models
    logger.info("Initializing models...")
    vedic_model = VedicModel()
    advanced_model = AdvancedModel()
    combined_model = CombinedModel()
    sports_model = SportsModel()

    # Train models
    logger.info("Training models...")
    try:
        vedic_model.train(train_data["vedic_format"])
        logger.info("Vedic model trained successfully")
        advanced_model.train(train_data["vedic_format"])
        logger.info("Advanced model trained successfully")
        combined_model.train(train_data["vedic_format"])
        logger.info("Combined model trained successfully")
        sports_model.train(train_data["sports_format"])
        logger.info("Sports model trained successfully")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return

    results = []
    high_confidence_results = []

    logger.info("Evaluating models on test set...")
    for _, game in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            game_data = prepare_game_data(game)
            if not game_data:
                continue

            try:
                # Get predictions from each model
                model_predictions = {}

                # Vedic Model
                try:
                    vedic_pred = vedic_model.predict(game_data)
                    model_predictions["vedic"] = {
                        "home_win_prob": vedic_pred.get("home_win_probability", 0.5),
                        "confidence": vedic_pred.get("confidence_score", 0.0),
                    }
                except Exception as e:
                    logger.warning(f"Vedic model prediction failed: {str(e)}")
                    model_predictions["vedic"] = {
                        "home_win_prob": 0.5,
                        "confidence": 0.0,
                    }

                # Advanced Model
                try:
                    advanced_pred = advanced_model.predict(game_data)
                    model_predictions["advanced"] = {
                        "home_win_prob": advanced_pred.get("home_win_probability", 0.5),
                        "confidence": advanced_pred.get("confidence_score", 0.0),
                    }
                except Exception as e:
                    logger.warning(f"Advanced model prediction failed: {str(e)}")
                    model_predictions["advanced"] = {
                        "home_win_prob": 0.5,
                        "confidence": 0.0,
                    }

                # Combined Model
                try:
                    combined_pred = combined_model.predict(game_data)
                    model_predictions["combined"] = {
                        "home_win_prob": combined_pred.get("home_win_probability", 0.5),
                        "confidence": combined_pred.get("confidence_score", 0.0),
                    }
                except Exception as e:
                    logger.warning(f"Combined model prediction failed: {str(e)}")
                    model_predictions["combined"] = {
                        "home_win_prob": 0.5,
                        "confidence": 0.0,
                    }

                # Sports Model
                try:
                    sports_pred = sports_model.predict(game_data)
                    model_predictions["sports"] = {
                        "home_win_prob": sports_pred.get("home_win_probability", 0.5),
                        "confidence": sports_pred.get("confidence_score", 0.0),
                    }
                except Exception as e:
                    logger.warning(f"Sports model prediction failed: {str(e)}")
                    model_predictions["sports"] = {
                        "home_win_prob": 0.5,
                        "confidence": 0.0,
                    }

                # Determine actual winner
                home_score = int(game["home_score"])
                away_score = int(game["away_score"])
                actual_winner = "home" if home_score > away_score else "away"

                # Calculate prediction accuracy
                predictions_correct = {
                    model: (pred["home_win_prob"] > 0.5) == (actual_winner == "home")
                    for model, pred in model_predictions.items()
                }

                # Calculate agreement and confidence
                all_agree = (
                    len(
                        set(
                            pred["home_win_prob"] > 0.5
                            for pred in model_predictions.values()
                        )
                    )
                    == 1
                )
                high_confidence = all(
                    abs(pred["home_win_prob"] - 0.5) >= 0.2
                    and pred["confidence"] >= 0.7
                    for pred in model_predictions.values()
                )

                # Record results
                result = {
                    "game_id": game.get("game_id", ""),
                    "game_date": game_data["game_date"],
                    "home_team": game_data["game_info"]["home_team"],
                    "away_team": game_data["game_info"]["away_team"],
                    "actual_winner": actual_winner,
                    "home_score": home_score,
                    "away_score": away_score,
                    "all_agree": all_agree,
                    "high_confidence": high_confidence,
                }

                # Add individual model results
                for model, pred in model_predictions.items():
                    result.update(
                        {
                            f"{model}_home_win_prob": pred["home_win_prob"],
                            f"{model}_confidence": pred["confidence"],
                            f"{model}_correct": predictions_correct[model],
                        }
                    )

                results.append(result)

                if high_confidence:
                    high_confidence_results.append(result)

            except Exception as e:
                logger.warning(f"Error processing game predictions: {str(e)}")
                continue

        except Exception as e:
            logger.warning(f"Error processing game data: {str(e)}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    high_confidence_df = pd.DataFrame(high_confidence_results)

    # Calculate overall accuracy for each model
    logger.info("\nOverall Model Accuracy:")
    for model in ["vedic", "advanced", "combined", "sports"]:
        accuracy = results_df[f"{model}_correct"].mean()
        confidence = results_df[f"{model}_confidence"].mean()
        logger.info(
            f"{model.capitalize()} Model: {accuracy:.3f} accuracy, {confidence:.3f} avg confidence"
        )

    # Calculate high confidence accuracy
    if len(high_confidence_results) > 0:
        logger.info("\nHigh Confidence Predictions:")
        for model in ["vedic", "advanced", "combined", "sports"]:
            accuracy = high_confidence_df[f"{model}_correct"].mean()
            logger.info(f"{model.capitalize()} Model: {accuracy:.3f} accuracy")
        logger.info(f"Number of high confidence games: {len(high_confidence_results)}")

    # Save results
    logger.info("\nSaving results...")
    results_df.to_csv("evaluation_results.csv", index=False)
    high_confidence_df.to_csv("high_confidence_results.csv", index=False)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    evaluate_models()

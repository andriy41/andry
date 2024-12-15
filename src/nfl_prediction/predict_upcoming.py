import pandas as pd
from datetime import datetime, timedelta
import pytz
from models.combined_ml.combined_model import CombinedMLModel
from models.sports_only.sports_model import SportsOnlyModel
from models.advanced_system.advanced_model import AdvancedSystemModel
from models.vedic_basic.vedic_model import VedicBasicModel
import json
import requests
from typing import List, Dict, Any, Optional


def load_training_data():
    """Load historical game data for training"""
    try:
        # Create mock training data for now
        training_data = {"games": [], "labels": []}

        # Add some sample games with NFL-specific stats
        sample_games = [
            {
                "home_team": "San Francisco 49ers",
                "away_team": "Seattle Seahawks",
                "winner": "San Francisco 49ers",
                "home_score": 28,
                "away_score": 21,
                "season": 2023,
                "week": 14,
                "home_stats": {
                    # Basic offensive stats
                    "points_per_game": 28.5,
                    "points_allowed_per_game": 15.8,
                    "passing_yards_per_game": 285.3,
                    "rushing_yards_per_game": 145.2,
                    "total_offense_per_game": 430.5,
                    "total_defense_per_game": 310.2,
                    # Advanced metrics
                    "yards_per_play": 6.2,
                    "dvoa_offense": 25.4,
                    "dvoa_defense": -15.2,
                    "dvoa_special_teams": 3.5,
                    "turnover_differential": 0.7,
                    "pressure_rate": 32.5,
                    "stuff_rate": 18.2,
                    # Situational stats
                    "sacks_per_game": 2.8,
                    "third_down_pct": 45.5,
                    "red_zone_pct": 65.2,
                    "fourth_down_pct": 62.5,
                    # Success metrics
                    "win_pct": 0.846,
                    "home_win_pct": 0.857,
                    "away_win_pct": 0.833,
                    "conference_win_pct": 0.800,
                    "division_win_pct": 0.750,
                    "strength_of_victory": 0.520,
                    # Form and trends
                    "last_5_games_form": 0.800,
                    "points_trend": 3.5,
                    "yards_trend": 25.8,
                },
                "away_stats": {
                    # Basic offensive stats
                    "points_per_game": 21.3,
                    "points_allowed_per_game": 23.1,
                    "passing_yards_per_game": 245.6,
                    "rushing_yards_per_game": 118.4,
                    "total_offense_per_game": 364.0,
                    "total_defense_per_game": 348.7,
                    # Advanced metrics
                    "yards_per_play": 5.4,
                    "dvoa_offense": 5.8,
                    "dvoa_defense": 2.1,
                    "dvoa_special_teams": -0.5,
                    "turnover_differential": -0.1,
                    "pressure_rate": 28.4,
                    "stuff_rate": 15.8,
                    # Situational stats
                    "sacks_per_game": 2.1,
                    "third_down_pct": 38.5,
                    "red_zone_pct": 55.8,
                    "fourth_down_pct": 48.2,
                    # Success metrics
                    "win_pct": 0.385,
                    "home_win_pct": 0.429,
                    "away_win_pct": 0.333,
                    "conference_win_pct": 0.400,
                    "division_win_pct": 0.333,
                    "strength_of_victory": 0.505,
                    # Form and trends
                    "last_5_games_form": 0.400,
                    "points_trend": -2.1,
                    "yards_trend": -15.4,
                },
                "injuries": {
                    "home_qb_impact": 0.0,
                    "away_qb_impact": 0.8,  # Significant impact due to QB injury
                    "home_offense_impact": 0.2,
                    "away_offense_impact": 0.3,
                    "home_defense_impact": 0.1,
                    "away_defense_impact": 0.1,
                },
                "head_to_head": {"home_wins_last_5": 0.8, "avg_point_diff": 8.5},
                "weather": {
                    "temperature": 65,
                    "wind_speed": 8,
                    "precipitation_chance": 0,
                },
                "venue": {"name": "Levi's Stadium", "is_dome": False, "altitude": 0},
                "days_rest": 7,
                "is_division_game": True,
                "is_conference_game": True,
                "playoff_implications": True,
                "home_qb_rating": 105.8,
                "away_qb_rating": 89.5,
            }
            # Add more sample games here...
        ]

        for game in sample_games:
            training_data["games"].append(game)
            training_data["labels"].append(
                1 if game["winner"] == game["home_team"] else 0
            )

        return training_data
    except Exception as e:
        print(f"Error loading training data: {e}")
        return {"games": [], "labels": []}


def get_nfl_season_type() -> int:
    """Determine the current NFL season type (1=preseason, 2=regular season, 3=postseason)"""
    current_date = datetime.now(pytz.timezone("US/Eastern"))

    # 2024 NFL Season dates
    preseason_start = datetime(2024, 8, 1, tzinfo=pytz.timezone("US/Eastern"))
    regular_season_start = datetime(2024, 9, 5, tzinfo=pytz.timezone("US/Eastern"))
    regular_season_end = datetime(2025, 1, 7, tzinfo=pytz.timezone("US/Eastern"))

    if current_date < preseason_start:
        return 1  # Offseason/Preseason
    elif current_date < regular_season_start:
        return 1  # Preseason
    elif current_date < regular_season_end:
        return 2  # Regular Season
    else:
        return 3  # Postseason


def get_nfl_week() -> int:
    """Calculate the current NFL week based on the date"""
    current_date = datetime.now(pytz.timezone("US/Eastern"))
    season_type = get_nfl_season_type()

    if season_type == 2:  # Regular Season
        # Week 1 started on September 5, 2024
        season_start = datetime(2024, 9, 5, tzinfo=pytz.timezone("US/Eastern"))
        days_since_start = (current_date - season_start).days
        current_week = (days_since_start // 7) + 1
        return min(max(current_week, 1), 18)  # Ensure week is between 1 and 18
    elif season_type == 3:  # Postseason
        # Calculate playoff weeks (19-22)
        season_end = datetime(2025, 1, 7, tzinfo=pytz.timezone("US/Eastern"))
        days_since_end = (current_date - season_end).days
        playoff_week = (days_since_end // 7) + 19
        return min(max(playoff_week, 19), 22)
    else:
        return 1  # Default to week 1 for preseason


def fetch_espn_games(
    year: Optional[int] = None,
    week: Optional[int] = None,
    season_type: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch NFL games from ESPN API"""
    if year is None:
        year = 2024  # Current NFL season
    if week is None:
        week = get_nfl_week()
    if season_type is None:
        season_type = get_nfl_season_type()

    print(f"Fetching games from ESPN API:")
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

    games = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "events" not in data:
            print("No events found in ESPN API response")
            return games

        for event in data["events"]:
            try:
                competition = event["competitions"][0]
                home_team = (
                    competition["competitors"][0]
                    if competition["competitors"][0]["homeAway"] == "home"
                    else competition["competitors"][1]
                )
                away_team = (
                    competition["competitors"][1]
                    if competition["competitors"][0]["homeAway"] == "home"
                    else competition["competitors"][0]
                )

                # Parse game time
                game_date = datetime.strptime(event["date"], "%Y-%m-%dT%H:%MZ")
                game_date = pytz.utc.localize(game_date).astimezone(
                    pytz.timezone("US/Eastern")
                )

                # Get venue information
                venue = competition.get("venue", {}).get("fullName", "")

                # Get broadcast information
                broadcast = (
                    competition.get("broadcasts", [{}])[0].get("names", [""])[0]
                    if competition.get("broadcasts")
                    else ""
                )

                # Get odds information
                odds = (
                    competition.get("odds", [{}])[0].get("details", "")
                    if competition.get("odds")
                    else ""
                )

                game_info = {
                    "date": game_date,
                    "home_team": home_team["team"]["abbreviation"],
                    "away_team": away_team["team"]["abbreviation"],
                    "week": week,
                    "season": year,
                    "season_type": season_type,
                    "event_id": event["id"],
                    "venue": venue,
                    "broadcast": broadcast,
                    "odds": odds,
                    "status": event["status"]["type"]["name"],
                }
                games.append(game_info)
                print(f"Found game: {game_info}")

            except (KeyError, IndexError) as e:
                print(f"Error parsing game data: {e}")
                continue

    except requests.exceptions.RequestException as e:
        print(f"Error fetching games from ESPN API: {e}")
        return games

    return games


def get_upcoming_games() -> List[Dict[str, Any]]:
    """Get list of upcoming NFL games for the current week"""
    # Fetch games from ESPN API
    games = fetch_espn_games()

    if not games:
        print("Warning: Could not fetch games from ESPN API. Using backup static data.")
        # Fallback to static data (previous implementation)
        eastern = pytz.timezone("US/Eastern")
        games = [
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-16 16:30", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Indianapolis Colts",
                "away_team": "Pittsburgh Steelers",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Cleveland Browns",
                "away_team": "Chicago Bears",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Green Bay Packers",
                "away_team": "Tampa Bay Buccaneers",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Tennessee Titans",
                "away_team": "Houston Texans",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Los Angeles Rams",
                "away_team": "Washington Commanders",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Miami Dolphins",
                "away_team": "New York Jets",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "New Orleans Saints",
                "away_team": "New York Giants",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Carolina Panthers",
                "away_team": "Atlanta Falcons",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 16:05", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Arizona Cardinals",
                "away_team": "San Francisco 49ers",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 16:05", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Denver Broncos",
                "away_team": "Detroit Lions",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 16:25", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Buffalo Bills",
                "away_team": "Dallas Cowboys",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-17 20:20", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Jacksonville Jaguars",
                "away_team": "Baltimore Ravens",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-18 20:15", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Seattle Seahawks",
                "away_team": "Philadelphia Eagles",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-16 20:30", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Las Vegas Raiders",
                "away_team": "Los Angeles Chargers",
                "week": 15,
                "season": 2023,
            },
            {
                "date": eastern.localize(
                    datetime.strptime("2023-12-16 13:00", "%Y-%m-%d %H:%M")
                ),
                "home_team": "Cincinnati Bengals",
                "away_team": "Minnesota Vikings",
                "week": 15,
                "season": 2023,
            },
        ]

    # Sort games by date
    games.sort(key=lambda x: x["date"])

    return games


def predict_games():
    """Predict upcoming NFL games using our ensemble of models"""
    # Initialize models
    combined_model = CombinedMLModel()
    sports_model = SportsOnlyModel()
    advanced_model = AdvancedSystemModel()
    vedic_model = VedicBasicModel()

    # Load and train with historical data
    training_data = load_training_data()

    print("Training models...")
    try:
        combined_model.train(training_data)
        sports_model.train(training_data)
        advanced_model.train(training_data)
        vedic_model.train(training_data)
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Get upcoming games
    games = get_upcoming_games()

    # Make predictions
    predictions = []
    for game in games:
        try:
            # Enhance game data with NFL-specific features
            enhanced_game = {
                **game,
                "injuries": {
                    "home_qb_impact": 0.0,
                    "away_qb_impact": 0.0,
                    "home_offense_impact": 0.0,
                    "away_offense_impact": 0.0,
                    "home_defense_impact": 0.0,
                    "away_defense_impact": 0.0,
                },
                "head_to_head": {
                    "home_wins_last_5": 0.5,  # Default to neutral
                    "avg_point_diff": 0.0,
                },
                "venue": {"is_dome": False, "altitude": 0},
            }

            # Get predictions from each model
            combined_pred = combined_model.predict(enhanced_game)
            sports_pred = sports_model.predict(enhanced_game)
            advanced_pred = advanced_model.predict(enhanced_game)
            vedic_pred = vedic_model.predict(enhanced_game)

            # Calculate ensemble confidence
            confidence_scores = [
                combined_pred["confidence_score"],
                sports_pred["confidence_score"],
                advanced_pred["confidence_score"],
                vedic_pred["confidence_score"],
            ]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

            # Determine consensus prediction
            predictions_list = [
                combined_pred["predicted_winner"],
                sports_pred["predicted_winner"],
                advanced_pred["predicted_winner"],
                vedic_pred["predicted_winner"],
            ]

            # Count votes for each team
            home_votes = predictions_list.count(game["home_team"])
            away_votes = predictions_list.count(game["away_team"])

            # Calculate consensus probability
            total_votes = home_votes + away_votes
            consensus_prob = (
                max(home_votes, away_votes) / total_votes if total_votes > 0 else 0.5
            )

            predictions.append(
                {
                    "game_time": game["date"].strftime("%Y-%m-%d %H:%M %Z"),
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "consensus_winner": game["home_team"]
                    if home_votes >= away_votes
                    else game["away_team"],
                    "consensus_probability": consensus_prob,
                    "average_confidence": avg_confidence,
                    "model_agreement": max(home_votes, away_votes)
                    / len(predictions_list),
                    "individual_predictions": {
                        "combined_model": {
                            "winner": combined_pred["predicted_winner"],
                            "confidence": combined_pred["confidence_score"],
                        },
                        "sports_model": {
                            "winner": sports_pred["predicted_winner"],
                            "confidence": sports_pred["confidence_score"],
                        },
                        "advanced_model": {
                            "winner": advanced_pred["predicted_winner"],
                            "confidence": advanced_pred["confidence_score"],
                        },
                        "vedic_model": {
                            "winner": vedic_pred["predicted_winner"],
                            "confidence": vedic_pred["confidence_score"],
                        },
                    },
                }
            )

        except Exception as e:
            print(
                f"Error predicting game {game['home_team']} vs {game['away_team']}: {e}"
            )
            continue

    # Sort predictions by game time
    predictions.sort(
        key=lambda x: datetime.strptime(x["game_time"], "%Y-%m-%d %H:%M %Z")
    )

    return predictions


def format_predictions(predictions):
    """Format predictions for display"""
    print("\nNFL Week 15 Predictions:")
    print("=" * 80)

    for pred in predictions:
        print(f"\n{pred['game_time']}")
        print(f"{pred['away_team']} @ {pred['home_team']}")
        print(f"Predicted Winner: {pred['consensus_winner']}")
        print(f"Consensus Probability: {pred['consensus_probability']:.2%}")
        print(f"Average Confidence: {pred['average_confidence']:.2%}")
        print(f"Model Agreement: {pred['model_agreement']:.2%}")
        print("-" * 40)
        print("Individual Model Predictions:")
        for model, prediction in pred["individual_predictions"].items():
            print(f"  {model}: {prediction['winner']} ({prediction['confidence']:.2%})")
        print("-" * 80)


if __name__ == "__main__":
    predictions = predict_games()
    if predictions:
        format_predictions(predictions)
    else:
        print("Failed to generate predictions.")

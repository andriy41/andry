import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_historical_games(start_year=2018, end_year=2023):
    """Generate historical NFL game data with realistic patterns."""
    all_games = []
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

    # Team strength ratings (based on historical performance)
    global team_ratings
    team_ratings = {
        "KC": 0.85,
        "BUF": 0.80,
        "SF": 0.82,
        "PHI": 0.80,
        "BAL": 0.78,
        "DAL": 0.75,
        "CIN": 0.75,
        "LAR": 0.72,
        "GB": 0.70,
        "MIN": 0.68,
        "NE": 0.65,
        "TB": 0.65,
        "TEN": 0.63,
        "LV": 0.60,
        "NO": 0.62,
        "MIA": 0.65,
        "LAC": 0.68,
        "DEN": 0.58,
        "CLE": 0.60,
        "PIT": 0.65,
        "IND": 0.55,
        "SEA": 0.68,
        "ARI": 0.52,
        "CHI": 0.50,
        "NYG": 0.58,
        "WAS": 0.52,
        "CAR": 0.48,
        "ATL": 0.55,
        "DET": 0.62,
        "JAX": 0.58,
        "NYJ": 0.55,
        "HOU": 0.50,
    }

    # Generate games for each season
    for year in range(start_year, end_year + 1):
        season_start = datetime(year, 9, 1)  # Approximate season start

        # Regular season (17 weeks)
        for week in range(1, 18):
            # Generate 16 games per week
            game_date = season_start + timedelta(days=(week - 1) * 7)
            teams_this_week = teams.copy()
            random.shuffle(teams_this_week)

            for i in range(0, len(teams_this_week), 2):
                home_team = teams_this_week[i]
                away_team = teams_this_week[i + 1]

                # Calculate score based on team ratings and home field advantage
                home_rating = team_ratings[home_team]
                away_rating = team_ratings[away_team]
                home_advantage = 0.1  # Home field advantage factor

                # Base expected scores
                home_exp_score = 24 * (home_rating + home_advantage)
                away_exp_score = 24 * away_rating

                # Add randomness
                home_score = int(np.random.normal(home_exp_score, 7))
                away_score = int(np.random.normal(away_exp_score, 7))

                # Ensure non-negative scores
                home_score = max(0, home_score)
                away_score = max(0, away_score)

                # Weather factors
                month = game_date.month
                temperature = np.random.normal(
                    75 - abs(month - 8) * 3, 5
                )  # Cooler in later months
                wind_speed = np.random.exponential(8)
                precipitation = np.random.choice([0, 0, 0, 1], p=[0.8, 0.1, 0.05, 0.05])

                game = {
                    "date": game_date.strftime("%Y-%m-%d"),
                    "season": year,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "temperature": temperature,
                    "wind_speed": wind_speed,
                    "precipitation": precipitation,
                    "is_playoff": 0,
                }
                all_games.append(game)

            # Adjust game date for next week
            game_date += timedelta(days=7)

        # Add playoff games
        playoff_teams = sorted(team_ratings.items(), key=lambda x: x[1], reverse=True)[
            :12
        ]
        playoff_teams = [team[0] for team in playoff_teams]

        # Wild card round (6 games)
        for i in range(0, 6, 2):
            game_date = datetime(year + 1, 1, 7 + i)  # Early January
            game = generate_playoff_game(
                playoff_teams[i], playoff_teams[i + 1], game_date, year
            )
            all_games.append(game)

        # Divisional round (4 games)
        winners = playoff_teams[:4]  # Top seeds plus wild card winners
        for i in range(0, 4, 2):
            game_date = datetime(year + 1, 1, 14 + i)
            game = generate_playoff_game(winners[i], winners[i + 1], game_date, year)
            all_games.append(game)

        # Conference championships (2 games)
        conf_winners = winners[:2]
        game_date = datetime(year + 1, 1, 21)
        game = generate_playoff_game(conf_winners[0], conf_winners[1], game_date, year)
        all_games.append(game)

        # Super Bowl
        game_date = datetime(year + 1, 2, 4)
        game = generate_playoff_game(
            conf_winners[0], conf_winners[1], game_date, year, is_superbowl=True
        )
        all_games.append(game)

    return pd.DataFrame(all_games)


def generate_playoff_game(home_team, away_team, game_date, season, is_superbowl=False):
    """Generate a playoff game with appropriate scoring patterns."""
    global team_ratings
    home_rating = team_ratings[home_team]
    away_rating = team_ratings[away_team]

    # Playoff games tend to be closer
    home_exp_score = 24 * (home_rating + (0.05 if not is_superbowl else 0))
    away_exp_score = 24 * away_rating

    # Add randomness (less than regular season)
    home_score = int(np.random.normal(home_exp_score, 5))
    away_score = int(np.random.normal(away_exp_score, 5))

    # Ensure non-negative scores
    home_score = max(0, home_score)
    away_score = max(0, away_score)

    return {
        "date": game_date.strftime("%Y-%m-%d"),
        "season": season,
        "week": "P" if not is_superbowl else "SB",
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "temperature": np.random.normal(45, 5),  # Cold weather in playoffs
        "wind_speed": np.random.exponential(8),
        "precipitation": np.random.choice([0, 0, 0, 1], p=[0.7, 0.2, 0.05, 0.05]),
        "is_playoff": 1,
    }


if __name__ == "__main__":
    # Generate historical games
    historical_games = generate_historical_games(2018, 2023)

    # Save to CSV
    historical_games.to_csv("nfl_historical_games.csv", index=False)

    # Print some statistics
    print(f"Total games generated: {len(historical_games)}")
    print("\nWin percentages by team (home games):")
    for team in historical_games["home_team"].unique():
        team_games = historical_games[historical_games["home_team"] == team]
        wins = len(team_games[team_games["home_score"] > team_games["away_score"]])
        win_pct = wins / len(team_games)
        print(f"{team}: {win_pct:.3f}")

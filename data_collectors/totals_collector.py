import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Optional
import json
import time


class NFLTotalsCollector:
    """Collects and processes NFL game totals and scoring data"""

    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.eastern_tz = pytz.timezone("US/Eastern")

    def collect_game_data(self) -> List[Dict[str, Any]]:
        """Collect detailed game data for totals prediction"""
        games_data = []

        # Fetch current games from scoreboard
        url = f"{self.base_url}/scoreboard"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            for event in data.get("events", []):
                game_data = self._process_game_data(event)
                if game_data:
                    games_data.append(game_data)
                    print(
                        f"Collected data for {game_data['away_team']} @ {game_data['home_team']}"
                    )

        except Exception as e:
            print(f"Error collecting game data: {e}")

        return games_data

    def _process_game_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process raw game data into structured format"""
        try:
            competition = event["competitions"][0]

            # Get teams data
            home_team = next(
                team
                for team in competition["competitors"]
                if team["homeAway"] == "home"
            )
            away_team = next(
                team
                for team in competition["competitors"]
                if team["homeAway"] == "away"
            )

            # Get odds data
            odds = next(
                (
                    odds
                    for odds in competition.get("odds", [])
                    if odds.get("provider", {}).get("name") == "ESPN BET"
                ),
                {},
            )
            total_line = odds.get("overUnder")

            # Get venue and weather data
            venue = competition.get("venue", {})
            weather = competition.get("weather", {})

            # Get game time
            game_date = datetime.strptime(event["date"], "%Y-%m-%dT%H:%MZ")
            game_date = pytz.utc.localize(game_date).astimezone(self.eastern_tz)

            # Get team stats
            home_stats = self._get_team_stats(home_team)
            away_stats = self._get_team_stats(away_team)

            game_data = {
                "game_id": event["id"],
                "date": game_date.strftime("%Y-%m-%d %H:%M"),
                "week": event.get("week", {}).get("number"),
                # Teams
                "home_team": home_team["team"]["abbreviation"],
                "away_team": away_team["team"]["abbreviation"],
                # Team Stats
                "home_stats": home_stats,
                "away_stats": away_stats,
                # Venue and Weather
                "venue": {
                    "name": venue.get("fullName"),
                    "indoor": venue.get("indoor", False),
                    "city": venue.get("address", {}).get("city"),
                    "state": venue.get("address", {}).get("state"),
                },
                "weather": {
                    "temperature": weather.get("temperature"),
                    "conditions": weather.get("displayValue"),
                    "wind": weather.get("windSpeed"),
                },
                # Odds
                "total_line": total_line,
                "spread": odds.get("spread"),
                "over_odds": odds.get("overOdds"),
                "under_odds": odds.get("underOdds"),
                # Game Context
                "is_primetime": self._is_primetime(game_date),
                "broadcast": competition.get("broadcasts", [{}])[0].get("names", [""])[
                    0
                ],
            }

            return game_data

        except Exception as e:
            print(f"Error processing game data: {e}")
            return None

    def _get_team_stats(self, team: Dict[str, Any]) -> Dict[str, Any]:
        """Extract team statistics"""
        stats = {}

        # Get basic stats
        for stat in team.get("statistics", []):
            name = stat.get("name")
            value = stat.get("value")
            if name and value is not None:
                stats[name.lower().replace(" ", "_")] = value

        # Get records
        stats["record"] = team.get("records", [{}])[0].get("summary", "")

        return stats

    def _is_primetime(self, game_time: datetime) -> bool:
        """Determine if game is in primetime"""
        hour = game_time.hour
        return hour >= 20  # 8 PM ET or later

    def get_game_data(self, game_id: str) -> Dict[str, Any]:
        """Get comprehensive data for a specific game"""

        # For Super Bowl LVIII, use mock data
        if game_id == "401547268":
            return {
                "game_id": game_id,
                "date": "2024-02-11 18:30",
                "home_team": "San Francisco 49ers",
                "away_team": "Kansas City Chiefs",
                "venue": {"name": "Allegiant Stadium", "indoor": True},
                "temperature": 72,
                "total_line": 49.5,
                "home_stats": {
                    "points_per_game": 28.9,
                    "points_allowed": 17.5,
                    "yards_per_play": 6.1,
                    "third_down_pct": 0.47,
                    "red_zone_pct": 0.68,
                    "qb_rating": 98.5,
                    "defensive_efficiency": 0.85,
                    "sacks_per_game": 2.8,
                    "turnovers_forced": 1.5,
                    "plays_per_game": 63.5,
                    "time_of_possession": 31.2,
                    "hurry_up_rate": 0.12,
                    "first_half_points": 15.8,
                    "second_half_points": 13.1,
                    "over_rate": 0.56,
                    "avg_total_points": 46.4,
                },
                "away_stats": {
                    "points_per_game": 23.4,
                    "points_allowed": 19.2,
                    "yards_per_play": 5.8,
                    "third_down_pct": 0.45,
                    "red_zone_pct": 0.63,
                    "qb_rating": 95.8,
                    "defensive_efficiency": 0.82,
                    "sacks_per_game": 2.6,
                    "turnovers_forced": 1.3,
                    "plays_per_game": 64.8,
                    "time_of_possession": 30.8,
                    "hurry_up_rate": 0.14,
                    "first_half_points": 12.5,
                    "second_half_points": 10.9,
                    "over_rate": 0.48,
                    "avg_total_points": 42.6,
                },
                "is_division_game": False,
                "is_primetime": True,
                "days_rest": 14,
                "playoff_implications": 1.0,
                "matchup_historical_avg": 44.5,
            }

        # For other games, try to fetch from API
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard/{game_id}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            game_data = response.json()

            # Extract basic game info
            result = {
                "game_id": game_id,
                "date": game_data["date"],
                "home_team": game_data["competitions"][0]["competitors"][0]["team"][
                    "name"
                ],
                "away_team": game_data["competitions"][0]["competitors"][1]["team"][
                    "name"
                ],
                "venue": {
                    "name": game_data["competitions"][0]["venue"]["fullName"],
                    "indoor": game_data["competitions"][0]["venue"].get(
                        "indoor", False
                    ),
                },
            }

            # Get weather data if available
            if "weather" in game_data["competitions"][0]:
                weather = game_data["competitions"][0]["weather"]
                result.update(
                    {
                        "temperature": float(weather.get("temperature", 70)),
                        "wind_speed": float(weather.get("windSpeed", 0)),
                        "precipitation_chance": float(weather.get("precipitation", 0))
                        / 100,
                    }
                )

            # Get total line if available
            odds = game_data["competitions"][0].get("odds", [{}])[0]
            if "overUnder" in odds:
                result["total_line"] = float(odds["overUnder"])

            # Add team stats
            home_stats = self._get_team_stats_by_name(result["home_team"])
            away_stats = self._get_team_stats_by_name(result["away_team"])

            result["home_stats"] = home_stats
            result["away_stats"] = away_stats

            # Add game context
            result.update(self._get_game_context(game_data))

            return result

        except Exception as e:
            print(f"Error fetching game data: {str(e)}")
            return None

    def _get_team_stats_by_name(self, team_name: str) -> Dict[str, float]:
        """Get current season stats for a team"""
        # This would typically fetch from a database or API
        # For now, returning mock data
        return {
            "points_per_game": 24.5,
            "points_allowed": 20.3,
            "yards_per_play": 5.8,
            "third_down_pct": 0.42,
            "red_zone_pct": 0.65,
            "qb_rating": 95.0,
            "defensive_efficiency": 0.72,
            "sacks_per_game": 2.5,
            "turnovers_forced": 1.2,
            "plays_per_game": 65.0,
            "time_of_possession": 30.5,
            "hurry_up_rate": 0.15,
            "first_half_points": 13.2,
            "second_half_points": 11.3,
            "over_rate": 0.53,
            "avg_total_points": 44.8,
        }

    def _get_game_context(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get contextual information about the game"""
        return {
            "is_division_game": self._check_division_game(game_data),
            "is_primetime": self._is_primetime(game_data.get("date", "")),
            "days_rest": 7,  # Default to 7 for now
            "playoff_implications": 1.0,  # Default to high implications
            "matchup_historical_avg": 45.5,  # Mock historical average
        }

    def _check_division_game(self, game_data: Dict[str, Any]) -> bool:
        """Check if it's a division game"""
        try:
            home_div = game_data["competitions"][0]["competitors"][0]["team"][
                "division"
            ]["name"]
            away_div = game_data["competitions"][0]["competitors"][1]["team"][
                "division"
            ]["name"]
            return home_div == away_div
        except:
            return False

    def _is_primetime(self, game_time: str) -> bool:
        """Check if it's a primetime game"""
        try:
            time = datetime.strptime(game_time, "%Y-%m-%dT%H:%M:%SZ")
            hour = time.hour
            return hour >= 19  # 7 PM or later
        except:
            return False

    def save_data(self, data: List[Dict[str, Any]], filename: str):
        """Save collected data to JSON file"""
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load collected data from JSON file"""
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except:
            return []

    def get_week_14_2024_data(self):
        """Get data for NFL Week 14, 2024"""
        games = [
            {
                "game_id": "NE_PIT_2024_14",
                "home_team": "Pittsburgh Steelers",
                "away_team": "New England Patriots",
                "date": "2024-12-05 20:15",  # Thursday Night Football
                "venue": "Acrisure Stadium",
                "vegas_total": 37.5,
                "home_stats": {
                    "points_per_game": 21.5,
                    "points_allowed": 19.8,
                    "yards_per_play": 5.1,
                    "third_down_pct": 0.39,
                    "red_zone_pct": 0.54,
                    "qb_rating": 85.5,
                    "defensive_efficiency": 0.82,
                    "sacks_per_game": 3.1,
                    "turnovers_forced": 1.7,
                    "plays_per_game": 61.5,
                    "time_of_possession": 30.2,
                    "hurry_up_rate": 0.11,
                    "first_half_points": 10.8,
                    "second_half_points": 10.7,
                    "over_rate": 0.42,
                    "avg_total_points": 41.3,
                },
                "away_stats": {
                    "points_per_game": 18.2,
                    "points_allowed": 22.5,
                    "yards_per_play": 4.9,
                    "third_down_pct": 0.35,
                    "red_zone_pct": 0.48,
                    "qb_rating": 78.5,
                    "defensive_efficiency": 0.75,
                    "sacks_per_game": 2.4,
                    "turnovers_forced": 1.3,
                    "plays_per_game": 60.8,
                    "time_of_possession": 29.5,
                    "hurry_up_rate": 0.13,
                    "first_half_points": 9.1,
                    "second_half_points": 9.1,
                    "over_rate": 0.38,
                    "avg_total_points": 40.7,
                },
                "is_division_game": False,
                "is_primetime": True,
                "days_rest": 7,
                "playoff_implications": 0.6,
                "matchup_historical_avg": 39.5,
                "temperature": 35,
                "wind_speed": 12,
                "is_dome": False,
                "precipitation_chance": 0.3,
                "altitude": 0,
            },
            {
                "game_id": "LAR_BAL_2024_14",
                "home_team": "Baltimore Ravens",
                "away_team": "Los Angeles Rams",
                "date": "2024-12-08 13:00",
                "venue": "M&T Bank Stadium",
                "vegas_total": 46.5,
                "home_stats": {
                    "points_per_game": 27.8,
                    "points_allowed": 18.5,
                    "yards_per_play": 5.9,
                    "third_down_pct": 0.45,
                    "red_zone_pct": 0.65,
                    "qb_rating": 98.5,
                    "defensive_efficiency": 0.88,
                    "sacks_per_game": 3.3,
                    "turnovers_forced": 1.8,
                    "plays_per_game": 64.5,
                    "time_of_possession": 31.8,
                    "hurry_up_rate": 0.14,
                    "first_half_points": 14.5,
                    "second_half_points": 13.3,
                    "over_rate": 0.54,
                    "avg_total_points": 46.3,
                },
                "away_stats": {
                    "points_per_game": 24.2,
                    "points_allowed": 22.8,
                    "yards_per_play": 5.6,
                    "third_down_pct": 0.41,
                    "red_zone_pct": 0.58,
                    "qb_rating": 92.5,
                    "defensive_efficiency": 0.76,
                    "sacks_per_game": 2.6,
                    "turnovers_forced": 1.4,
                    "plays_per_game": 62.5,
                    "time_of_possession": 30.2,
                    "hurry_up_rate": 0.16,
                    "first_half_points": 12.5,
                    "second_half_points": 11.7,
                    "over_rate": 0.52,
                    "avg_total_points": 47.0,
                },
                "is_division_game": False,
                "is_primetime": False,
                "days_rest": 7,
                "playoff_implications": 0.8,
                "matchup_historical_avg": 45.5,
                "temperature": 42,
                "wind_speed": 8,
                "is_dome": False,
                "precipitation_chance": 0.2,
                "altitude": 0,
            },
        ]
        return games


def collect_current_totals():
    """Collect totals data for current games"""
    collector = NFLTotalsCollector()
    games = collector.collect_game_data()

    if games:
        # Save with current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collector.save_data(games, f"nfl_totals_current_{timestamp}.json")
        print(f"\nCollected data for {len(games)} games")

        # Print summary
        print("\nCurrent Totals:")
        for game in games:
            print(f"\n{game['away_team']} @ {game['home_team']}")
            print(f"Total Line: {game['total_line']}")
            print(f"Weather: {game['weather']['conditions']}")
            print(f"Broadcast: {game['broadcast']}")

    return games


if __name__ == "__main__":
    print("Collecting current NFL totals data...")
    collect_current_totals()

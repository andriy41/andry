"""
NFL data fetcher using ESPN API endpoints
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLDataFetcher:
    """Fetches NFL data from ESPN APIs"""

    BASE_URLS = {
        "scoreboard": "site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
        "teams": "site.api.espn.com/apis/site/v2/sports/football/nfl/teams",
        "game_summary": "site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
        "odds": "sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event_id}/competitions/{event_id}/odds",
        "probabilities": "sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event_id}/competitions/{event_id}/probabilities",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        )
        self.rate_limit_delay = 0.5  # Increased delay between requests
        self._cache = {}  # Simple memory cache

    def _make_request(self, url: str, cache_key: str = None) -> Dict[str, Any]:
        """Make a request with caching and rate limiting"""
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        time.sleep(self.rate_limit_delay)
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                data = response.json()

                if cache_key:
                    self._cache[cache_key] = data
                return data
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to fetch data after {max_retries} attempts: {e}"
                    )
                    raise
                time.sleep(retry_delay * (attempt + 1))

    def fetch_games_by_dates(
        self, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Fetch games between dates (YYYYMMDD format)"""
        url = f"https://{self.BASE_URLS['scoreboard']}?limit=1000&dates={start_date}-{end_date}"
        return self._make_request(url)

    def fetch_season_games(self, year: int) -> List[Dict[str, Any]]:
        """Fetch all games for a season"""
        # Season typically runs from September to February
        start_date = f"{year}0901"
        end_date = f"{year+1}0228"
        return self.fetch_games_by_dates(start_date, end_date)

    def fetch_game_details(self, game_id: str) -> Dict[str, Any]:
        """Fetch detailed information for a specific game"""
        url = f"https://{self.BASE_URLS['game_summary']}?event={game_id}"
        return self._make_request(url, cache_key=game_id)

    def fetch_game_odds(self, game_id: str) -> Dict[str, Any]:
        """Fetch betting odds for a game"""
        url = f"https://{self.BASE_URLS['odds']}".format(event_id=game_id)
        return self._make_request(url, cache_key=game_id)

    def fetch_win_probabilities(self, game_id: str) -> Dict[str, Any]:
        """Fetch win probabilities for a game"""
        url = f"https://{self.BASE_URLS['probabilities']}".format(event_id=game_id)
        return self._make_request(url, cache_key=game_id)

    def fetch_training_data(
        self, years: List[int], include_odds: bool = True
    ) -> pd.DataFrame:
        """Fetch and prepare training data for multiple seasons"""
        all_games = []

        for year in years:
            logger.info(f"Fetching games for {year} season...")
            try:
                games = self.fetch_season_games(year)

                for game in games:
                    try:
                        game_id = game["id"]
                        details = self.fetch_game_details(game_id)

                        # Enhanced game info with more features
                        game_data = {
                            "game_id": game_id,
                            "game_time": game["date"],
                            "home_team": game["competitions"][0]["competitors"][0][
                                "team"
                            ]["abbreviation"],
                            "away_team": game["competitions"][0]["competitors"][1][
                                "team"
                            ]["abbreviation"],
                            "home_score": int(
                                game["competitions"][0]["competitors"][0]["score"]
                            ),
                            "away_score": int(
                                game["competitions"][0]["competitors"][1]["score"]
                            ),
                            "stadium_latitude": float(
                                details["gameInfo"]["venue"]["latitude"]
                            ),
                            "stadium_longitude": float(
                                details["gameInfo"]["venue"]["longitude"]
                            ),
                            "season": year,
                            "week": game["week"]["number"],
                            "home_win": int(
                                game["competitions"][0]["competitors"][0]["winner"]
                            ),
                            "venue_name": details["gameInfo"]["venue"]["fullName"],
                            "venue_indoor": details["gameInfo"]["venue"].get(
                                "indoor", False
                            ),
                            "neutral_site": game["competitions"][0].get(
                                "neutralSite", False
                            ),
                            "game_type": game["competitions"][0]
                            .get("type", {})
                            .get("abbreviation", "REG"),
                            "attendance": details["gameInfo"].get("attendance"),
                            "weather": details["gameInfo"].get("weather"),
                            "temperature": details["gameInfo"].get("temperature"),
                            "wind_speed": details["gameInfo"].get("windSpeed"),
                        }

                        # Add team records
                        for competitor in game["competitions"][0]["competitors"]:
                            team_type = (
                                "home" if competitor["homeAway"] == "home" else "away"
                            )
                            record = competitor.get("records", [{}])[0]
                            game_data.update(
                                {
                                    f"{team_type}_wins": int(
                                        record.get("summary", "0-0").split("-")[0]
                                    ),
                                    f"{team_type}_losses": int(
                                        record.get("summary", "0-0").split("-")[1]
                                    ),
                                }
                            )

                        # Add odds if requested
                        if include_odds:
                            try:
                                odds = self.fetch_game_odds(game_id)
                                if odds.get("items"):
                                    game_data.update(
                                        {
                                            "spread": odds["items"][0].get("spread"),
                                            "over_under": odds["items"][0].get(
                                                "overUnder"
                                            ),
                                            "home_moneyline": odds["items"][0].get(
                                                "homeMoneyLine"
                                            ),
                                            "away_moneyline": odds["items"][0].get(
                                                "awayMoneyLine"
                                            ),
                                        }
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not fetch odds for game {game_id}: {e}"
                                )

                        # Add win probabilities
                        try:
                            probs = self.fetch_win_probabilities(game_id)
                            if probs.get("items"):
                                game_data.update(
                                    {
                                        "home_win_probability": probs["items"][0].get(
                                            "homeWinPercentage"
                                        ),
                                        "away_win_probability": probs["items"][0].get(
                                            "awayWinPercentage"
                                        ),
                                    }
                                )
                        except Exception as e:
                            logger.warning(
                                f"Could not fetch probabilities for game {game_id}: {e}"
                            )

                        all_games.append(game_data)
                        time.sleep(self.rate_limit_delay)

                    except Exception as e:
                        logger.error(f"Error processing game {game_id}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error fetching season {year}: {e}")
                continue

        return pd.DataFrame(all_games)

    def prepare_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features and labels for model training"""
        # Convert game time to datetime
        df["game_time"] = pd.to_datetime(df["game_time"])

        # Create features dictionary format expected by our models
        features = []
        labels = []

        for _, row in df.iterrows():
            game_dict = {
                "game_time": row["game_time"],
                "stadium_location": {
                    "latitude": row["stadium_latitude"],
                    "longitude": row["stadium_longitude"],
                },
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
            features.append(game_dict)
            labels.append(row["home_win"])

        return pd.DataFrame(features), pd.Series(labels)

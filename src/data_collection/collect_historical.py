"""
Collect NFL historical data from 2019 to 2024
"""
import requests
import pandas as pd
from datetime import datetime
import time
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLHistoricalCollector:
    def __init__(self):
        self.base_urls = {
            "scoreboard": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            "game_summary": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
            "teams": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams",
            "odds": "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{}/competitions/{}/odds",
        }
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        )
        # Rate limiting parameters
        self.requests_per_minute = 20  # Conservative rate limit
        self.last_request_time = time.time()

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        # Ensure minimum time between requests (60 seconds / requests_per_minute)
        min_interval = 60.0 / self.requests_per_minute
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, url):
        """Make a rate-limited request"""
        self._wait_for_rate_limit()
        return self.session.get(url)

    def fetch_season_games(self, year):
        """Fetch all games for a specific season"""
        logger.info(f"Fetching games for {year} season")

        start_date = f"{year}0901"
        end_date = f"{year+1}0228"

        url = f"{self.base_urls['scoreboard']}?limit=1000&dates={start_date}-{end_date}"
        response = self._make_request(url)
        response.raise_for_status()

        return response.json().get("events", [])

    def fetch_game_details(self, game_id):
        """Fetch detailed information for a specific game"""
        try:
            url = f"{self.base_urls['game_summary']}?event={game_id}"
            response = self._make_request(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching details for game {game_id}: {e}")
            return {}

    def fetch_game_odds(self, game_id):
        """Fetch betting odds for a game"""
        try:
            url = self.base_urls["odds"].format(game_id, game_id)
            response = self._make_request(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching odds for game {game_id}: {e}")
            return {}

    def process_game_data(self, game, details, odds=None):
        """Process raw game data into structured format"""
        try:
            # Basic game info
            game_data = {
                "game_id": game["id"],
                "season": game["season"].get("year"),
                "week": game["week"].get("number"),
                "game_date": game["date"],
            }

            # Venue info
            venue = details.get("gameInfo", {}).get("venue", {})
            game_data.update(
                {
                    "venue_name": venue.get("fullName", "Unknown"),
                    "venue_indoor": venue.get("indoor", None),
                    "venue_latitude": venue.get("latitude", None),
                    "venue_longitude": venue.get("longitude", None),
                }
            )

            # Team info and scores
            competitors = game.get("competitions", [{}])[0].get("competitors", [])
            if not competitors:
                logger.warning(f"No competitors found for game {game['id']}")
                return None

            for team in competitors:
                is_home = team.get("homeAway") == "home"
                prefix = "home" if is_home else "away"

                team_data = team.get("team", {})
                records = team.get("records", [{}])
                statistics = team.get("statistics", [{}])

                game_data.update(
                    {
                        f"{prefix}_team": team_data.get("abbreviation", "UNK"),
                        f"{prefix}_score": int(team.get("score", 0)),
                        f"{prefix}_record": records[0].get("summary", "0-0")
                        if records
                        else "0-0",
                        f"{prefix}_win_percentage": float(statistics[0].get("value", 0))
                        if statistics
                        else 0.0,
                    }
                )

            # Game stats from boxscore
            if "boxscore" in details:
                for team in details.get("boxscore", {}).get("teams", []):
                    is_home = team.get("team", {}).get("homeAway") == "home"
                    prefix = "home" if is_home else "away"

                    for stat in team.get("statistics", []):
                        name = stat.get("name", "").lower().replace(" ", "_")
                        if name:
                            game_data[f"{prefix}_{name}"] = stat.get(
                                "displayValue", "0"
                            )

            # Betting odds
            if odds and "items" in odds:
                odds_data = odds.get("items", [{}])[0]
                game_data.update(
                    {
                        "spread": odds_data.get("spread"),
                        "over_under": odds_data.get("overUnder"),
                        "home_moneyline": odds_data.get("homeMoneyLine"),
                        "away_moneyline": odds_data.get("awayMoneyLine"),
                    }
                )

            return game_data

        except Exception as e:
            logger.error(f"Error processing game {game.get('id', 'unknown')}: {str(e)}")
            return None

    def collect_seasons(self, start_year=2019, end_year=2024):
        """Collect data for multiple seasons"""
        all_games = []
        total_processed = 0

        for year in range(start_year, end_year + 1):
            try:
                games = self.fetch_season_games(year)
                logger.info(f"Found {len(games)} games for {year} season")

                successful_games = 0
                for i, game in enumerate(games, 1):
                    try:
                        logger.info(
                            f"Processing game {i}/{len(games)} for {year} season (Game ID: {game['id']})"
                        )

                        # Fetch additional data
                        details = self.fetch_game_details(game["id"])
                        odds = self.fetch_game_odds(game["id"])

                        # Process game data
                        game_data = self.process_game_data(game, details, odds)
                        if game_data:
                            all_games.append(game_data)
                            successful_games += 1
                            total_processed += 1

                    except Exception as e:
                        logger.error(f"Error processing game {game['id']}: {e}")
                        continue

                logger.info(
                    f"Completed {year} season: {successful_games}/{len(games)} games processed successfully"
                )

                # Save progress after each season
                if all_games:
                    df = pd.DataFrame(all_games)
                    output_file = f"data/nfl_games_{start_year}_{end_year}_progress.csv"
                    df.to_csv(output_file, index=False)
                    logger.info(
                        f"Progress saved to {output_file} ({len(df)} total games)"
                    )

            except Exception as e:
                logger.error(f"Error fetching {year} season: {e}")
                continue

        return pd.DataFrame(all_games)


def main():
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Initialize collector and fetch data
    collector = NFLHistoricalCollector()
    df = collector.collect_seasons(2019, 2024)

    # Save final dataset
    output_file = "data/nfl_games_2019_2024.csv"
    df.to_csv(output_file, index=False)

    logger.info(f"Data collection complete. Collected {len(df)} games.")
    logger.info(f"Data saved to {output_file}")

    # Save data summary
    summary = {
        "total_games": len(df),
        "games_per_season": df["season"].value_counts().to_dict(),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }

    with open("data/collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

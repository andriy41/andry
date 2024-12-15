import requests
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime


class ESPNAPIClient:
    """Client for accessing ESPN's NFL API endpoints with rate limiting and caching."""

    BASE_URLS = {
        "core": "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl",
        "site": "https://site.api.espn.com/apis/site/v2/sports/football/nfl",
        "site_web": "https://site.web.api.espn.com/apis",
    }

    def __init__(self, cache_dir: str = "cache", rate_limit_delay: float = 0.5):
        self.session = requests.Session()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ESPNAPIClient")

    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _get_cache_path(self, endpoint: str) -> Path:
        """Generate a cache file path for an endpoint."""
        safe_endpoint = endpoint.replace("/", "_").replace("?", "_")
        return self.cache_dir / f"{safe_endpoint}.json"

    def _get_cached_response(self, cache_path: Path) -> Optional[Dict]:
        """Get cached response if it exists and is fresh."""
        if cache_path.exists():
            try:
                with cache_path.open("r") as f:
                    data = json.load(f)
                    return data
            except json.JSONDecodeError:
                return None
        return None

    def _cache_response(self, cache_path: Path, data: Dict):
        """Cache API response."""
        with cache_path.open("w") as f:
            json.dump(data, f)

    def get(
        self, endpoint: str, base_type: str = "core", use_cache: bool = True
    ) -> Dict[str, Any]:
        """Make a GET request to the ESPN API with rate limiting and caching."""
        cache_path = self._get_cache_path(endpoint)

        if use_cache:
            cached = self._get_cached_response(cache_path)
            if cached is not None:
                self.logger.debug(f"Using cached response for {endpoint}")
                return cached

        self._rate_limit()

        url = f"{self.BASE_URLS[base_type]}/{endpoint}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            self._cache_response(cache_path, data)
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            raise

    # Basic Data Endpoints
    def get_athletes(self, limit: int = 1000, active: bool = True) -> Dict[str, Any]:
        """Get list of NFL athletes."""
        return self.get(f"athletes?limit={limit}&active={str(active).lower()}")

    def get_teams(self, limit: int = 32) -> Dict[str, Any]:
        """Get list of NFL teams."""
        return self.get(f"teams?limit={limit}")

    def get_positions(self, limit: int = 75) -> Dict[str, Any]:
        """Get list of NFL positions."""
        return self.get(f"positions?limit={limit}")

    def get_venues(self, limit: int = 700) -> Dict[str, Any]:
        """Get list of NFL venues."""
        return self.get(f"venues?limit={limit}")

    # Team-related Endpoints
    def get_team_info(self, team_id: int) -> Dict[str, Any]:
        """Get detailed team information."""
        return self.get(f"teams/{team_id}", base_type="site")

    def get_team_roster(
        self, team_id: int, season: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get team roster with optional season filter."""
        endpoint = f"teams/{team_id}/roster"
        if season:
            endpoint += f"?season={season}"
        return self.get(endpoint, base_type="site")

    def get_team_schedule(
        self, team_id: int, season: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get team schedule."""
        endpoint = f"teams/{team_id}/schedule"
        if season:
            endpoint += f"?season={season}"
        return self.get(endpoint, base_type="site")

    def get_team_statistics(
        self, team_id: int, season: int, season_type: int = 2
    ) -> Dict[str, Any]:
        """Get team statistics."""
        return self.get(
            f"seasons/{season}/types/{season_type}/teams/{team_id}/statistics"
        )

    def get_team_injuries(self, team_id: int) -> Dict[str, Any]:
        """Get team injuries."""
        return self.get(f"teams/{team_id}/injuries")

    def get_team_depth_charts(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get team depth charts."""
        return self.get(f"seasons/{season}/teams/{team_id}/depthcharts")

    # Game-related Endpoints
    def get_scoreboard(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get scoreboard data. Date format: YYYYMMDD"""
        endpoint = "scoreboard"
        if date:
            endpoint += f"?dates={date}"
        return self.get(endpoint, base_type="site")

    def get_game_summary(self, game_id: str) -> Dict[str, Any]:
        """Get detailed game summary."""
        return self.get(f"summary?event={game_id}", base_type="site")

    def get_game_plays(self, game_id: str, limit: int = 400) -> Dict[str, Any]:
        """Get all plays for a game."""
        return self.get(f"events/{game_id}/competitions/{game_id}/plays?limit={limit}")

    def get_game_probabilities(self, game_id: str, limit: int = 200) -> Dict[str, Any]:
        """Get win probabilities for a game."""
        return self.get(
            f"events/{game_id}/competitions/{game_id}/probabilities?limit={limit}"
        )

    # Season and Calendar Endpoints
    def get_calendar(self) -> Dict[str, Any]:
        """Get NFL calendar."""
        return self.get("calendar")

    def get_season_info(self, season: int) -> Dict[str, Any]:
        """Get season information."""
        return self.get(f"seasons/{season}")

    def get_season_types(self, season: int) -> Dict[str, Any]:
        """Get season types (pre/regular/post)."""
        return self.get(f"seasons/{season}/types")

    def get_season_weeks(self, season: int, season_type: int) -> Dict[str, Any]:
        """Get weeks in a season."""
        return self.get(f"seasons/{season}/types/{season_type}/weeks")

    # Player-related Endpoints
    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        """Get detailed player information."""
        return self.get(f"athletes/{player_id}")

    def get_player_stats(
        self, player_id: int, season: int, season_type: int = 2
    ) -> Dict[str, Any]:
        """Get player statistics."""
        return self.get(
            f"seasons/{season}/types/{season_type}/athletes/{player_id}/statistics"
        )

    def get_player_gamelog(self, player_id: int) -> Dict[str, Any]:
        """Get player game log."""
        return self.get(
            f"common/v3/sports/football/nfl/athletes/{player_id}/gamelog",
            base_type="site_web",
        )

    # Odds and Betting Endpoints
    def get_game_odds(self, game_id: str) -> Dict[str, Any]:
        """Get game odds."""
        return self.get(f"events/{game_id}/competitions/{game_id}/odds")

    def get_futures(self, season: int) -> Dict[str, Any]:
        """Get futures odds."""
        return self.get(f"seasons/{season}/futures")

    # News and Updates
    def get_news(self) -> Dict[str, Any]:
        """Get NFL news."""
        return self.get("news", base_type="site")

    def get_transactions(self) -> Dict[str, Any]:
        """Get NFL transactions."""
        return self.get("transactions")

    # Statistics and Leaders
    def get_current_leaders(self) -> Dict[str, Any]:
        """Get current statistical leaders."""
        return self.get("leaders")

    def get_season_leaders(self, season: int, season_type: int = 2) -> Dict[str, Any]:
        """Get season statistical leaders."""
        return self.get(f"seasons/{season}/types/{season_type}/leaders")


# Example usage:
if __name__ == "__main__":
    client = ESPNAPIClient()
    try:
        # Example: Get team statistics
        stats = client.get_team_statistics(1, 2023)  # team_id 1, 2023 season
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.utils.espn_api import ESPNAPIClient
import logging

logger = logging.getLogger(__name__)


class ESPNDataFetcher:
    """Fetches and processes NFL data from ESPN APIs for the Advanced Model"""

    # ESPN API Base URLs
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    STATS_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    FANTASY_URL = "https://fantasy.espn.com/apis/v3/games/ffl"

    def __init__(self):
        self.client = ESPNAPIClient()
        self.teams_cache = {}
        self.players_cache = {}
        self.season_cache = {}

    def get_game_data(self, game_id: str) -> Dict[str, Any]:
        """Get comprehensive game data including advanced stats"""
        try:
            # Basic game data
            game_summary = self.client.get_game_summary(game_id)

            # Advanced game stats
            detailed_stats = self._get_detailed_game_stats(game_id)
            play_by_play = self._get_play_by_play_data(game_id)
            probabilities = self._get_win_probability_data(game_id)
            matchup_stats = self._get_matchup_stats(game_id)
            drive_charts = self._get_drive_charts(game_id)
            player_stats = self._get_game_player_stats(game_id)

            return {
                "game_id": game_id,
                "summary": game_summary,
                "detailed_stats": detailed_stats,
                "play_by_play": play_by_play,
                "probabilities": probabilities,
                "matchup_stats": matchup_stats,
                "drive_charts": drive_charts,
                "player_stats": player_stats,
                "weather": self._get_weather_data(game_id),
                "stadium": self._get_stadium_data(game_id),
                "injuries": self._get_game_injuries(game_id),
                "odds": self._get_game_odds(game_id),
                "power_index": self._get_team_power_indexes(game_id),
            }
        except Exception as e:
            logger.error(f"Error fetching game data for {game_id}: {str(e)}")
            return {}

    def _get_detailed_game_stats(self, game_id: str) -> Dict[str, Any]:
        """Get detailed game statistics"""
        endpoint = (
            f"{self.STATS_URL}/events/{game_id}/competitions/{game_id}/statistics"
        )
        return self.client.make_request(endpoint)

    def _get_play_by_play_data(self, game_id: str) -> Dict[str, Any]:
        """Get play-by-play data with advanced analytics"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/plays"
        plays = self.client.make_request(endpoint)

        # Enrich with advanced play analytics
        if plays and "plays" in plays:
            for play in plays["plays"]:
                play["analytics"] = self._analyze_play(play)
        return plays

    def _get_win_probability_data(self, game_id: str) -> Dict[str, Any]:
        """Get detailed win probability data"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/winprobability"
        return self.client.make_request(endpoint)

    def _get_matchup_stats(self, game_id: str) -> Dict[str, Any]:
        """Get head-to-head matchup statistics"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/matchupStats"
        return self.client.make_request(endpoint)

    def _get_drive_charts(self, game_id: str) -> Dict[str, Any]:
        """Get drive charts and efficiency metrics"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/drives"
        return self.client.make_request(endpoint)

    def _get_game_player_stats(self, game_id: str) -> Dict[str, Any]:
        """Get detailed player statistics for the game"""
        endpoint = f"{self.STATS_URL}/events/{game_id}/competitions/{game_id}/competitors/statistics/players"
        return self.client.make_request(endpoint)

    def _get_weather_data(self, game_id: str) -> Dict[str, Any]:
        """Get detailed weather data for the game"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/weather"
        return self.client.make_request(endpoint)

    def _get_stadium_data(self, game_id: str) -> Dict[str, Any]:
        """Get stadium information and conditions"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/venue"
        return self.client.make_request(endpoint)

    def _get_game_injuries(self, game_id: str) -> Dict[str, Any]:
        """Get injury reports for both teams"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/injuries"
        return self.client.make_request(endpoint)

    def _get_game_odds(self, game_id: str) -> Dict[str, Any]:
        """Get betting odds and line movements"""
        endpoint = f"{self.BASE_URL}/summary/{game_id}/odds"
        return self.client.make_request(endpoint)

    def _get_team_power_indexes(self, game_id: str) -> Dict[str, Any]:
        """Get ESPN's Football Power Index (FPI) data"""
        endpoint = f"{self.STATS_URL}/powerindex"
        return self.client.make_request(endpoint)

    def get_team_season_stats(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get comprehensive team statistics for a season"""
        cache_key = f"{team_id}_{season}"
        if cache_key in self.season_cache:
            return self.season_cache[cache_key]

        try:
            # Basic team stats
            basic_stats = self._get_team_basic_stats(team_id, season)
            advanced_stats = self._get_team_advanced_stats(team_id, season)
            roster_stats = self._get_team_roster_stats(team_id, season)
            schedule_stats = self._get_team_schedule_stats(team_id, season)
            situational_stats = self._get_team_situational_stats(team_id, season)

            combined_stats = {
                "team_id": team_id,
                "season": season,
                "basic_stats": basic_stats,
                "advanced_stats": advanced_stats,
                "roster_stats": roster_stats,
                "schedule_stats": schedule_stats,
                "situational_stats": situational_stats,
                "rankings": self._get_team_rankings(team_id, season),
                "trends": self._get_team_trends(team_id, season),
                "injuries": self._get_team_injuries(team_id),
                "draft_picks": self._get_team_draft_picks(team_id, season),
                "power_index": self._get_team_power_index(team_id),
                "playoff_odds": self._get_team_playoff_odds(team_id),
            }

            self.season_cache[cache_key] = combined_stats
            return combined_stats

        except Exception as e:
            logger.error(f"Error fetching season stats for team {team_id}: {str(e)}")
            return {}

    def _get_team_basic_stats(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get basic team statistics"""
        endpoint = f"{self.STATS_URL}/seasons/{season}/teams/{team_id}/statistics"
        return self.client.make_request(endpoint)

    def _get_team_advanced_stats(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get advanced team analytics"""
        endpoint = f"{self.STATS_URL}/seasons/{season}/teams/{team_id}/advancedstats"
        return self.client.make_request(endpoint)

    def _get_team_roster_stats(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get detailed roster statistics"""
        endpoint = f"{self.STATS_URL}/seasons/{season}/teams/{team_id}/roster"
        return self.client.make_request(endpoint)

    def _get_team_schedule_stats(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get schedule statistics and metrics"""
        endpoint = f"{self.BASE_URL}/teams/{team_id}/schedule"
        return self.client.make_request(endpoint)

    def _get_team_situational_stats(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get situational statistics (red zone, third down, etc.)"""
        endpoint = f"{self.STATS_URL}/seasons/{season}/teams/{team_id}/situationalstats"
        return self.client.make_request(endpoint)

    def _get_team_rankings(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get team rankings in various categories"""
        endpoint = f"{self.STATS_URL}/seasons/{season}/teams/{team_id}/rankings"
        return self.client.make_request(endpoint)

    def _get_team_trends(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get team performance trends"""
        endpoint = f"{self.STATS_URL}/seasons/{season}/teams/{team_id}/trends"
        return self.client.make_request(endpoint)

    def _get_team_injuries(self, team_id: int) -> Dict[str, Any]:
        """Get team injury report"""
        endpoint = f"{self.BASE_URL}/teams/{team_id}/injuries"
        return self.client.make_request(endpoint)

    def _get_team_draft_picks(self, team_id: int, season: int) -> Dict[str, Any]:
        """Get team draft picks information"""
        endpoint = f"{self.BASE_URL}/teams/{team_id}/draftpicks"
        return self.client.make_request(endpoint)

    def _get_team_power_index(self, team_id: int) -> Dict[str, Any]:
        """Get team's ESPN Football Power Index (FPI)"""
        endpoint = f"{self.STATS_URL}/powerindex/teams/{team_id}"
        return self.client.make_request(endpoint)

    def _get_team_playoff_odds(self, team_id: int) -> Dict[str, Any]:
        """Get team's playoff odds and scenarios"""
        endpoint = f"{self.STATS_URL}/teams/{team_id}/playoffscenarios"
        return self.client.make_request(endpoint)

    def _analyze_play(self, play: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single play for advanced metrics"""
        return {
            "success_rate": self._calculate_success_rate(play),
            "expected_points": self._calculate_expected_points(play),
            "win_probability_added": self._calculate_wpa(play),
            "air_yards": self._calculate_air_yards(play),
            "yards_after_catch": self._calculate_yac(play),
            "pressure_rate": self._calculate_pressure_rate(play),
        }

    def _calculate_success_rate(self, play: Dict[str, Any]) -> float:
        """Calculate play success rate based on down and distance"""
        yards_gained = play.get("yards", 0)
        down = play.get("down", 1)
        distance = play.get("distance", 10)

        if down == 1:
            return yards_gained >= 0.4 * distance
        elif down == 2:
            return yards_gained >= 0.6 * distance
        else:
            return yards_gained >= distance

    def _calculate_expected_points(self, play: Dict[str, Any]) -> float:
        """Calculate expected points for the play"""
        # Implement expected points model based on field position and game situation
        field_position = play.get("fieldPosition", 50)
        down = play.get("down", 1)
        distance = play.get("distance", 10)

        # Basic EP model (can be enhanced with more sophisticated calculations)
        base_ep = (100 - field_position) * 0.07
        down_factor = 1 - ((down - 1) * 0.2)
        distance_factor = 1 - (distance / 20)

        return base_ep * down_factor * distance_factor

    def _calculate_wpa(self, play: Dict[str, Any]) -> float:
        """Calculate Win Probability Added for the play"""
        pre_wp = play.get("preWinProbability", 0.5)
        post_wp = play.get("postWinProbability", 0.5)
        return post_wp - pre_wp

    def _calculate_air_yards(self, play: Dict[str, Any]) -> float:
        """Calculate air yards for passing plays"""
        if play.get("type", {}).get("text", "").lower() == "pass":
            return float(play.get("airYards", 0))
        return 0.0

    def _calculate_yac(self, play: Dict[str, Any]) -> float:
        """Calculate yards after catch for passing plays"""
        if play.get("type", {}).get("text", "").lower() == "pass":
            total_yards = float(play.get("yards", 0))
            air_yards = self._calculate_air_yards(play)
            return max(0, total_yards - air_yards)
        return 0.0

    def _calculate_pressure_rate(self, play: Dict[str, Any]) -> float:
        """Calculate quarterback pressure rate"""
        if play.get("type", {}).get("text", "").lower() == "pass":
            return 1.0 if play.get("underPressure", False) else 0.0
        return 0.0

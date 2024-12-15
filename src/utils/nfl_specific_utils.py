"""NFL specific utilities for prediction system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLUtils:
    def __init__(self):
        """Initialize NFL utilities."""
        self.conferences = {
            "AFC": ["North", "South", "East", "West"],
            "NFC": ["North", "South", "East", "West"],
        }
        self.playoff_spots = {
            "division_winners": 4,  # per conference
            "wild_cards": 3,  # per conference
        }
        self.game_types = [
            "Regular Season",
            "Wild Card",
            "Divisional",
            "Conference",
            "Super Bowl",
        ]

    def calculate_playoff_probability(
        self, team_stats, games_remaining, conference_standings
    ):
        """Calculate team's probability of making playoffs."""
        try:
            current_wins = team_stats["wins"]
            current_losses = team_stats["losses"]
            current_ties = team_stats.get("ties", 0)
            win_percentage = (current_wins + 0.5 * current_ties) / (
                current_wins + current_losses + current_ties
            )

            # Project final record
            projected_wins = current_wins + (games_remaining * win_percentage)

            # Historical playoff threshold (usually around 9-10 wins)
            playoff_threshold = 9.5

            if projected_wins < playoff_threshold:
                return max(0.1, (projected_wins / playoff_threshold) * 0.5)
            else:
                return min(1.0, projected_wins / playoff_threshold)
        except Exception as e:
            logger.error(f"Error calculating playoff probability: {str(e)}")
            return None

    def analyze_weather_impact(self, game_weather, team_stats):
        """Analyze impact of weather conditions on team performance."""
        try:
            weather_factors = {
                "temperature": game_weather["temperature"],
                "wind_speed": game_weather["wind_speed"],
                "precipitation": game_weather["precipitation"],
                "is_dome": game_weather["is_dome"],
            }

            # Calculate weather impact score (0-1)
            if weather_factors["is_dome"]:
                return {
                    "weather_impact_score": 0,
                    "passing_game_impact": 0,
                    "running_game_impact": 0,
                    "kicking_game_impact": 0,
                }

            # Weather impacts
            temp_impact = self._calculate_temperature_impact(
                weather_factors["temperature"]
            )
            wind_impact = self._calculate_wind_impact(weather_factors["wind_speed"])
            precip_impact = self._calculate_precipitation_impact(
                weather_factors["precipitation"]
            )

            return {
                "weather_impact_score": (temp_impact + wind_impact + precip_impact) / 3,
                "passing_game_impact": wind_impact * 0.7 + precip_impact * 0.3,
                "running_game_impact": temp_impact * 0.3 + precip_impact * 0.7,
                "kicking_game_impact": wind_impact * 0.8 + precip_impact * 0.2,
            }
        except Exception as e:
            logger.error(f"Error analyzing weather impact: {str(e)}")
            return None

    def _calculate_temperature_impact(self, temperature):
        """Calculate impact of temperature on game performance."""
        try:
            # Assume optimal temperature is 70°F
            temp_diff = abs(temperature - 70)
            return min(1, temp_diff / 50)  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Error calculating temperature impact: {str(e)}")
            return None

    def _calculate_wind_impact(self, wind_speed):
        """Calculate impact of wind on game performance."""
        try:
            # Wind speed in mph
            return min(1, wind_speed / 25)  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Error calculating wind impact: {str(e)}")
            return None

    def _calculate_precipitation_impact(self, precipitation):
        """Calculate impact of precipitation on game performance."""
        try:
            # Precipitation in inches
            return min(1, precipitation / 0.5)  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Error calculating precipitation impact: {str(e)}")
            return None

    def calculate_qb_rating(self, qb_stats):
        """Calculate comprehensive quarterback rating."""
        try:
            # Traditional passer rating calculation
            a = ((qb_stats["completions"] / qb_stats["attempts"]) - 0.3) * 5
            b = ((qb_stats["yards"] / qb_stats["attempts"]) - 3) * 0.25
            c = (qb_stats["touchdowns"] / qb_stats["attempts"]) * 20
            d = 2.375 - ((qb_stats["interceptions"] / qb_stats["attempts"]) * 25)

            passer_rating = ((a + b + c + d) / 6) * 100

            # Additional metrics
            return {
                "passer_rating": passer_rating,
                "qbr": qb_stats.get("qbr", None),  # ESPN's QBR if available
                "completion_percentage": (
                    qb_stats["completions"] / qb_stats["attempts"]
                )
                * 100,
                "yards_per_attempt": qb_stats["yards"] / qb_stats["attempts"],
                "touchdown_percentage": (qb_stats["touchdowns"] / qb_stats["attempts"])
                * 100,
                "interception_percentage": (
                    qb_stats["interceptions"] / qb_stats["attempts"]
                )
                * 100,
            }
        except Exception as e:
            logger.error(f"Error calculating QB rating: {str(e)}")
            return None

    def analyze_injury_impact(self, team_injuries, depth_chart):
        """Analyze impact of injuries on team performance."""
        try:
            position_weights = {
                "QB": 0.3,
                "WR": 0.1,
                "RB": 0.1,
                "TE": 0.05,
                "OL": 0.15,
                "DL": 0.1,
                "LB": 0.1,
                "DB": 0.1,
            }

            impact_score = 0
            affected_positions = {}

            for injury in team_injuries:
                position = injury["position"]
                player_depth = injury["depth_chart_position"]

                # Calculate individual player impact
                position_impact = position_weights.get(position, 0.05)
                depth_factor = 1 / max(player_depth, 1)  # Higher impact for starters

                impact_score += position_impact * depth_factor

                # Track affected positions
                affected_positions[position] = affected_positions.get(position, 0) + 1

            return {
                "total_impact_score": min(1, impact_score),
                "affected_positions": affected_positions,
                "key_positions_affected": [
                    pos
                    for pos, count in affected_positions.items()
                    if position_weights.get(pos, 0) >= 0.1
                ],
            }
        except Exception as e:
            logger.error(f"Error analyzing injury impact: {str(e)}")
            return None

    def analyze_schedule_strength(self, schedule, team_records):
        """Analyze strength of schedule and future opponents."""
        try:
            past_opponents = schedule[schedule["date"] < datetime.now()]
            future_opponents = schedule[schedule["date"] >= datetime.now()]

            past_sos = np.mean(
                [
                    team_records[opp]["win_percentage"]
                    for opp in past_opponents["opponent"]
                ]
            )
            future_sos = np.mean(
                [
                    team_records[opp]["win_percentage"]
                    for opp in future_opponents["opponent"]
                ]
            )

            return {
                "past_strength_of_schedule": past_sos,
                "future_strength_of_schedule": future_sos,
                "sos_trend": future_sos - past_sos,
                "tough_games_remaining": len(
                    [
                        opp
                        for opp in future_opponents["opponent"]
                        if team_records[opp]["win_percentage"] > 0.6
                    ]
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing schedule strength: {str(e)}")
            return None

    def calculate_momentum_score(self, recent_games):
        """Calculate team's momentum based on recent performance."""
        try:
            weights = np.array(
                [0.35, 0.25, 0.2, 0.15, 0.05]
            )  # Most recent games weighted higher
            recent_results = recent_games["won"].values[-5:]  # Last 5 games

            momentum_score = np.sum(weights * recent_results)

            return {
                "momentum_score": momentum_score,
                "winning_streak": self._calculate_streak(recent_games["won"].values),
                "points_trend": np.mean(np.diff(recent_games["points_scored"].values)),
                "defense_trend": np.mean(
                    np.diff(recent_games["points_allowed"].values)
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return None

    def _calculate_streak(self, results):
        """Calculate current streak (positive for wins, negative for losses)."""
        try:
            streak = 0
            current_result = results[-1]

            for result in reversed(results):
                if result == current_result:
                    streak += 1 if current_result else -1
                else:
                    break

            return streak
        except Exception as e:
            logger.error(f"Error calculating streak: {str(e)}")
            return 0

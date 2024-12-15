"""
Player matchup analysis using Vedic astrology
"""
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class NFLPlayerAnalyzer:
    """Analyze player matchups using Vedic astrology"""

    def __init__(self):
        self.logger = logging.getLogger("nfl_metrics")

        # Configure metric weights
        self.qb_metrics = {
            "completion_percentage": 0.15,
            "yards_per_attempt": 0.15,
            "touchdown_percentage": 0.15,
            "interception_percentage": -0.15,
            "qb_rating": 0.20,
            "rushing_yards": 0.10,
            "sack_percentage": -0.10,
        }

        self.rb_metrics = {
            "yards_per_carry": 0.25,
            "rushing_yards": 0.20,
            "touchdowns": 0.20,
            "fumbles": -0.15,
            "yards_after_contact": 0.20,
        }

        self.wr_metrics = {
            "receptions": 0.15,
            "yards_per_reception": 0.20,
            "catch_percentage": 0.20,
            "touchdowns": 0.15,
            "yards_after_catch": 0.15,
            "drops": -0.15,
        }

        # Astrological influence weights
        self.astro_influences = {
            "mars": 0.3,  # Physical performance
            "jupiter": 0.2,  # Success and expansion
            "sun": 0.15,  # Vitality and confidence
            "mercury": 0.15,  # Agility and decision-making
            "moon": 0.1,  # Emotional balance
            "venus": 0.05,  # Team chemistry
            "saturn": 0.05,  # Discipline and responsibility
        }

    def analyze_player_performance(
        self, player_data: Dict, game_time: datetime, astro_factors: Dict
    ) -> Dict[str, float]:
        """Analyze player performance considering both statistics and astrological factors"""

        # Get base performance metrics
        base_metrics = self._calculate_base_metrics(player_data)

        # Calculate astrological influence
        astro_influence = self._calculate_astro_influence(astro_factors)

        # Combine metrics with astrological influence
        final_metrics = {}
        for metric, value in base_metrics.items():
            final_metrics[metric] = value * (1 + astro_influence)

        # Add overall rating
        final_metrics["overall_rating"] = sum(final_metrics.values()) / len(
            final_metrics
        )

        return final_metrics

    def _calculate_base_metrics(self, player_data: Dict) -> Dict[str, float]:
        """Calculate base performance metrics based on position"""
        position = player_data.get("position", "").upper()
        stats = player_data.get("stats", {})

        if position == "QB":
            return self._calculate_qb_metrics(stats)
        elif position == "RB":
            return self._calculate_rb_metrics(stats)
        elif position in ["WR", "TE"]:
            return self._calculate_wr_metrics(stats)
        else:
            return {}

    def _calculate_qb_metrics(self, stats: Dict) -> Dict[str, float]:
        """Calculate quarterback-specific metrics"""
        metrics = {}
        for metric, weight in self.qb_metrics.items():
            if metric in stats:
                metrics[metric] = stats[metric] * weight
        return metrics

    def _calculate_rb_metrics(self, stats: Dict) -> Dict[str, float]:
        """Calculate running back-specific metrics"""
        metrics = {}
        for metric, weight in self.rb_metrics.items():
            if metric in stats:
                metrics[metric] = stats[metric] * weight
        return metrics

    def _calculate_wr_metrics(self, stats: Dict) -> Dict[str, float]:
        """Calculate receiver-specific metrics"""
        metrics = {}
        for metric, weight in self.wr_metrics.items():
            if metric in stats:
                metrics[metric] = stats[metric] * weight
        return metrics

    def _calculate_astro_influence(self, astro_factors: Dict) -> float:
        """Calculate overall astrological influence on player performance"""
        influence = 0.0
        for planet, weight in self.astro_influences.items():
            if planet in astro_factors:
                influence += astro_factors[planet] * weight
        return influence

    def analyze_matchup(
        self,
        game_time: datetime,
        stadium_location: Tuple[float, float],
        player1_data: Dict[str, Any],
        player2_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze matchup between two players"""
        try:
            from models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator

            calculator = NFLVedicCalculator()

            # Get planetary positions
            planet_positions = calculator.calculate_planet_positions(game_time)

            # Get zodiac strengths
            zodiac_strengths = calculator.calculate_zodiac_strengths(planet_positions)

            # Get moon nakshatra
            moon_nakshatra = calculator.calculate_moon_nakshatra(game_time)

            # Calculate player scores
            player1_score = calculator.calculate_team_score(
                player1_data["team"], planet_positions, zodiac_strengths, moon_nakshatra
            )
            player2_score = calculator.calculate_team_score(
                player2_data["team"], planet_positions, zodiac_strengths, moon_nakshatra
            )

            # Calculate advantage
            total_score = player1_score + player2_score
            if total_score > 0:
                advantage = (player1_score - player2_score) / total_score
            else:
                advantage = 0

            # Generate recommendations
            recommendations = []
            if abs(advantage) > 0.2:
                favored_player = (
                    player1_data["name"] if advantage > 0 else player2_data["name"]
                )
                recommendations.append(
                    f"Strong astrological advantage for {favored_player}"
                )
            else:
                recommendations.append("No clear astrological advantage")

            if moon_nakshatra in [1, 6, 11, 16, 21, 26]:  # Auspicious nakshatras
                recommendations.append("Favorable time for both players")
            elif moon_nakshatra in [4, 9, 14, 19, 24]:  # Challenging nakshatras
                recommendations.append("Challenging conditions for both players")

            return {
                "advantage": advantage,
                "player1_score": player1_score,
                "player2_score": player2_score,
                "recommendations": recommendations,
                "astrological_factors": {
                    "moon_nakshatra": moon_nakshatra,
                    "zodiac_strengths": zodiac_strengths,
                },
            }

        except Exception as e:
            self.logger.error(f"Error analyzing player matchup: {str(e)}")
            return {
                "advantage": 0,
                "recommendations": ["Unable to calculate astrological factors"],
                "error": str(e),
            }

    def _generate_matchup_recommendations(
        self, advantage: float, pos1: str, pos2: str
    ) -> List[str]:
        """Generate strategic recommendations based on matchup analysis"""
        recommendations = []

        if advantage > 0.2:
            recommendations.append(
                f"Strong advantage for {pos1} - exploit matchup early"
            )
        elif advantage < -0.2:
            recommendations.append(
                f"Challenging matchup - consider additional protection/support"
            )

        # Position-specific recommendations
        if pos1 == "QB":
            if advantage > 0:
                recommendations.append("Favorable conditions for deep passing game")
            else:
                recommendations.append("Focus on quick releases and short passes")
        elif pos1 == "RB":
            if advantage > 0:
                recommendations.append("Favorable conditions for outside runs")
            else:
                recommendations.append("Focus on inside runs and pass protection")

        return recommendations

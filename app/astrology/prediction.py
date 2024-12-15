"""
Main prediction interface for NFL games using astrological analysis
"""
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator
from .stadium_data import get_stadium_coordinates
from .analyzers.astro_analyzer import NFLAstroAnalyzer
from .analyzers.betting_analyzer import NFLBettingAnalyzer
from .analyzers.player_metrics import NFLPlayerAnalyzer

logger = logging.getLogger(__name__)


class NFLAstrologyPredictor:
    """NFL game prediction system using Vedic astrology"""

    def __init__(self):
        self.calculator = NFLVedicCalculator()
        self.astro_analyzer = NFLAstroAnalyzer()
        self.betting_analyzer = NFLBettingAnalyzer()
        self.player_analyzer = NFLPlayerAnalyzer()

    def predict_game(
        self,
        game_time: datetime,
        home_team: str,
        away_team: str,
        weather_conditions: Optional[str] = None,
        include_betting: bool = False,
    ) -> Dict:
        """Generate comprehensive prediction for an NFL game using Vedic astrology."""
        try:
            # Get planetary positions for game time
            planet_positions = self.calculator.calculate_planet_positions(game_time)

            # Get zodiac strengths
            zodiac_strengths = self.calculator.calculate_zodiac_strengths(
                planet_positions
            )

            # Get moon nakshatra
            moon_nakshatra = self.calculator.calculate_moon_nakshatra(game_time)

            # Calculate team scores
            home_score = self.calculator.calculate_team_score(
                home_team, planet_positions, zodiac_strengths, moon_nakshatra
            )
            away_score = self.calculator.calculate_team_score(
                away_team, planet_positions, zodiac_strengths, moon_nakshatra
            )

            # Normalize scores
            total_score = home_score + away_score
            if total_score > 0:
                home_probability = home_score / total_score
                away_probability = away_score / total_score
            else:
                home_probability = 0.5
                away_probability = 0.5

            # Calculate confidence based on multiple astrological factors
            planet_strengths = {
                planet: self.calculator.calculate_planet_strength(game_time, planet)
                for planet in ["Sun", "Moon", "Mars", "Jupiter", "Venus", "Saturn"]
            }

            # Weight different factors for confidence
            confidence_factors = [
                sum(planet_strengths.values())
                / (10 * len(planet_strengths)),  # Planetary strength (0-1)
                abs(home_probability - 0.5) * 2,  # Edge factor (0-1)
                max(zodiac_strengths.values()) / 10,  # Zodiac strength (0-1)
                1.0
                if moon_nakshatra in [1, 6, 11, 16, 21, 26]
                else 0.5,  # Auspicious nakshatra bonus
            ]

            # Calculate weighted confidence
            weights = [0.4, 0.3, 0.2, 0.1]  # Weights for each factor
            confidence = sum(f * w for f, w in zip(confidence_factors, weights))

            # Categorize confidence
            if confidence >= 0.8:
                confidence_category = "Strong"
            elif confidence >= 0.65:
                confidence_category = "Good"
            elif confidence >= 0.5:
                confidence_category = "Moderate"
            else:
                confidence_category = "Weak"

            result = {
                "base_prediction": {
                    "total_home_score": home_score,
                    "total_away_score": away_score,
                    "planetary_positions": planet_positions,
                    "zodiac_strengths": zodiac_strengths,
                    "moon_nakshatra": moon_nakshatra,
                }
            }

            if include_betting:
                result["betting_analysis"] = {
                    "home_probability": home_probability,
                    "away_probability": away_probability,
                    "confidence": confidence,
                    "confidence_category": confidence_category,
                    "confidence_factors": {
                        "planetary_strength": confidence_factors[0],
                        "edge_factor": confidence_factors[1],
                        "zodiac_strength": confidence_factors[2],
                        "nakshatra_bonus": confidence_factors[3],
                    },
                }

                # Generate betting recommendation
                edge = abs(home_probability - 0.5)
                if confidence >= 0.7 and edge >= 0.1:
                    pick = home_team if home_probability > 0.5 else away_team
                    result["betting_recommendation"] = {
                        "recommendation": f"Consider betting on {pick}",
                        "reason": f"Strong astrological indicators with {confidence:.1%} confidence",
                    }
                else:
                    result["betting_recommendation"] = {
                        "recommendation": "No strong bet",
                        "reason": "Insufficient astrological edge or confidence",
                    }

            return result

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None

    def analyze_player_matchup(
        self,
        game_time: datetime,
        stadium_location: Tuple[float, float],
        player1_data: Dict[str, Any],
        player2_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze matchup between two players using Vedic astrology"""
        try:
            # Get planetary positions
            planet_positions = self.calculator.calculate_planet_positions(game_time)

            # Get zodiac strengths
            zodiac_strengths = self.calculator.calculate_zodiac_strengths(
                planet_positions
            )

            # Get moon nakshatra
            moon_nakshatra = self.calculator.calculate_moon_nakshatra(game_time)

            # Calculate player scores
            player1_score = self.calculator.calculate_team_score(
                player1_data["team"], planet_positions, zodiac_strengths, moon_nakshatra
            )
            player2_score = self.calculator.calculate_team_score(
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
            logger.error(f"Error analyzing player matchup: {str(e)}")
            return {
                "advantage": 0,
                "recommendations": ["Unable to calculate astrological factors"],
                "error": str(e),
            }

    def get_favorable_times(
        self, team: str, start_date: datetime, end_date: datetime
    ) -> list:
        """Find favorable game times for a team"""
        coords = get_stadium_coordinates(team)
        if not coords:
            raise ValueError(f"Could not find stadium coordinates for {team}")

        return self.astro_analyzer.get_favorable_times(
            date=start_date,
            location=coords,
            window_hours=int((end_date - start_date).total_seconds() / 3600),
        )

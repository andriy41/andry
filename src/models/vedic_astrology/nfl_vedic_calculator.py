"""
Vedic astrology calculator for NFL predictions.
"""

from typing import Dict, Any, List, Optional
import swisseph as swe
from datetime import datetime
import pytz
import logging

from .calculations import (
    calculate_shadbala,
    calculate_vimshottari_dasha,
    calculate_ashtakavarga,
    calculate_yogas,
)


class NFLVedicCalculator:
    def __init__(self):
        self.eastern_tz = pytz.timezone("US/Eastern")

    def calculate_game_factors(
        self, game_time: datetime, lat: float, lon: float
    ) -> Dict[str, Any]:
        """
        Calculate Vedic astrological factors for a game.

        Args:
            game_time: Game start time (can be naive or timezone-aware)
            lat: Stadium latitude
            lon: Stadium longitude

        Returns:
            Dictionary containing Vedic factors
        """
        # Convert to eastern time if needed
        if game_time.tzinfo is None:
            game_time = self.eastern_tz.localize(game_time)
        elif game_time.tzinfo != self.eastern_tz:
            game_time = game_time.astimezone(self.eastern_tz)

        # Convert to Julian date
        jd = swe.julday(
            game_time.year,
            game_time.month,
            game_time.day,
            game_time.hour + game_time.minute / 60.0,
        )

        # Get planet positions
        planets = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
        planet_map = {
            "Sun": swe.SUN,
            "Moon": swe.MOON,
            "Mars": swe.MARS,
            "Mercury": swe.MERCURY,
            "Jupiter": swe.JUPITER,
            "Venus": swe.VENUS,
            "Saturn": swe.SATURN,
        }

        positions = {}
        for planet in planets:
            planet_info = swe.calc_ut(jd, planet_map[planet])[0]
            positions[planet] = {
                "longitude": planet_info[0],
                "latitude": planet_info[1],
                "speed": planet_info[3],
            }

        # Calculate house cusps
        house_cusps = self._calculate_house_cusps(jd, lat, lon)

        # Map planets to houses
        houses = {}
        for planet, pos in positions.items():
            for i in range(12):
                next_cusp = house_cusps[(i + 1) % 12]
                if (house_cusps[i] <= pos["longitude"] < next_cusp) or (
                    house_cusps[i] > next_cusp
                    and (
                        pos["longitude"] >= house_cusps[i]
                        or pos["longitude"] < next_cusp
                    )
                ):
                    houses[planet] = i + 1
                    break

        # Calculate Moon longitude for dasha
        moon_long = positions["Moon"]["longitude"]

        # Calculate various Vedic factors
        shadbala = calculate_shadbala(jd, lat, lon)
        dasha = calculate_vimshottari_dasha(moon_long, jd)
        ashtakavarga = calculate_ashtakavarga(
            "Sun", positions, houses
        )  # Calculate for Sun's perspective
        yogas = calculate_yogas(jd)

        return {
            "shadbala": shadbala,
            "dasha": dasha,
            "ashtakavarga": ashtakavarga,
            "yogas": yogas,
            "jd": jd,
            "moon_longitude": moon_long,
            "positions": positions,
            "houses": houses,
        }

    def predict_outcome(self, game_factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict game outcome based on Vedic factors.

        Args:
            game_factors: Dictionary of Vedic factors

        Returns:
            Dictionary containing predictions
        """
        # Calculate overall strength
        total_strength = (
            sum(game_factors["shadbala"].values())
            + len(game_factors["yogas"]) * 0.5
            + sum(game_factors["ashtakavarga"].values()) / 100.0
        )

        # Calculate confidence
        confidence = min(0.9, total_strength / 20.0)

        # Make predictions
        predicted_total = 45.0 + total_strength

        return {
            "predicted_total": predicted_total,
            "confidence": confidence,
            "recommendation": "OVER" if predicted_total > 45 else "UNDER",
        }

    def predict_game_outcome(
        self,
        game_time: datetime,
        home_team: str,
        away_team: str,
        stadium_lat: float = 40.0,
        stadium_lon: float = -74.0,
    ) -> Dict[str, Any]:
        """
        Predict game outcome using Vedic astrology.

        Args:
            game_time: Game start time
            home_team: Home team name
            away_team: Away team name
            stadium_lat: Stadium latitude (default: NYC area)
            stadium_lon: Stadium longitude (default: NYC area)

        Returns:
            Dictionary containing predictions
        """
        # Calculate game factors
        game_factors = self.calculate_game_factors(game_time, stadium_lat, stadium_lon)

        # Get base prediction
        base_pred = self.predict_outcome(game_factors)

        # Calculate team-specific adjustments based on Vedic factors
        home_strength = (
            game_factors["shadbala"].get("Sun", 0)
            + game_factors["shadbala"].get("Jupiter", 0)  # Sun represents home
            + game_factors["shadbala"].get(  # Jupiter represents favor
                "Mars", 0
            )  # Mars represents victory
        ) / 3.0

        away_strength = (
            game_factors["shadbala"].get("Moon", 0)
            + game_factors["shadbala"].get("Mercury", 0)  # Moon represents away
            + game_factors["shadbala"].get(  # Mercury represents travel
                "Venus", 0
            )  # Venus represents harmony
        ) / 3.0

        # Calculate spread and moneyline based on strengths
        spread = (home_strength - away_strength) * 7  # Convert to points
        home_win_prob = 0.5 + (home_strength - away_strength) * 0.3

        # Convert probability to moneyline
        if home_win_prob > 0.5:
            home_ml = -100 / (home_win_prob - 1)
            away_ml = (1 - home_win_prob) * 100
        else:
            home_ml = (1 - home_win_prob) * 100
            away_ml = -100 / (1 - home_win_prob - 1)

        return {
            "total_points": base_pred["predicted_total"],
            "total_confidence": base_pred["confidence"],
            "total_recommendation": base_pred["recommendation"],
            "spread": spread,
            "spread_confidence": min(0.9, abs(spread / 7)),
            "home_moneyline": int(home_ml),
            "away_moneyline": int(away_ml),
            "win_confidence": min(0.9, abs(home_win_prob - 0.5) * 2),
        }

    def _calculate_house_cusps(self, jd: float, lat: float, lon: float) -> List[float]:
        # Calculate house cusps using Placidus house system
        # This is a simplified implementation and may not be accurate for all latitudes
        ascendant = swe.houses(jd, lat, lon)[0]
        house_cusps = [ascendant + i * 30 for i in range(12)]
        return house_cusps

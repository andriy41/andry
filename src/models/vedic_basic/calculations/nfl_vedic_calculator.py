"""
NFL-specific Vedic astrology calculator
Adapted from the vedic_astrology_calculator project
"""
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)

import swisseph as swe
from typing import Dict, Any, List, Tuple
import math
from datetime import datetime, timezone
import logging

from src.models.vedic_basic.calculations.planetary_calculations import (
    calculate_planet_strength,
    calculate_planetary_alignment_score,
    calculate_sarvashtakavarga,
    check_shadbala,
    check_vimshottari_dasa,
    calculate_divisional_strength,
    calculate_bhava_chalit_aspects,
    calculate_special_lagnas,
    calculate_victory_yogas,
    calculate_nakshatra_tara,
    calculate_sublords,
    calculate_retrograde_impact,
    calculate_moon_phase,
    calculate_muhurta_score,
    calculate_hora_score,
    PLANETS,
    PLANET_MAP,
    HOUSE_LORDS,
    NAKSHATRA_LORDS,
)

logger = logging.getLogger(__name__)


class NFLVedicCalculator:
    """Calculate Vedic astrological factors for NFL games"""

    def __init__(self):
        # Initialize Swiss Ephemeris
        swe.set_ephe_path("/usr/share/ephe")  # This path may need to be adjusted

        # Define relevant planets for NFL games
        self.planets = PLANETS

        # Planet weights for NFL games
        self.planet_weights = {
            "sun": 0.15,
            "moon": 0.10,
            "mars": 0.20,  # Higher weight for Mars (aggression/competition)
            "mercury": 0.15,
            "jupiter": 0.15,
            "venus": 0.05,
            "saturn": 0.10,
            "rahu": 0.05,
            "ketu": 0.05,
        }

        # Define houses relevant to sports
        self.relevant_houses = {
            1: 0.15,  # Overall team strength
            3: 0.10,  # Short drives and quick plays
            6: 0.15,  # Defense and obstacles
            7: 0.15,  # Opposition and competition
            10: 0.20,  # Success and achievement
            11: 0.15,  # Gains and scoring
            2: 0.10,  # Possession and resources
        }

    def calculate_game_features(
        self,
        game_time: datetime,
        latitude: float,
        longitude: float,
        home_team: str,
        away_team: str,
    ) -> List[float]:
        """Calculate game features for the model"""
        try:
            # Convert to Julian date
            jd = self._datetime_to_jd(game_time)

            # Calculate planetary positions
            positions = {}
            for planet_name, planet_id in self.planets.items():
                try:
                    result = swe.calc_ut(jd, planet_id)
                    if isinstance(result, tuple) and len(result) >= 2:
                        values = result[0]  # First element contains the position values
                        if isinstance(values, (tuple, list)) and len(values) >= 3:
                            positions[planet_name] = {
                                "longitude": float(values[0]),
                                "latitude": float(values[1]),
                                "distance": float(values[2]),
                            }
                        else:
                            logger.warning(
                                f"Invalid values format for planet {planet_name}: {values}"
                            )
                            positions[planet_name] = {
                                "longitude": 0.0,
                                "latitude": 0.0,
                                "distance": 0.0,
                            }
                    else:
                        logger.warning(
                            f"Invalid result format for planet {planet_name}: {result}"
                        )
                        positions[planet_name] = {
                            "longitude": 0.0,
                            "latitude": 0.0,
                            "distance": 0.0,
                        }
                except Exception as e:
                    logger.error(f"Error calculating position for {planet_name}: {e}")
                    positions[planet_name] = {
                        "longitude": 0.0,
                        "latitude": 0.0,
                        "distance": 0.0,
                    }

            # Calculate basic strengths
            features = []

            # Add individual planet strengths
            for planet in [
                "sun",
                "moon",
                "mars",
                "mercury",
                "jupiter",
                "venus",
                "saturn",
            ]:
                strength = self._calculate_planet_strength(
                    positions[planet.lower()], jd, latitude, longitude
                )
                features.append(strength)

            # Add Rahu and Ketu strengths
            rahu_strength = self._calculate_planet_strength(
                positions["rahu"], jd, latitude, longitude
            )
            ketu_strength = self._calculate_planet_strength(
                positions["ketu"], jd, latitude, longitude
            )
            features.extend([rahu_strength, ketu_strength])

            # Calculate house lord strengths (assuming Aries ascendant for simplicity)
            ascendant_lord = self._calculate_planet_strength(
                positions["mars"], jd, latitude, longitude
            )
            tenth_lord = self._calculate_planet_strength(
                positions["saturn"], jd, latitude, longitude
            )
            features.extend([ascendant_lord, tenth_lord])

            # Calculate team-specific features
            home_yoga = self._calculate_team_yoga(home_team, positions)
            away_yoga = self._calculate_team_yoga(away_team, positions)

            # Calculate nakshatra scores
            moon_longitude = positions["moon"]["longitude"]
            home_nakshatra = (moon_longitude * 27 / 360) % 27
            away_nakshatra = (home_nakshatra + 14) % 27  # Opposite nakshatra
            home_nakshatra_score = (home_nakshatra + 1) / 27
            away_nakshatra_score = (away_nakshatra + 1) / 27

            # Calculate general features
            planetary_alignment = self._calculate_planetary_alignment(positions)
            moon_phase = self._calculate_moon_phase(positions)

            # Add team-specific and general features
            features.extend(
                [
                    home_yoga,
                    away_yoga,
                    home_nakshatra_score,
                    away_nakshatra_score,
                    planetary_alignment,
                    moon_phase,
                ]
            )

            return features

        except Exception as e:
            logger.error(f"Error calculating game features: {e}")
            # Return default features in case of error
            return [0.5] * 17  # 17 is the number of features we calculate

    def _datetime_to_jd(self, dt: datetime) -> float:
        """Convert datetime to Julian day"""
        return swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60.0)

    def _calculate_planet_strength(
        self, planet_data: Dict[str, float], jd: float, lat: float, lon: float
    ) -> float:
        """Calculate planetary strength based on position and dignity"""
        try:
            # Basic strength from longitude position (0-1)
            long_strength = (math.cos(math.radians(planet_data["longitude"])) + 1) / 2

            # Strength from latitude (closer to ecliptic is stronger)
            lat_strength = 1 - abs(planet_data["latitude"]) / 90

            # Strength from distance (closer is stronger)
            dist_strength = 1 / (1 + planet_data["distance"])

            # Combine strengths with weights
            total_strength = (
                0.5 * long_strength + 0.3 * lat_strength + 0.2 * dist_strength
            )

            return max(0.0, min(1.0, total_strength))  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error calculating planet strength: {e}")
            return 0.5  # Return neutral strength in case of error

    def _calculate_team_yoga(
        self, team: str, positions: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate yoga (planetary combination) strength for a team"""
        try:
            # Simple yoga calculation based on favorable planet combinations
            mars_jupiter_angle = abs(
                positions["mars"]["longitude"] - positions["jupiter"]["longitude"]
            )
            sun_moon_angle = abs(
                positions["sun"]["longitude"] - positions["moon"]["longitude"]
            )

            # Consider angles both ways around the zodiac
            mars_jupiter_angle = min(mars_jupiter_angle, 360 - mars_jupiter_angle)
            sun_moon_angle = min(sun_moon_angle, 360 - sun_moon_angle)

            # Strong yoga when planets are conjunct (0°) or in trine (120°)
            mars_jupiter_strength = (
                1
                - min(
                    abs(mars_jupiter_angle % 120), abs(120 - mars_jupiter_angle % 120)
                )
                / 60
            )
            sun_moon_strength = (
                1 - min(abs(sun_moon_angle % 120), abs(120 - sun_moon_angle % 120)) / 60
            )

            # Combine yoga strengths
            yoga_strength = 0.6 * mars_jupiter_strength + 0.4 * sun_moon_strength

            return max(0.0, min(1.0, yoga_strength))  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error calculating team yoga: {e}")
            return 0.5  # Return neutral strength in case of error

    def _calculate_planetary_alignment(
        self, positions: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate overall planetary alignment score"""
        try:
            # Count favorable aspects between planets
            favorable_aspects = 0
            total_aspects = 0

            planet_list = list(positions.keys())
            for i in range(len(planet_list)):
                for j in range(i + 1, len(planet_list)):
                    p1 = planet_list[i]
                    p2 = planet_list[j]

                    # Calculate angle between planets
                    angle = abs(positions[p1]["longitude"] - positions[p2]["longitude"])
                    angle = min(angle, 360 - angle)

                    # Check for favorable aspects (0°, 60°, 120°)
                    if (
                        angle < 10
                        or abs(angle - 60) < 10  # Conjunction
                        or abs(angle - 120) < 10  # Sextile
                    ):  # Trine
                        favorable_aspects += 1
                    total_aspects += 1

            return favorable_aspects / total_aspects if total_aspects > 0 else 0.5

        except Exception as e:
            logger.error(f"Error calculating planetary alignment: {e}")
            return 0.5  # Return neutral alignment in case of error

    def _calculate_moon_phase(self, positions: Dict[str, Dict[str, float]]) -> float:
        """Calculate moon phase score"""
        try:
            # Calculate angle between Sun and Moon
            angle = abs(positions["sun"]["longitude"] - positions["moon"]["longitude"])
            angle = min(angle, 360 - angle)

            # Convert to moon phase (0 = new moon, 0.5 = full moon, 1 = back to new moon)
            moon_phase = angle / 360

            # Score is higher near full moon (0.5) and lower near new moon (0 or 1)
            phase_score = 1 - abs(moon_phase - 0.5) * 2

            return max(0.0, min(1.0, phase_score))  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error calculating moon phase: {e}")
            return 0.5  # Return neutral phase in case of error

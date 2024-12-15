"""Planet-related calculations for NFL Vedic Astrology."""

import logging
from typing import Dict, Any, Optional, List

from .astro_constants import ZODIAC_SIGNS

logger = logging.getLogger(__name__)


def calculate_planet_strength(
    positions: Dict[str, Dict[str, float]],
    planet_name: str,
    home_data: Optional[Dict] = None,
    away_data: Optional[Dict] = None,
) -> float:
    """Calculate the strength of a planet at given time."""
    try:
        if planet_name not in positions:
            return 0.5

        planet_pos = positions[planet_name]
        strength = 1.0

        # Base strength from altitude
        altitude = planet_pos.get("altitude", 0)
        strength *= (
            (1 + (altitude / 90.0)) if altitude > 0 else (1 - abs(altitude / 90.0))
        )

        # Phase influence
        if "phase" in planet_pos:
            phase = planet_pos["phase"] / 100.0
            strength *= 0.5 + phase

        # Team data influence
        if home_data and away_data:
            home_sign = home_data.get("zodiac_sign", "")
            away_sign = away_data.get("zodiac_sign", "")
            planet_long = planet_pos.get("longitude", 0)

            # Check if planet is in team's sign
            if is_in_sign(planet_long, home_sign, ZODIAC_SIGNS):
                strength *= 1.2
            elif is_in_sign(planet_long, away_sign, ZODIAC_SIGNS):
                strength *= 0.8

        return min(max(strength, 0.1), 2.0)

    except Exception as e:
        logger.error(f"Error calculating planet strength: {str(e)}")
        return 0.5


def is_in_sign(longitude: float, sign: str, zodiac_signs: List[str]) -> bool:
    """Check if a celestial longitude is in a given zodiac sign."""
    try:
        if not sign or sign not in zodiac_signs:
            return False

        sign_num = zodiac_signs.index(sign)
        sign_start = sign_num * 30
        sign_end = sign_start + 30

        return sign_start <= longitude < sign_end

    except Exception as e:
        logger.error(f"Error checking sign position: {str(e)}")
        return False


def count_beneficial_aspects(positions: Dict[str, Dict[str, float]]) -> int:
    """Count the number of beneficial aspects between planets."""
    try:
        beneficial_count = 0
        planets = list(positions.keys())

        for i in range(len(planets)):
            for j in range(i + 1, len(planets)):
                p1, p2 = planets[i], planets[j]

                if "longitude" not in positions[p1] or "longitude" not in positions[p2]:
                    continue

                # Calculate angular separation
                angle = abs(positions[p1]["longitude"] - positions[p2]["longitude"])
                angle = min(angle, 360 - angle)  # Use shorter arc

                # Check for beneficial aspects (trine: 120°, sextile: 60°)
                if abs(angle - 120) <= 10 or abs(angle - 60) <= 6:
                    beneficial_count += 1

        return beneficial_count

    except Exception as e:
        logger.error(f"Error counting beneficial aspects: {str(e)}")
        return 0


def count_malefic_aspects(positions: Dict[str, Dict[str, float]]) -> int:
    """Count the number of malefic aspects between planets."""
    try:
        malefic_count = 0
        planets = list(positions.keys())

        for i in range(len(planets)):
            for j in range(i + 1, len(planets)):
                p1, p2 = planets[i], planets[j]

                if "longitude" not in positions[p1] or "longitude" not in positions[p2]:
                    continue

                # Calculate angular separation
                angle = abs(positions[p1]["longitude"] - positions[p2]["longitude"])
                angle = min(angle, 360 - angle)  # Use shorter arc

                # Check for malefic aspects (square: 90°, opposition: 180°)
                if abs(angle - 90) <= 8 or abs(angle - 180) <= 10:
                    malefic_count += 1

        return malefic_count

    except Exception as e:
        logger.error(f"Error counting malefic aspects: {str(e)}")
        return 0

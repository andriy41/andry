"""Team-related calculations for NFL Vedic Astrology."""

import logging
from typing import Dict, Optional, List

from .astro_constants import ZODIAC_SIGNS

logger = logging.getLogger(__name__)


def resolve_team_name(team_code: str, team_data: Dict, team_aliases: Dict) -> str:
    """Resolve team code to full team name, handling aliases."""
    try:
        if team_code in team_data:
            return team_code

        if team_code in team_aliases:
            return team_aliases[team_code]

        logger.warning(f"Unknown team code: {team_code}")
        return team_code

    except Exception as e:
        logger.error(f"Error resolving team name for {team_code}: {str(e)}")
        return team_code


def calculate_team_strength(
    team_data: Dict,
    planet_positions: Dict[str, Dict[str, float]],
    zodiac_signs: List[str],
    week_number: int,
) -> float:
    """Calculate overall astrological strength for a team."""
    try:
        if not team_data:
            return 0.5

        # Get team's ruling planet and zodiac sign
        ruling_planet = team_data.get("ruling_planet", "").lower()
        team_sign = team_data.get("zodiac_sign", "")

        # Calculate ruling planet strength (0-1)
        planet_strength = 0.5
        if ruling_planet in planet_positions:
            planet_data = planet_positions[ruling_planet]
            planet_strength = (
                planet_data.get("altitude", 0) / 90
                + (  # Altitude factor
                    1 if not planet_data.get("retrograde", False) else 0.5
                )
                + planet_data.get("phase", 0.5)  # Retrograde factor  # Phase factor
            ) / 3

        # Calculate zodiac sign strength (0-1)
        sign_strength = _calculate_sign_strength(
            team_sign, planet_positions, zodiac_signs
        )

        # Calculate week factor (games become more important later in season)
        week_factor = min(week_number / 18.0, 1.0)

        # Weight the factors
        weights = {"planet_strength": 0.4, "sign_strength": 0.35, "week_factor": 0.25}

        # Calculate final score
        score = (
            weights["planet_strength"] * planet_strength
            + weights["sign_strength"] * sign_strength
            + weights["week_factor"] * week_factor
        )

        # Ensure score is between 0 and 1
        return max(0.1, min(0.9, score))

    except Exception as e:
        logger.error(f"Error calculating team strength: {str(e)}")
        return 0.5


def _calculate_sign_strength(
    sign: str, planet_positions: Dict[str, Dict[str, float]], zodiac_signs: List[str]
) -> float:
    """Calculate the strength of a zodiac sign based on planetary positions."""
    try:
        if not sign or sign not in zodiac_signs:
            return 0.5

        sign_num = zodiac_signs.index(sign)
        sign_start = sign_num * 30
        sign_end = sign_start + 30

        # Calculate how many planets are in or aspect the sign
        strength = 0.0
        count = 0

        for planet, pos in planet_positions.items():
            if "longitude" not in pos:
                continue

            planet_long = pos["longitude"]

            # Planet in sign
            if sign_start <= planet_long < sign_end:
                strength += 1.0
                count += 1

            # Trine aspect (120°)
            trine1 = (planet_long + 120) % 360
            trine2 = (planet_long + 240) % 360
            if (sign_start <= trine1 < sign_end) or (sign_start <= trine2 < sign_end):
                strength += 0.5
                count += 1

        # Normalize strength
        return strength / max(count, 1) if count > 0 else 0.5

    except Exception as e:
        logger.error(f"Error calculating sign strength: {str(e)}")
        return 0.5

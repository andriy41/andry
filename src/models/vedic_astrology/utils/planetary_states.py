"""Module for calculating planetary states and dignities."""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Planetary dignities and debilitations
PLANETARY_DIGNITIES = {
    "Sun": {
        "exaltation": 10,  # Aries
        "debilitation": 190,  # Libra
        "own_signs": [120],  # Leo
        "friendly_signs": [0, 90, 150],  # Aries, Cancer, Sagittarius
        "neutral_signs": [30, 60, 180],  # Taurus, Gemini, Scorpio
        "enemy_signs": [210, 240, 270, 300, 330],  # Remaining signs
    },
    "Moon": {
        "exaltation": 33,  # Taurus
        "debilitation": 213,  # Scorpio
        "own_signs": [90],  # Cancer
        "friendly_signs": [0, 120, 150],  # Aries, Leo, Sagittarius
        "neutral_signs": [30, 60, 180],  # Taurus, Gemini, Libra
        "enemy_signs": [210, 240, 270, 300, 330],  # Remaining signs
    },
    "Mars": {
        "exaltation": 298,  # Capricorn
        "debilitation": 118,  # Cancer
        "own_signs": [0, 210],  # Aries, Scorpio
        "friendly_signs": [120, 150, 180],  # Leo, Sagittarius, Libra
        "neutral_signs": [30, 60, 90],  # Taurus, Gemini, Cancer
        "enemy_signs": [240, 270, 300, 330],  # Remaining signs
    },
    "Mercury": {
        "exaltation": 165,  # Virgo
        "debilitation": 345,  # Pisces
        "own_signs": [60, 165],  # Gemini, Virgo
        "friendly_signs": [120, 180, 240],  # Leo, Libra, Sagittarius
        "neutral_signs": [0, 30, 90],  # Aries, Taurus, Cancer
        "enemy_signs": [150, 210, 270, 300, 330],  # Remaining signs
    },
    "Jupiter": {
        "exaltation": 95,  # Cancer
        "debilitation": 275,  # Capricorn
        "own_signs": [150, 330],  # Sagittarius, Pisces
        "friendly_signs": [0, 120, 180],  # Aries, Leo, Libra
        "neutral_signs": [30, 60, 90],  # Taurus, Gemini, Cancer
        "enemy_signs": [210, 240, 270, 300],  # Remaining signs
    },
    "Venus": {
        "exaltation": 357,  # Pisces
        "debilitation": 177,  # Virgo
        "own_signs": [30, 180],  # Taurus, Libra
        "friendly_signs": [60, 150, 240],  # Gemini, Sagittarius, Aquarius
        "neutral_signs": [90, 120, 210],  # Cancer, Leo, Scorpio
        "enemy_signs": [0, 270, 300, 330],  # Remaining signs
    },
    "Saturn": {
        "exaltation": 200,  # Libra
        "debilitation": 20,  # Aries
        "own_signs": [270, 300],  # Aquarius, Capricorn
        "friendly_signs": [60, 180, 240],  # Gemini, Libra, Aquarius
        "neutral_signs": [90, 120, 150],  # Cancer, Leo, Sagittarius
        "enemy_signs": [0, 30, 210, 330],  # Remaining signs
    },
    "Rahu": {
        "exaltation": 60,  # Gemini
        "debilitation": 240,  # Sagittarius
        "own_signs": [30],  # Taurus
        "friendly_signs": [0, 90, 180],  # Aries, Cancer, Libra
        "neutral_signs": [120, 150, 210],  # Leo, Virgo, Scorpio
        "enemy_signs": [270, 300, 330],  # Remaining signs
    },
    "Ketu": {
        "exaltation": 240,  # Sagittarius
        "debilitation": 60,  # Gemini
        "own_signs": [210],  # Scorpio
        "friendly_signs": [0, 120, 180],  # Aries, Leo, Libra
        "neutral_signs": [30, 90, 150],  # Taurus, Cancer, Virgo
        "enemy_signs": [270, 300, 330],  # Remaining signs
    },
}


def calculate_planet_state(
    planet: str,
    longitude: float,
    house: int,
    nakshatra: str,
    is_retrograde: bool,
    all_positions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate the complete state of a planet including dignity, combustion, and aspects.

    Args:
        planet: Name of the planet
        longitude: Longitude of the planet
        house: House position (1-12)
        nakshatra: Current nakshatra
        is_retrograde: Whether planet is retrograde
        all_positions: Dictionary of all planetary positions

    Returns:
        Dictionary containing planet state information
    """
    try:
        state = {
            "planet": planet,
            "longitude": longitude,
            "house": house,
            "nakshatra": nakshatra,
            "is_retrograde": is_retrograde,
            "is_exalted": False,
            "is_debilitated": False,
            "is_in_own_sign": False,
            "is_in_friendly_sign": False,
            "is_in_neutral_sign": False,
            "is_in_enemy_sign": False,
            "is_in_dig_bala": False,
            "is_combust": False,
            "in_planetary_war": False,
            "strength": 0.0,
        }

        # Get dignity information
        if planet in PLANETARY_DIGNITIES:
            dignity = PLANETARY_DIGNITIES[planet]
            sign_long = longitude % 30

            # Check exaltation/debilitation
            if abs(longitude - dignity["exaltation"]) < 30:
                state["is_exalted"] = True
            elif abs(longitude - dignity["debilitation"]) < 30:
                state["is_debilitated"] = True

            # Check sign placement
            if longitude in dignity["own_signs"]:
                state["is_in_own_sign"] = True
            elif longitude in dignity["friendly_signs"]:
                state["is_in_friendly_sign"] = True
            elif longitude in dignity["neutral_signs"]:
                state["is_in_neutral_sign"] = True
            else:
                state["is_in_enemy_sign"] = True

            # Check dig bala (directional strength)
            if (
                (house == 1 and planet in ["Sun", "Mars"])
                or (house == 4 and planet in ["Moon", "Mercury"])
                or (house == 7 and planet in ["Saturn", "Jupiter"])
                or (house == 10 and planet in ["Venus", "Mercury"])
            ):
                state["is_in_dig_bala"] = True

        # Check combustion with Sun
        if planet != "Sun" and "Sun" in all_positions:
            sun_long = all_positions["Sun"]["longitude"]
            distance = abs(longitude - sun_long)
            if distance <= 3:  # Within 3 degrees
                state["is_combust"] = True

        # Check planetary war
        for other_planet, other_pos in all_positions.items():
            if other_planet != planet:
                other_long = other_pos["longitude"]
                if abs(longitude - other_long) < 1:  # Within 1 degree
                    state["in_planetary_war"] = True
                    break

        # Calculate overall strength
        strength = 1.0
        if state["is_exalted"]:
            strength *= 1.5
        if state["is_debilitated"]:
            strength *= 0.5
        if state["is_in_own_sign"]:
            strength *= 1.25
        if state["is_in_friendly_sign"]:
            strength *= 1.1
        if state["is_in_enemy_sign"]:
            strength *= 0.8
        if state["is_in_dig_bala"]:
            strength *= 1.25
        if state["is_combust"]:
            strength *= 0.75
        if state["in_planetary_war"]:
            strength *= 0.75
        if state["is_retrograde"]:
            if planet in ["Jupiter", "Venus", "Mercury"]:  # Benefics
                strength *= 1.25
            else:
                strength *= 0.75

        state["strength"] = min(max(strength, 0.1), 2.0)  # Cap between 0.1 and 2.0

        return state

    except Exception as e:
        logger.error(f"Error calculating planet state for {planet}: {str(e)}")
        return {"planet": planet, "error": str(e), "strength": 1.0}  # Default strength

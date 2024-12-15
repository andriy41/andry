import swisseph as swe
from typing import Dict, Any, List, Tuple
import logging
from src.models.vedic_basic.calculations.planetary_calculations import (
    calculate_planet_strength,
    calculate_house_lord_strength,
    calculate_planetary_alignment_score,
    calculate_ashtakavarga_points,
    calculate_dignity_score,
    calculate_aspect_strength,
    calculate_house_placement_score,
    PLANETS,
    PLANET_MAP,
    ASPECT_ORBS,
)

logger = logging.getLogger(__name__)

# Constants for NFL predictions
FAVORITE_HOUSES = {1, 3, 6, 10, 11}  # Houses favoring the favorite team
UNDERDOG_HOUSES = {7, 9, 12, 4, 5}  # Houses favoring the underdog
VICTORY_HOUSES = {1, 10, 11}  # Houses indicating victory
DEFEAT_HOUSES = {6, 8, 12}  # Houses indicating defeat

# Sport-specific planet weights
SPORT_PLANET_WEIGHTS = {
    "Sun": 1.5,  # Leadership and vitality
    "Moon": 2.0,  # Public support and momentum
    "Mars": 1.8,  # Energy and aggression
    "Mercury": 1.2,  # Strategy and adaptability
    "Jupiter": 1.5,  # Success and expansion
    "Venus": 1.2,  # Harmony and team chemistry
    "Saturn": 1.3,  # Discipline and defense
    "Rahu": 1.4,  # Unexpected developments
    "Ketu": 1.4,  # Hidden strengths/weaknesses
}


def calculate_team_strength(
    team_type: str, positions: Dict[str, float], jd: float, lat: float, lon: float
) -> float:
    """Calculate overall team strength based on planetary positions"""
    total_strength = 0.0
    total_weight = 0.0

    # Get relevant houses based on team type
    relevant_houses = FAVORITE_HOUSES if team_type == "favorite" else UNDERDOG_HOUSES

    # Calculate strength from planet positions
    for planet, weight in SPORT_PLANET_WEIGHTS.items():
        if planet not in positions:
            continue

        # Get planet's house
        house = int(positions[planet] / 30) + 1

        # Calculate base strength
        strength = calculate_planet_strength(planet, jd, lat, lon)

        # Adjust based on house placement
        if house in relevant_houses:
            strength *= 1.2
        elif house in (VICTORY_HOUSES if team_type == "favorite" else DEFEAT_HOUSES):
            strength *= 1.1

        total_strength += strength * weight
        total_weight += weight

    # Normalize strength
    if total_weight > 0:
        return total_strength / total_weight
    return 0.5


def calculate_game_outcome_strength(
    positions: Dict[str, float], jd: float, lat: float, lon: float
) -> Tuple[float, float]:
    """Calculate strength factors for game outcome"""
    # Calculate strength for victory houses
    victory_strength = 0.0
    for house in VICTORY_HOUSES:
        lord_strength = calculate_house_lord_strength(house, jd, lat, lon)
        victory_strength += lord_strength
    victory_strength /= len(VICTORY_HOUSES)

    # Calculate strength for defeat houses
    defeat_strength = 0.0
    for house in DEFEAT_HOUSES:
        lord_strength = calculate_house_lord_strength(house, jd, lat, lon)
        defeat_strength += lord_strength
    defeat_strength /= len(DEFEAT_HOUSES)

    return victory_strength, defeat_strength


def calculate_team_yoga(team: str, jd: float, positions: Dict[str, float]) -> float:
    """Calculate yoga (planetary combinations) strength for a team"""
    yoga_strength = 0.0

    # Check for Raja Yoga (strong 1st, 5th, 9th, 10th houses)
    raja_houses = [1, 5, 9, 10]
    for house in raja_houses:
        lord_strength = calculate_house_lord_strength(
            house, jd, 0, 0
        )  # Using 0,0 as coordinates since this is general
        yoga_strength += lord_strength * 0.25

    # Check for Dhana Yoga (2nd and 11th houses - wealth/gains)
    dhana_houses = [2, 11]
    for house in dhana_houses:
        lord_strength = calculate_house_lord_strength(house, jd, 0, 0)
        yoga_strength += lord_strength * 0.15

    # Check for Vipreet Raja Yoga (6th, 8th, 12th houses - turning negatives to positives)
    vipreet_houses = [6, 8, 12]
    vipreet_strength = 0
    for house in vipreet_houses:
        lord_strength = calculate_house_lord_strength(house, jd, 0, 0)
        vipreet_strength += lord_strength
    yoga_strength += (vipreet_strength / len(vipreet_houses)) * 0.1

    return min(1.0, yoga_strength)


def calculate_game_timing_factors(jd: float) -> Dict[str, float]:
    """Calculate timing-based factors for the game"""
    factors = {}

    # Moon phase calculation
    moon_pos = swe.calc_ut(jd, swe.MOON)[0][0]
    sun_pos = swe.calc_ut(jd, swe.SUN)[0][0]
    moon_phase = abs(moon_pos - sun_pos) % 360

    # Score moon phase (full moon = highest, new moon = lowest)
    if 170 <= moon_phase <= 190:  # Full moon
        factors["moon_phase"] = 1.0
    elif moon_phase <= 10 or moon_phase >= 350:  # New moon
        factors["moon_phase"] = 0.3
    else:
        factors["moon_phase"] = 0.6

    # Planetary hour influence
    hour = int((jd * 24) % 24)
    planetary_day = int(jd + 0.5) % 7

    # Simplified planetary hour ruler calculation
    hour_ruler = (planetary_day + hour) % 7
    factors["hour_strength"] = 0.5 + (hour_ruler / 14)  # Normalize to 0.5-1.0

    return factors


def calculate_confidence_score(
    positions: Dict[str, float],
    jd: float,
    lat: float,
    lon: float,
    favorite_strength: float,
    underdog_strength: float,
) -> float:
    """Calculate overall confidence score for the prediction"""

    # Get timing factors
    timing_factors = calculate_game_timing_factors(jd)

    # Calculate planetary alignment score
    alignment_score = calculate_planetary_alignment_score(positions)

    # Get victory/defeat strengths
    victory_strength, defeat_strength = calculate_game_outcome_strength(
        positions, jd, lat, lon
    )

    # Calculate final confidence score
    confidence = (
        favorite_strength * 0.25
        + (1 - underdog_strength) * 0.25
        + alignment_score * 0.15
        + timing_factors["moon_phase"] * 0.15
        + victory_strength * 0.1
        + (1 - defeat_strength) * 0.1
    )

    return max(0.0, min(1.0, confidence))


def calculate_game_strength(game_data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate overall game strength factors."""
    try:
        # Extract required data
        jd = game_data.get("julian_day", 0)
        lat = game_data.get("latitude", 0)
        lon = game_data.get("longitude", 0)

        # Calculate team strengths
        home_strength = calculate_team_strength(
            "home", game_data.get("planet_positions", {}), jd, lat, lon
        )
        away_strength = calculate_team_strength(
            "away", game_data.get("planet_positions", {}), jd, lat, lon
        )

        # Calculate game outcome factors
        outcome_strength = calculate_game_outcome_strength(
            game_data.get("planet_positions", {}), jd, lat, lon
        )

        # Calculate timing factors
        timing_factors = calculate_game_timing_factors(jd)

        # Calculate confidence score
        confidence = calculate_confidence_score(
            game_data.get("planet_positions", {}),
            jd,
            lat,
            lon,
            home_strength,
            away_strength,
        )

        return {
            "home_team_strength": home_strength,
            "away_team_strength": away_strength,
            "outcome_strength": outcome_strength,
            "timing_score": timing_factors,
            "confidence_score": confidence,
        }

    except Exception as e:
        logger.error(f"Error calculating game strength: {e}")
        return {
            "home_team_strength": 0.5,
            "away_team_strength": 0.5,
            "outcome_strength": 0.5,
            "timing_score": 0.5,
            "confidence_score": 0.5,
        }

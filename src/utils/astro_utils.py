"""
Utility functions for astrological calculations.
"""

import swisseph as swe
from datetime import datetime, timedelta
import math
import logging

# Initialize ephemeris data path
swe.set_ephe_path("/Users/space/Downloads/NFL_Project/ephe")

# Initialize logger
logger = logging.getLogger(__name__)


def get_planet_positions(date):
    """Get positions of major planets for a given date"""
    try:
        jd = swe.julday(
            date.year,
            date.month,
            date.day,
            date.hour + date.minute / 60.0 + date.second / 3600.0,
        )

        planets = {
            "Sun": swe.SUN,
            "Moon": swe.MOON,
            "Mars": swe.MARS,
            "Mercury": swe.MERCURY,
            "Jupiter": swe.JUPITER,
            "Venus": swe.VENUS,
            "Saturn": swe.SATURN,
        }

        positions = {}
        for planet, planet_id in planets.items():
            try:
                result = swe.calc_ut(jd, planet_id)
                if result is None:
                    raise ValueError(f"No result returned for planet {planet}")

                # Extract values from result
                values = result[0] if isinstance(result, tuple) else result
                if not isinstance(values, (list, tuple)) or len(values) < 6:
                    values = [0.0] * 6  # Default values if not enough returned

                positions[planet] = {
                    "longitude": float(values[0]),
                    "latitude": float(values[1]),
                    "distance": float(values[2]),
                    "speed": float(
                        values[3]
                    ),  # Negative speed indicates retrograde motion
                    "declination": float(values[4]) if len(values) > 4 else 0.0,
                    "right_ascension": float(values[5]) if len(values) > 5 else 0.0,
                }
            except Exception as e:
                logger.error(
                    f"Error calculating position for planet {planet}: {str(e)}"
                )
                positions[planet] = {
                    "longitude": 0.0,
                    "latitude": 0.0,
                    "distance": 0.0,
                    "speed": 0.0,
                    "declination": 0.0,
                    "right_ascension": 0.0,
                }

        return positions

    except Exception as e:
        logger.error(f"Error in get_planet_positions: {str(e)}")
        return {}


def is_retrograde(planet_data):
    """Check if a planet is retrograde based on its speed"""
    return planet_data["speed"] < 0


def get_dignity_score(planet, position):
    """Calculate dignity score based on planet's position"""
    # Exaltation points for planets
    exaltation_points = {
        "Sun": 0,  # Aries
        "Moon": 30,  # Taurus
        "Mars": 300,  # Capricorn
        "Mercury": 150,  # Virgo
        "Jupiter": 90,  # Cancer
        "Venus": 330,  # Pisces
        "Saturn": 180,  # Libra
    }

    # Calculate how close planet is to its exaltation point
    if planet in exaltation_points:
        diff = abs(position["longitude"] - exaltation_points[planet])
        if diff > 180:
            diff = 360 - diff

        # Score decreases linearly with distance from exaltation point
        score = max(0, 1 - diff / 90)
        return score

    return 0


def calculate_astrological_advantage(date, home_team: str, away_team: str) -> float:
    """Calculate astrological advantage for a game."""
    positions = get_planet_positions(date)

    # Calculate basic planetary strength
    home_strength = 0
    away_strength = 0

    # Mars (competition) aspects
    mars_aspects = calculate_mars_aspects(positions)
    home_strength += mars_aspects["home"]
    away_strength += mars_aspects["away"]

    # Jupiter (luck) aspects
    jupiter_aspects = calculate_jupiter_aspects(positions)
    home_strength += jupiter_aspects["home"]
    away_strength += jupiter_aspects["away"]

    # Moon phase influence
    moon_influence = calculate_moon_influence(positions)
    home_strength += moon_influence["home"]
    away_strength += moon_influence["away"]

    # Calculate final advantage (-1 to 1, positive favors home team)
    total_strength = home_strength + away_strength
    if total_strength == 0:
        return 0

    advantage = (home_strength - away_strength) / total_strength
    return max(-1, min(1, advantage))  # Clamp between -1 and 1


def calculate_mars_aspects(positions):
    """Calculate Mars aspects for home and away teams."""
    mars_pos = positions["Mars"]["longitude"]

    aspects = {"home": 0, "away": 0}

    # Check aspects with Jupiter (expansion)
    jupiter_pos = positions["Jupiter"]["longitude"]
    angle = abs(mars_pos - jupiter_pos)
    if angle > 180:
        angle = 360 - angle

    # Strong aspects
    if angle < 10 or abs(angle - 120) < 10:  # Conjunction or trine
        aspects["home"] += 0.5

    return aspects


def calculate_jupiter_aspects(positions):
    """Calculate Jupiter aspects for home and away teams."""
    jupiter_pos = positions["Jupiter"]["longitude"]

    aspects = {"home": 0, "away": 0}

    # Check aspects with Sun (vitality)
    sun_pos = positions["Sun"]["longitude"]
    angle = abs(jupiter_pos - sun_pos)
    if angle > 180:
        angle = 360 - angle

    # Strong aspects
    if angle < 10 or abs(angle - 120) < 10:  # Conjunction or trine
        aspects["home"] += 0.5

    return aspects


def calculate_moon_influence(positions):
    """Calculate Moon's influence on the game."""
    moon_pos = positions["Moon"]["longitude"]
    sun_pos = positions["Sun"]["longitude"]

    # Calculate moon phase
    phase_angle = moon_pos - sun_pos
    if phase_angle < 0:
        phase_angle += 360

    influences = {"home": 0, "away": 0}

    # Waxing moon favors home team
    if phase_angle < 180:
        influences["home"] += 0.3
    else:
        influences["away"] += 0.3

    return influences


def calculate_yogas(date):
    """Calculate strength of victory-related yogas"""
    positions = get_planet_positions(date)

    score = 0

    # Check for Jupiter-Sun relationship (Raja Yoga)
    jupiter_sun_diff = abs(
        positions["Jupiter"]["longitude"] - positions["Sun"]["longitude"]
    )
    if jupiter_sun_diff > 180:
        jupiter_sun_diff = 360 - jupiter_sun_diff

    if jupiter_sun_diff < 10:  # Conjunction
        score += 0.3
    elif abs(jupiter_sun_diff - 120) < 10:  # Trine
        score += 0.2

    # Check for Mars-Sun relationship (Victory Yoga)
    mars_sun_diff = abs(positions["Mars"]["longitude"] - positions["Sun"]["longitude"])
    if mars_sun_diff > 180:
        mars_sun_diff = 360 - mars_sun_diff

    if mars_sun_diff < 10:  # Conjunction
        score += 0.25
    elif abs(mars_sun_diff - 120) < 10:  # Trine
        score += 0.15

    # Check for exalted planets
    for planet, pos in positions.items():
        dignity = get_dignity_score(planet, pos)
        score += dignity * 0.1

    return min(score, 1.0)  # Normalize to 0-1


def calculate_planetary_hour_strength(date):
    """Calculate the strength of planetary hour influence"""
    # Get hour of day (0-23)
    hour = date.hour

    # Planetary hours have different strengths
    # For sports: Mars hours are strongest, followed by Sun and Jupiter
    planetary_hour_weights = {
        "Mars": 1.0,
        "Sun": 0.8,
        "Jupiter": 0.7,
        "Venus": 0.5,
        "Mercury": 0.4,
        "Saturn": 0.3,
        "Moon": 0.6,
    }

    # Calculate planetary hour ruler (simplified)
    day_of_week = date.weekday()
    hour_index = (day_of_week * 24 + hour) % 7

    # Map hour index to planet
    planets = ["Mars", "Sun", "Venus", "Mercury", "Moon", "Saturn", "Jupiter"]
    ruling_planet = planets[hour_index]

    return planetary_hour_weights[ruling_planet]


def calculate_mars_jupiter_relationship(positions):
    """Calculate the relationship between Mars and Jupiter"""
    mars_pos = positions["Mars"]["longitude"]
    jupiter_pos = positions["Jupiter"]["longitude"]

    # Calculate angular distance
    diff = abs(mars_pos - jupiter_pos)
    if diff > 180:
        diff = 360 - diff

    # Score based on aspects (0°, 60°, 90°, 120°, 180°)
    aspect_orbs = {
        0: 10,  # Conjunction
        60: 6,  # Sextile
        90: 8,  # Square
        120: 10,  # Trine
        180: 10,  # Opposition
    }

    score = 0
    for aspect, orb in aspect_orbs.items():
        aspect_diff = abs(diff - aspect)
        if aspect_diff <= orb:
            # Closer to exact aspect = stronger score
            strength = 1 - (aspect_diff / orb)
            if aspect in [0, 120]:  # Favorable aspects
                score += strength * 0.3
            elif aspect == 60:  # Mild favorable
                score += strength * 0.2
            else:  # Challenging aspects
                score -= strength * 0.1

    # Normalize score to 0-1 range
    return (score + 0.5) / 1.5


def calculate_shadow_planet_influence(date):
    """Calculate the influence of shadow planets (Rahu/Ketu) for sports outcomes"""
    try:
        jd = swe.julday(
            date.year,
            date.month,
            date.day,
            date.hour + date.minute / 60.0 + date.second / 3600.0,
        )

        # Get Rahu (North Node) position
        result = swe.calc_ut(jd, swe.MEAN_NODE)
        if result is None:
            return 0.5  # Neutral influence if calculation fails

        # Extract values from result
        values = result[0] if isinstance(result, tuple) else result
        if not isinstance(values, (list, tuple)) or len(values) < 1:
            return 0.5

        rahu_long = float(values[0])

        # Ketu is exactly opposite Rahu
        ketu_long = (rahu_long + 180) % 360

        # Get positions of Sun and Moon for additional calculations
        sun_result = swe.calc_ut(jd, swe.SUN)
        moon_result = swe.calc_ut(jd, swe.MOON)

        if sun_result is None or moon_result is None:
            return 0.5

        sun_values = sun_result[0] if isinstance(sun_result, tuple) else sun_result
        moon_values = moon_result[0] if isinstance(moon_result, tuple) else moon_result

        if not isinstance(sun_values, (list, tuple)) or not isinstance(
            moon_values, (list, tuple)
        ):
            return 0.5

        sun_long = float(sun_values[0])
        moon_long = float(moon_values[0])

        # Calculate aspects between shadow planets and luminaries
        rahu_sun_aspect = abs(rahu_long - sun_long) % 360
        rahu_moon_aspect = abs(rahu_long - moon_long) % 360
        ketu_sun_aspect = abs(ketu_long - sun_long) % 360
        ketu_moon_aspect = abs(ketu_long - moon_long) % 360

        # Define favorable and unfavorable aspects
        favorable_aspects = [0, 60, 120]  # Conjunction, sextile, trine
        unfavorable_aspects = [90, 180]  # Square, opposition
        orb = 8  # Degrees of orb allowed

        influence = 0.5  # Start with neutral influence

        # Check aspects and adjust influence
        for aspect in favorable_aspects:
            if any(abs(a - aspect) <= orb for a in [rahu_sun_aspect, rahu_moon_aspect]):
                influence += 0.1
            if any(abs(a - aspect) <= orb for a in [ketu_sun_aspect, ketu_moon_aspect]):
                influence += 0.1

        for aspect in unfavorable_aspects:
            if any(abs(a - aspect) <= orb for a in [rahu_sun_aspect, rahu_moon_aspect]):
                influence -= 0.1
            if any(abs(a - aspect) <= orb for a in [ketu_sun_aspect, ketu_moon_aspect]):
                influence -= 0.1

        # Normalize influence to 0-1 range
        return max(0.0, min(1.0, influence))

    except Exception as e:
        logger.error(f"Error calculating shadow planet influence: {str(e)}")
        return 0.5  # Return neutral influence on error

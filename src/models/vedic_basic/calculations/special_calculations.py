"""Special Vedic astrological calculations for NFL predictions."""

import swisseph as swe
from typing import Dict, Any, List, Tuple
from datetime import datetime
import math


def calculate_muhurta_score(jd: float, lat: float, lon: float) -> float:
    """
    Calculate the Muhurta (timing) score for a given time.

    Args:
        jd: Julian day number
        lat: Latitude
        lon: Longitude

    Returns:
        float: Muhurta score between 0 and 1
    """
    try:
        # Get current time info
        ut = swe.jdut1_to_utc(jd)
        hour = ut[3] + ut[4] / 60.0

        # Calculate day of week (0=Sunday, 6=Saturday)
        day_of_week = int(jd + 1.5) % 7

        # Divide day into 30 muhurtas
        muhurta_num = int((hour / 24.0) * 30)

        # Auspicious muhurtas by day of week (simplified)
        auspicious_muhurtas = {
            0: [1, 4, 10, 13, 16, 22, 25],  # Sunday
            1: [2, 5, 11, 14, 17, 23, 26],  # Monday
            2: [3, 6, 12, 15, 18, 24, 27],  # Tuesday
            3: [4, 7, 13, 16, 19, 25, 28],  # Wednesday
            4: [5, 8, 14, 17, 20, 26, 29],  # Thursday
            5: [6, 9, 15, 18, 21, 27, 30],  # Friday
            6: [7, 10, 16, 19, 22, 28, 1],  # Saturday
        }

        # Calculate base score
        if muhurta_num in auspicious_muhurtas.get(day_of_week, []):
            score = 0.8
        else:
            score = 0.4

        # Adjust for sunrise/sunset
        sunrise, sunset = calculate_sunrise_sunset(jd, lat, lon)
        if abs(hour - sunrise) < 1 or abs(hour - sunset) < 1:
            score += 0.2

        return min(max(score, 0.0), 1.0)

    except Exception as e:
        return 0.5  # Default score on error


def calculate_hora_score(jd: float) -> float:
    """
    Calculate the Hora (planetary hour) score.

    Args:
        jd: Julian day number

    Returns:
        float: Hora score between 0 and 1
    """
    try:
        ut = swe.jdut1_to_utc(jd)
        hour = ut[3]
        day_of_week = int(jd + 1.5) % 7

        # Planetary rulers of hours
        hora_rulers = {
            0: ["sun", "venus", "mercury", "moon", "saturn", "jupiter", "mars"] * 4,
            1: ["moon", "saturn", "jupiter", "mars", "sun", "venus", "mercury"] * 4,
            2: ["mars", "sun", "venus", "mercury", "moon", "saturn", "jupiter"] * 4,
            3: ["mercury", "moon", "saturn", "jupiter", "mars", "sun", "venus"] * 4,
            4: ["jupiter", "mars", "sun", "venus", "mercury", "moon", "saturn"] * 4,
            5: ["venus", "mercury", "moon", "saturn", "jupiter", "mars", "sun"] * 4,
            6: ["saturn", "jupiter", "mars", "sun", "venus", "mercury", "moon"] * 4,
        }

        hora = hour % 24
        current_ruler = hora_rulers[day_of_week][hora]

        # Beneficial planets score higher
        planet_scores = {
            "sun": 0.7,
            "moon": 0.8,
            "mars": 0.6,
            "mercury": 0.7,
            "jupiter": 0.9,
            "venus": 0.8,
            "saturn": 0.5,
        }

        return planet_scores.get(current_ruler, 0.5)

    except Exception as e:
        return 0.5


def calculate_nakshatra_score(jd: float) -> float:
    """
    Calculate the Nakshatra (lunar mansion) score.

    Args:
        jd: Julian day number

    Returns:
        float: Nakshatra score between 0 and 1
    """
    try:
        # Get Moon's position
        moon_pos = swe.calc_ut(jd, swe.MOON)[0]
        moon_long = moon_pos[0]

        # Calculate Nakshatra number (0-26)
        nakshatra = int(moon_long * 27 / 360)

        # Nakshatras favorable for competition/victory
        favorable_nakshatras = [
            0,  # Ashwini - Victory
            3,  # Rohini - Success
            5,  # Mrigashira - Swift action
            7,  # Punarvasu - Renewal
            10,  # Magha - Power
            11,  # Purva Phalguni - Victory
            14,  # Chitra - Success
            15,  # Swati - Independent action
            16,  # Vishakha - Focused achievement
            18,  # Jyeshtha - Leadership
            20,  # Purva Ashadha - Early victory
            22,  # Shravana - Learning from success
            24,  # Purva Bhadrapada - Spiritual victory
            26,  # Revati - Completion
        ]

        # Calculate score
        if nakshatra in favorable_nakshatras:
            base_score = 0.8
        else:
            base_score = 0.4

        # Adjust for pada (quarter)
        pada = int((moon_long % (360 / 27)) * 4 / (360 / 27))
        pada_adjustment = [0.1, 0.05, -0.05, -0.1][pada]

        return min(max(base_score + pada_adjustment, 0.0), 1.0)

    except Exception as e:
        return 0.5


def calculate_sunrise_sunset(jd: float, lat: float, lon: float) -> Tuple[float, float]:
    """
    Calculate sunrise and sunset times.

    Args:
        jd: Julian day number
        lat: Latitude
        lon: Longitude

    Returns:
        Tuple[float, float]: (sunrise hour, sunset hour)
    """
    try:
        # Get sunrise and sunset
        result = swe.rise_trans(
            jd - 1, swe.SUN, lon, lat, rsmi=swe.CALC_RISE | swe.BIT_DISC_CENTER
        )
        sunrise_jd = result[1][0]  # Julian day of sunrise

        result = swe.rise_trans(
            jd - 1, swe.SUN, lon, lat, rsmi=swe.CALC_SET | swe.BIT_DISC_CENTER
        )
        sunset_jd = result[1][0]  # Julian day of sunset

        # Convert to hours
        sunrise = (sunrise_jd % 1) * 24
        sunset = (sunset_jd % 1) * 24

        return sunrise, sunset

    except Exception as e:
        # Return approximate values if calculation fails
        return 6.0, 18.0

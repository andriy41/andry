"""Planetary calculations for Vedic astrology."""

import swisseph as swe
from datetime import datetime
import math
import os
from typing import List, Dict, Any, Union, Tuple
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Constants
PLANETS = [
    "Sun",
    "Moon",
    "Mars",
    "Mercury",
    "Jupiter",
    "Venus",
    "Saturn",
    "Rahu",
    "Ketu",
]
PLANET_MAP = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,  # North Node
    "Ketu": swe.TRUE_NODE,  # South Node
}

# Default location for NFL games (approximate center of USA)
LATITUDE = 39.8283  # Degrees North
LONGITUDE = -98.5795  # Degrees West

# House systems
HOUSE_SYSTEMS = {"Placidus": b"P", "Koch": b"K", "Equal": b"E", "Whole Sign": b"W"}

# Aspect orbs
ASPECT_ORBS = {
    0: 10,  # Conjunction
    60: 6,  # Sextile
    90: 8,  # Square
    120: 10,  # Trine
    180: 10,  # Opposition
}

# Nakshatras and their lords
NAKSHATRA_LORDS = {
    1: "Mars",
    2: "Venus",
    3: "Mercury",
    4: "Moon",
    5: "Saturn",
    6: "Jupiter",
    7: "Mars",
    8: "Venus",
    9: "Mercury",
    10: "Moon",
    11: "Saturn",
    12: "Jupiter",
    13: "Mars",
    14: "Venus",
    15: "Mercury",
    16: "Moon",
    17: "Saturn",
    18: "Jupiter",
    19: "Mars",
    20: "Venus",
    21: "Mercury",
    22: "Moon",
    23: "Saturn",
    24: "Jupiter",
    25: "Mars",
    26: "Venus",
    27: "Mercury",
}

# House lords (based on natural zodiac)
HOUSE_LORDS = {
    1: "Mars",  # Aries
    2: "Venus",  # Taurus
    3: "Mercury",  # Gemini
    4: "Moon",  # Cancer
    5: "Sun",  # Leo
    6: "Mercury",  # Virgo
    7: "Venus",  # Libra
    8: "Mars",  # Scorpio
    9: "Jupiter",  # Sagittarius
    10: "Saturn",  # Capricorn
    11: "Saturn",  # Aquarius
    12: "Jupiter",  # Pisces
}

# Muhurta periods and their lords
MUHURTA_LORDS = {
    0: "Sun",
    1: "Venus",
    2: "Mercury",
    3: "Moon",
    4: "Saturn",
    5: "Jupiter",
    6: "Mars",
    7: "Sun",
    8: "Venus",
    9: "Mercury",
    10: "Moon",
    11: "Saturn",
    12: "Jupiter",
    13: "Mars",
    14: "Sun",
}

# Hora lords
HORA_LORDS = {
    "Sun": ["Sun", "Venus"],
    "Mon": ["Moon", "Saturn"],
    "Tue": ["Mars", "Mercury"],
    "Wed": ["Mercury", "Jupiter"],
    "Thu": ["Jupiter", "Mars"],
    "Fri": ["Venus", "Moon"],
    "Sat": ["Saturn", "Sun"],
}

NAKSHATRAS = list(range(1, 28))  # 27 Nakshatras
HOUSES = list(range(1, 13))  # 12 Houses


def init_ephemeris():
    """Initialize the ephemeris data."""
    try:
        ephe_path = os.path.join(os.path.dirname(__file__), "ephe")
        if not os.path.exists(ephe_path):
            os.makedirs(ephe_path)
        swe.set_ephe_path(ephe_path)
        swe.set_sid_mode(swe.SIDM_LAHIRI)
    except Exception as e:
        print(f"Warning: Could not initialize ephemeris: {str(e)}")


def get_planet_positions(date, lat=None, lon=None):
    """Get positions of major planets for a given date."""
    try:
        jd = swe.julday(
            date.year,
            date.month,
            date.day,
            date.hour + date.minute / 60.0 + date.second / 3600.0,
        )

        # Set geographic position if provided
        if lat is not None and lon is not None:
            swe.set_topo(lat, lon, 0)

        # Set sidereal mode
        swe.set_sid_mode(swe.SIDM_LAHIRI)

        planets = {
            "Sun": swe.SUN,
            "Moon": swe.MOON,
            "Mars": swe.MARS,
            "Mercury": swe.MERCURY,
            "Jupiter": swe.JUPITER,
            "Venus": swe.VENUS,
            "Saturn": swe.SATURN,
            "Rahu": swe.MEAN_NODE,  # North Node
            "Ketu": None,  # South Node (calculated from Rahu)
        }

        positions = {}
        for planet_name, planet_id in planets.items():
            if planet_id is not None:
                flags = (
                    swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_SIDEREAL | swe.FLG_TOPOCTR
                )
                try:
                    result = swe.calc_ut(jd, planet_id, flags)
                    if result is None:
                        raise ValueError(f"No result returned for planet {planet_name}")

                    # Swiss Ephemeris returns a tuple where first element is a list of values
                    values = result[0] if isinstance(result, tuple) else result
                    if not isinstance(values, (list, tuple)) or len(values) < 6:
                        values = [0.0] * 6  # Default values if not enough returned

                    positions[planet_name] = {
                        "longitude": float(values[0]),
                        "latitude": float(values[1]),
                        "distance": float(values[2]),
                        "speed_long": float(values[3]),
                        "speed_lat": float(values[4]),
                        "speed_dist": float(values[5]),
                    }

                    # Calculate retrograde status
                    positions[planet_name]["is_retrograde"] = (
                        positions[planet_name]["speed_long"] < 0
                    )

                    # Calculate zodiac sign and degree
                    long = positions[planet_name]["longitude"]
                    positions[planet_name]["sign"] = int(long / 30) + 1
                    positions[planet_name]["degree"] = long % 30

                except Exception as e:
                    logger.error(
                        f"Error calculating position for {planet_name}: {str(e)}"
                    )
                    positions[planet_name] = {
                        "longitude": 0.0,
                        "latitude": 0.0,
                        "distance": 0.0,
                        "speed_long": 0.0,
                        "speed_lat": 0.0,
                        "speed_dist": 0.0,
                        "is_retrograde": False,
                        "sign": 1,
                        "degree": 0.0,
                    }

            # Calculate Ketu (opposite to Rahu)
            if planet_name == "Ketu" and "Rahu" in positions:
                rahu_long = positions["Rahu"]["longitude"]
                ketu_long = (rahu_long + 180) % 360
                positions["Ketu"] = {
                    "longitude": ketu_long,
                    "latitude": -positions["Rahu"]["latitude"],
                    "distance": positions["Rahu"]["distance"],
                    "speed_long": -positions["Rahu"]["speed_long"],
                    "speed_lat": -positions["Rahu"]["speed_lat"],
                    "speed_dist": positions["Rahu"]["speed_dist"],
                    "is_retrograde": positions["Rahu"]["is_retrograde"],
                    "sign": int(ketu_long / 30) + 1,
                    "degree": ketu_long % 30,
                }

        return positions

    except Exception as e:
        logger.error(f"Error in get_planet_positions: {str(e)}")
        return {}


def calculate_planet_strength(
    date: datetime, planet: str, lat: float = None, lon: float = None
) -> List[float]:
    """Calculate the strength of a planet based on its position and aspects.

    Args:
        date: Datetime of the game
        planet: Name of the planet
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List containing planet strength (0-1)
    """
    try:
        # Use default coordinates if none provided
        lat = lat if lat is not None else LATITUDE
        lon = lon if lon is not None else LONGITUDE

        # Convert datetime to Julian day
        jd = swe.julday(
            date.year,
            date.month,
            date.day,
            date.hour + date.minute / 60.0 + date.second / 3600.0,
        )

        # Get planet position
        if planet not in PLANET_MAP:
            logger.error(f"Invalid planet name: {planet}")
            return [0.0]

        planet_id = PLANET_MAP[planet]
        try:
            # Calculate planet position
            result = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
            if result is None:
                raise ValueError(f"No result returned for planet {planet}")

            # Extract values from result
            values = result[0] if isinstance(result, tuple) else result
            if not isinstance(values, (list, tuple)) or len(values) < 1:
                values = [0.0]  # Default values if not enough returned

            longitude = float(values[0])  # Longitude is first value

            # Calculate various strength factors
            dignity = calculate_dignity_score(longitude, planet)
            aspects = calculate_aspect_strength(longitude, jd)
            house = int(longitude / 30) + 1  # Get house number (1-12)
            placement = calculate_house_placement_score(longitude, house)

            # Combine factors with weights
            strength = (
                dignity * 0.4
                + aspects * 0.3  # Dignity has highest weight
                + placement * 0.3  # Aspects second  # House placement third
            )

            return [max(0.0, min(1.0, strength))]  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error calculating {planet} position: {str(e)}")
            return [0.0]

    except Exception as e:
        logger.error(f"Error in calculate_planet_strength: {str(e)}")
        return [0.0]


def calculate_sarvashtakavarga(date, lat=None, lon=None):
    """Calculate Sarvashtakavarga (total Ashtakavarga points)."""
    try:
        positions = get_planet_positions(date, lat, lon)
        if not positions:
            return [0.0]

        total_strength = 0

        for planet in PLANETS:
            if planet not in positions:
                continue

            planet_data = positions[planet]
            if not isinstance(planet_data, dict):
                continue

            # Calculate various strengths
            sthana_bala = calculate_house_placement_score(
                planet_data.get("longitude", 0.0), planet_data.get("house", 1)
            )

            # Angular houses get more strength
            dig_bala = 1.5 if planet_data.get("house", 1) in [1, 4, 7, 10] else 1.0

            # Retrograde planets get less strength
            kala_bala = 0.8 if planet_data.get("is_retrograde", False) else 1.0

            # Combine strengths and normalize
            planet_strength = (sthana_bala + dig_bala + kala_bala) / 3.5
            total_strength += planet_strength

        # Return normalized total strength
        return [max(0.0, min(1.0, total_strength / len(PLANETS)))]

    except Exception as e:
        logger.error(f"Error calculating Sarvashtakavarga: {str(e)}")
        return [0.0]


def check_shadbala(date, lat=None, lon=None):
    """Calculate Shadbala (six-fold strength) for planets."""
    try:
        positions = get_planet_positions(date, lat, lon)
        total_strength = 0

        for planet in PLANETS:
            if planet not in positions:
                continue

            # Calculate various strengths
            sthana_bala = calculate_house_placement_score(
                positions[planet]["longitude"], positions[planet]["house"]
            )
            dig_bala = 1.5 if positions[planet]["house"] in [1, 4, 7, 10] else 1.0
            kala_bala = 1.0  # Time strength (simplified)

            # Combine strengths
            planet_strength = (
                sthana_bala + dig_bala + kala_bala
            ) / 3.5  # Normalize to 0-1
            total_strength += planet_strength

        return [total_strength / len(PLANETS)]

    except Exception as e:
        logger.error(f"Error calculating Shadbala: {str(e)}")
        return [0.0]


def calculate_bhava_chalit_aspects(date, lat=None, lon=None):
    """Calculate aspects using Bhava Chalit system."""
    try:
        positions = get_planet_positions(date, lat, lon)
        total_aspects = 0

        # Calculate aspects for each house
        for p1 in PLANETS[:-1]:  # Exclude Ketu as it's always opposite Rahu
            pos1 = positions.get(p1, {}).get("longitude", 0)

            for p2 in PLANETS:
                if p2 not in positions or p2 == p1:
                    continue

                # Calculate angular distance
                angle = abs(pos1 - positions[p2]["longitude"])
                if angle > 180:
                    angle = 360 - angle

                # Check major aspects (60°, 90°, 120°, 180°)
                aspects = [60, 90, 120, 180]
                for aspect in aspects:
                    if abs(angle - aspect) <= 6:  # 6° orb
                        total_aspects += 1

        return [total_aspects / (len(PLANETS) * (len(PLANETS) - 1))]  # Return as list

    except Exception as e:
        logger.error(f"Error calculating Bhava Chalit aspects: {str(e)}")
        return [0.0]


def calculate_special_lagna_strength(positions, lagna_type, lat=None, lon=None):
    """Calculate strength of special lagna.

    Args:
        positions: Dictionary of planetary positions
        lagna_type: Type of lagna ('hora', 'ghati', 'vighati')
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        Strength score between 0 and 1
    """
    try:
        # Get ascendant position
        asc_long = 0  # Default to 0 if no ascendant
        for house in range(1, 13):
            if get_house_lord(house) == "Sun":  # Sun rules the ascendant
                asc_long = (house - 1) * 30
                break

        # Calculate lagna position based on type
        if lagna_type == "hora":
            lagna_pos = (asc_long + 30) % 360
        elif lagna_type == "ghati":
            lagna_pos = (asc_long + 45) % 360
        else:  # vighati
            lagna_pos = (asc_long + 60) % 360

        # Calculate strength based on planetary aspects
        strength = 0
        for planet in PLANETS:
            if planet not in positions:
                continue

            planet_pos = positions[planet]["longitude"]
            angle = abs(planet_pos - lagna_pos)
            if angle > 180:
                angle = 360 - angle

            # Check aspects
            if angle < 10:  # Conjunction
                strength += 1.0
            elif abs(angle - 120) < 10:  # Trine
                strength += 0.75
            elif abs(angle - 60) < 10:  # Sextile
                strength += 0.5

        return strength / len(PLANETS)  # Normalize

    except Exception as e:
        logger.error(f"Error calculating {lagna_type} lagna strength: {str(e)}")
        return 0.0


def calculate_special_lagnas(date, lat=None, lon=None):
    """Calculate additional special lagnas.

    Args:
        date: Datetime of the game
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List of special lagna strengths
    """
    try:
        positions = get_planet_positions(date, lat, lon)
        if not positions:
            return [0.0] * 3  # Return zeros for all three lagnas

        # Calculate special lagnas (Hora, Ghati, Vighati)
        lagna_scores = []
        for lagna_type in ["hora", "ghati", "vighati"]:
            score = calculate_special_lagna_strength(positions, lagna_type, lat, lon)
            lagna_scores.append(score)

        return lagna_scores

    except Exception as e:
        logger.error(f"Error calculating special lagnas: {str(e)}")
        return [0.0] * 3


def calculate_moon_phase(dt: datetime, lat=None, lon=None) -> List[float]:
    """Calculate the moon phase strength.

    Args:
        dt: Datetime of the game
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List containing moon phase strength (0-1)
    """
    try:
        jd = swe.julday(
            dt.year, dt.month, dt.day, dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        )

        # Get Sun and Moon positions
        result_sun = swe.calc_ut(jd, swe.SUN)
        result_moon = swe.calc_ut(jd, swe.MOON)

        if result_sun is None or result_moon is None:
            return [0.5]

        # Extract longitudes
        sun_values = result_sun[0] if isinstance(result_sun, tuple) else result_sun
        moon_values = result_moon[0] if isinstance(result_moon, tuple) else result_moon

        if not isinstance(sun_values, (list, tuple)) or not isinstance(
            moon_values, (list, tuple)
        ):
            return [0.5]

        sun_long = float(sun_values[0])
        moon_long = float(moon_values[0])

        # Calculate phase angle
        phase_angle = (moon_long - sun_long) % 360

        # Calculate strength based on phase
        # Full moon (around 180°) and new moon (around 0° or 360°) are considered strong
        strength = 1.0 - abs(180 - phase_angle) / 180.0

        return [max(0.1, min(1.0, strength))]

    except Exception as e:
        logger.error(f"Error calculating moon phase: {str(e)}")
        return [0.5]


def calculate_retrograde_impact(date, lat=None, lon=None):
    """Calculate the impact of retrograde planets."""
    try:
        positions = get_planet_positions(date, lat, lon)
        retro_impact = 0
        retro_count = 0

        for planet in PLANETS:
            if planet not in positions:
                continue

            if positions[planet].get("is_retrograde", False):
                retro_count += 1

                # Check house placement
                house = positions[planet]["house"]
                if house in [1, 4, 5, 7, 9, 10]:
                    retro_impact += 0.5
                else:
                    retro_impact -= 0.5

        if retro_count == 0:
            return [1.0]  # Neutral impact if no retrogrades

        return [(retro_impact / retro_count + 1) / 2]  # Normalize to 0-1
    except Exception as e:
        logger.error(f"Error calculating retrograde impact: {str(e)}")
        return [0.0]


def calculate_planetary_alignment_score(date, lat=None, lon=None):
    """Calculate the alignment score between planets."""
    positions = get_planet_positions(date, lat, lon)

    # Calculate aspects and alignments between planets
    alignment_score = 0

    # Check conjunctions and aspects
    for i, planet1 in enumerate(
        PLANETS[:-1]
    ):  # Exclude Ketu as it's always opposite Rahu
        pos1 = positions.get(planet1, {}).get("longitude", 0)

        for planet2 in PLANETS[i + 1 :]:
            pos2 = positions.get(planet2, {}).get("longitude", 0)

            # Calculate angular distance
            angle = abs(pos1 - pos2)
            if angle > 180:
                angle = 360 - angle

            # Check major aspects (0°, 60°, 90°, 120°, 180°)
            aspects = [0, 60, 90, 120, 180]
            for aspect in aspects:
                orb = 6  # Allowing 6° orb
                if abs(angle - aspect) <= orb:
                    # Weight the aspects differently
                    weights = {0: 1.0, 60: 0.5, 90: 0.75, 120: 0.75, 180: 1.0}
                    alignment_score += weights[aspect] * (1 - abs(angle - aspect) / orb)

    # Normalize score between 0 and 1
    max_possible_score = (
        len(PLANETS) * (len(PLANETS) - 1) / 2
    )  # Maximum number of aspects
    alignment_score = min(1.0, alignment_score / max_possible_score)

    return [alignment_score]


def calculate_victory_yogas(
    dt: datetime, team: str = None, lat=None, lon=None
) -> List[float]:
    """Calculate victory yoga scores based on planetary positions

    Args:
        dt: Datetime of the game
        team: Optional team name for team-specific calculations
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List containing victory yoga score
    """
    try:
        jd = swe.julday(
            dt.year, dt.month, dt.day, dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        )

        # Get Sun and Moon positions
        result_sun = swe.calc_ut(jd, swe.SUN)
        result_moon = swe.calc_ut(jd, swe.MOON)
        result_jupiter = swe.calc_ut(jd, swe.JUPITER)

        if result_sun is None or result_moon is None or result_jupiter is None:
            return [0.5]

        # Extract longitudes
        sun_values = result_sun[0] if isinstance(result_sun, tuple) else result_sun
        moon_values = result_moon[0] if isinstance(result_moon, tuple) else result_moon
        jupiter_values = (
            result_jupiter[0] if isinstance(result_jupiter, tuple) else result_jupiter
        )

        if (
            not isinstance(sun_values, (list, tuple))
            or not isinstance(moon_values, (list, tuple))
            or not isinstance(jupiter_values, (list, tuple))
        ):
            return [0.5]

        sun_long = float(sun_values[0])
        moon_long = float(moon_values[0])
        jupiter_long = float(jupiter_values[0])

        # Calculate strength based on planetary positions and aspects
        strength = calculate_victory_yoga_strength(sun_long, moon_long, jupiter_long)

        return [max(0.0, min(1.0, strength))]

    except Exception as e:
        logger.error(f"Error calculating victory yogas: {str(e)}")
        return [0.5]


def calculate_special_lagnas(dt: datetime, lat=None, lon=None) -> List[float]:
    """Calculate special lagna positions and strengths

    Args:
        dt: Datetime of the game
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List of special lagna strengths
    """
    try:
        # Initialize ephemeris
        swe.set_ephe_path()

        # Convert datetime to Julian day
        jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60.0)

        # Calculate Arudha Lagna (AL)
        ascendant = swe.houses(jd, lat, lon)[0][0]
        lord_of_asc = get_house_lord(ascendant)
        lord_pos = swe.calc_ut(jd, PLANET_MAP[lord_of_asc])[0][0]
        al_pos = (2 * lord_pos - ascendant) % 360

        # Calculate Hora Lagna (HL)
        sun_pos = swe.calc_ut(jd, swe.SUN)[0][0]
        hl_pos = (sun_pos + (dt.hour * 15)) % 360

        # Calculate Ghati Lagna (GL)
        gl_pos = (sun_pos + (dt.minute * 0.25)) % 360

        # Calculate strengths based on aspects and positions
        al_strength = calculate_lagna_strength(al_pos, jd, lat, lon)
        hl_strength = calculate_lagna_strength(hl_pos, jd, lat, lon)
        gl_strength = calculate_lagna_strength(gl_pos, jd, lat, lon)

        return [al_strength, hl_strength, gl_strength]

    except Exception as e:
        logger.error(f"Error calculating special lagnas: {str(e)}")
        return [0.0, 0.0, 0.0]


def calculate_lagna_strength(
    lagna_pos: float, jd: float, lat: float, lon: float
) -> float:
    """Calculate the strength of a lagna position

    Args:
        lagna_pos: Position of the lagna in degrees
        jd: Julian day
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        Strength score between 0 and 1
    """
    try:
        strength = 0.0

        # Check benefic aspects
        for planet in ["Jupiter", "Venus", "Mercury"]:
            planet_pos = swe.calc_ut(jd, PLANET_MAP[planet])[0][0]
            aspect_angle = abs(planet_pos - lagna_pos) % 360

            # Favorable aspects (trine, sextile)
            if any(abs(aspect_angle - angle) <= 8 for angle in [60, 120]):
                strength += 0.2

        # Check malefic aspects (reduce strength)
        for planet in ["Saturn", "Mars", "Rahu"]:
            planet_pos = swe.calc_ut(jd, PLANET_MAP[planet])[0][0]
            aspect_angle = abs(planet_pos - lagna_pos) % 360

            # Difficult aspects (square, opposition)
            if any(abs(aspect_angle - angle) <= 8 for angle in [90, 180]):
                strength -= 0.15

        # Normalize strength to 0-1 range
        strength = max(0.0, min(1.0, strength + 0.5))  # Add 0.5 as base strength

        return strength

    except Exception as e:
        logger.error(f"Error calculating lagna strength: {str(e)}")
        return 0.0


def get_nakshatra_lord(nakshatra_num):
    """Get the ruling planet of a nakshatra (1-27)."""
    lords = [
        "Ketu",
        "Venus",
        "Sun",
        "Moon",
        "Mars",
        "Rahu",
        "Jupiter",
        "Saturn",
        "Mercury",
    ] * 3
    return lords[(nakshatra_num - 1) % 9]


def get_house_lord(house_num: int) -> str:
    """Get the ruling planet of a house (1-12).

    Args:
        house_num: House number (1-12)

    Returns:
        Name of the ruling planet
    """
    # Natural zodiac house lords
    house_lords = {
        1: "Mars",  # Aries
        2: "Venus",  # Taurus
        3: "Mercury",  # Gemini
        4: "Moon",  # Cancer
        5: "Sun",  # Leo
        6: "Mercury",  # Virgo
        7: "Venus",  # Libra
        8: "Mars",  # Scorpio
        9: "Jupiter",  # Sagittarius
        10: "Saturn",  # Capricorn
        11: "Saturn",  # Aquarius
        12: "Jupiter",  # Pisces
    }

    house_num = ((house_num - 1) % 12) + 1
    return house_lords.get(house_num, "Sun")  # Default to Sun if invalid house


def calculate_house_lord_strength(
    house_num: int, date: datetime, lat: float = None, lon: float = None
) -> List[float]:
    """Calculate the strength of a house lord.

    Args:
        house_num: House number (1-12)
        date: Datetime of the game
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List containing house lord strength (0-1)
    """
    try:
        # Use default coordinates if none provided
        lat = lat if lat is not None else LATITUDE
        lon = lon if lon is not None else LONGITUDE

        # Convert datetime to Julian day
        jd = swe.julday(
            date.year,
            date.month,
            date.day,
            date.hour + date.minute / 60.0 + date.second / 3600.0,
        )

        # Get house lord
        house_lord = get_house_lord(house_num)
        if not house_lord or house_lord not in PLANET_MAP:
            return [0.0]

        # Get planet position
        planet_id = PLANET_MAP[house_lord]
        try:
            result = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
            if result is None:
                raise ValueError(f"No result returned for planet {house_lord}")

            # Extract values from result
            values = result[0] if isinstance(result, tuple) else result
            if not isinstance(values, (list, tuple)) or len(values) < 1:
                values = [0.0]  # Default values if not enough returned

            longitude = float(values[0])  # Longitude is first value

            # Calculate various strength factors
            dignity = calculate_dignity_score(longitude, house_lord)
            aspects = calculate_aspect_strength(longitude, jd)
            current_house = int(longitude / 30) + 1
            placement = calculate_house_placement_score(longitude, current_house)

            # Combine factors with weights
            strength = dignity * 0.4 + aspects * 0.3 + placement * 0.3

            return [max(0.0, min(1.0, strength))]

        except Exception as e:
            logger.error(f"Error calculating house lord position: {str(e)}")
            return [0.0]

    except Exception as e:
        logger.error(f"Error in calculate_house_lord_strength: {str(e)}")
        return [0.0]


def calculate_dignity_score(longitude: float, planet: str) -> float:
    """Calculate dignity score based on planet's position."""
    sign = int(longitude / 30)
    degree = longitude % 30

    # Basic dignity scores
    if planet == HOUSE_LORDS.get(sign + 1):  # Planet rules this sign
        return 1.0
    elif planet == HOUSE_LORDS.get((sign + 6) % 12 + 1):  # Planet rules opposite sign
        return 0.2
    else:
        return 0.5


def calculate_aspect_strength(longitude: float, jd: float) -> float:
    """Calculate aspect strength from other planets."""
    total_strength = 0.0
    count = 0

    for planet in PLANETS:
        if planet in PLANET_MAP:
            try:
                result = swe.calc_ut(jd, PLANET_MAP[planet], swe.FLG_SWIEPH)
                if result is None:
                    continue

                # Extract values from result
                values = result[0] if isinstance(result, tuple) else result
                if not isinstance(values, (list, tuple)) or len(values) < 1:
                    continue

                other_long = float(values[0])  # Longitude is first value
                aspect_diff = abs(longitude - other_long) % 360

                # Check major aspects
                for aspect, orb in ASPECT_ORBS.items():
                    if abs(aspect_diff - aspect) <= orb:
                        if aspect in [0, 60, 120]:  # Favorable aspects
                            total_strength += 1.0
                        else:  # Challenging aspects
                            total_strength += 0.3
                        count += 1
                        break

            except Exception as e:
                logger.warning(
                    f"Error calculating aspect for planet {planet}: {str(e)}"
                )
                continue

    return total_strength / (count + 1) if count > 0 else 0.5


def calculate_house_placement_score(longitude: float, house: int) -> float:
    """Calculate the strength of a planet based on its house placement and longitude."""
    try:
        # Base house strength
        house_strength = {
            1: 1.0,  # Ascendant - strongest
            10: 0.9,  # MC - very strong
            7: 0.8,  # Descendant - strong
            4: 0.7,  # IC - moderately strong
            5: 0.6,
            9: 0.6,  # Trine houses - good
            2: 0.5,
            11: 0.5,  # Succedent houses - neutral
            3: 0.4,
            6: 0.4,
            12: 0.4,  # Cadent houses - weak
        }.get(
            house, 0.3
        )  # Default for unknown house

        # Position within house (each house is 30 degrees)
        house_position = longitude % 30
        position_strength = (
            abs(15 - house_position) / 15.0
        )  # Strongest at middle of house

        # Combine and normalize
        total_strength = (house_strength + position_strength) / 2.0
        return max(0.0, min(1.0, total_strength))

    except Exception as e:
        logger.error(f"Error calculating house placement score: {str(e)}")
        return 0.0


def calculate_sublords(date, lat=None, lon=None):
    """Calculate sublord influences."""
    try:
        positions = get_planet_positions(date, lat, lon)
        total_strength = 0
        for planet in PLANETS:
            if planet not in positions:
                continue

            pos = positions[planet]["longitude"]

            # Calculate sublord strengths
            star_lord = get_nakshatra_lord(int(pos * 27 / 360) + 1)
            sub_lord = get_nakshatra_lord(int(pos * 81 / 360) + 1)
            sub_sub_lord = get_nakshatra_lord(int(pos * 243 / 360) + 1)

            # Add strength based on benefic/malefic nature
            benefics = ["Jupiter", "Venus", "Mercury", "Moon"]
            malefics = ["Saturn", "Mars", "Rahu", "Ketu"]

            for lord in [star_lord, sub_lord, sub_sub_lord]:
                if lord in benefics:
                    total_strength += 1
                elif lord in malefics:
                    total_strength += 0.5

        # Normalize and return as list
        max_strength = len(PLANETS) * 3  # 3 lords per planet
        return [total_strength / max_strength]

    except Exception as e:
        logger.error(f"Error calculating sublords: {str(e)}")
        return [0.0]


def calculate_nakshatra_tara(date, lat=None, lon=None):
    """Calculate Nakshatra Tara (lunar mansion) strength."""
    try:
        positions = get_planet_positions(date, lat, lon)
        if "Moon" not in positions:
            return [0.0]

        moon_nakshatra = positions["Moon"]["nakshatra"]
        tara_number = ((moon_nakshatra - 1) % 9) + 1

        # Tara strengths (1-9 scale)
        tara_strengths = {
            1: 1.0,  # Janma (Birth) - Neutral
            2: 0.5,  # Sampat (Wealth) - Good
            3: 0.3,  # Vipat (Danger) - Bad
            4: 0.8,  # Kshema (Well-being) - Good
            5: 0.4,  # Pratyak (Obstacle) - Bad
            6: 0.7,  # Sadhaka (Achievement) - Good
            7: 0.2,  # Vadha (Destruction) - Bad
            8: 0.9,  # Mitra (Friend) - Good
            9: 0.6,  # Ati-Mitra (Great friend) - Good
        }

        return [tara_strengths.get(tara_number, 0.0)]

    except Exception as e:
        logger.error(f"Error calculating Nakshatra Tara: {str(e)}")
        return [0.0]


def calculate_muhurta_score(date: datetime, lat=None, lon=None) -> List[float]:
    """Calculate the auspiciousness of the muhurta (time period).

    Args:
        date: Datetime of the game
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List containing muhurta score (0-1)
    """
    try:
        # Get positions of key planets
        positions = get_planet_positions(date, lat, lon)

        # Calculate muhurta based on:
        # 1. Day of week ruler
        # 2. Hora (hour) ruler
        # 3. Planetary positions

        weekday = date.weekday()  # 0-6 (Monday-Sunday)
        hour = date.hour

        # Day rulers (Sun-Saturn)
        day_rulers = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
        day_ruler = day_rulers[weekday]

        # Get day ruler strength
        day_strength = calculate_planet_strength(date, day_ruler, lat, lon)[0]

        # Get hora ruler
        hora_num = (hour % 12) + 1
        hora_ruler = get_house_lord(hora_num)
        hora_strength = calculate_planet_strength(date, hora_ruler, lat, lon)[0]

        # Calculate overall muhurta score
        score = (day_strength + hora_strength) / 2

        return [min(1.0, max(0.0, score))]

    except Exception as e:
        logger.error(f"Error calculating muhurta score: {str(e)}")
        return [0.0]


def calculate_hora_score(date: datetime, lat=None, lon=None) -> List[float]:
    """Calculate the strength of the current hora (planetary hour).

    Args:
        date: Datetime of the game
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        List containing hora strength (0-1)
    """
    try:
        hour = date.hour
        is_day = 6 <= hour < 18  # Rough day/night division

        # Get hora number (1-12)
        hora_num = (hour % 12) + 1

        # Different rulers for day and night
        if is_day:
            hora_rulers = [
                "Sun",
                "Venus",
                "Mercury",
                "Moon",
                "Saturn",
                "Jupiter",
                "Mars",
            ]
        else:
            hora_rulers = [
                "Moon",
                "Saturn",
                "Jupiter",
                "Mars",
                "Sun",
                "Venus",
                "Mercury",
            ]

        # Get current hora ruler
        hora_ruler = hora_rulers[hora_num % 7]

        # Calculate hora strength based on:
        # 1. Ruler planet strength
        # 2. Aspect relationships
        strength = calculate_planet_strength(date, hora_ruler, lat, lon)[0]

        # Get aspects to hora ruler
        positions = get_planet_positions(date, lat, lon)
        ruler_pos = positions[hora_ruler]["longitude"]

        aspect_score = 0
        num_aspects = 0

        for planet in PLANETS:
            if planet == hora_ruler:
                continue

            planet_pos = positions[planet]["longitude"]
            angle = abs(ruler_pos - planet_pos) % 360

            # Check for major aspects
            for aspect_angle, orb in ASPECT_ORBS.items():
                if abs(angle - aspect_angle) <= orb:
                    # Benefic aspects add strength, malefic aspects reduce it
                    if planet in ["Jupiter", "Venus", "Moon"]:
                        aspect_score += 0.1
                    elif planet in ["Saturn", "Mars", "Rahu", "Ketu"]:
                        aspect_score -= 0.05
                    num_aspects += 1
                    break

        if num_aspects > 0:
            aspect_score /= num_aspects

        final_score = (strength + max(0, aspect_score)) / 2
        return [min(1.0, max(0.0, final_score))]

    except Exception as e:
        logger.error(f"Error calculating hora score: {str(e)}")
        return [0.0]


def calculate_ashtakavarga_points(
    planet_positions: Dict[str, float], jd: float
) -> float:
    """Calculate Ashtakavarga points for all planets."""
    try:
        total_points = 0

        # Calculate points for each planet
        for planet in PLANETS:
            if planet in PLANET_MAP:
                ret, data = swe.calc_ut(jd, PLANET_MAP[planet], swe.FLG_SWIEPH)
                if ret >= 0:
                    longitude = data[0]
                    house = int(longitude / 30) + 1

                    # Basic points based on house placement
                    if house in [1, 4, 7, 10]:  # Angular houses
                        total_points += 1.0
                    elif house in [2, 5, 8, 11]:  # Succedent houses
                        total_points += 0.75
                    else:  # Cadent houses
                        total_points += 0.5

                    # Additional points for aspects
                    total_points += calculate_aspect_strength(longitude, jd)

        # Normalize points to 0-1 range
        max_possible_points = len(PLANETS) * 2  # Maximum points possible
        normalized_points = total_points / max_possible_points

        return min(max(normalized_points, 0.0), 1.0)

    except Exception as e:
        logger.error(f"Error calculating Ashtakavarga points: {e}")
        return 0.0


def check_vimshottari_dasa(date, lat=None, lon=None):
    """Check Vimshottari Dasa influences."""
    try:
        positions = get_planet_positions(date, lat, lon)
        if "Moon" not in positions:
            return [0.0]

        moon_nakshatra = positions["Moon"]["nakshatra"]
        dasa_lord = get_nakshatra_lord(moon_nakshatra)

        strength = 1.0
        if dasa_lord in positions:
            # Check if dasa lord is strong
            if positions[dasa_lord]["house"] in [1, 4, 5, 7, 9, 10]:
                strength *= 1.3
            if not positions[dasa_lord].get("is_retrograde", False):
                strength *= 1.2

        return [strength]
    except Exception as e:
        logger.error(f"Error checking Vimshottari Dasa: {str(e)}")
        return [0.0]


def calculate_divisional_strength(date, lat=None, lon=None):
    """Calculate strength across multiple divisional charts."""
    try:
        positions = get_planet_positions(date, lat, lon)
        total_strength = 0
        for planet in PLANETS:
            if planet in positions:
                long = positions[planet]["longitude"]

                # D1 (Rashi) - Birth chart
                rashi_strength = 1.0

                # D9 (Navamsa) - 9th division
                nav_long = (long * 9) % 360
                nav_house = int(nav_long / 30) + 1
                if nav_house in [1, 4, 5, 7, 9, 10]:
                    rashi_strength *= 1.2

                total_strength += rashi_strength

        return [total_strength / len(PLANETS)]  # Normalize

    except Exception as e:
        logger.error(f"Error calculating divisional strength: {str(e)}")
        return [0.0]


# Initialize ephemeris when module is loaded
init_ephemeris()

"""
Module for calculating planetary positions in Vedic astrology.
"""

import swisseph as swe
import logging
from typing import Dict, Tuple, Any, Optional
import ctypes
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
    "True Node": swe.TRUE_NODE,
}


def setup_ephemeris(ephe_path: str = None) -> None:
    """
    Set up the Swiss Ephemeris with the given path.

    Args:
        ephe_path (str): Path to the ephemeris files.
    """
    if ephe_path is None:
        # Use default path relative to project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        ephe_path = os.path.join(project_root, "ephe")

    swe.set_ephe_path(ephe_path)
    logger.info(f"Swiss Ephemeris set up with path: {ephe_path}")


def validate_inputs(julian_day: float, lat: float, lon: float) -> None:
    """Validate input parameters for planetary calculations."""
    if not isinstance(julian_day, (int, float)) or julian_day < 0:
        raise ValueError("Julian Day must be a positive number")

    if not isinstance(lat, (int, float)) or lat < -90 or lat > 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")

    if not isinstance(lon, (int, float)) or lon < -180 or lon > 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")


def calculate_planet_positions(
    julian_day: float, lat: float, lon: float
) -> Tuple[Dict[str, float], Dict[str, int], Optional[Dict[str, Any]]]:
    """
    Calculate planetary positions and houses.

    Args:
        julian_day (float): Julian day number.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        Tuple[Dict[str, float], Dict[str, int], Optional[Dict[str, Any]]]:
            Positions, houses, and astrological points.
    """
    # Validate inputs
    validate_inputs(julian_day, lat, lon)

    houses, ascmc = swe.houses(julian_day, lat, lon, b"W")  # 'W' for Whole Sign houses
    ascendant = houses[0]

    positions = {"Ascendant": ascendant}
    houses_dict = {"Ascendant": 1}

    for planet_name, planet_id in PLANETS.items():
        try:
            result = swe.calc_ut(
                julian_day, planet_id, flags=swe.FLG_SWIEPH | swe.FLG_SIDEREAL
            )
            position = result[0][0]
            positions[planet_name] = float(position)

            # Calculate house placement
            house = int((position - ascendant) / 30) + 1
            if house <= 0:
                house += 12
            houses_dict[planet_name] = house

        except Exception as e:
            logger.error(f"Error calculating position for {planet_name}: {str(e)}")
            positions[planet_name] = 0.0
            houses_dict[planet_name] = 1

    # Calculate additional points
    try:
        sunrise, sunset = calculate_sunrise_sunset(julian_day, lat, lon)
        gulika = calculate_gulika(julian_day, sunrise, sunset, lat, lon)
        upaketu = calculate_upaketu(julian_day, sunrise, sunset, lat, lon)
        yogi = calculate_yogi_point(ascendant, positions["Moon"], positions["Sun"])

        points = {"Gulika": gulika, "Upaketu": upaketu, "Yogi": yogi}
    except Exception as e:
        logger.error(f"Error calculating additional points: {str(e)}")
        points = None

    return positions, houses_dict, points


def calculate_sunrise_sunset(
    julian_day: float, lat: float, lon: float
) -> Tuple[float, float]:
    """
    Calculate sunrise and sunset times.

    Args:
        julian_day (float): Julian day number.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        Tuple[float, float]: Sunrise and sunset times as Julian day fractions.
    """
    geopos = (ctypes.c_double * 3)(float(lon), float(lat), 0.0)
    rsmi_rise = swe.CALC_RISE | swe.BIT_HINDU_RISING
    rsmi_set = swe.CALC_SET | swe.BIT_HINDU_RISING

    try:
        sunrise = swe.rise_trans(julian_day, swe.SUN, rsmi_rise, geopos, 0.0, 0.0)[1][0]
        sunset = swe.rise_trans(julian_day, swe.SUN, rsmi_set, geopos, 0.0, 0.0)[1][0]
        logger.info(f"Calculated sunrise: {sunrise}, sunset: {sunset}")
        return sunrise, sunset
    except swe.Error as e:
        logger.error(f"Error calculating sunrise/sunset: {e}")
        return None, None


def calculate_gulika(
    julian_day: float, sunrise: float, sunset: float, lat: float, lon: float
) -> float:
    """
    Calculate the position of Gulika.

    Args:
        julian_day (float): Julian day number.
        sunrise (float): Sunrise time as Julian day fraction.
        sunset (float): Sunset time as Julian day fraction.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        float: Gulika position in degrees.
    """
    if sunrise is None or sunset is None:
        logger.error("Cannot calculate Gulika: sunrise or sunset is None")
        return None

    day_duration = sunset - sunrise
    part_duration = day_duration / 8
    day_of_week = int(swe.day_of_week(julian_day))
    gulika_parts = [7, 6, 5, 4, 3, 2, 1, 0]
    gulika_part = gulika_parts[day_of_week]
    gulika_time = sunrise + (part_duration * gulika_part)

    try:
        houses, _ = swe.houses(gulika_time, lat, lon, b"P")
        gulika_position = (houses[0] + (gulika_part * 45)) % 360
        logger.info(f"Calculated Gulika position: {gulika_position}")
        return gulika_position
    except swe.Error as e:
        logger.error(f"Error calculating Gulika position: {e}")
        return None


def calculate_upaketu(
    julian_day: float, sunrise: float, sunset: float, lat: float, lon: float
) -> float:
    """
    Calculate the position of Upaketu.

    Args:
        julian_day (float): Julian day number.
        sunrise (float): Sunrise time as Julian day fraction.
        sunset (float): Sunset time as Julian day fraction.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        float: Upaketu position in degrees.
    """
    if sunrise is None or sunset is None:
        logger.error("Cannot calculate Upaketu: sunrise or sunset is None")
        return None

    day_duration = sunset - sunrise
    part_duration = day_duration / 8
    day_of_week = int(swe.day_of_week(julian_day))
    upaketu_parts = [5, 4, 3, 2, 1, 7, 6]
    upaketu_part = upaketu_parts[day_of_week]
    upaketu_time = sunrise + (part_duration * upaketu_part)

    try:
        houses, _ = swe.houses(upaketu_time, lat, lon, b"P")
        upaketu_position = (houses[0] + (upaketu_part * 30)) % 360
        logger.info(f"Calculated Upaketu position: {upaketu_position}")
        return upaketu_position
    except swe.Error as e:
        logger.error(f"Error calculating Upaketu position: {e}")
        return None


def calculate_yogi_point(ascendant: float, moon: float, sun: float) -> float:
    """
    Calculate the Yogi Point.

    Args:
        ascendant (float): Ascendant position in degrees.
        moon (float): Moon position in degrees.
        sun (float): Sun position in degrees.

    Returns:
        float: Yogi Point position in degrees.
    """
    yogi_point = (ascendant + moon - sun + 360) % 360
    logger.info(f"Calculated Yogi Point: {yogi_point}")
    return yogi_point


def get_planet_name(planet_id: int) -> str:
    """
    Get the name of a planet from its Swiss Ephemeris ID.

    Args:
        planet_id (int): Swiss Ephemeris planet ID.

    Returns:
        str: Name of the planet.
    """
    return next((name for name, pid in PLANETS.items() if pid == planet_id), "Unknown")


def calculate_ayanamsa(julian_day: float) -> float:
    """
    Calculate the ayanamsa (precession of the equinoxes) for a given date.

    Args:
        julian_day (float): Julian day number.

    Returns:
        float: Ayanamsa value in degrees.
    """
    try:
        ayanamsa = swe.get_ayanamsa(julian_day)
        logger.info(f"Calculated ayanamsa: {ayanamsa}")
        return ayanamsa
    except swe.Error as e:
        logger.error(f"Error calculating ayanamsa: {e}")
        return None


def main():
    # Example usage
    setup_ephemeris()

    year, month, day = 2023, 7, 15
    hour, minute, second = 12, 0, 0
    latitude, longitude = 40.7128, -74.0060  # New York City

    julian_day = swe.julday(year, month, day, hour + minute / 60 + second / 3600)

    try:
        positions, houses, _ = calculate_planet_positions(
            julian_day, latitude, longitude
        )
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        return

    print("Planetary Positions and Houses:")
    for planet, position in positions.items():
        print(f"{planet}: {position:.2f}° (House {houses.get(planet, 'N/A')})")

    sunrise, sunset = calculate_sunrise_sunset(julian_day, latitude, longitude)
    print(f"\nSunrise: {sunrise}")
    print(f"Sunset: {sunset}")

    gulika = calculate_gulika(julian_day, sunrise, sunset, latitude, longitude)
    upaketu = calculate_upaketu(julian_day, sunrise, sunset, latitude, longitude)
    print(f"\nGulika position: {gulika:.2f}°")
    print(f"Upaketu position: {upaketu:.2f}°")

    ayanamsa = calculate_ayanamsa(julian_day)
    print(f"\nAyanamsa: {ayanamsa:.2f}°")


if __name__ == "__main__":
    main()

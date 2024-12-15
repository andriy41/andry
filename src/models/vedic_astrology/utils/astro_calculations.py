"""Utility functions for astrological calculations."""

import ephem
from datetime import datetime
import pytz
from typing import Dict, Any, Optional, List, Tuple


def calculate_planet_positions(dt: datetime) -> Dict[str, Dict[str, float]]:
    """Calculate positions of major planets for given date/time."""
    observer = ephem.Observer()
    observer.date = dt

    positions = {}
    planets = {
        "sun": ephem.Sun(),
        "moon": ephem.Moon(),
        "mars": ephem.Mars(),
        "jupiter": ephem.Jupiter(),
        "saturn": ephem.Saturn(),
        "venus": ephem.Venus(),
        "mercury": ephem.Mercury(),
    }

    for name, planet in planets.items():
        planet.compute(observer)
        positions[name] = {
            "longitude": float(planet.hlong),
            "latitude": float(planet.hlat),
            "phase": float(planet.phase),
            "altitude": float(planet.alt),
            "azimuth": float(planet.az),
        }

    return positions


def calculate_moon_phase(dt: datetime) -> float:
    """Calculate the moon phase (0-1) for the given datetime."""
    moon = ephem.Moon()
    observer = ephem.Observer()
    observer.date = dt
    moon.compute(observer)
    return moon.phase / 100.0


def calculate_nakshatra(moon_longitude: float) -> str:
    """Calculate the Nakshatra (lunar mansion) based on moon's longitude."""
    # Each nakshatra spans 13°20' (13.33333... degrees)
    nakshatra_span = 13 + (20 / 60)
    nakshatra_names = [
        "Ashwini",
        "Bharani",
        "Krittika",
        "Rohini",
        "Mrigashira",
        "Ardra",
        "Punarvasu",
        "Pushya",
        "Ashlesha",
        "Magha",
        "Purva Phalguni",
        "Uttara Phalguni",
        "Hasta",
        "Chitra",
        "Swati",
        "Vishakha",
        "Anuradha",
        "Jyeshtha",
        "Moola",
        "Purva Ashadha",
        "Uttara Ashadha",
        "Shravana",
        "Dhanishta",
        "Shatabhisha",
        "Purva Bhadrapada",
        "Uttara Bhadrapada",
        "Revati",
    ]

    nakshatra_num = int(moon_longitude / nakshatra_span)
    return nakshatra_names[nakshatra_num % 27]


def parse_game_datetime(
    game_date: str, game_time: str, timezone: str
) -> Optional[datetime]:
    """Parse game date and time into a datetime object."""
    try:
        tz = pytz.timezone(timezone)
        dt_str = f"{game_date} {game_time}"
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        return tz.localize(dt)
    except Exception as e:
        return None


def calculate_aspects(
    planet_positions: Dict[str, Dict[str, float]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Calculate aspects between planets."""
    aspects = {}

    # Major aspects and their orbs (in degrees)
    aspect_types = {
        "conjunction": {"angle": 0, "orb": 10},
        "opposition": {"angle": 180, "orb": 10},
        "trine": {"angle": 120, "orb": 8},
        "square": {"angle": 90, "orb": 8},
        "sextile": {"angle": 60, "orb": 6},
    }

    for p1 in planet_positions:
        aspects[p1] = []
        p1_long = planet_positions[p1]["longitude"]

        for p2 in planet_positions:
            if p1 >= p2:  # Avoid duplicate aspects
                continue

            p2_long = planet_positions[p2]["longitude"]

            # Calculate the shortest angular distance between planets
            diff = abs(p1_long - p2_long)
            if diff > 180:
                diff = 360 - diff

            # Check each aspect type
            for aspect_name, aspect_data in aspect_types.items():
                orb = abs(diff - aspect_data["angle"])
                if orb <= aspect_data["orb"]:
                    aspects[p1].append(
                        {
                            "planet": p2,
                            "type": aspect_name,
                            "orb": orb,
                            "exact": orb <= 1,
                            "applying": is_aspect_applying(
                                p1_long, p2_long, aspect_data["angle"]
                            ),
                        }
                    )

    return aspects


def is_aspect_applying(p1_long: float, p2_long: float, aspect_angle: float) -> bool:
    """Determine if an aspect is applying or separating."""
    # This is a simplified version - in reality, we'd need to consider
    # the planets' speeds, but for our purposes this approximation is sufficient
    target_angle = (p1_long + aspect_angle) % 360
    diff = (target_angle - p2_long) % 360
    return diff <= 180

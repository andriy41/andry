"""
Handles astrological calculations for the Vedic model.
"""
import swisseph as swe
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
import os
import logging
import ephem
import math

# Initialize ephemeris data path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ephe_path = os.path.join(project_root, "data", "ephe")
os.makedirs(ephe_path, exist_ok=True)
swe.set_ephe_path(ephe_path)

# Initialize logger
logger = logging.getLogger(__name__)


class AstroCalculator:
    def __init__(self):
        """Initialize the AstroCalculator with planet definitions."""
        # Define planet IDs according to Swiss Ephemeris
        self.planets = {
            "Sun": swe.SUN,  # SE_SUN
            "Moon": swe.MOON,  # SE_MOON
            "Mars": swe.MARS,  # SE_MARS
            "Mercury": swe.MERCURY,  # SE_MERCURY
            "Jupiter": swe.JUPITER,  # SE_JUPITER
            "Venus": swe.VENUS,  # SE_VENUS
            "Saturn": swe.SATURN,  # SE_SATURN
            "Rahu": swe.MEAN_NODE,  # SE_MEAN_NODE (North Node)
            "Ketu": swe.TRUE_NODE,  # SE_TRUE_NODE (South Node)
        }

        # Define auspicious houses for each planet
        self.auspicious_houses = {
            "Sun": [1, 5, 9],
            "Moon": [2, 4, 6],
            "Mars": [1, 4, 7, 10],
            "Mercury": [1, 5, 9],
            "Jupiter": [1, 5, 9],
            "Venus": [2, 4, 6],
            "Saturn": [3, 6, 11],
            "Rahu": [3, 6, 11],
            "Ketu": [3, 6, 11],
        }

        # Initialize stadium coordinates
        self.stadium_coords = {
            "ARI": {"lat": 33.5276, "lon": -112.2626},  # State Farm Stadium
            "ATL": {"lat": 33.7554, "lon": -84.4009},  # Mercedes-Benz Stadium
            "BAL": {"lat": 39.2780, "lon": -76.6227},  # M&T Bank Stadium
            "BUF": {"lat": 42.7738, "lon": -78.7870},  # Highmark Stadium
            "CAR": {"lat": 35.2258, "lon": -80.8529},  # Bank of America Stadium
            "CHI": {"lat": 41.8623, "lon": -87.6167},  # Soldier Field
            "CIN": {"lat": 39.0955, "lon": -84.5161},  # Paycor Stadium
            "CLE": {"lat": 41.5061, "lon": -81.6995},  # FirstEnergy Stadium
            "DAL": {"lat": 32.7473, "lon": -97.0945},  # AT&T Stadium
            "DEN": {"lat": 39.7439, "lon": -105.0201},  # Empower Field at Mile High
            "DET": {"lat": 42.3400, "lon": -83.0456},  # Ford Field
            "GB": {"lat": 44.5013, "lon": -88.0622},  # Lambeau Field
            "HOU": {"lat": 29.6847, "lon": -95.4107},  # NRG Stadium
            "IND": {"lat": 39.7601, "lon": -86.1639},  # Lucas Oil Stadium
            "JAX": {"lat": 30.3239, "lon": -81.6373},  # TIAA Bank Field
            "KC": {"lat": 39.0489, "lon": -94.4839},  # Arrowhead Stadium
            "LAC": {"lat": 33.8644, "lon": -118.2611},  # SoFi Stadium
            "LAR": {"lat": 33.8644, "lon": -118.2611},  # SoFi Stadium
            "LV": {"lat": 36.0909, "lon": -115.1833},  # Allegiant Stadium
            "MIA": {"lat": 25.9580, "lon": -80.2389},  # Hard Rock Stadium
            "MIN": {"lat": 44.9735, "lon": -93.2575},  # U.S. Bank Stadium
            "NE": {"lat": 42.0909, "lon": -71.2643},  # Gillette Stadium
            "NO": {"lat": 29.9511, "lon": -90.0814},  # Caesars Superdome
            "NYG": {"lat": 40.8135, "lon": -74.0745},  # MetLife Stadium
            "NYJ": {"lat": 40.8135, "lon": -74.0745},  # MetLife Stadium
            "PHI": {"lat": 39.9008, "lon": -75.1675},  # Lincoln Financial Field
            "PIT": {"lat": 40.4468, "lon": -80.0158},  # Acrisure Stadium
            "SEA": {"lat": 47.5952, "lon": -122.3316},  # Lumen Field
            "SF": {"lat": 37.4032, "lon": -121.9697},  # Levi's Stadium
            "TB": {"lat": 27.9759, "lon": -82.5033},  # Raymond James Stadium
            "TEN": {"lat": 36.1665, "lon": -86.7713},  # Nissan Stadium
            "WAS": {"lat": 38.9076, "lon": -76.8645},  # FedEx Field
            "WSH": {"lat": 38.9076, "lon": -76.8645},  # FedEx Field (alias for WAS)
            "AFC": {
                "lat": 25.9580,
                "lon": -80.2389,
            },  # Use Miami coordinates for AFC games
            "NFC": {
                "lat": 25.9580,
                "lon": -80.2389,
            },  # Use Miami coordinates for NFC games
        }

    def get_stadium_coordinates(self, team):
        """Get the latitude and longitude for a team's stadium."""
        if team in self.stadium_coords:
            return self.stadium_coords[team]
        return None

    def calculate_planet_positions(self, date, coords):
        """Calculate planetary positions for a given date and location"""
        try:
            # Extract coordinates
            lat = coords.get("lat", 0)
            lon = coords.get("lon", 0)

            # Calculate astrological positions
            positions = {}

            # Use ephem to calculate planetary positions
            observer = ephem.Observer()
            observer.lat = str(lat)
            observer.lon = str(lon)
            observer.date = date

            # Calculate positions for major planets
            planets = {
                "Sun": ephem.Sun(),
                "Moon": ephem.Moon(),
                "Mars": ephem.Mars(),
                "Mercury": ephem.Mercury(),
                "Jupiter": ephem.Jupiter(),
                "Venus": ephem.Venus(),
                "Saturn": ephem.Saturn(),
            }

            # Calculate position for each planet
            for name, planet in planets.items():
                planet.compute(observer)
                positions[name] = math.degrees(planet.ra)  # Convert to degrees

            return positions

        except Exception as e:
            logger.error(f"Error calculating planetary positions: {str(e)}")
            return None

    def calculate_house_positions(self, timestamp, lat, lon):
        """Calculate house positions for a given time and location."""
        try:
            # Convert timestamp to Julian Day
            dt = pd.to_datetime(timestamp)
            utc_time = dt.tz_localize("UTC") if dt.tz is None else dt.tz_convert("UTC")
            jd = swe.julday(
                utc_time.year,
                utc_time.month,
                utc_time.day,
                utc_time.hour + utc_time.minute / 60.0 + utc_time.second / 3600.0,
            )

            # Convert coordinates to float
            lat = float(lat)
            lon = float(lon)

            # Set geographic location
            swe.set_topo(float(lon), float(lat), 0)

            # Calculate house cusps using Placidus system
            result = swe.houses(float(jd), float(lat), float(lon), b"P")

            if result and len(result) >= 2:
                # Extract values from the result tuple
                cusps = result[0]  # House cusps (13 values)
                ascmc = result[1]  # Additional points (10 values)

                # Create house positions dictionary
                house_data = {
                    "cusps": [float(cusp) for cusp in cusps],  # House cusps
                    "ascendant": float(ascmc[0]),  # Ascendant
                    "midheaven": float(ascmc[1]),  # Midheaven (MC)
                    "armc": float(ascmc[2]),  # ARMC
                    "vertex": float(ascmc[3]),  # Vertex
                    "equatorial_ascendant": float(ascmc[4]),  # Equatorial Ascendant
                    "co_ascendant_koch": float(ascmc[5]),  # Co-Ascendant (Koch)
                    "co_ascendant_munkasey": float(ascmc[6]),  # Co-Ascendant (Munkasey)
                    "polar_ascendant": float(ascmc[7]),  # Polar Ascendant
                }

                return house_data

        except Exception as e:
            logger.error(f"Error calculating house positions: {str(e)}")

        # Return default values if calculation fails
        return {
            "cusps": [0.0] * 13,
            "ascendant": 0.0,
            "midheaven": 0.0,
            "armc": 0.0,
            "vertex": 0.0,
            "equatorial_ascendant": 0.0,
            "co_ascendant_koch": 0.0,
            "co_ascendant_munkasey": 0.0,
            "polar_ascendant": 0.0,
        }

    def calculate_planet_strength(self, planet_name, planet, house_data):
        """Calculate the strength of a planet based on its position and aspects."""
        try:
            if (
                not isinstance(planet, dict)
                or "longitude" not in planet
                or "speed_longitude" not in planet
            ):
                logger.error(f"Invalid planet data format for {planet_name}")
                return 0.5  # Return base strength for invalid data

            strength = 0.5  # Base strength

            # Base strength from zodiacal position
            sign = (
                int(float(planet["longitude"]) / 30) + 1
            )  # Calculate zodiac sign (1-12)

            # Adjust strength based on sign element
            if sign in [1, 5, 9]:  # Fire signs (Aries, Leo, Sagittarius)
                strength += 0.2
            elif sign in [2, 6, 10]:  # Earth signs (Taurus, Virgo, Capricorn)
                strength += 0.15
            elif sign in [3, 7, 11]:  # Air signs (Gemini, Libra, Aquarius)
                strength += 0.1
            else:  # Water signs (Cancer, Scorpio, Pisces)
                strength += 0.05

            # Adjust for retrograde motion
            if float(planet["speed_longitude"]) < 0:
                strength *= 0.8

            # Ensure strength is within bounds
            return max(0.0, min(1.0, strength))

        except Exception as e:
            logger.error(
                f"Error calculating planet strength for {planet_name}: {str(e)}"
            )
            return 0.5  # Return base strength on error

    def find_house(self, longitude, cusps):
        """Find which house a planet is in based on its longitude."""
        for i in range(12):
            next_i = (i + 1) % 12
            if cusps[i] <= longitude < cusps[next_i]:
                return i + 1
            # Handle case crossing 0°
            if cusps[i] > cusps[next_i] and (
                longitude >= cusps[i] or longitude < cusps[next_i]
            ):
                return i + 1
        return 1  # Default to 1st house if not found

    def calculate_aspects(self, planet_positions):
        """Calculate aspects between planets."""
        aspects = {}

        # Define major aspects and their orbs
        major_aspects = {
            "conjunction": {"angle": 0, "orb": 10},
            "opposition": {"angle": 180, "orb": 10},
            "trine": {"angle": 120, "orb": 8},
            "square": {"angle": 90, "orb": 8},
            "sextile": {"angle": 60, "orb": 6},
        }

        # Calculate aspects between each pair of planets
        planets = list(planet_positions.keys())
        for i, planet1 in enumerate(planets):
            if planet_positions[planet1] is None:
                continue

            for planet2 in planets[i + 1 :]:
                if planet_positions[planet2] is None:
                    continue

                # Calculate angular difference
                long1 = planet_positions[planet1]["longitude"]
                long2 = planet_positions[planet2]["longitude"]
                diff = abs(long1 - long2)
                if diff > 180:
                    diff = 360 - diff

                # Check for aspects
                for aspect_name, aspect_data in major_aspects.items():
                    orb = abs(diff - aspect_data["angle"])
                    if orb <= aspect_data["orb"]:
                        key = f"{planet1}-{planet2}"
                        aspects[key] = {
                            "type": aspect_name,
                            "orb": orb,
                            "exact": orb <= 1,
                        }

        return aspects

    def calculate_auspicious_score(self, planet_positions, houses):
        """Calculate auspicious score based on planet positions and house placements."""
        score = 0

        if not houses or "cusps" not in houses:
            return 0

        cusps = houses["cusps"]

        for planet, pos in planet_positions.items():
            if pos is None:
                continue

            # Get house position
            longitude = pos["longitude"]
            house = self.find_house(longitude, cusps)

            # Add score for auspicious house placement
            if (
                planet in self.auspicious_houses
                and house in self.auspicious_houses[planet]
            ):
                score += 1

            # Add score for retrograde motion (some planets are considered beneficial when retrograde)
            if pos["speed_longitude"] < 0:
                if planet in ["Jupiter", "Saturn"]:
                    score += 0.5
                elif planet in ["Mars", "Mercury", "Venus"]:
                    score -= 0.5

        return score

    def calculate_all_features(
        self, timestamp, lat, lon, home_team=None, away_team=None
    ):
        """Calculate all astrological features for a given time and location"""
        features = {}

        try:
            # Calculate Julian day
            jd = self.calculate_julian_day(timestamp)
            if not jd:
                raise ValueError("Failed to calculate Julian day")

            # Get planet positions
            planet_positions = self.calculate_planet_positions(
                timestamp, {"lat": lat, "lon": lon}
            )
            if not planet_positions:
                raise ValueError("Failed to calculate planet positions")

            # Get house positions
            house_data = self.calculate_house_positions(timestamp, lat, lon)
            if not house_data:
                raise ValueError("Failed to calculate house positions")

            # Calculate basic planetary strengths
            for planet in [
                "Sun",
                "Moon",
                "Mars",
                "Mercury",
                "Jupiter",
                "Venus",
                "Saturn",
                "Rahu",
                "Ketu",
            ]:
                try:
                    if planet in planet_positions:
                        strength = self.calculate_planet_strength(
                            planet, planet_positions[planet], house_data
                        )
                        features[f"{planet.lower()}_strength"] = float(strength)
                    else:
                        features[f"{planet.lower()}_strength"] = 0.5
                except Exception as e:
                    logger.error(f"Error calculating strength for {planet}: {str(e)}")
                    features[f"{planet.lower()}_strength"] = 0.5

            # Calculate enhanced Vedic features
            features.update(
                {
                    "sarvashtakavarga_score": 0.5,  # Placeholder
                    "shadbala_score": 0.5,  # Placeholder
                    "vimshottari_dasa_score": 0.5,  # Placeholder
                    "divisional_strength": 0.5,  # Placeholder
                    "bhava_chalit_score": 0.5,  # Placeholder
                    "special_lagnas_score": 0.5,  # Placeholder
                    "victory_yogas_score": 0.5,  # Placeholder
                    "nakshatra_tara_score": 0.5,  # Placeholder
                    "sublords_score": 0.5,  # Placeholder
                    "retrograde_impact": 0.5,  # Placeholder
                }
            )

            # Calculate house strengths
            features.update(
                {
                    "ascendant_lord_strength": self.calculate_house_strength(
                        1, house_data
                    ),
                    "tenth_lord_strength": self.calculate_house_strength(
                        10, house_data
                    ),
                    "fifth_lord_strength": self.calculate_house_strength(5, house_data),
                    "ninth_lord_strength": self.calculate_house_strength(9, house_data),
                }
            )

            # Calculate team-specific features
            features.update(
                {
                    "home_team_yoga": 0.5,  # Placeholder
                    "away_team_yoga": 0.5,  # Placeholder
                    "home_nakshatra_score": 0.5,  # Placeholder
                    "away_nakshatra_score": 0.5,  # Placeholder
                }
            )

            # Calculate timing factors
            features.update(
                {
                    "planetary_alignment": 0.5,  # Placeholder
                    "moon_phase_score": 0.5,  # Placeholder
                    "muhurta_score": 0.5,  # Placeholder
                    "hora_score": 0.5,  # Placeholder
                }
            )

            return features

        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")

            # Return default features
            default_features = {
                "sun_strength": 0.5,
                "moon_strength": 0.5,
                "mars_strength": 0.5,
                "mercury_strength": 0.5,
                "jupiter_strength": 0.5,
                "venus_strength": 0.5,
                "saturn_strength": 0.5,
                "rahu_strength": 0.5,
                "ketu_strength": 0.5,
                "sarvashtakavarga_score": 0.5,
                "shadbala_score": 0.5,
                "vimshottari_dasa_score": 0.5,
                "divisional_strength": 0.5,
                "bhava_chalit_score": 0.5,
                "special_lagnas_score": 0.5,
                "victory_yogas_score": 0.5,
                "nakshatra_tara_score": 0.5,
                "sublords_score": 0.5,
                "retrograde_impact": 0.5,
                "ascendant_lord_strength": 0.5,
                "tenth_lord_strength": 0.5,
                "fifth_lord_strength": 0.5,
                "ninth_lord_strength": 0.5,
                "home_team_yoga": 0.5,
                "away_team_yoga": 0.5,
                "home_nakshatra_score": 0.5,
                "away_nakshatra_score": 0.5,
                "planetary_alignment": 0.5,
                "moon_phase_score": 0.5,
                "muhurta_score": 0.5,
                "hora_score": 0.5,
            }
            return default_features

    def calculate_julian_day(self, timestamp):
        """Convert timestamp to Julian Day"""
        try:
            dt = pd.to_datetime(timestamp)
            utc_time = dt.tz_localize("UTC") if dt.tz is None else dt.tz_convert("UTC")
            jd = swe.julday(
                utc_time.year,
                utc_time.month,
                utc_time.day,
                utc_time.hour + utc_time.minute / 60.0 + utc_time.second / 3600.0,
            )
            return jd
        except Exception as e:
            logger.error(f"Error converting to Julian Day: {str(e)}")
            return None

    def calculate_house_strength(self, house_num, house_data):
        """Calculate the strength of a house."""
        if not house_data or "cusps" not in house_data:
            return 0.5

        try:
            # Basic strength based on angular houses
            if house_num in [1, 4, 7, 10]:  # Angular houses
                strength = 1.0
            elif house_num in [2, 5, 8, 11]:  # Succedent houses
                strength = 0.8
            else:  # Cadent houses
                strength = 0.6

            return strength
        except Exception as e:
            logger.error(f"Error calculating house strength: {str(e)}")
            return 0.5

    def calculate_sarvashtakavarga_score(self, planet_positions):
        """Calculate Sarvashtakavarga score."""
        return 0.5  # Placeholder

    def calculate_shadbala_score(self, planet_positions, house_data):
        """Calculate Shadbala score."""
        return 0.5  # Placeholder

    def calculate_vimshottari_dasa_score(self, jd):
        """Calculate Vimshottari Dasa score."""
        return 0.5  # Placeholder

    def calculate_divisional_strength(self, planet_positions):
        """Calculate divisional chart strength."""
        return 0.5  # Placeholder

    def calculate_bhava_chalit_score(self, house_data):
        """Calculate Bhava Chalit score."""
        return 0.5  # Placeholder

    def calculate_special_lagnas_score(self, jd, lat, lon):
        """Calculate special Lagnas score."""
        return 0.5  # Placeholder

    def calculate_victory_yogas_score(self, planet_positions, house_data):
        """Calculate victory Yogas score."""
        return 0.5  # Placeholder

    def calculate_nakshatra_score(self, moon_longitude):
        """Calculate Nakshatra score."""
        return 0.5  # Placeholder

    def calculate_sublords_score(self, planet_positions):
        """Calculate sublords score."""
        return 0.5  # Placeholder

    def calculate_retrograde_impact(self, planet_positions):
        """Calculate retrograde impact score."""
        try:
            score = 0.5
            retrograde_count = 0

            for planet_name, planet in planet_positions.items():
                if planet and planet.get("speed_longitude", 0) < 0:
                    retrograde_count += 1

            if retrograde_count > 0:
                score -= 0.1 * retrograde_count

            return max(0.1, min(1.0, score))
        except Exception as e:
            logger.error(f"Error calculating retrograde impact: {str(e)}")
            return 0.5

    def calculate_team_yoga(self, team, planet_positions):
        """Calculate team Yoga score."""
        return 0.5  # Placeholder

    def calculate_team_nakshatra(self, team, planet_positions):
        """Calculate team Nakshatra score."""
        return 0.5  # Placeholder

    def calculate_planetary_alignment(self, planet_positions):
        """Calculate planetary alignment score."""
        return 0.5  # Placeholder

    def calculate_moon_phase_score(self, jd):
        """Calculate Moon phase score."""
        return 0.5  # Placeholder

    def calculate_muhurta_score(self, jd):
        """Calculate Muhurta score."""
        return 0.5  # Placeholder

    def calculate_hora_score(self, jd):
        """Calculate Hora score."""
        return 0.5  # Placeholder

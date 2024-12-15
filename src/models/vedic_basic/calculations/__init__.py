"""
NFL Vedic Astrology Calculations
"""

from .planetary_calculations import (
    calculate_planet_strength,
    calculate_planetary_alignment_score,
    calculate_sarvashtakavarga,
    check_shadbala,
    check_vimshottari_dasa,
    calculate_divisional_strength,
    calculate_bhava_chalit_aspects,
    calculate_special_lagnas,
    calculate_victory_yogas,
    calculate_nakshatra_tara,
    calculate_sublords,
    calculate_retrograde_impact,
    calculate_moon_phase,
    calculate_muhurta_score,
    calculate_hora_score,
    get_house_lord,
    get_nakshatra_lord,
    PLANETS,
    LATITUDE,
    LONGITUDE,
    HOUSE_LORDS,
    NAKSHATRA_LORDS,
    MUHURTA_LORDS,
    HORA_LORDS,
)
from .dashas import calculate_vimshottari_dasha
from .ashtakavarga import calculate_ashtakavarga
from .sports_calculations import calculate_game_strength

# Import NFLVedicCalculator last to avoid circular imports
from .nfl_vedic_calculator import NFLVedicCalculator

__all__ = [
    "NFLVedicCalculator",
    "calculate_planet_strength",
    "calculate_planetary_alignment_score",
    "calculate_sarvashtakavarga",
    "check_shadbala",
    "check_vimshottari_dasa",
    "calculate_divisional_strength",
    "calculate_bhava_chalit_aspects",
    "calculate_special_lagnas",
    "calculate_victory_yogas",
    "calculate_nakshatra_tara",
    "calculate_sublords",
    "calculate_retrograde_impact",
    "calculate_moon_phase",
    "calculate_muhurta_score",
    "calculate_hora_score",
    "get_house_lord",
    "get_nakshatra_lord",
    "calculate_vimshottari_dasha",
    "calculate_ashtakavarga",
    "calculate_game_strength",
    "PLANETS",
    "LATITUDE",
    "LONGITUDE",
    "HOUSE_LORDS",
    "NAKSHATRA_LORDS",
    "MUHURTA_LORDS",
    "HORA_LORDS",
]

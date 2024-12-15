"""
Vedic Astrology package for NFL predictions.
"""

# Import core functionality first
from .nfl_vedic_calculator import NFLVedicCalculator

# Import calculation functions
from .calculations.shadbala import calculate_shadbala
from .calculations.vimshottari_dasha import calculate_vimshottari_dasha
from .calculations.ashtakavarga import calculate_ashtakavarga
from .calculations.yogas import calculate_yogas

# Import submodules
from . import calculations
from . import predictions
from . import utils
from . import training

__all__ = [
    "NFLVedicCalculator",
    "calculate_shadbala",
    "calculate_vimshottari_dasha",
    "calculate_ashtakavarga",
    "calculate_yogas",
]

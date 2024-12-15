"""Configuration parameters for Vedic astrology calculations."""

from typing import Dict, Any

# Swiss Ephemeris configuration
EPHEMERIS_CONFIG = {
    "path": "ephe",  # Path to ephemeris files
    "ayanamsa": "SIDM_LAHIRI",  # Default ayanamsa system
    "house_system": "P",  # Placidus house system
    "zodiac_type": "sidereal",  # Use sidereal zodiac
}

# Planetary weights for strength calculations
PLANET_WEIGHTS = {
    "SUN": 0.15,
    "MOON": 0.15,
    "MARS": 0.20,  # Higher weight for sports/competition
    "MERCURY": 0.10,
    "JUPITER": 0.15,
    "VENUS": 0.10,
    "SATURN": 0.15,
}

# Aspect configurations
ASPECT_CONFIG = {
    "major_aspects": [0, 60, 90, 120, 180],  # Degrees
    "orbs": {
        0: 10,  # Conjunction
        60: 6,  # Sextile
        90: 8,  # Square
        120: 8,  # Trine
        180: 10,  # Opposition
    },
    "weights": {
        0: 1.0,  # Conjunction
        60: 0.5,  # Sextile
        90: 0.8,  # Square
        120: 0.7,  # Trine
        180: 0.9,  # Opposition
    },
}

# House significance for sports predictions
HOUSE_WEIGHTS = {
    1: 0.15,  # Team's overall strength
    2: 0.05,  # Financial resources
    3: 0.10,  # Short travels, communication
    5: 0.15,  # Sports, performance
    6: 0.10,  # Injuries, challenges
    7: 0.15,  # Opponents
    8: 0.05,  # Hidden factors
    10: 0.15,  # Success, achievement
    11: 0.10,  # Team objectives
}

# Dignity point system
DIGNITY_POINTS = {
    "own_sign": 1.0,
    "exaltation": 1.0,
    "friendly_sign": 0.5,
    "neutral_sign": 0.0,
    "enemy_sign": -0.5,
    "debilitation": -1.0,
}

# Time sensitivity settings
TIME_SENSITIVITY = {
    "critical_degree": 29,  # Last degree of sign
    "gandanta_orb": 2,  # Orb for water-fire sign transitions
    "hora_weight": 0.1,  # Weight for hora consideration
}

# Feature calculation settings
FEATURE_CONFIG = {
    "lookback_period": 30,  # Days to look back for trends
    "momentum_window": 7,  # Days for momentum calculation
    "strength_threshold": 0.6,  # Minimum strength for significance
    "confidence_threshold": 0.7,  # Minimum confidence for predictions
}

# Model training parameters
TRAINING_CONFIG = {
    "min_confidence": 0.6,  # Minimum confidence to include in training
    "max_features": 20,  # Maximum number of features to use
    "feature_selection": "mutual_info",  # Feature selection method
    "cross_validation_folds": 5,
}


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "ephemeris": EPHEMERIS_CONFIG,
        "planets": PLANET_WEIGHTS,
        "aspects": ASPECT_CONFIG,
        "houses": HOUSE_WEIGHTS,
        "dignity": DIGNITY_POINTS,
        "time": TIME_SENSITIVITY,
        "features": FEATURE_CONFIG,
        "training": TRAINING_CONFIG,
    }

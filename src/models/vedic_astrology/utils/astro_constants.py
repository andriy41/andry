"""Constants for astrological calculations."""

# Planet influences and their weights
PLANET_INFLUENCES = {
    "sun": {
        "aspect": "strength",
        "weight": 1.0,
        "houses": [1, 5, 9],
        "strength_factors": ["altitude", "phase"],
    },
    "moon": {
        "aspect": "momentum",
        "weight": 1.2,
        "houses": [2, 4, 7],
        "strength_factors": ["phase", "speed"],
    },
    "mars": {
        "aspect": "strength",
        "weight": 1.1,
        "houses": [1, 3, 10],
        "strength_factors": ["altitude", "retrograde"],
    },
    "jupiter": {
        "aspect": "harmony",
        "weight": 1.3,
        "houses": [5, 9, 11],
        "strength_factors": ["altitude", "speed"],
    },
    "saturn": {
        "aspect": "strength",
        "weight": 0.9,
        "houses": [4, 7, 10],
        "strength_factors": ["altitude", "retrograde"],
    },
    "venus": {
        "aspect": "harmony",
        "weight": 1.1,
        "houses": [2, 7, 11],
        "strength_factors": ["phase", "speed"],
    },
    "mercury": {
        "aspect": "momentum",
        "weight": 1.0,
        "houses": [3, 6, 9],
        "strength_factors": ["speed", "retrograde"],
    },
}

# Astrological aspects and their influences
ASPECTS = {
    0: {"name": "conjunction", "orb": 10, "weight": 1.0},  # 0 degrees
    60: {"name": "sextile", "orb": 6, "weight": 0.5},  # 60 degrees
    90: {"name": "square", "orb": 8, "weight": -0.8},  # 90 degrees
    120: {"name": "trine", "orb": 8, "weight": 0.8},  # 120 degrees
    180: {"name": "opposition", "orb": 10, "weight": -0.6},  # 180 degrees
}

# Zodiac signs
ZODIAC_SIGNS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]

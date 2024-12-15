"""Configuration package for Vedic astrology calculations."""

from .model_config import (
    EPHEMERIS_CONFIG,
    PLANET_WEIGHTS,
    ASPECT_CONFIG,
    HOUSE_WEIGHTS,
    DIGNITY_POINTS,
    TIME_SENSITIVITY,
    FEATURE_CONFIG,
    TRAINING_CONFIG,
    get_config,
)

__all__ = [
    "EPHEMERIS_CONFIG",
    "PLANET_WEIGHTS",
    "ASPECT_CONFIG",
    "HOUSE_WEIGHTS",
    "DIGNITY_POINTS",
    "TIME_SENSITIVITY",
    "FEATURE_CONFIG",
    "TRAINING_CONFIG",
    "get_config",
]

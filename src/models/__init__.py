"""
NFL prediction models package.
"""

from .total_prediction.total_model import TotalPredictionModel
from .total_prediction.neuro_total_model import NeuroTotalModel
from .total_prediction.astro_total_model import AstroTotalModel
from .total_prediction.vedic_total_model import VedicTotalModel
from .total_prediction.ml_total_model import MLTotalModel
from .total_prediction.stats_total_model import StatsTotalModel

from .vedic_astrology import NFLVedicCalculator as VedicAstrologyCalculator
from .vedic_basic import (
    VedicModel,
    NFLVedicCalculator as VedicBasicCalculator,
    train_model as train_vedic_model,
)

__all__ = [
    "TotalPredictionModel",
    "NeuroTotalModel",
    "AstroTotalModel",
    "VedicTotalModel",
    "MLTotalModel",
    "StatsTotalModel",
    "VedicAstrologyCalculator",
    "VedicModel",
    "VedicBasicCalculator",
    "train_vedic_model",
]

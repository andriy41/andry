"""
API endpoints for NFL prediction system
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import joblib
import os

from models.ensemble import NFLEnsemble
from models.vedic_basic.calculations.nfl_vedic_calculator import NFLVedicCalculator
from data_collection.nfl_data_collector import NFLDataCollector

app = FastAPI(title="NFL Prediction API")

# Initialize components
ensemble = NFLEnsemble()
vedic_calc = NFLVedicCalculator()
data_collector = NFLDataCollector()

# Model paths
MODEL_DIR = "saved_models"
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble")


class GameData(BaseModel):
    game_time: str  # ISO format
    home_team: str
    away_team: str
    stadium_latitude: float
    stadium_longitude: float


class TrainingData(BaseModel):
    games: List[GameData]
    labels: List[int]  # 1 for home team win, 0 for away team win


class PredictionRequest(BaseModel):
    game: GameData


class PredictionResponse(BaseModel):
    predicted_winner: str
    win_probability: float
    model_predictions: Dict[str, float]
    vedic_factors: Dict[str, Any]


@app.post("/train")
async def train_model(data: TrainingData):
    """Train the ensemble model with historical data"""
    try:
        # Convert game data to features
        X = []
        for game in data.games:
            game_time = datetime.fromisoformat(game.game_time)
            features = vedic_calc.calculate_game_features(
                game_time,
                game.stadium_latitude,
                game.stadium_longitude,
                game.home_team,
                game.away_team,
            )
            X.append(features)

        X = np.array(X)
        y = np.array(data.labels)

        # Train ensemble
        metrics = ensemble.train(X, y)

        # Save trained models
        os.makedirs(MODEL_DIR, exist_ok=True)
        ensemble.save_models(ENSEMBLE_PATH)

        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_game(request: PredictionRequest):
    """Make prediction for a single game"""
    try:
        # Convert game data to features
        game_time = datetime.fromisoformat(request.game.game_time)
        features = vedic_calc.calculate_game_features(
            game_time,
            request.game.stadium_latitude,
            request.game.stadium_longitude,
            request.game.home_team,
            request.game.away_team,
        )

        # Make prediction
        prediction = ensemble.predict(np.array([features]))

        # Get additional Vedic factors
        vedic_factors = vedic_calc.calculate_game_factors(
            game_time,
            {
                "latitude": request.game.stadium_latitude,
                "longitude": request.game.stadium_longitude,
            },
        )

        return {
            "predicted_winner": prediction["predicted_winner"],
            "win_probability": float(prediction["win_probability"]),
            "model_predictions": {
                k: float(v) for k, v in prediction["model_predictions"].items()
            },
            "vedic_factors": vedic_factors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load-models")
async def load_saved_models():
    """Load previously trained models"""
    try:
        ensemble.load_models(ENSEMBLE_PATH)
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collect-data")
async def collect_training_data(seasons: Optional[List[int]] = None):
    """Collect NFL game data for training"""
    try:
        # Default to last 5 seasons if none specified
        if not seasons:
            current_year = datetime.now().year
            seasons = list(range(current_year - 5, current_year))

        # Collect data
        data = data_collector.collect_seasons(seasons)
        return {"status": "success", "data_collected": len(data), "seasons": seasons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

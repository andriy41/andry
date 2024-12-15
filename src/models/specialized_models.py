"""Specialized prediction models for different aspects of NFL games."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import joblib
import ephem  # For astronomical calculations
from geopy.geocoders import Nominatim  # For geographical coordinates
from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)

# NFL Teams data with their locations
NFL_TEAMS = {
    "ARI": {
        "name": "Arizona Cardinals",
        "city": "Glendale",
        "state": "AZ",
        "lat": 33.5275,
        "lon": -112.2625,
    },
    "ATL": {
        "name": "Atlanta Falcons",
        "city": "Atlanta",
        "state": "GA",
        "lat": 33.7555,
        "lon": -84.4010,
    },
    "BAL": {
        "name": "Baltimore Ravens",
        "city": "Baltimore",
        "state": "MD",
        "lat": 39.2780,
        "lon": -76.6227,
    },
    "BUF": {
        "name": "Buffalo Bills",
        "city": "Orchard Park",
        "state": "NY",
        "lat": 42.7738,
        "lon": -78.7870,
    },
    "CAR": {
        "name": "Carolina Panthers",
        "city": "Charlotte",
        "state": "NC",
        "lat": 35.2258,
        "lon": -80.8528,
    },
    "CHI": {
        "name": "Chicago Bears",
        "city": "Chicago",
        "state": "IL",
        "lat": 41.8623,
        "lon": -87.6167,
    },
    "CIN": {
        "name": "Cincinnati Bengals",
        "city": "Cincinnati",
        "state": "OH",
        "lat": 39.0955,
        "lon": -84.5160,
    },
    "CLE": {
        "name": "Cleveland Browns",
        "city": "Cleveland",
        "state": "OH",
        "lat": 41.5061,
        "lon": -81.6995,
    },
    "DAL": {
        "name": "Dallas Cowboys",
        "city": "Arlington",
        "state": "TX",
        "lat": 32.7473,
        "lon": -97.0945,
    },
    "DEN": {
        "name": "Denver Broncos",
        "city": "Denver",
        "state": "CO",
        "lat": 39.7439,
        "lon": -105.0201,
    },
    "DET": {
        "name": "Detroit Lions",
        "city": "Detroit",
        "state": "MI",
        "lat": 42.3400,
        "lon": -83.0456,
    },
    "GB": {
        "name": "Green Bay Packers",
        "city": "Green Bay",
        "state": "WI",
        "lat": 44.5013,
        "lon": -88.0622,
    },
    "HOU": {
        "name": "Houston Texans",
        "city": "Houston",
        "state": "TX",
        "lat": 29.6847,
        "lon": -95.4107,
    },
    "IND": {
        "name": "Indianapolis Colts",
        "city": "Indianapolis",
        "state": "IN",
        "lat": 39.7601,
        "lon": -86.1639,
    },
    "JAX": {
        "name": "Jacksonville Jaguars",
        "city": "Jacksonville",
        "state": "FL",
        "lat": 30.3239,
        "lon": -81.6373,
    },
    "KC": {
        "name": "Kansas City Chiefs",
        "city": "Kansas City",
        "state": "MO",
        "lat": 39.0489,
        "lon": -94.4839,
    },
    "LAC": {
        "name": "Los Angeles Chargers",
        "city": "Inglewood",
        "state": "CA",
        "lat": 33.9534,
        "lon": -118.3387,
    },
    "LAR": {
        "name": "Los Angeles Rams",
        "city": "Inglewood",
        "state": "CA",
        "lat": 33.9534,
        "lon": -118.3387,
    },
    "LV": {
        "name": "Las Vegas Raiders",
        "city": "Las Vegas",
        "state": "NV",
        "lat": 36.0909,
        "lon": -115.1833,
    },
    "MIA": {
        "name": "Miami Dolphins",
        "city": "Miami Gardens",
        "state": "FL",
        "lat": 25.9580,
        "lon": -80.2389,
    },
    "MIN": {
        "name": "Minnesota Vikings",
        "city": "Minneapolis",
        "state": "MN",
        "lat": 44.9735,
        "lon": -93.2575,
    },
    "NE": {
        "name": "New England Patriots",
        "city": "Foxborough",
        "state": "MA",
        "lat": 42.0909,
        "lon": -71.2643,
    },
    "NO": {
        "name": "New Orleans Saints",
        "city": "New Orleans",
        "state": "LA",
        "lat": 29.9511,
        "lon": -90.0814,
    },
    "NYG": {
        "name": "New York Giants",
        "city": "East Rutherford",
        "state": "NJ",
        "lat": 40.8135,
        "lon": -74.0745,
    },
    "NYJ": {
        "name": "New York Jets",
        "city": "East Rutherford",
        "state": "NJ",
        "lat": 40.8135,
        "lon": -74.0745,
    },
    "PHI": {
        "name": "Philadelphia Eagles",
        "city": "Philadelphia",
        "state": "PA",
        "lat": 39.9008,
        "lon": -75.1674,
    },
    "PIT": {
        "name": "Pittsburgh Steelers",
        "city": "Pittsburgh",
        "state": "PA",
        "lat": 40.4468,
        "lon": -80.0158,
    },
    "SEA": {
        "name": "Seattle Seahawks",
        "city": "Seattle",
        "state": "WA",
        "lat": 47.5952,
        "lon": -122.3316,
    },
    "SF": {
        "name": "San Francisco 49ers",
        "city": "Santa Clara",
        "state": "CA",
        "lat": 37.4032,
        "lon": -121.9697,
    },
    "TB": {
        "name": "Tampa Bay Buccaneers",
        "city": "Tampa",
        "state": "FL",
        "lat": 27.9759,
        "lon": -82.5033,
    },
    "TEN": {
        "name": "Tennessee Titans",
        "city": "Nashville",
        "state": "TN",
        "lat": 36.1665,
        "lon": -86.7713,
    },
    "WAS": {
        "name": "Washington Commanders",
        "city": "Landover",
        "state": "MD",
        "lat": 38.9076,
        "lon": -76.8645,
    },
}


@dataclass
class TeamLocation:
    """Team location data."""

    city: str
    latitude: float
    longitude: float
    timezone: str


class FormBasedModel:
    """Model focusing on team form and momentum."""

    def __init__(self):
        """Initialize form-based model."""
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.01
        )
        self.scaler = StandardScaler()

    def prepare_features(self, game_data: pd.DataFrame) -> np.ndarray:
        """Prepare form-based features."""
        features = []

        # Recent form (last 5 games)
        features.extend(
            [
                game_data["home_last_5_wins"].iloc[0] / 5,
                game_data["away_last_5_wins"].iloc[0] / 5,
                game_data["home_last_5_points_for"].iloc[0]
                / game_data["home_last_5_points_against"].iloc[0],
                game_data["away_last_5_points_for"].iloc[0]
                / game_data["away_last_5_points_against"].iloc[0],
            ]
        )

        # Momentum indicators
        features.extend(
            [
                game_data["home_win_streak"].iloc[0] / 5,  # Normalized to 5 games
                game_data["away_win_streak"].iloc[0] / 5,
                game_data["home_points_trend"].iloc[0],  # Point differential trend
                game_data["away_points_trend"].iloc[0],
            ]
        )

        # Rest and schedule
        features.extend(
            [
                game_data["home_rest_days"].iloc[0] / 7,  # Normalized to a week
                game_data["away_rest_days"].iloc[0] / 7,
                game_data["home_schedule_strength"].iloc[0],
                game_data["away_schedule_strength"].iloc[0],
            ]
        )

        return np.array(features).reshape(1, -1)


class InjuryAwareModel:
    """Model that focuses on injury impacts."""

    def __init__(self):
        """Initialize injury-aware model."""
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_split=5
        )
        self.scaler = StandardScaler()
        self.position_weights = {
            "QB": 0.3,
            "WR": 0.15,
            "RB": 0.15,
            "TE": 0.1,
            "OL": 0.1,
            "DL": 0.05,
            "LB": 0.05,
            "DB": 0.1,
        }

    def calculate_injury_impact(self, injuries: List[Dict[str, str]]) -> float:
        """Calculate total injury impact based on position weights."""
        total_impact = 0.0
        for injury in injuries:
            position = injury["position"]
            status = injury["status"]

            # Convert status to impact factor
            if status == "Out":
                factor = 1.0
            elif status == "Doubtful":
                factor = 0.75
            elif status == "Questionable":
                factor = 0.5
            else:
                factor = 0.0

            total_impact += self.position_weights.get(position, 0.05) * factor

        return total_impact

    def prepare_features(self, game_data: pd.DataFrame) -> np.ndarray:
        """Prepare injury-aware features."""
        features = []

        # Injury impacts
        features.extend(
            [
                self.calculate_injury_impact(game_data["home_injuries"].iloc[0]),
                self.calculate_injury_impact(game_data["away_injuries"].iloc[0]),
            ]
        )

        # Team depth metrics
        features.extend(
            [
                game_data["home_depth_rating"].iloc[0],
                game_data["away_depth_rating"].iloc[0],
            ]
        )

        # Historical injury performance
        features.extend(
            [
                game_data["home_injury_win_rate"].iloc[0],
                game_data["away_injury_win_rate"].iloc[0],
            ]
        )

        return np.array(features).reshape(1, -1)


class MatchupModel:
    """Model focusing on head-to-head and matchup-specific factors."""

    def __init__(self):
        """Initialize matchup model."""
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.01
        )
        self.scaler = StandardScaler()

    def prepare_features(self, game_data: pd.DataFrame) -> np.ndarray:
        """Prepare matchup-specific features."""
        features = []

        # Head-to-head history
        features.extend(
            [
                game_data["h2h_home_wins"].iloc[0]
                / max(game_data["h2h_games"].iloc[0], 1),
                game_data["h2h_home_points"].iloc[0]
                / max(game_data["h2h_games"].iloc[0], 1),
                game_data["recent_h2h_home_wins"].iloc[0] / 3,  # Last 3 meetings
            ]
        )

        # Matchup-specific metrics
        features.extend(
            [
                game_data["home_pass_off_vs_pass_def"].iloc[0],
                game_data["home_rush_off_vs_rush_def"].iloc[0],
                game_data["away_pass_off_vs_pass_def"].iloc[0],
                game_data["away_rush_off_vs_rush_def"].iloc[0],
            ]
        )

        # Style matchup
        features.extend(
            [
                game_data["home_style_advantage"].iloc[0],
                game_data["away_style_advantage"].iloc[0],
            ]
        )

        return np.array(features).reshape(1, -1)


class AstrologyModel:
    """Model incorporating astrological and astronomical factors."""

    def __init__(self):
        """Initialize astrology model."""
        self.geolocator = Nominatim(user_agent="nfl_predictions")
        self.tf = TimezoneFinder()
        self.team_locations: Dict[str, TeamLocation] = self._initialize_team_locations()

    def _initialize_team_locations(self) -> Dict[str, TeamLocation]:
        """Initialize NFL team locations."""
        team_locations = {}
        for team in NFL_TEAMS:  # You'll need to define NFL_TEAMS
            try:
                location = self.geolocator.geocode(
                    f"{NFL_TEAMS[team]['city']}, {NFL_TEAMS[team]['state']}"
                )
                timezone = self.tf.timezone_at(
                    lat=location.latitude, lng=location.longitude
                )
                team_locations[team] = TeamLocation(
                    city=NFL_TEAMS[team]["city"],
                    latitude=location.latitude,
                    longitude=location.longitude,
                    timezone=timezone,
                )
            except Exception as e:
                logger.error(
                    f"Error getting location for {NFL_TEAMS[team]['city']}: {str(e)}"
                )

        return team_locations

    def calculate_planetary_positions(
        self, game_time: datetime, location: TeamLocation
    ) -> Dict[str, float]:
        """Calculate planetary positions for a given time and location."""
        observer = ephem.Observer()
        observer.lat = str(location.latitude)
        observer.lon = str(location.longitude)
        observer.date = game_time

        positions = {}
        for planet in ["Sun", "Moon", "Mars", "Jupiter", "Saturn"]:
            planet_obj = getattr(ephem, planet)()
            planet_obj.compute(observer)
            positions[planet] = {
                "alt": float(planet_obj.alt) * 180 / np.pi,  # Convert to degrees
                "az": float(planet_obj.az) * 180 / np.pi,
            }

        return positions

    def prepare_features(self, game_data: pd.DataFrame) -> np.ndarray:
        """Prepare astrological features."""
        features = []

        game_time = pd.to_datetime(game_data["game_time"].iloc[0])
        home_team = game_data["home_team"].iloc[0]
        away_team = game_data["away_team"].iloc[0]

        # Get planetary positions for both teams
        if home_team in self.team_locations and away_team in self.team_locations:
            home_planets = self.calculate_planetary_positions(
                game_time, self.team_locations[home_team]
            )
            away_planets = self.calculate_planetary_positions(
                game_time, self.team_locations[away_team]
            )

            # Add planetary features
            for planet in ["Sun", "Moon", "Mars", "Jupiter", "Saturn"]:
                features.extend(
                    [
                        home_planets[planet]["alt"],
                        home_planets[planet]["az"],
                        away_planets[planet]["alt"],
                        away_planets[planet]["az"],
                    ]
                )

            # Calculate aspects between planets
            for p1 in ["Sun", "Moon", "Mars"]:
                for p2 in ["Jupiter", "Saturn"]:
                    home_aspect = abs(home_planets[p1]["az"] - home_planets[p2]["az"])
                    away_aspect = abs(away_planets[p1]["az"] - away_planets[p2]["az"])
                    features.extend([home_aspect, away_aspect])

        return np.array(features).reshape(1, -1)


class EnsemblePredictor:
    """Ensemble predictor combining all specialized models."""

    def __init__(self):
        """Initialize ensemble predictor."""
        self.form_model = FormBasedModel()
        self.injury_model = InjuryAwareModel()
        self.matchup_model = MatchupModel()
        self.astrology_model = AstrologyModel()

        self.model_weights = {
            "form": 0.3,
            "injury": 0.25,
            "matchup": 0.35,
            "astrology": 0.1,
        }

    def predict(self, game_data: pd.DataFrame) -> Dict[str, any]:
        """Make ensemble prediction."""
        predictions = {}
        confidences = {}

        # Get predictions from each model
        try:
            form_features = self.form_model.prepare_features(game_data)
            predictions["form"] = self.form_model.model.predict(form_features)[0]
            confidences["form"] = self.form_model.model.predict_proba(form_features)[0][
                1
            ]
        except:
            predictions["form"] = 0.5
            confidences["form"] = 0.0

        try:
            injury_features = self.injury_model.prepare_features(game_data)
            predictions["injury"] = self.injury_model.model.predict(injury_features)[0]
            confidences["injury"] = np.mean(
                [
                    tree.predict(injury_features)[0]
                    for tree in self.injury_model.model.estimators_
                ]
            )
        except:
            predictions["injury"] = 0.5
            confidences["injury"] = 0.0

        try:
            matchup_features = self.matchup_model.prepare_features(game_data)
            predictions["matchup"] = self.matchup_model.model.predict(matchup_features)[
                0
            ]
            confidences["matchup"] = self.matchup_model.model.predict_proba(
                matchup_features
            )[0][1]
        except:
            predictions["matchup"] = 0.5
            confidences["matchup"] = 0.0

        try:
            astro_features = self.astrology_model.prepare_features(game_data)
            # Astrology model provides influence factors rather than direct predictions
            astro_influence = np.mean(astro_features)  # Simplified for example
            predictions["astrology"] = 0.5 + (astro_influence * 0.1)  # Small influence
            confidences["astrology"] = abs(astro_influence)
        except:
            predictions["astrology"] = 0.5
            confidences["astrology"] = 0.0

        # Calculate weighted prediction
        weighted_pred = sum(
            pred * self.model_weights[model] for model, pred in predictions.items()
        )

        # Calculate overall confidence
        agreement_factor = 1 - np.std(list(predictions.values()))
        confidence_factor = np.mean(list(confidences.values()))
        overall_confidence = agreement_factor * confidence_factor

        # Check high confidence criteria
        same_prediction = all(pred > 0.5 for pred in predictions.values()) or all(
            pred < 0.5 for pred in predictions.values()
        )
        high_confidence = all(conf > 0.8 for conf in confidences.values())
        very_high_confidence = any(conf > 0.85 for conf in confidences.values())

        is_high_confidence_pick = (
            same_prediction and high_confidence and very_high_confidence
        )

        return {
            "prediction": weighted_pred,
            "confidence": overall_confidence,
            "is_high_confidence": is_high_confidence_pick,
            "model_predictions": predictions,
            "model_confidences": confidences,
            "agreement_factor": agreement_factor,
        }

    def save(self, path: str):
        """Save ensemble predictor state."""
        state = {
            "form_model": self.form_model,
            "injury_model": self.injury_model,
            "matchup_model": self.matchup_model,
            "model_weights": self.model_weights,
        }
        joblib.dump(state, path)

    def load(self, path: str):
        """Load ensemble predictor state."""
        state = joblib.load(path)
        self.form_model = state["form_model"]
        self.injury_model = state["injury_model"]
        self.matchup_model = state["matchup_model"]
        self.model_weights = state["model_weights"]

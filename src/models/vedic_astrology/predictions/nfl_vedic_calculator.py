"""NFL Vedic Astrology Calculator for game predictions."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pytz

from ...vedic_astrology.utils.astro_calculations import (
    calculate_planet_positions,
    calculate_moon_phase,
    parse_game_datetime,
)
from ...vedic_astrology.utils.planet_calculations import (
    calculate_planet_strength,
    count_beneficial_aspects,
    count_malefic_aspects,
)
from ...vedic_astrology.utils.team_calculations import (
    resolve_team_name,
    calculate_team_strength,
)
from ...vedic_astrology.utils.astro_constants import (
    PLANET_INFLUENCES,
    ASPECTS,
    ZODIAC_SIGNS,
)
from ...vedic_astrology.data.team_data import TEAM_DATA, TEAM_ALIASES
from ...vedic_astrology.utils.varga_calculations import (
    calculate_varga_strength,
    calculate_varga_aspects,
    calculate_team_varga_strength,
)
from ...vedic_astrology.utils.sublord_calculations import SublordCalculator
from ...vedic_astrology.utils.sublord_array_analysis import SublordArrayAnalyzer

logger = logging.getLogger(__name__)


class NFLVedicCalculator:
    """Calculator for NFL game predictions using Vedic astrology."""

    def __init__(self):
        """Initialize the Vedic calculator."""
        self.logger = logging.getLogger(__name__)
        self.team_data = TEAM_DATA
        self.team_aliases = TEAM_ALIASES
        self.planet_influences = PLANET_INFLUENCES
        self.aspects = ASPECTS
        self.zodiac_signs = ZODIAC_SIGNS
        self.house_signs = {}
        self.house_degrees = {}

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when calculations fail."""
        return {
            "moon_phase": 0.5,
            "home_planet_strength": 0.5,
            "away_planet_strength": 0.5,
            "beneficial_aspects": 0,
            "malefic_aspects": 0,
            "home_strength": 0.5,
            "away_strength": 0.5,
            "raja_yoga": 0,
            "dhana_yoga": 0,
            "vipreet_yoga": 0,
            "kesari_yoga": 0,
        }

    def calculate_game_features(
        self,
        *,
        game_date: str,
        game_time: str,
        timezone: str,
        home_team: str,
        away_team: str,
        week_number: int = 1,
        is_playoff: bool = False,
    ) -> Dict[str, float]:
        """Calculate all enhanced Vedic features for a game."""
        try:
            # Parse game datetime
            dt = parse_game_datetime(game_date, game_time, timezone)
            if dt is None:
                self.logger.error("Invalid game datetime")
                return self._get_default_features()

            # Calculate planetary positions
            positions = calculate_planet_positions(dt)
            if not positions:
                self.logger.error("Could not calculate planetary positions")
                return self._get_default_features()

            # Resolve team names
            home_team = resolve_team_name(home_team, self.team_data, self.team_aliases)
            away_team = resolve_team_name(away_team, self.team_data, self.team_aliases)

            # Get team data
            home_data = self.team_data.get(home_team, {})
            away_data = self.team_data.get(away_team, {})

            if not home_data or not away_data:
                self.logger.error(f"Missing team data for {home_team} or {away_team}")
                return self._get_default_features()

            # Calculate basic features
            features = {
                "moon_phase": calculate_moon_phase(positions),
                "beneficial_aspects": count_beneficial_aspects(positions),
                "malefic_aspects": count_malefic_aspects(positions),
            }

            # Calculate team strengths using Varga charts
            home_planets = ["Sun", "Mars", "Jupiter"]  # Traditional victory planets
            away_planets = ["Saturn", "Mercury", "Venus"]  # Challenge planets

            # Calculate Varga-based planet strengths
            varga_strengths = calculate_varga_strength(positions)
            features["home_planet_strength"] = sum(
                varga_strengths.get(p, 0) for p in home_planets
            ) / len(home_planets)
            features["away_planet_strength"] = sum(
                varga_strengths.get(p, 0) for p in away_planets
            ) / len(away_planets)

            # Calculate team strengths
            features["home_strength"] = calculate_team_varga_strength(
                positions, home_planets
            )
            features["away_strength"] = calculate_team_varga_strength(
                positions, away_planets
            )

            # Calculate yogas (combinations)
            features["raja_yoga"] = self._calculate_raja_yoga(
                positions, is_playoff, week_number
            )
            features["dhana_yoga"] = self._calculate_dhana_yoga(positions)
            features["vipreet_yoga"] = self._calculate_vipreet_yoga(positions)
            features["kesari_yoga"] = self._calculate_kesari_yoga(positions)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating game features: {e}")
            return self._get_default_features()

    def _calculate_raja_yoga(
        self, positions: Dict[str, Dict[str, float]], is_playoff: bool, week_number: int
    ) -> float:
        """Calculate Raja Yoga strength (special combination indicating victory)."""
        try:
            base_strength = 0.0

            # Check Jupiter-Sun conjunction or trine
            jupiter_pos = positions.get("jupiter", {}).get("longitude", 0)
            sun_pos = positions.get("sun", {}).get("longitude", 0)

            angle = abs(jupiter_pos - sun_pos)
            angle = min(angle, 360 - angle)

            if angle < 10 or abs(angle - 120) < 10:  # Conjunction or trine
                base_strength += 0.5

            # Enhance for playoff games
            if is_playoff:
                base_strength *= 1.5

            # Adjust for week number (stronger in later weeks)
            week_factor = min(week_number / 18, 1.0)
            base_strength *= 1 + week_factor

            return min(base_strength, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating Raja Yoga: {str(e)}")
            return 0.0

    def _calculate_dhana_yoga(self, positions: Dict[str, Dict[str, float]]) -> float:
        """Calculate Dhana Yoga strength (wealth/success combination)."""
        try:
            jupiter_pos = positions.get("jupiter", {})
            venus_pos = positions.get("venus", {})

            if not jupiter_pos or not venus_pos:
                return 0.0

            # Check for mutual aspect or conjunction
            angle = abs(jupiter_pos.get("longitude", 0) - venus_pos.get("longitude", 0))
            angle = min(angle, 360 - angle)

            if angle < 10:  # Conjunction
                return 1.0
            elif abs(angle - 120) < 10:  # Trine
                return 0.7
            elif abs(angle - 60) < 6:  # Sextile
                return 0.5

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating Dhana Yoga: {str(e)}")
            return 0.0

    def _calculate_vipreet_yoga(self, positions: Dict[str, Dict[str, float]]) -> float:
        """Calculate Vipreet Yoga strength (unexpected reversal combination)."""
        try:
            saturn_pos = positions.get("saturn", {})
            mars_pos = positions.get("mars", {})

            if not saturn_pos or not mars_pos:
                return 0.0

            # Check for mutual aspect
            angle = abs(saturn_pos.get("longitude", 0) - mars_pos.get("longitude", 0))
            angle = min(angle, 360 - angle)

            if angle < 10:  # Conjunction
                return 1.0
            elif abs(angle - 90) < 8:  # Square
                return 0.8
            elif abs(angle - 180) < 10:  # Opposition
                return 0.6

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating Vipreet Yoga: {str(e)}")
            return 0.0

    def _calculate_kesari_yoga(self, positions: Dict[str, Dict[str, float]]) -> float:
        """Calculate Kesari Yoga strength (victory combination)."""
        try:
            jupiter_pos = positions.get("jupiter", {})
            moon_pos = positions.get("moon", {})

            if not jupiter_pos or not moon_pos:
                return 0.0

            # Check for mutual aspect
            angle = abs(jupiter_pos.get("longitude", 0) - moon_pos.get("longitude", 0))
            angle = min(angle, 360 - angle)

            if angle < 10:  # Conjunction
                return 1.0
            elif abs(angle - 120) < 10:  # Trine
                return 0.8
            elif abs(angle - 60) < 6:  # Sextile
                return 0.6

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating Kesari Yoga: {str(e)}")
            return 0.0

    def predict_influence(
        self,
        game_date: str,
        game_time: str,
        timezone: str,
        home_team: str,
        away_team: str,
        week_number: int = 1,
        is_playoff: bool = False,
    ) -> Dict[str, float]:
        """Predict the astrological influence on a game's outcome."""
        try:
            features = self.calculate_game_features(
                game_date=game_date,
                game_time=game_time,
                timezone=timezone,
                home_team=home_team,
                away_team=away_team,
                week_number=week_number,
                is_playoff=is_playoff,
            )

            # Calculate home team advantage based on astrological factors
            home_advantage = (
                features["home_planet_strength"] * 0.3
                + features["home_strength"] * 0.2
                + (features["beneficial_aspects"] - features["malefic_aspects"]) * 0.1
                + features["raja_yoga"] * 0.2
            )

            # Calculate away team strength
            away_strength = (
                features["away_planet_strength"] * 0.3 + features["away_strength"] * 0.2
            )

            # Calculate special combinations influence
            special_combinations = (
                features["dhana_yoga"] * 0.4
                + features["vipreet_yoga"] * 0.2
                + features["kesari_yoga"] * 0.2
            )

            # Calculate moon phase influence (peaks at new and full moon)
            moon_influence = abs(features["moon_phase"] - 0.5) * 2

            # Combine all factors
            total_influence = (
                home_advantage * 0.4
                + (1 - away_strength) * 0.3
                + special_combinations * 0.2  # Invert away strength
                + moon_influence * 0.1
            )

            # Normalize to [-1, 1] range where:
            # -1: Strong away team advantage
            #  0: Neutral
            # +1: Strong home team advantage
            total_influence = max(min(total_influence, 1), -1)

            return {
                "total_influence": total_influence,
                "home_advantage": home_advantage,
                "away_strength": away_strength,
                "special_combinations": special_combinations,
                "moon_influence": moon_influence,
                "features": features,
            }

        except Exception as e:
            self.logger.error(f"Error predicting influence: {str(e)}")
            return {
                "total_influence": 0,
                "home_advantage": 0,
                "away_strength": 0,
                "special_combinations": 0,
                "moon_influence": 0,
                "features": self._get_default_features(),
            }

    def predict_winner(
        self,
        game_date: str,
        game_time: str,
        timezone: str,
        home_team: str,
        away_team: str,
        week_number: int = 1,
        is_playoff: bool = False,
    ) -> Dict[str, Any]:
        """Predict the winner of a game based on astrological influences."""
        try:
            influence = self.predict_influence(
                game_date,
                game_time,
                timezone,
                home_team,
                away_team,
                week_number,
                is_playoff,
            )
            total_influence = influence["total_influence"]

            # Convert influence to win probability
            # total_influence is in [-1, 1], convert to [0, 1]
            home_win_prob = (total_influence + 1) / 2

            return {
                "predicted_winner": home_team if home_win_prob > 0.5 else away_team,
                "home_win_probability": home_win_prob,
                "away_win_probability": 1 - home_win_prob,
                "confidence": abs(home_win_prob - 0.5)
                * 2,  # Scale confidence to [0, 1]
                "influence_factors": influence,
            }

        except Exception as e:
            self.logger.error(f"Error predicting winner: {str(e)}")
            return {
                "predicted_winner": None,
                "home_win_probability": 0.5,
                "away_win_probability": 0.5,
                "confidence": 0,
                "influence_factors": None,
            }

    def predict_score(
        self,
        game_date: str,
        game_time: str,
        timezone: str,
        home_team: str,
        away_team: str,
        week_number: int = 1,
        is_playoff: bool = False,
    ) -> Dict[str, Any]:
        """Predict the score range for a game based on astrological influences."""
        try:
            influence = self.predict_influence(
                game_date,
                game_time,
                timezone,
                home_team,
                away_team,
                week_number,
                is_playoff,
            )
            features = influence["features"]

            # Base scores (NFL average points per game)
            base_home_score = 23
            base_away_score = 20

            # Adjust scores based on astrological factors
            home_adjustment = (
                (features["home_planet_strength"] - 0.5) * 10
                + (features["home_strength"] - 0.5) * 8
                + (features["beneficial_aspects"] - features["malefic_aspects"]) * 2
            )

            away_adjustment = (features["away_planet_strength"] - 0.5) * 10 + (
                features["away_strength"] - 0.5
            ) * 8

            # Special combinations adjustments
            special_adjustment = (
                features["raja_yoga"] * 4
                + features["dhana_yoga"] * 3
                + features["vipreet_yoga"] * 2
                + features["kesari_yoga"] * 2
            )

            # Moon phase affects overall scoring
            moon_factor = abs(features["moon_phase"] - 0.5) * 2
            total_adjustment = moon_factor * 6

            # Calculate predicted scores
            predicted_home = (
                base_home_score
                + home_adjustment
                + special_adjustment
                + total_adjustment
            )
            predicted_away = base_away_score + away_adjustment + total_adjustment

            # Ensure scores are positive and reasonable
            predicted_home = max(min(round(predicted_home), 50), 0)
            predicted_away = max(min(round(predicted_away), 50), 0)

            return {
                "predicted_home_score": predicted_home,
                "predicted_away_score": predicted_away,
                "predicted_total": predicted_home + predicted_away,
                "score_confidence": influence["total_influence"],
                "influence_factors": influence,
            }

        except Exception as e:
            self.logger.error(f"Error predicting score: {str(e)}")
            return {
                "predicted_home_score": 0,
                "predicted_away_score": 0,
                "predicted_total": 0,
                "score_confidence": 0,
                "influence_factors": None,
            }

    def calculate_sky_effects(self, houses, planet_strengths):
        from ...vedic_astrology.utils.varga_calculations import (
            calculate_varga_strength,
            calculate_varga_aspects,
        )
        from ...vedic_astrology.utils.sublord_calculations import SublordCalculator
        from ...vedic_astrology.utils.sublord_array_analysis import SublordArrayAnalyzer

        sky_effects = {house: 0 for house in range(1, 13)}
        favorite_houses = {1, 3, 6, 10, 11}
        underdog_houses = {7, 9, 12, 4, 5}

        favorite_sky_score = 0
        underdog_sky_score = 0

        benefics = ["Jupiter", "Venus", "Mercury", "Moon", "Yogi Point"]
        malefics = ["Sun", "Mars", "Saturn"]
        nodes = ["Rahu", "Ketu", "Gulika", "Upaketu"]

        # Calculate Varga strengths and aspects
        planet_positions = {
            planet: houses.get(planet, 0) for planet in planet_strengths.keys()
        }
        varga_strengths = calculate_varga_strength(planet_positions)
        varga_aspects = calculate_varga_aspects(planet_positions)

        # Initialize calculators
        sublord_calc = SublordCalculator()
        array_analyzer = SublordArrayAnalyzer()

        # Collect sublord information for all houses
        house_sublords = {}
        for house in range(1, 13):
            house_sign = self.get_house_sign(house)
            house_degree = self.get_house_degree(house)
            if house_sign and house_degree is not None:
                sublord_info = sublord_calc.get_sublord(house_sign, house_degree)
                if "error" not in sublord_info:
                    house_sublords[house] = sublord_info

        # Analyze sublord array patterns
        array_analysis = array_analyzer.analyze_sublord_array(house_sublords)
        array_strength_factor = array_analysis["strength_factor"]

        # Log array analysis results
        logging.debug(f"Sublord Array Analysis: {array_analysis['array_details']}")
        if array_analysis["complete_array"]:
            logging.info(
                f"Found complete sublord array favoring {array_analysis['complete_array']}"
            )
        if array_analysis["key_houses_array"]:
            logging.info(
                f"Found key houses array favoring {array_analysis['key_houses_array']}"
            )

        def calculate_effect_strength(planets, planet_strengths, varga_strengths):
            base_strength = sum(
                3 * planet_strengths.get(planet, 1) for planet in planets
            )
            varga_boost = sum(2 * varga_strengths.get(planet, 0) for planet in planets)
            return base_strength + varga_boost

        def update_scores(house, effect, favorite_houses, underdog_houses):
            if house in favorite_houses:
                return effect, 0
            elif house in underdog_houses:
                return 0, effect
            return 0, 0

        for house in range(1, 13):
            prev_house = 12 if house == 1 else house - 1
            next_house = 1 if house == 12 else house + 1

            # Calculate sublord strength for the house
            house_sign = self.get_house_sign(house)
            house_degree = self.get_house_degree(house)
            if house_sign and house_degree is not None:
                sublord_info = sublord_calc.get_sublord(house_sign, house_degree)
                sublord_strength = sublord_calc.calculate_house_sublord_strength(
                    house, sublord_info, planet_positions, varga_strengths
                )
            else:
                sublord_strength = 0.5  # Default if we can't calculate

            # Apply array strength factor to sublord strength
            sublord_strength *= array_strength_factor

            benefics_around = [
                planet
                for planet in benefics
                if houses.get(planet) in [prev_house, next_house]
            ]
            malefics_around = [
                planet
                for planet in malefics
                if houses.get(planet) in [prev_house, next_house]
            ]
            nodes_around = [
                planet
                for planet in nodes
                if houses.get(planet) in [prev_house, next_house]
            ]

            # Enhanced effect calculation using Varga strengths and sublord strength
            if len(benefics_around) >= 2:
                effect = calculate_effect_strength(
                    benefics_around, planet_strengths, varga_strengths
                )
                # Add aspect influence
                aspect_boost = sum(
                    varga_aspects.get(planet, 0) for planet in benefics_around
                )
                effect *= 1 + aspect_boost * 0.2  # 20% boost from aspects
                # Modify effect by sublord strength
                effect *= 0.5 + sublord_strength  # Sublord influence
                sky_effects[house] = effect
                fav_score, und_score = update_scores(
                    house, effect, favorite_houses, underdog_houses
                )
                favorite_sky_score += fav_score
                underdog_sky_score += und_score

            elif len(malefics_around) >= 2:
                effect = -calculate_effect_strength(
                    malefics_around, planet_strengths, varga_strengths
                )
                # Add aspect influence
                aspect_boost = sum(
                    varga_aspects.get(planet, 0) for planet in malefics_around
                )
                effect *= 1 + aspect_boost * 0.2  # 20% boost from aspects
                # Modify effect by sublord strength
                effect *= 0.5 + sublord_strength  # Sublord influence
                sky_effects[house] = effect
                fav_score, und_score = update_scores(
                    house, effect, favorite_houses, underdog_houses
                )
                favorite_sky_score += fav_score
                underdog_sky_score += und_score

            # Enhanced node influence using Varga calculations and sublord strength
            if nodes_around:
                node_varga_strength = sum(
                    varga_strengths.get(node, 0) for node in nodes_around
                )
                node_effect = (
                    node_varga_strength * sublord_strength
                )  # Combine with sublord strength
                if sky_effects[house] > 0:
                    sky_effects[house] *= (
                        1 - node_effect * 0.3
                    )  # Reduced positive effect
                elif sky_effects[house] < 0:
                    sky_effects[house] *= (
                        1 + node_effect * 0.3
                    )  # Enhanced negative effect

            # Special consideration for angular houses (1, 7)
            if house in [1, 7]:
                malefics_around.extend(
                    [
                        planet
                        for planet in nodes
                        if houses.get(planet) in [prev_house, next_house]
                    ]
                )
                if len(malefics_around) >= 2:
                    effect = -calculate_effect_strength(
                        malefics_around, planet_strengths, varga_strengths
                    )
                    # Add Varga-based influence
                    varga_influence = sum(
                        varga_strengths.get(planet, 0) for planet in malefics_around
                    )
                    # Include sublord strength in the calculation
                    effect *= 1 + varga_influence * 0.25 * sublord_strength
                    sky_effects[house] = effect
                    if house == 1:
                        favorite_sky_score += effect
                    else:
                        underdog_sky_score += effect

        # Apply array points to scores
        if array_analysis["complete_array"] == "favorite":
            favorite_sky_score += array_analysis["points"]
        elif array_analysis["complete_array"] == "underdog":
            underdog_sky_score += array_analysis["points"]
        elif array_analysis["key_houses_array"] == "favorite":
            favorite_sky_score += array_analysis["points"]
        elif array_analysis["key_houses_array"] == "underdog":
            underdog_sky_score += array_analysis["points"]

        overall_sky_effect = sum(sky_effects.values())
        logging.debug(
            f"Calculated SKY effects with Varga, Sublords, and Array Analysis: {sky_effects}"
        )
        logging.debug(
            f"Favorite SKY score: {favorite_sky_score}, Underdog SKY score: {underdog_sky_score}"
        )
        logging.debug(f"Overall SKY effect: {overall_sky_effect}")

        return sky_effects, favorite_sky_score, underdog_sky_score, overall_sky_effect

    def check_victory_houses(
        self, houses, positions, cusp_values, sky_effects, spot_plays
    ):
        """
        Calculate strength based on planetary states and configurations in victory houses.
        Integrates Varga, sublord, and planetary state calculations.
        """
        from ...vedic_astrology.utils.varga_calculations import calculate_varga_strength
        from ...vedic_astrology.utils.sublord_calculations import SublordCalculator
        from ...vedic_astrology.utils.sublord_array_analysis import SublordArrayAnalyzer
        from ...vedic_astrology.utils.planetary_states import calculate_planet_state
        from ...vedic_astrology.utils.stolen_cusps import (
            determine_stolen_cusps,
            calculate_stolen_cusp_impact,
        )

        favorite_houses = {1, 3, 6, 10, 11}
        underdog_houses = {7, 9, 12, 4, 5}

        # Initialize calculators
        sublord_calc = SublordCalculator()
        array_analyzer = SublordArrayAnalyzer()

        # Get house cusps and ascendant
        cusps = self.get_house_cusps()
        ascmc = {"ascendant": self.get_ascendant()}

        # Analyze stolen cusps
        stolen_cusps = determine_stolen_cusps(cusps, ascmc)
        stolen_cusp_impact = calculate_stolen_cusp_impact(stolen_cusps, houses)

        # Calculate Varga strengths
        planet_positions = {
            planet: positions[planet]["longitude"] for planet in positions
        }
        varga_strengths = calculate_varga_strength(planet_positions)

        # Calculate planetary states
        planet_states = {}
        for planet, data in positions.items():
            state = calculate_planet_state(
                planet=planet,
                longitude=data["longitude"],
                house=houses[planet],
                nakshatra=data.get("nakshatra", ""),
                is_retrograde=data.get("is_retrograde", False),
                all_positions=positions,
            )
            planet_states[planet] = state

        # Collect sublord information
        house_sublords = {}
        for house in range(1, 13):
            house_sign = self.get_house_sign(house)
            house_degree = self.get_house_degree(house)
            if house_sign and house_degree is not None:
                sublord_info = sublord_calc.get_sublord(house_sign, house_degree)
                if "error" not in sublord_info:
                    house_sublords[house] = sublord_info

        # Analyze sublord array patterns
        array_analysis = array_analyzer.analyze_sublord_array(house_sublords)
        array_strength_factor = array_analysis["strength_factor"]

        # Calculate victory house scores
        favorite_score = 0
        underdog_score = 0

        for planet, house in houses.items():
            state = planet_states[planet]

            # Calculate base strength from planetary state
            base_strength = state["strength"]

            # Add Varga strength
            if planet in varga_strengths:
                base_strength = (base_strength + varga_strengths[planet]) / 2

            # Add sublord influence if available
            if house in house_sublords:
                sublord_strength = sublord_calc.calculate_house_sublord_strength(
                    house, house_sublords[house], planet_positions, varga_strengths
                )
                base_strength *= 0.5 + sublord_strength

            # Apply array strength factor
            base_strength *= array_strength_factor

            # Score for victory houses
            if house in favorite_houses:
                favorite_score += base_strength
                if house in [1, 10]:  # Angular houses
                    favorite_score += base_strength * 0.5
            elif house in underdog_houses:
                underdog_score += base_strength
                if house in [4, 7]:  # Angular houses
                    underdog_score += base_strength * 0.5

        # Add cuspal influences
        favorite_cuspal_score = sum(
            influence
            for planet, influence in cusp_values.items()
            if houses[planet] in favorite_houses
        )
        underdog_cuspal_score = sum(
            influence
            for planet, influence in cusp_values.items()
            if houses[planet] in underdog_houses
        )

        # Add sky effects
        (
            sky_effects,
            favorite_sky_score,
            underdog_sky_score,
            _,
        ) = self.calculate_sky_effects(houses, planet_states)

        # Add spot play points
        spot_plays, favorite_spot_points, underdog_spot_points = self.check_spot_plays(
            houses, positions
        )

        # Add array points
        if array_analysis["complete_array"] == "favorite":
            favorite_score += array_analysis["points"]
        elif array_analysis["complete_array"] == "underdog":
            underdog_score += array_analysis["points"]
        elif array_analysis["key_houses_array"] == "favorite":
            favorite_score += array_analysis["points"]
        elif array_analysis["key_houses_array"] == "underdog":
            underdog_score += array_analysis["points"]

        # Combine all scores with weights
        favorite_total = (
            favorite_score * 1.0
            + favorite_cuspal_score * 0.3  # Base victory house score
            + favorite_sky_score * 0.2  # Reduced weight for cuspal
            + favorite_spot_points * 0.5  # Reduced weight for sky effects
            + (  # Medium weight for spot plays
                stolen_cusp_impact if stolen_cusp_impact > 0 else 0
            )
            * 0.4  # Positive stolen cusp impact
        )

        underdog_total = (
            underdog_score * 1.0
            + underdog_cuspal_score * 0.3  # Base victory house score
            + underdog_sky_score * 0.2  # Reduced weight for cuspal
            + underdog_spot_points * 0.5  # Reduced weight for sky effects
            + (  # Medium weight for spot plays
                abs(stolen_cusp_impact) if stolen_cusp_impact < 0 else 0
            )
            * 0.4  # Negative stolen cusp impact
        )

        # Determine result
        if favorite_total > underdog_total:
            result = "Favorite has a stronger indication of victory."
        elif underdog_total > favorite_total:
            result = "Underdog has a stronger indication of victory."
        else:
            result = "Both teams have equal strength indications."

        # Log detailed analysis
        logging.debug(f"Victory result: {result}")
        logging.debug(
            f"Favorite score: {favorite_total}, Underdog score: {underdog_total}"
        )
        logging.debug(f"Planet states: {planet_states}")
        logging.debug(f"Sublord array analysis: {array_analysis['array_details']}")
        logging.debug(f"Spot plays: {spot_plays}")
        logging.debug(f"Stolen cusps: {stolen_cusps}")
        logging.debug(f"Stolen cusp impact: {stolen_cusp_impact}")

        return result, favorite_total, underdog_total, planet_states

    def get_house_sign(self, house_number: int) -> str:
        """Get the zodiac sign of a house."""
        try:
            return self.house_signs[house_number]
        except (KeyError, AttributeError):
            return None

    def get_house_degree(self, house_number: int) -> float:
        """Get the degree within the sign for a house cusp."""
        try:
            return self.house_degrees[house_number]
        except (KeyError, AttributeError):
            return None

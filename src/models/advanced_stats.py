"""Advanced statistical features for NFL predictions including ELO, power rankings, and Bayesian priors."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Container for team statistics."""

    elo: float = 1500
    wins: int = 0
    losses: int = 0
    points_for: float = 0
    points_against: float = 0
    quality_wins: int = 0  # Wins against teams with winning records
    strength_of_schedule: float = 0
    power_ranking: float = 0
    variance: float = 100  # For Bayesian updating
    schedule_difficulty: List[float] = None  # Future schedule difficulty

    def __post_init__(self):
        self.schedule_difficulty = []


class NFLAdvancedStats:
    """Calculate and track advanced NFL statistics."""

    def __init__(self):
        """Initialize the advanced stats calculator."""
        # ELO parameters
        self.K = 32  # Base K-factor
        self.HOME_ADVANTAGE = 50  # ELO points for home advantage
        self.K_MULTIPLIER = {
            "playoff": 1.5,  # Playoff games matter more
            "division": 1.2,  # Division games matter more
            "conference": 1.1,  # Conference games matter slightly more
        }

        # Bayesian parameters
        self.PRIOR_VARIANCE = 100
        self.PERFORMANCE_VARIANCE = 50

        # Monte Carlo parameters
        self.monte_carlo_sims = 10000  # Increased from 1000
        self.scaler = MinMaxScaler()

        # Team stats
        self.team_stats: Dict[str, TeamStats] = {}
        self.prior_means: Dict[str, float] = {}
        self.prior_stds: Dict[str, float] = {}

        # Historical data
        self.historical_games: List[Dict] = []

    def _calculate_game_importance(self, game: pd.Series) -> float:
        """Calculate the importance multiplier for a game."""
        importance = 1.0

        # Convert week to int if it's a string number
        week = game["week"]
        if isinstance(week, str) and week.isdigit():
            week = int(week)

        # Late season games are more important
        if isinstance(week, int) and week > 12:
            importance *= 1.2

        # Playoff games are most important
        if game.get("is_playoff", False):
            importance *= 1.5

        # Increase importance for division and conference games
        if game.get("is_division_game", False):
            importance *= self.K_MULTIPLIER["division"]
        elif game.get("is_conference_game", False):
            importance *= self.K_MULTIPLIER["conference"]

        return importance

    def _update_elo(self, game: pd.Series):
        """Update ELO ratings with sophisticated adjustments."""
        home_team = game["home_team"]
        away_team = game["away_team"]

        # Get current ELO ratings
        home_elo = self.team_stats[home_team].elo
        away_elo = self.team_stats[away_team].elo

        # Calculate expected win probability
        expected_home = 1 / (
            1 + 10 ** ((away_elo - (home_elo + self.HOME_ADVANTAGE)) / 400)
        )

        # Calculate actual outcome
        home_score = game["home_score"]
        away_score = game["away_score"]
        actual_home = 1 if home_score > away_score else 0

        # Calculate margin of victory multiplier
        mov = abs(home_score - away_score)
        expected_mov = (
            abs(expected_home - 0.5) * 20
        )  # Expected margin based on ELO difference
        mov_multiplier = np.log(max(mov, 1) + 1) * (
            2.2 / (1 if mov < expected_mov else (mov - expected_mov) * 0.4 + 1)
        )

        # Calculate game importance
        importance_multiplier = self._calculate_game_importance(game)

        # Update ratings
        K = self.K * importance_multiplier * mov_multiplier
        elo_change = K * (actual_home - expected_home)

        self.team_stats[home_team].elo += elo_change
        self.team_stats[away_team].elo -= elo_change

        # Update variances based on prediction accuracy
        prediction_error = abs(actual_home - expected_home)
        variance_update = prediction_error * self.PRIOR_VARIANCE
        self.team_stats[home_team].variance = (
            self.team_stats[home_team].variance * 0.95 + variance_update * 0.05
        )
        self.team_stats[away_team].variance = (
            self.team_stats[away_team].variance * 0.95 + variance_update * 0.05
        )

    def _update_power_rankings(self):
        """Update power rankings with sophisticated metrics."""
        # Calculate basic stats for normalization
        wins = np.array([stats.wins for stats in self.team_stats.values()])
        points_for = np.array([stats.points_for for stats in self.team_stats.values()])
        points_against = np.array(
            [stats.points_against for stats in self.team_stats.values()]
        )
        elos = np.array([stats.elo for stats in self.team_stats.values()])

        # Normalize components
        norm_wins = self.scaler.fit_transform(wins.reshape(-1, 1)).flatten()
        norm_point_diff = self.scaler.fit_transform(
            (points_for - points_against).reshape(-1, 1)
        ).flatten()
        norm_elo = self.scaler.fit_transform(elos.reshape(-1, 1)).flatten()

        for idx, (team, stats) in enumerate(self.team_stats.items()):
            # Calculate components
            win_component = norm_wins[idx]
            point_diff_component = norm_point_diff[idx]
            elo_component = norm_elo[idx]
            quality_component = stats.quality_wins / max(stats.wins, 1)
            sos_component = stats.strength_of_schedule

            # Calculate future strength of schedule
            future_sos = (
                np.mean(stats.schedule_difficulty) if stats.schedule_difficulty else 0.5
            )

            # Weighted combination
            stats.power_ranking = (
                0.30 * win_component
                + 0.20 * point_diff_component  # Win percentage
                + 0.20 * elo_component  # Point differential
                + 0.15 * quality_component  # ELO rating
                + 0.10 * sos_component  # Quality wins
                + 0.05  # Strength of schedule
                * future_sos  # Future schedule difficulty
            )

    def _update_bayesian_priors(self):
        """Update Bayesian priors with sophisticated modeling."""
        for team, stats in self.team_stats.items():
            # Calculate team strength indicators
            games_played = stats.wins + stats.losses
            if games_played == 0:
                continue

            # Point differential per game
            point_diff = (stats.points_for - stats.points_against) / games_played

            # Win percentage with Bayesian smoothing
            win_pct = (stats.wins + 1) / (games_played + 2)  # Add pseudo-counts

            # ELO rating normalized
            elo_factor = (stats.elo - 1500) / 400

            # Quality wins ratio
            quality_ratio = stats.quality_wins / max(stats.wins, 1)

            # Combine factors with uncertainty
            prior_mean = (
                0.35 * win_pct
                + 0.25 * elo_factor
                + 0.20 * (point_diff / 10)
                + 0.20 * quality_ratio  # Normalize point differential
            )

            # Update prior mean with exponential smoothing
            old_mean = self.prior_means.get(team, 0.5)
            self.prior_means[team] = 0.8 * prior_mean + 0.2 * old_mean

            # Calculate prior standard deviation
            confidence_factors = [
                1 / np.sqrt(games_played + 1),  # More games = less uncertainty
                stats.variance
                / self.PRIOR_VARIANCE,  # Higher variance = more uncertainty
                abs(point_diff) / 20,  # Extreme point differentials = more uncertainty
                abs(elo_factor),  # Extreme ELO ratings = more uncertainty
            ]
            self.prior_stds[team] = (
                np.mean(confidence_factors) * 0.2
            )  # Scale to reasonable range

    def simulate_game(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Run Monte Carlo simulation with sophisticated modeling."""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            raise ValueError("Both teams must have stats available")

        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]

        # Get Bayesian priors
        home_mean = self.prior_means.get(home_team, 0.5)
        away_mean = self.prior_means.get(away_team, 0.5)
        home_std = self.prior_stds.get(home_team, 0.2)
        away_std = self.prior_stds.get(away_team, 0.2)

        # Simulation results
        home_wins = 0
        total_points = []
        spreads = []
        home_scores = []
        away_scores = []

        for _ in range(self.monte_carlo_sims):
            # Sample team strengths from prior distributions
            home_strength = np.random.normal(home_mean, home_std)
            away_strength = np.random.normal(away_mean, away_std)

            # Add home field advantage (normally distributed)
            home_advantage = np.random.normal(0.1, 0.02)  # Mean=10%, SD=2%
            home_strength += home_advantage

            # Calculate base point expectations
            home_exp = home_stats.points_for / max(
                home_stats.wins + home_stats.losses, 1
            )
            away_exp = away_stats.points_for / max(
                away_stats.wins + away_stats.losses, 1
            )

            # Simulate scores with team strength adjustments
            home_score = np.random.poisson(home_exp * (1 + home_strength))
            away_score = np.random.poisson(away_exp * (1 + away_strength))

            # Record results
            home_wins += 1 if home_score > away_score else 0
            total_points.append(home_score + away_score)
            spreads.append(home_score - away_score)
            home_scores.append(home_score)
            away_scores.append(away_score)

        # Calculate sophisticated metrics
        win_prob = home_wins / self.monte_carlo_sims
        win_prob_std = np.std([1 if x > 0 else 0 for x in spreads])

        return {
            "win_prob": win_prob,
            "win_prob_std": win_prob_std,
            "expected_total": np.mean(total_points),
            "total_std": np.std(total_points),
            "expected_spread": np.mean(spreads),
            "spread_std": np.std(spreads),
            "home_score_dist": {
                "mean": np.mean(home_scores),
                "std": np.std(home_scores),
                "percentiles": np.percentile(home_scores, [25, 50, 75]),
            },
            "away_score_dist": {
                "mean": np.mean(away_scores),
                "std": np.std(away_scores),
                "percentiles": np.percentile(away_scores, [25, 50, 75]),
            },
        }

    def update_team_stats(self, game_data: pd.DataFrame):
        """Update team statistics based on game results."""
        # Sort games chronologically
        game_data = game_data.sort_values("date")

        # Reset or initialize team stats
        self.team_stats.clear()
        self.historical_games.clear()

        # Process each game
        for _, game in game_data.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]

            # Initialize teams if needed
            for team in [home_team, away_team]:
                if team not in self.team_stats:
                    self.team_stats[team] = TeamStats()

            # Update basic stats
            home_score = game["home_score"]
            away_score = game["away_score"]
            home_win = home_score > away_score

            if home_win:
                self.team_stats[home_team].wins += 1
                self.team_stats[away_team].losses += 1
                # Check if it's a quality win
                if self.team_stats[away_team].wins > self.team_stats[away_team].losses:
                    self.team_stats[home_team].quality_wins += 1
            else:
                self.team_stats[away_team].wins += 1
                self.team_stats[home_team].losses += 1
                if self.team_stats[home_team].wins > self.team_stats[home_team].losses:
                    self.team_stats[away_team].quality_wins += 1

            self.team_stats[home_team].points_for += home_score
            self.team_stats[home_team].points_against += away_score
            self.team_stats[away_team].points_for += away_score
            self.team_stats[away_team].points_against += home_score

            # Update ELO ratings
            self._update_elo(game)

            # Store game for historical reference
            self.historical_games.append(game.to_dict())

        # Update strength of schedule
        self._update_strength_of_schedule()

        # Update power rankings
        self._update_power_rankings()

        # Update Bayesian priors
        self._update_bayesian_priors()

    def _update_strength_of_schedule(self):
        """Calculate strength of schedule for each team."""
        for team, stats in self.team_stats.items():
            # Get all opponents
            opponents = []
            for game in self.historical_games:
                if game["home_team"] == team:
                    opponents.append(game["away_team"])
                elif game["away_team"] == team:
                    opponents.append(game["home_team"])

            # Calculate opponent winning percentage
            opp_records = []
            for opp in opponents:
                opp_stats = self.team_stats[opp]
                opp_wins = opp_stats.wins
                opp_losses = opp_stats.losses
                # Remove head-to-head games
                h2h_games = sum(
                    1
                    for g in self.historical_games
                    if (g["home_team"] == team and g["away_team"] == opp)
                    or (g["away_team"] == team and g["home_team"] == opp)
                )
                if opp_wins + opp_losses > h2h_games:
                    opp_records.append((opp_wins, opp_losses - h2h_games))

            if opp_records:
                total_wins = sum(w for w, _ in opp_records)
                total_games = sum(w + l for w, l in opp_records)
                stats.strength_of_schedule = (
                    total_wins / total_games if total_games > 0 else 0.5
                )
            else:
                stats.strength_of_schedule = 0.5

    def get_team_features(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Get advanced statistical features for a matchup."""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            raise ValueError("Both teams must have stats available")

        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]

        # Run simulation
        sim_results = self.simulate_game(home_team, away_team)

        return {
            "home_elo": home_stats.elo,
            "away_elo": away_stats.elo,
            "home_power_ranking": home_stats.power_ranking,
            "away_power_ranking": away_stats.power_ranking,
            "home_sos": home_stats.strength_of_schedule,
            "away_sos": away_stats.strength_of_schedule,
            "home_prior_mean": self.prior_means.get(home_team, 0.5),
            "away_prior_mean": self.prior_means.get(away_team, 0.5),
            "home_prior_std": self.prior_stds.get(home_team, 0.2),
            "away_prior_std": self.prior_stds.get(away_team, 0.2),
            "sim_win_prob": sim_results["win_prob"],
            "sim_spread": sim_results["expected_spread"],
            "sim_total": sim_results["expected_total"],
            "sim_confidence": 1 - sim_results["win_prob_std"],
            "home_quality_wins_ratio": home_stats.quality_wins
            / max(home_stats.wins, 1),
            "away_quality_wins_ratio": away_stats.quality_wins
            / max(away_stats.wins, 1),
            "home_point_diff_per_game": (
                home_stats.points_for - home_stats.points_against
            )
            / max(home_stats.wins + home_stats.losses, 1),
            "away_point_diff_per_game": (
                away_stats.points_for - away_stats.points_against
            )
            / max(away_stats.wins + away_stats.losses, 1),
        }

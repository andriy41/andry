import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_weighted_form(
    games_df: pd.DataFrame,
    team: str,
    date: datetime,
    window: int = 8,
    decay_factor: float = 0.85,
) -> float:
    """
    Calculate weighted form with exponential decay for recency.

    Args:
        games_df: DataFrame containing historical games
        team: Team to calculate form for
        date: Current date
        window: Number of games to look back
        decay_factor: Weight decay factor for older games
    """
    team_games = (
        games_df[
            ((games_df["home_team"] == team) | (games_df["away_team"] == team))
            & (games_df["game_datetime"] < date)
        ]
        .sort_values("game_datetime", ascending=False)
        .head(window)
    )

    if len(team_games) == 0:
        return 0.5

    weights = np.array([decay_factor**i for i in range(len(team_games))])
    weights = weights / weights.sum()

    wins = []
    point_margins = []

    for _, game in team_games.iterrows():
        is_home = game["home_team"] == team
        points_scored = game["home_score"] if is_home else game["away_score"]
        points_allowed = game["away_score"] if is_home else game["home_score"]

        win = points_scored > points_allowed
        margin = points_scored - points_allowed

        wins.append(win)
        point_margins.append(margin)

    weighted_win_rate = np.average(wins, weights=weights)
    weighted_margin = np.average(point_margins, weights=weights)

    # Combine win rate and point margin into a single form metric
    return 0.7 * weighted_win_rate + 0.3 * (weighted_margin / 20)


def calculate_head_to_head(
    games_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    date: datetime,
    window: int = 5,
    decay_factor: float = 0.8,
) -> float:
    """
    Calculate head-to-head performance with recency weighting.

    Args:
        games_df: DataFrame containing historical games
        home_team: Home team
        away_team: Away team
        date: Current date
        window: Number of h2h games to consider
        decay_factor: Weight decay factor for older games
    """
    h2h_games = (
        games_df[
            (
                (
                    (games_df["home_team"] == home_team)
                    & (games_df["away_team"] == away_team)
                )
                | (
                    (games_df["home_team"] == away_team)
                    & (games_df["away_team"] == home_team)
                )
            )
            & (games_df["game_datetime"] < date)
        ]
        .sort_values("game_datetime", ascending=False)
        .head(window)
    )

    if len(h2h_games) == 0:
        return 0.5

    weights = np.array([decay_factor**i for i in range(len(h2h_games))])
    weights = weights / weights.sum()

    home_wins = []
    for _, game in h2h_games.iterrows():
        if game["home_team"] == home_team:
            home_wins.append(game["home_score"] > game["away_score"])
        else:
            home_wins.append(game["away_score"] > game["home_score"])

    return np.average(home_wins, weights=weights)


def calculate_win_streak(games_df: pd.DataFrame, team: str, date: datetime) -> int:
    """
    Calculate current win/loss streak (positive for wins, negative for losses).

    Args:
        games_df: DataFrame containing historical games
        team: Team to calculate streak for
        date: Current date
    """
    team_games = games_df[
        ((games_df["home_team"] == team) | (games_df["away_team"] == team))
        & (games_df["game_datetime"] < date)
    ].sort_values("game_datetime", ascending=False)

    if len(team_games) == 0:
        return 0

    streak = 0
    last_result = None

    for _, game in team_games.iterrows():
        is_home = game["home_team"] == team
        points_scored = game["home_score"] if is_home else game["away_score"]
        points_allowed = game["away_score"] if is_home else game["home_score"]

        won = points_scored > points_allowed

        if last_result is None:
            last_result = won
            streak = 1 if won else -1
        elif won == last_result:
            streak = streak + 1 if won else streak - 1
        else:
            break

    return streak


def calculate_rest_advantage(games_df: pd.DataFrame, team: str, date: datetime) -> int:
    """
    Calculate days of rest before game.

    Args:
        games_df: DataFrame containing historical games
        team: Team to calculate rest for
        date: Game date
    """
    last_game = (
        games_df[
            ((games_df["home_team"] == team) | (games_df["away_team"] == team))
            & (games_df["game_datetime"] < date)
        ]
        .sort_values("game_datetime", ascending=False)
        .head(1)
    )

    if len(last_game) == 0:
        return 7  # Default to a week if no previous game

    last_game_date = last_game.iloc[0]["game_datetime"]
    return (date - last_game_date).days


def calculate_home_away_performance(
    games_df: pd.DataFrame, team: str, is_home: bool, date: datetime, window: int = 10
) -> float:
    """
    Calculate team's performance specifically at home or away.

    Args:
        games_df: DataFrame containing historical games
        team: Team to analyze
        is_home: Whether to calculate home or away performance
        date: Current date
        window: Number of games to consider
    """
    if is_home:
        team_games = (
            games_df[
                (games_df["home_team"] == team) & (games_df["game_datetime"] < date)
            ]
            .sort_values("game_datetime", ascending=False)
            .head(window)
        )
    else:
        team_games = (
            games_df[
                (games_df["away_team"] == team) & (games_df["game_datetime"] < date)
            ]
            .sort_values("game_datetime", ascending=False)
            .head(window)
        )

    if len(team_games) == 0:
        return 0.5

    if is_home:
        wins = (team_games["home_score"] > team_games["away_score"]).mean()
        point_diff = (team_games["home_score"] - team_games["away_score"]).mean()
    else:
        wins = (team_games["away_score"] > team_games["home_score"]).mean()
        point_diff = (team_games["away_score"] - team_games["home_score"]).mean()

    return 0.7 * wins + 0.3 * (point_diff / 20)


def prepare_enhanced_features(
    games_df: pd.DataFrame, date: datetime, home_team: str, away_team: str
) -> Dict[str, float]:
    """
    Prepare enhanced feature set for prediction.

    Args:
        games_df: DataFrame containing historical games
        date: Game date
        home_team: Home team
        away_team: Away team
    """
    features = {}

    # Calculate form
    features["home_form"] = calculate_weighted_form(games_df, home_team, date)
    features["away_form"] = calculate_weighted_form(games_df, away_team, date)

    # Head-to-head history
    features["h2h_advantage"] = calculate_head_to_head(
        games_df, home_team, away_team, date
    )

    # Win/loss streaks
    features["home_streak"] = calculate_win_streak(games_df, home_team, date)
    features["away_streak"] = calculate_win_streak(games_df, away_team, date)

    # Rest advantage
    features["home_rest"] = calculate_rest_advantage(games_df, home_team, date)
    features["away_rest"] = calculate_rest_advantage(games_df, away_team, date)
    features["rest_advantage"] = features["home_rest"] - features["away_rest"]

    # Home/Away specific performance
    features["home_home_performance"] = calculate_home_away_performance(
        games_df, home_team, True, date
    )
    features["away_away_performance"] = calculate_home_away_performance(
        games_df, away_team, False, date
    )

    return features


def enhance_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance training data with additional features
    """
    data = data.copy()

    # Ensure datetime column is in datetime format
    data["game_datetime"] = pd.to_datetime(data["game_datetime"])

    # Sort by game datetime for proper feature calculation
    data = data.sort_values("game_datetime").copy()

    # Calculate basic stats
    data["win_pct"] = calculate_rolling_win_percentage(data)
    data["loss_pct"] = 1 - data["win_pct"]
    data["points_per_game"] = calculate_rolling_average(data, "points_scored", window=5)
    data["points_allowed_per_game"] = calculate_rolling_average(
        data, "points_allowed", window=5
    )

    # Calculate home/away stats
    data["home_win_pct"] = calculate_home_win_percentage(data)
    data["away_win_pct"] = calculate_away_win_percentage(data)

    # Calculate division and conference stats
    data["div_win_pct"] = calculate_division_win_percentage(data)
    data["conf_win_pct"] = calculate_conference_win_percentage(data)

    # Calculate quarter-by-quarter scoring
    for quarter in range(1, 5):
        data[f"q{quarter}_points_per_game"] = calculate_rolling_average(
            data, f"q{quarter}_points", window=5
        )

    # Calculate margin stats
    data["avg_margin"] = calculate_rolling_average(data, "point_margin", window=5)
    data["avg_margin_home"] = calculate_home_margin_average(data)
    data["avg_margin_away"] = calculate_away_margin_average(data)

    # Calculate consistency metrics
    data["scoring_consistency"] = calculate_scoring_consistency(data)
    data["defense_consistency"] = calculate_defense_consistency(data)

    # Calculate season-phase performance
    data["early_season_win_pct"] = calculate_early_season_win_pct(data)
    data["mid_season_win_pct"] = calculate_mid_season_win_pct(data)
    data["late_season_win_pct"] = calculate_late_season_win_pct(data)

    # Calculate advanced metrics
    data["points_per_drive"] = calculate_points_per_drive(data)
    data["yards_per_play"] = calculate_yards_per_play(data)
    data["strength_of_schedule"] = calculate_strength_of_schedule(data)

    # Calculate momentum and efficiency
    data["momentum"] = calculate_momentum_score(data)
    data["off_efficiency"] = calculate_offensive_efficiency(data)
    data["third_down_conv"] = calculate_third_down_conversion(data)

    # Calculate feature differences for home vs away
    feature_columns = [
        "win_pct",
        "loss_pct",
        "points_per_game",
        "points_allowed_per_game",
        "home_win_pct",
        "away_win_pct",
        "div_win_pct",
        "conf_win_pct",
        "q1_points_per_game",
        "q2_points_per_game",
        "q3_points_per_game",
        "q4_points_per_game",
        "avg_margin",
        "avg_margin_home",
        "avg_margin_away",
        "scoring_consistency",
        "defense_consistency",
        "early_season_win_pct",
        "mid_season_win_pct",
        "late_season_win_pct",
        "points_per_drive",
        "yards_per_play",
        "momentum",
        "off_efficiency",
        "third_down_conv",
    ]

    for feature in feature_columns:
        # Create home and away versions of each feature
        data[f"home_{feature}"] = data[feature]
        data[f"away_{feature}"] = data.groupby("away_team")[feature].shift(1)

        # Calculate the difference (home - away)
        data[f"{feature}_diff"] = data[f"home_{feature}"] - data[f"away_{feature}"]

    # Add game context features
    data["is_division_game"] = data["home_team"].str[:3] == data["away_team"].str[:3]
    data["is_conference_game"] = data["home_team"].str[3] == data["away_team"].str[3]
    data[
        "is_home_game"
    ] = True  # Since we're always calculating from home team perspective

    return data.fillna(0)


def calculate_team_stats(team_games: pd.DataFrame, team: str) -> Dict[str, float]:
    """Calculate comprehensive statistics for a team."""
    stats = {}

    try:
        # Basic win/loss stats
        wins = sum(
            (team_games["home_team"] == team)
            & (team_games["home_score"] > team_games["away_score"])
            | (team_games["away_team"] == team)
            & (team_games["away_score"] > team_games["home_score"])
        )
        losses = sum(
            (team_games["home_team"] == team)
            & (team_games["home_score"] < team_games["away_score"])
            | (team_games["away_team"] == team)
            & (team_games["away_score"] < team_games["home_score"])
        )
        ties = sum(
            (team_games["home_team"] == team)
            & (team_games["home_score"] == team_games["away_score"])
            | (team_games["away_team"] == team)
            & (team_games["away_score"] == team_games["home_score"])
        )
        games = len(team_games)

        # Calculate all the basic stats
        stats.update(calculate_basic_stats(team_games, team, wins, losses, ties, games))

        # Calculate scoring stats
        stats.update(calculate_scoring_stats(team_games, team, games))

        # Calculate situational stats
        stats.update(calculate_situational_stats(team_games, team))

        # Calculate consistency metrics
        stats.update(calculate_consistency_metrics(team_games, team))

        return stats

    except Exception as e:
        logger.error(f"Error calculating team stats: {str(e)}")
        return create_default_team_stats()


def calculate_rankings(team_stats: Dict[str, Dict[str, float]]) -> None:
    """Calculate rankings for all numeric stats."""
    try:
        if not team_stats:
            return

        # Get all numeric stats from first team
        first_team = next(iter(team_stats.values()))
        numeric_stats = [
            stat
            for stat, value in first_team.items()
            if isinstance(value, (int, float))
        ]

        for stat in numeric_stats:
            reverse = not ("allowed" in stat or "loss" in stat or "std" in stat)
            sorted_teams = sorted(
                team_stats.items(), key=lambda x: x[1][stat], reverse=reverse
            )
            for rank, (team, _) in enumerate(sorted_teams, 1):
                team_stats[team][f"{stat}_rank"] = rank

    except Exception as e:
        logger.error(f"Error calculating rankings: {str(e)}")


def create_default_features() -> Dict[str, float]:
    """Create default features when calculation fails."""
    return {
        "win_pct": 0.5,
        "loss_pct": 0.5,
        "tie_pct": 0.0,
        "points_per_game": 20.0,
        "points_allowed_per_game": 20.0,
        # Add all other default features...
    }


def create_default_team_stats() -> Dict[str, float]:
    """Create default team stats when calculation fails."""
    return {
        "win_pct": 0.5,
        "loss_pct": 0.5,
        "tie_pct": 0.0,
        "points_per_game": 20.0,
        "points_allowed_per_game": 20.0,
        # Add all other default stats...
    }


def calculate_basic_stats(
    team_games: pd.DataFrame, team: str, wins: int, losses: int, ties: int, games: int
) -> Dict[str, float]:
    """Calculate basic win/loss stats."""
    stats = {}

    stats["win_pct"] = wins / games if games > 0 else 0
    stats["loss_pct"] = losses / games if games > 0 else 0
    stats["tie_pct"] = ties / games if games > 0 else 0

    return stats


def calculate_scoring_stats(
    team_games: pd.DataFrame, team: str, games: int
) -> Dict[str, float]:
    """Calculate scoring stats."""
    stats = {}

    points_scored = sum(
        team_games[team_games["home_team"] == team]["home_score"].fillna(0)
        + team_games[team_games["away_team"] == team]["away_score"].fillna(0)
    )
    points_allowed = sum(
        team_games[team_games["home_team"] == team]["away_score"].fillna(0)
        + team_games[team_games["away_team"] == team]["home_score"].fillna(0)
    )

    stats["points_per_game"] = points_scored / games if games > 0 else 0
    stats["points_allowed_per_game"] = points_allowed / games if games > 0 else 0
    stats["point_diff"] = (points_scored - points_allowed) / games if games > 0 else 0

    return stats


def calculate_situational_stats(
    team_games: pd.DataFrame, team: str
) -> Dict[str, float]:
    """Calculate situational stats."""
    stats = {}

    home_games = team_games[team_games["home_team"] == team]
    away_games = team_games[team_games["away_team"] == team]

    home_wins = sum(home_games["home_score"] > home_games["away_score"])
    away_wins = sum(away_games["away_score"] > away_games["home_score"])

    stats["home_win_pct"] = home_wins / len(home_games) if len(home_games) > 0 else 0
    stats["away_win_pct"] = away_wins / len(away_games) if len(away_games) > 0 else 0
    stats["home_road_diff"] = (
        home_wins / len(home_games) if len(home_games) > 0 else 0
    ) - (away_wins / len(away_games) if len(away_games) > 0 else 0)

    return stats


def calculate_consistency_metrics(
    team_games: pd.DataFrame, team: str
) -> Dict[str, float]:
    """Calculate consistency metrics."""
    stats = {}

    margins = []
    for _, game in team_games.iterrows():
        if game["home_team"] == team:
            margin = game["home_score"] - game["away_score"]
        else:
            margin = game["away_score"] - game["home_score"]
        margins.append(margin)

    margins = np.array(margins)
    stats["avg_margin"] = np.mean(margins) if len(margins) > 0 else 0
    stats["margin_std"] = np.std(margins) if len(margins) > 0 else 0

    return stats


def prepare_game_features(
    row: pd.Series, data: pd.DataFrame, season_stats: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Prepare enhanced features for a single game."""
    features = {}

    # Calculate form
    features["home_form"] = calculate_weighted_form(
        data, row["home_team"], row["game_datetime"]
    )
    features["away_form"] = calculate_weighted_form(
        data, row["away_team"], row["game_datetime"]
    )

    # Head-to-head history
    features["h2h_advantage"] = calculate_head_to_head(
        data, row["home_team"], row["away_team"], row["game_datetime"]
    )

    # Win/loss streaks
    features["home_streak"] = calculate_win_streak(
        data, row["home_team"], row["game_datetime"]
    )
    features["away_streak"] = calculate_win_streak(
        data, row["away_team"], row["game_datetime"]
    )

    # Rest advantage
    features["home_rest"] = calculate_rest_advantage(
        data, row["home_team"], row["game_datetime"]
    )
    features["away_rest"] = calculate_rest_advantage(
        data, row["away_team"], row["game_datetime"]
    )
    features["rest_advantage"] = features["home_rest"] - features["away_rest"]

    # Home/Away specific performance
    features["home_home_performance"] = calculate_home_away_performance(
        data, row["home_team"], True, row["game_datetime"]
    )
    features["away_away_performance"] = calculate_home_away_performance(
        data, row["away_team"], False, row["game_datetime"]
    )

    # Season stats
    if (
        row["season"] in season_stats
        and row["home_team"] in season_stats[row["season"]]
    ):
        home_stats = season_stats[row["season"]][row["home_team"]]
        for stat, value in home_stats.items():
            features[f"home_{stat}"] = value
    else:
        # Default values for new teams
        for stat in ["win_pct", "home_win_pct", "away_win_pct"]:
            features[f"home_{stat}"] = 0.5
        for stat in ["points_per_game", "points_allowed_per_game"]:
            features[f"home_{stat}"] = 20  # League average approximation
        for stat in ["total_games", "home_games", "away_games"]:
            features[f"home_{stat}"] = 0
        for stat in ["win_pct_rank", "points_scored_rank", "points_allowed_rank"]:
            features[f"home_{stat}"] = 16  # Middle rank

    if (
        row["season"] in season_stats
        and row["away_team"] in season_stats[row["season"]]
    ):
        away_stats = season_stats[row["season"]][row["away_team"]]
        for stat, value in away_stats.items():
            features[f"away_{stat}"] = value
    else:
        # Default values for new teams
        for stat in ["win_pct", "home_win_pct", "away_win_pct"]:
            features[f"away_{stat}"] = 0.5
        for stat in ["points_per_game", "points_allowed_per_game"]:
            features[f"away_{stat}"] = 20  # League average approximation
        for stat in ["total_games", "home_games", "away_games"]:
            features[f"away_{stat}"] = 0
        for stat in ["win_pct_rank", "points_scored_rank", "points_allowed_rank"]:
            features[f"away_{stat}"] = 16  # Middle rank

    # Calculate differentials
    for stat in [
        "win_pct",
        "points_per_game",
        "points_allowed_per_game",
        "home_win_pct",
        "away_win_pct",
    ]:
        features[f"{stat}_diff"] = features[f"home_{stat}"] - features[f"away_{stat}"]

    # Add streak-based features
    features["home_streak"] = features.get("home_streak", 0)
    features["away_streak"] = features.get("away_streak", 0)
    features["streak_diff"] = features["home_streak"] - features["away_streak"]

    # Add matchup-specific features
    features["is_division_game"] = 1 if row.get("is_division_game", False) else 0
    features["is_conference_game"] = 1 if row.get("is_conference_game", False) else 0
    features["is_primetime"] = 1 if row.get("is_primetime", False) else 0

    # Add rest and travel features
    features["home_rest_days"] = features.get("home_rest", 7)
    features["away_rest_days"] = features.get("away_rest", 7)
    features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]

    return features


def calculate_strength_of_schedule(
    games_df: pd.DataFrame, team: str, date: datetime, window: int = 16
) -> float:
    """
    Calculate strength of schedule based on opponent win rates.

    Args:
        games_df: DataFrame containing historical games
        team: Team to calculate SOS for
        date: Current date
        window: Number of games to consider
    """
    team_games = (
        games_df[
            ((games_df["home_team"] == team) | (games_df["away_team"] == team))
            & (games_df["game_datetime"] < date)
        ]
        .sort_values("game_datetime", ascending=False)
        .head(window)
    )

    if len(team_games) == 0:
        return 0.5

    opponent_win_rates = []
    for _, game in team_games.iterrows():
        opponent = game["away_team"] if game["home_team"] == team else game["home_team"]
        opp_games = games_df[
            ((games_df["home_team"] == opponent) | (games_df["away_team"] == opponent))
            & (games_df["game_datetime"] < game["game_datetime"])
        ]

        if len(opp_games) == 0:
            opponent_win_rates.append(0.5)
            continue

        wins = 0
        for _, opp_game in opp_games.iterrows():
            is_home = opp_game["home_team"] == opponent
            points_scored = (
                opp_game["home_score"] if is_home else opp_game["away_score"]
            )
            points_allowed = (
                opp_game["away_score"] if is_home else opp_game["home_score"]
            )
            if points_scored > points_allowed:
                wins += 1

        win_rate = wins / len(opp_games)
        opponent_win_rates.append(win_rate)

    return np.mean(opponent_win_rates)


def calculate_efficiency_metrics(
    games_df: pd.DataFrame, team: str, date: datetime, window: int = 8
) -> Dict[str, float]:
    """
    Calculate offensive and defensive efficiency metrics.

    Args:
        games_df: DataFrame containing historical games
        team: Team to calculate metrics for
        date: Current date
        window: Number of games to consider
    """
    team_games = (
        games_df[
            ((games_df["home_team"] == team) | (games_df["away_team"] == team))
            & (games_df["game_datetime"] < date)
        ]
        .sort_values("game_datetime", ascending=False)
        .head(window)
    )

    if len(team_games) == 0:
        return {
            "points_per_drive": 2.0,
            "points_allowed_per_drive": 2.0,
            "yards_per_play": 5.0,
            "yards_allowed_per_play": 5.0,
        }

    total_points = 0
    total_points_allowed = 0
    total_drives = 0
    total_yards = 0
    total_yards_allowed = 0
    total_plays = 0
    total_plays_against = 0

    for _, game in team_games.iterrows():
        is_home = game["home_team"] == team
        points = game["home_team_score"] if is_home else game["away_team_score"]
        points_allowed = game["away_team_score"] if is_home else game["home_team_score"]

        # Estimate drives and plays based on scoring
        estimated_drives = max(8, points / 3)  # Minimum 8 drives per game
        estimated_plays = max(50, points * 4)  # Minimum 50 plays per game

        total_points += points
        total_points_allowed += points_allowed
        total_drives += estimated_drives
        total_yards += estimated_plays * 5  # Estimate 5 yards per play
        total_yards_allowed += estimated_plays * 5
        total_plays += estimated_plays
        total_plays_against += estimated_plays

    return {
        "points_per_drive": total_points / total_drives,
        "points_allowed_per_drive": total_points_allowed / total_drives,
        "yards_per_play": total_yards / total_plays,
        "yards_allowed_per_play": total_yards_allowed / total_plays_against,
    }


def calculate_season_phase_performance(
    games_df: pd.DataFrame, team: str, date: datetime
) -> Dict[str, float]:
    """
    Calculate team performance in different phases of the season.

    Args:
        games_df: DataFrame containing historical games
        team: Team to analyze
        date: Current date
    """

    def get_season_phase(week: int) -> str:
        if week <= 6:
            return "early"
        elif week <= 12:
            return "mid"
        else:
            return "late"

    team_games = games_df[
        ((games_df["home_team"] == team) | (games_df["away_team"] == team))
        & (games_df["game_datetime"] < date)
    ]

    if len(team_games) == 0:
        return {
            "early_season_win_pct": 0.5,
            "mid_season_win_pct": 0.5,
            "late_season_win_pct": 0.5,
        }

    phase_wins = {"early": 0, "mid": 0, "late": 0}
    phase_games = {"early": 0, "mid": 0, "late": 0}

    for _, game in team_games.iterrows():
        phase = get_season_phase(game["week"])
        is_home = game["home_team"] == team
        points = game["home_score"] if is_home else game["away_score"]
        points_allowed = game["away_score"] if is_home else game["home_score"]

        phase_games[phase] += 1
        if points > points_allowed:
            phase_wins[phase] += 1

    return {
        "early_season_win_pct": phase_wins["early"] / max(1, phase_games["early"]),
        "mid_season_win_pct": phase_wins["mid"] / max(1, phase_games["mid"]),
        "late_season_win_pct": phase_wins["late"] / max(1, phase_games["late"]),
    }


def calculate_momentum_score(
    games_df: pd.DataFrame, team: str, date: datetime, window: int = 5
) -> float:
    """
    Calculate momentum score based on recent performance trend

    Args:
        games_df: DataFrame containing historical games
        team: Team to analyze
        date: Current date
        window: Number of games to look back

    Returns:
        float: Momentum score between -1 and 1
    """
    try:
        # Filter games before the current date
        mask = (games_df["date"] < date) & (
            (games_df["home_team"] == team) | (games_df["away_team"] == team)
        )
        recent_games = games_df[mask].sort_values("date", ascending=False).head(window)

        if len(recent_games) == 0:
            return 0.0

        # Calculate weighted results (more recent games count more)
        weights = np.exp(-np.arange(len(recent_games)) / 2)  # Exponential decay
        weights = weights / weights.sum()

        results = []
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                result = 1 if game["home_score"] > game["away_score"] else -1
            else:
                result = 1 if game["away_score"] > game["home_score"] else -1
            results.append(result)

        momentum = np.sum(weights * results)
        return float(momentum)

    except Exception as e:
        logging.error(f"Error calculating momentum score for {team}: {str(e)}")
        return 0.0


def calculate_injury_impact(team: str, injuries: Dict[str, float]) -> float:
    """Calculate impact of injuries based on player importance"""
    if not injuries:
        return 0.0

    # Weights for different positions
    position_weights = {
        "QB": 0.3,
        "WR": 0.15,
        "RB": 0.15,
        "OL": 0.1,
        "DL": 0.1,
        "LB": 0.1,
        "DB": 0.1,
    }

    total_impact = 0.0
    for position, players in injuries.items():
        if position in position_weights:
            total_impact += position_weights[position] * len(players)

    return min(total_impact, 1.0)  # Cap at 1.0


def calculate_third_down_stats(
    team_games: pd.DataFrame, team: str
) -> Tuple[float, int]:
    """Calculate third down conversion rate"""
    if len(team_games) == 0:
        return 0.0, 0

    # Use league average if not available
    league_avg_conversion = 0.392  # NFL average third down conversion rate
    return league_avg_conversion, len(team_games)


def calculate_rolling_win_percentage(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate rolling win percentage for each team"""

    def team_win_pct(games):
        if len(games) == 0:
            return 0.0
        wins = sum((games["home_team_score"] > games["away_team_score"]).astype(int))
        return wins / len(games)

    win_pcts = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        current_team = data.iloc[i]["home_team"]
        home_games = prev_games[prev_games["home_team"] == current_team].tail(window)
        win_pcts.iloc[i] = team_win_pct(home_games)

    return win_pcts


def calculate_rolling_average(
    data: pd.DataFrame, column: str, window: int = 5
) -> pd.Series:
    """Calculate rolling average for a specific column"""
    return data[column].rolling(window=window, min_periods=1).mean()


def calculate_home_win_percentage(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate home win percentage"""
    home_wins = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        current_team = data.iloc[i]["home_team"]
        home_games = prev_games[prev_games["home_team"] == current_team].tail(window)
        if len(home_games) == 0:
            home_wins.iloc[i] = 0.5
        else:
            wins = sum(
                (home_games["home_team_score"] > home_games["away_team_score"]).astype(
                    int
                )
            )
            home_wins.iloc[i] = wins / len(home_games)

    return home_wins


def calculate_away_win_percentage(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate away win percentage"""
    away_wins = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        current_team = data.iloc[i]["away_team"]
        away_games = prev_games[prev_games["away_team"] == current_team].tail(window)
        if len(away_games) == 0:
            away_wins.iloc[i] = 0.5
        else:
            wins = sum(
                (away_games["away_team_score"] > away_games["home_team_score"]).astype(
                    int
                )
            )
            away_wins.iloc[i] = wins / len(away_games)

    return away_wins


def calculate_division_win_percentage(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate division win percentage"""
    div_wins = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        current_team = data.iloc[i]["home_team"]
        div_games = prev_games[
            (
                (prev_games["home_team"] == current_team)
                | (prev_games["away_team"] == current_team)
            )
            & (prev_games["is_division_game"])
        ].tail(window)

        if len(div_games) == 0:
            div_wins.iloc[i] = 0.5
        else:
            wins = sum(
                (
                    (div_games["home_team"] == current_team)
                    & (div_games["home_team_score"] > div_games["away_team_score"])
                )
                | (
                    (div_games["away_team"] == current_team)
                    & (div_games["away_team_score"] > div_games["home_team_score"])
                )
            )
            div_wins.iloc[i] = wins / len(div_games)

    return div_wins


def calculate_conference_win_percentage(
    data: pd.DataFrame, window: int = 5
) -> pd.Series:
    """Calculate conference win percentage"""
    conf_wins = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        current_team = data.iloc[i]["home_team"]
        conf_games = prev_games[
            (
                (prev_games["home_team"] == current_team)
                | (prev_games["away_team"] == current_team)
            )
            & (prev_games["is_conference_game"])
        ].tail(window)

        if len(conf_games) == 0:
            conf_wins.iloc[i] = 0.5
        else:
            wins = sum(
                (
                    (conf_games["home_team"] == current_team)
                    & (conf_games["home_team_score"] > conf_games["away_team_score"])
                )
                | (
                    (conf_games["away_team"] == current_team)
                    & (conf_games["away_team_score"] > conf_games["home_team_score"])
                )
            )
            conf_wins.iloc[i] = wins / len(conf_games)

    return conf_wins


def calculate_early_season_win_pct(data: pd.DataFrame) -> pd.Series:
    """Calculate win percentage in early season games (weeks 1-6)"""
    early_wins = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        early_games = prev_games[prev_games["week"] <= 6]
        if len(early_games) == 0:
            early_wins.iloc[i] = 0.5
        else:
            wins = sum(
                (
                    early_games["home_team_score"] > early_games["away_team_score"]
                ).astype(int)
            )
            early_wins.iloc[i] = wins / len(early_games)

    return early_wins


def calculate_mid_season_win_pct(data: pd.DataFrame) -> pd.Series:
    """Calculate win percentage in mid-season games (weeks 7-12)"""
    mid_wins = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        mid_games = prev_games[(prev_games["week"] > 6) & (prev_games["week"] <= 12)]
        if len(mid_games) == 0:
            mid_wins.iloc[i] = 0.5
        else:
            wins = sum(
                (mid_games["home_team_score"] > mid_games["away_team_score"]).astype(
                    int
                )
            )
            mid_wins.iloc[i] = wins / len(mid_games)

    return mid_wins


def calculate_late_season_win_pct(data: pd.DataFrame) -> pd.Series:
    """Calculate win percentage in late-season games (weeks 13+)"""
    late_wins = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        late_games = prev_games[prev_games["week"] > 12]
        if len(late_games) == 0:
            late_wins.iloc[i] = 0.5
        else:
            wins = sum(
                (late_games["home_team_score"] > late_games["away_team_score"]).astype(
                    int
                )
            )
            late_wins.iloc[i] = wins / len(late_games)

    return late_wins


def calculate_strength_of_schedule(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate strength of schedule based on opponent win percentages"""
    sos = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        prev_games = data.iloc[:i]
        current_opponent = data.iloc[i]["away_team"]
        opponent_games = prev_games[
            (prev_games["home_team"] == current_opponent)
            | (prev_games["away_team"] == current_opponent)
        ].tail(window)

        if len(opponent_games) == 0:
            sos.iloc[i] = 0.5
        else:
            opponent_wins = sum(
                (
                    opponent_games["home_team_score"]
                    > opponent_games["away_team_score"]
                ).astype(int)
            )
            sos.iloc[i] = opponent_wins / len(opponent_games)

    return sos


def calculate_offensive_efficiency(data: pd.DataFrame) -> pd.Series:
    """Calculate offensive efficiency composite score"""
    return (
        0.4 * data["points_per_game"]
        + 0.3 * data["yards_per_play"]
        + 0.3 * data["third_down_conv"]
    )


def calculate_third_down_conversion(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate third down conversion rate"""
    return (
        data["third_downs_converted"].rolling(window=window, min_periods=1).mean()
        / data["third_downs_attempted"].rolling(window=window, min_periods=1).mean()
    )


def calculate_points_per_drive(data: pd.DataFrame) -> pd.Series:
    """Estimate points per drive"""
    return data["points_per_game"] / 12  # Assuming average of 12 drives per game


def calculate_yards_per_play(data: pd.DataFrame) -> pd.Series:
    """Calculate yards per play"""
    return data["total_yards"] / data["total_plays"]


def calculate_scoring_consistency(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate scoring consistency (lower value means more consistent)"""
    return data["points_per_game"].rolling(window=window, min_periods=1).std()


def calculate_defense_consistency(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Calculate defensive consistency (lower value means more consistent)"""
    return data["points_allowed_per_game"].rolling(window=window, min_periods=1).std()


def calculate_win_percentages(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate various win percentages for each team

    Returns:
        Dict containing different win percentage metrics
    """
    try:
        results = {}

        # Split into smaller functions for better organization
        results["overall"] = _calculate_overall_win_pct(data)
        results["home"] = _calculate_home_win_pct(data)
        results["away"] = _calculate_away_win_pct(data)
        results["last_5"] = _calculate_recent_win_pct(data, 5)
        results["last_10"] = _calculate_recent_win_pct(data, 10)

        return results

    except Exception as e:
        logging.error(f"Error calculating win percentages: {str(e)}")
        return _create_default_win_percentages()


def _calculate_overall_win_pct(data: pd.DataFrame) -> pd.Series:
    """Helper function to calculate overall win percentage"""
    wins = data["wins"]
    losses = data["losses"]
    ties = data.get("ties", 0)  # Some datasets might not have ties
    return (wins + 0.5 * ties) / (wins + losses + ties)


def _calculate_home_win_pct(data: pd.DataFrame) -> pd.Series:
    """Helper function to calculate home win percentage"""
    home_wins = data["home_wins"]
    home_games = data["home_games"]
    return home_wins / home_games


def _calculate_away_win_pct(data: pd.DataFrame) -> pd.Series:
    """Helper function to calculate away win percentage"""
    away_wins = data["away_wins"]
    away_games = data["away_games"]
    return away_wins / away_games


def _calculate_recent_win_pct(data: pd.DataFrame, window: int) -> pd.Series:
    """Helper function to calculate recent win percentage"""
    recent_wins = data[f"last_{window}_wins"]
    return recent_wins / window


def _create_default_win_percentages() -> Dict[str, pd.Series]:
    """Create default win percentages when calculation fails"""
    return {
        "overall": pd.Series(dtype=float),
        "home": pd.Series(dtype=float),
        "away": pd.Series(dtype=float),
        "last_5": pd.Series(dtype=float),
        "last_10": pd.Series(dtype=float),
    }

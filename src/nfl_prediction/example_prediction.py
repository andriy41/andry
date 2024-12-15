"""
Example script showing how to use the NFL Astrological Prediction System
"""
from datetime import datetime, timedelta
from app.astrology import (
    NFLAstrologyPredictor,
    NFLAstroAnalyzer,
    NFLBettingAnalyzer,
    NFLPlayerAnalyzer,
)


def main():
    # Create predictor instance
    predictor = NFLAstrologyPredictor()

    # Example game prediction
    game_time = datetime.now()
    home_team = "Kansas City Chiefs"
    away_team = "San Francisco 49ers"
    weather = "Clear, 72F, Wind 5mph"

    # Get comprehensive prediction
    prediction = predictor.predict_game(
        game_time=game_time,
        home_team=home_team,
        away_team=away_team,
        weather_conditions=weather,
        include_betting=True,
    )

    # Print results
    print("\nGame Prediction:")
    print(f"{home_team} vs {away_team}")
    print(f"Time: {game_time}")
    print(f"Weather: {weather}\n")

    # Base prediction
    base = prediction["base_prediction"]
    print("Base Prediction:")
    print(f"Home Team Strength: {base['total_home_score']:.2f}")
    print(f"Away Team Strength: {base['total_away_score']:.2f}")

    # Betting analysis
    if "betting_analysis" in prediction:
        betting = prediction["betting_analysis"]
        print("\nBetting Analysis:")
        print(f"Home Win Probability: {betting['home_probability']:.2%}")
        print(f"Away Win Probability: {betting['away_probability']:.2%}")
        print(f"Confidence: {betting['confidence']:.2%}")

        # Betting recommendation
        rec = prediction["betting_recommendation"]
        print(f"\nRecommendation: {rec['recommendation']}")
        print(f"Reason: {rec['reason']}")

    # Example player matchup analysis
    print("\nExample Player Matchup:")
    player1 = {"name": "Patrick Mahomes", "position": "QB", "team": home_team}
    player2 = {"name": "Brock Purdy", "position": "QB", "team": away_team}

    matchup = predictor.analyze_player_matchup(
        game_time=game_time,
        stadium_location=(39.0489, -94.4839),  # Arrowhead Stadium
        player1_data=player1,
        player2_data=player2,
    )

    print(f"Matchup: {player1['name']} vs {player2['name']}")
    print(f"Advantage: {matchup['advantage']:.2f}")
    print("Recommendations:")
    for rec in matchup["recommendations"]:
        print(f"- {rec}")

    # Find favorable times
    print("\nFavorable Game Times:")
    start_date = datetime.now()
    end_date = start_date + timedelta(days=7)
    favorable_times = predictor.get_favorable_times(
        team=home_team, start_date=start_date, end_date=end_date
    )

    print(f"\nBest times for {home_team} games next week:")
    for time in favorable_times[:3]:
        print(f"- {time}")


if __name__ == "__main__":
    main()

from models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator
from datetime import datetime
import pytz


def test_prediction():
    # Initialize the calculator
    calculator = NFLVedicCalculator()

    # Example game: Let's predict a game between Kansas City Chiefs and San Francisco 49ers
    game_date = "2024-12-08"  # This Sunday
    game_time = "13:00:00"  # 1 PM
    timezone = "America/New_York"
    home_team = "KC"  # Kansas City Chiefs
    away_team = "SF"  # San Francisco 49ers

    # Get prediction
    prediction = calculator.predict_influence(
        game_date=game_date,
        game_time=game_time,
        timezone=timezone,
        home_team=home_team,
        away_team=away_team,
    )

    # Print results
    print("\nNFL Game Prediction using Vedic Astrology")
    print("==========================================")
    print(f"Game: {away_team} @ {home_team}")
    print(f"Date: {game_date} {game_time} {timezone}")
    print("\nPrediction Results:")
    print(f"Home Team Win Probability: {prediction['home_win_probability']:.2%}")
    print(f"Prediction Confidence: {prediction['confidence']:.2%}")

    # Print detailed features
    print("\nAstrological Features:")
    features = prediction["features"]
    print(f"Moon Phase Score: {features['moon_phase_score']:.3f}")
    print(f"Beneficial Aspects: {features['beneficial_aspects']}")
    print(f"Malefic Aspects: {features['malefic_aspects']}")
    print(f"Home Team Score: {features['home_team_score']:.3f}")
    print(f"Away Team Score: {features['away_team_score']:.3f}")
    print(f"Planetary Alignment Score: {features['planetary_alignment_score']:.3f}")


if __name__ == "__main__":
    test_prediction()

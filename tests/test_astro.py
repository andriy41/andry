from predict_total import TotalPredictor
from datetime import datetime


def test_astro_predictions():
    """Test astrological predictions for upcoming games"""
    predictor = TotalPredictor()

    # Test game data
    test_game = {
        "game_id": "TEST_GAME_1",
        "home_team": "Baltimore Ravens",
        "away_team": "Los Angeles Rams",
        "date": "2024-01-28 15:00",
        "venue": "M&T Bank Stadium",
        "vegas_total": 44.5,
        "home_stats": {
            "points_per_game": 28.4,
            "points_allowed": 16.5,
            "yards_per_play": 5.9,
            "third_down_pct": 0.44,
            "red_zone_pct": 0.63,
            "qb_rating": 102.3,
            "defensive_efficiency": 0.89,
            "sacks_per_game": 3.2,
            "turnovers_forced": 1.9,
            "plays_per_game": 63.5,
            "time_of_possession": 31.5,
            "hurry_up_rate": 0.12,
            "first_half_points": 15.2,
            "second_half_points": 13.2,
            "over_rate": 0.53,
            "avg_total_points": 44.9,
        },
        "away_stats": {
            "points_per_game": 22.1,
            "points_allowed": 17.3,
            "yards_per_play": 5.5,
            "third_down_pct": 0.41,
            "red_zone_pct": 0.58,
            "qb_rating": 95.8,
            "defensive_efficiency": 0.85,
            "sacks_per_game": 2.8,
            "turnovers_forced": 1.6,
            "plays_per_game": 62.8,
            "time_of_possession": 30.8,
            "hurry_up_rate": 0.14,
            "first_half_points": 11.5,
            "second_half_points": 10.6,
            "over_rate": 0.47,
            "avg_total_points": 39.4,
        },
        "is_division_game": False,
        "is_primetime": True,
        "days_rest": 7,
        "playoff_implications": 1.0,
        "matchup_historical_avg": 42.5,
        "temperature": 45,
        "wind_speed": 8,
        "is_dome": False,
        "precipitation_chance": 0.2,
        "altitude": 0,
    }

    print("\nTesting Astrological Predictions")
    print("=" * 50)

    try:
        # Get predictions from each model
        prediction = predictor.predict(test_game)

        print(f"\nGame: {test_game['away_team']} @ {test_game['home_team']}")
        print(f"Date: {test_game['date']}")
        print(f"Vegas Total: {test_game['vegas_total']}")

        print("\nModel Predictions:")
        for model, total in prediction["model_predictions"].items():
            conf = prediction["model_confidences"][model]
            print(f"- {model.title()}: {total:.1f} (Confidence: {conf:.2f})")

        if "astro" in prediction["model_predictions"]:
            astro_pred = predictor.models["astro"].predict(test_game)
            if "factors" in astro_pred:
                print("\nAstrological Factors:")
                for factor, value in astro_pred["factors"].items():
                    print(f"- {factor.replace('_', ' ').title()}: {value:.3f}")

        consensus = prediction["consensus"]
        print(f"\nConsensus Total: {consensus['total']:.1f}")
        print(f"Rating: {consensus['rating']}")
        print(f"Explanation: {consensus['explanation']}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    test_astro_predictions()

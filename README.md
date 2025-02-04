# NFL Prediction System

A dedicated prediction system for NFL games.

## Directory Structure
- `data/`: Raw and processed data
- `models/`: Model definitions and saved models
- `prediction/`: Prediction scripts and interfaces
- `evaluation/`: Model evaluation tools
- `utils/`: Utility functions
- `scripts/`: Helper scripts
- `logs/`: Log files

## Setup
1. Install requirements
2. Configure data sources
3. Train models
4. Make predictions

## Models

### Vedic Astrology Model
The system includes a Vedic astrology-based prediction model that analyzes planetary positions and their influence on game outcomes. Key features:

- Planetary position analysis using Swiss Ephemeris
- Team strength calculations based on foundation dates
- Astrological factor integration with traditional stats
- Confidence-based prediction filtering

#### Vedic Model Setup
1. Install Swiss Ephemeris:
```bash
pip install -r requirements.txt
```
2. Download ephemeris files:
```bash
mkdir -p ephe
cd ephe
wget https://www.astro.com/ftp/swisseph/ephe/seas_18.se1
wget https://www.astro.com/ftp/swisseph/ephe/semo_18.se1
wget https://www.astro.com/ftp/swisseph/ephe/sepl_18.se1
```
3. Configure stadium coordinates:
Update `src/models/vedic_astrology/data/stadium_coordinates.py` with accurate stadium data.

4. Run tests:
```bash
python -m pytest src/models/vedic_astrology/tests/
```
#### Model Training
```bash
python src/models/train_all_models.py
```
The Vedic model combines traditional astrological principles with modern machine learning techniques for a unique approach to NFL game prediction.

## Usage
See prediction scripts in the `prediction/` directory for examples.
#   a n d r y  
 
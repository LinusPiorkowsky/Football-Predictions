import os
import json
from pathlib import Path
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create all necessary directories"""
    base_dir = Path(__file__).parent
    
    directories = [
        'data',
        'data/2024_25',
        'data/2025_26',
        'predictions',
        'results',
        'static/css',
        'static/js',
        'templates',
        'models',
        'logs'
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def download_initial_fixtures():
    """Download initial fixtures for testing"""
    try:
        url = "https://www.football-data.co.uk/fixtures.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        fixtures_dir = Path(__file__).parent / 'data' / '2025_26'
        fixtures_file = fixtures_dir / 'fixtures.csv'
        
        with open(fixtures_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        logger.info(f"Downloaded fixtures to {fixtures_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading fixtures: {e}")
        return False

def create_sample_predictions():
    """Create sample predictions for testing"""
    sample_predictions = {
        "predictions": [
            {
                "Div": "D1",
                "Date": "2025-02-14",
                "Time": "20:30",
                "HomeTeam": "Bayern Munich",
                "AwayTeam": "Borussia Dortmund",
                "Prediction": "H",
                "Prob_H": 0.65,
                "Prob_D": 0.20,
                "Prob_A": 0.15,
                "B365H": 1.80,
                "B365D": 3.50,
                "B365A": 4.50,
                "Confidence": 0.65,
                "High_conf": 1,
                "1X_prob": 0.85,
                "X2_prob": 0.35,
                "1X_odds": 1.18,
                "X2_odds": 2.86
            }
        ]
    }
    
    predictions_dir = Path(__file__).parent / 'predictions'
    sample_file = predictions_dir / 'sample_predictions.json'
    
    with open(sample_file, 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    logger.info(f"Created sample predictions at {sample_file}")

def create_config_file():
    """Create configuration file"""
    config = {
        "app": {
            "name": "Football AI Predictions",
            "version": "1.0.0",
            "debug": False
        },
        "data": {
            "current_season": "2025_26",
            "leagues": ["D1", "E0", "F1", "I1", "SP1"],
            "update_schedule": {
                "predictions": ["Tuesday 14:00", "Friday 18:00"],
                "results": ["Daily 23:00"]
            }
        },
        "model": {
            "type": "ensemble",
            "algorithms": ["RandomForest", "GradientBoosting"],
            "confidence_threshold": 0.65
        }
    }
    
    config_file = Path(__file__).parent / 'config.json'
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created config file at {config_file}")

def main():
    """Run all setup tasks"""
    logger.info("Starting Football AI App Setup...")
    
    # Create directories
    setup_directories()
    
    # Download initial data
    download_initial_fixtures()
    
    # Create sample files
    create_sample_predictions()
    
    # Create config
    create_config_file()
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the app with: python app.py")

if __name__ == "__main__":
    main()
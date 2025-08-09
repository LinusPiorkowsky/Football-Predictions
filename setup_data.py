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
    create_sample_predictions()import os
import json
from pathlib import Path
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create all necessary directories with .gitkeep files"""
    base_dir = Path(__file__).parent
    
    # Directories that need to exist
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
        
        # Add .gitkeep to empty directories
        gitkeep = dir_path / '.gitkeep'
        if not any(dir_path.iterdir()) and not gitkeep.exists():
            gitkeep.touch()
        
        logger.info(f"✓ Created directory: {directory}")

def download_initial_data():
    """Download initial fixtures for current season"""
    try:
        # Current season directory
        season_dir = Path(__file__).parent / 'data' / '2025_26'
        season_dir.mkdir(parents=True, exist_ok=True)
        
        # Download fixtures
        logger.info("Downloading current fixtures...")
        fixtures_url = "https://www.football-data.co.uk/fixtures.csv"
        response = requests.get(fixtures_url, timeout=30)
        
        if response.status_code == 200:
            fixtures_file = season_dir / 'fixtures.csv'
            with open(fixtures_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.info(f"✓ Downloaded fixtures to {fixtures_file}")
        else:
            logger.warning("Could not download fixtures - will retry on first run")
        
        # Try to download current season Bundesliga data
        logger.info("Attempting to download Bundesliga data...")
        bundesliga_url = "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
        response = requests.get(bundesliga_url, timeout=30)
        
        if response.status_code == 200:
            bundesliga_file = season_dir / 'D1.csv'
            with open(bundesliga_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.info(f"✓ Downloaded Bundesliga data to {bundesliga_file}")
        else:
            logger.info("No current season data available yet - this is normal for new seasons")
        
        return True
        
    except Exception as e:
        logger.warning(f"Could not download initial data: {e}")
        logger.info("Data will be downloaded on first run")
        return False

def create_sample_data():
    """Create sample data for testing"""
    
    # Sample prediction for testing
    sample_prediction = """Div,Date,Weekday,Time,HomeTeam,AwayTeam,Prediction,B365H,B365D,B365A,Prob_H,Prob_D,Prob_A,Confidence,High_conf
D1,2025-02-15,Saturday,15:30,Bayern Munich,Borussia Dortmund,H,1.65,4.0,5.0,0.68,0.22,0.10,0.68,1
D1,2025-02-15,Saturday,15:30,RB Leipzig,Bayer Leverkusen,D,2.30,3.40,3.00,0.35,0.40,0.25,0.40,0
D1,2025-02-15,Saturday,18:30,Eintracht Frankfurt,Wolfsburg,A,2.80,3.30,2.50,0.25,0.30,0.45,0.45,0
"""
    
    predictions_dir = Path(__file__).parent / 'predictions'
    sample_file = predictions_dir / 'sample_predictions.csv'
    
    with open(sample_file, 'w') as f:
        f.write(sample_prediction)
    
    logger.info(f"✓ Created sample predictions at {sample_file}")
    
    # Sample results for testing
    sample_results = """Date,Time,Div,HomeTeam,AwayTeam,Prediction,Actual_Result,Score,FinalBet,IsCorrect,High_conf
2025-02-08,15:30,D1,Borussia Dortmund,Werder Bremen,H,H,2-1,H,1,1
2025-02-08,15:30,D1,Hoffenheim,Union Berlin,A,D,1-1,X2,1,0
2025-02-08,18:30,D1,Bayern Munich,Holstein Kiel,H,H,4-0,H,1,1
"""
    
    results_dir = Path(__file__).parent / 'results'
    sample_file = results_dir / 'sample_results.csv'
    
    with open(sample_file, 'w') as f:
        f.write(sample_results)
    
    logger.info(f"✓ Created sample results at {sample_file}")

def create_config():
    """Create configuration file"""
    config = {
        "app": {
            "name": "Football AI Predictions",
            "version": "1.0.0",
            "debug": False,
            "secret_key": "change-this-in-production"
        },
        "data": {
            "current_season": "2025_26",
            "previous_season": "2024_25",
            "base_url": "https://www.football-data.co.uk",
            "leagues": {
                "D1": "Bundesliga",
                "E0": "Premier League",
                "F1": "Ligue 1",
                "I1": "Serie A",
                "SP1": "La Liga"
            },
            "primary_league": "D1"
        },
        "schedule": {
            "predictions": {
                "tuesday": "14:00",
                "friday": "18:00"
            },
            "results": {
                "daily": "23:00"
            }
        },
        "model": {
            "type": "ensemble",
            "algorithms": ["RandomForest", "GradientBoosting"],
            "confidence_threshold": 0.65,
            "high_confidence_threshold": 0.70,
            "features": [
                "home_goals_avg",
                "away_goals_avg",
                "home_form",
                "away_form",
                "head_to_head",
                "league_position"
            ]
        }
    }
    
    config_file = Path(__file__).parent / 'config.json'
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Created config.json")

def check_requirements():
    """Check if all required files exist"""
    base_dir = Path(__file__).parent
    
    required_files = [
        'app.py',
        'prediction_agent.py',
        'requirements.txt',
        'models/predictor.py',
        'models/comparator.py',
        'templates/index.html',
        'static/css/style.css'
    ]
    
    missing = []
    for file in required_files:
        file_path = base_dir / file
        if not file_path.exists():
            missing.append(file)
    
    if missing:
        logger.warning(f"⚠️  Missing files: {', '.join(missing)}")
        logger.info("Make sure all files are in place before running the app")
    else:
        logger.info("✓ All required files present")

def main():
    """Run all setup tasks"""
    print("=" * 50)
    print("🚀 Football AI Predictions - Setup")
    print("=" * 50)
    
    # Create directories
    logger.info("\n📁 Creating directory structure...")
    setup_directories()
    
    # Download initial data
    logger.info("\n📥 Downloading initial data...")
    download_initial_data()
    
    # Create sample data
    logger.info("\n📝 Creating sample data...")
    create_sample_data()
    
    # Create config
    logger.info("\n⚙️  Creating configuration...")
    create_config()
    
    # Check requirements
    logger.info("\n🔍 Checking requirements...")
    check_requirements()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: python app.py")
    print("3. Visit: http://localhost:5000")
    print("\nFor Railway deployment:")
    print("1. Push to GitHub")
    print("2. Connect to Railway")
    print("3. Deploy!")

if __name__ == "__main__":
    main()
    
    # Create config
    create_config_file()
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the app with: python app.py")

if __name__ == "__main__":
    main()
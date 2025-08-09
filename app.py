# app.py - Production Ready Version
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PREDICTION_DIR = BASE_DIR / "predictions"
RESULT_DIR = BASE_DIR / "results"
CURRENT_SEASON = "2025_26"
CURRENT_SEASON_DIR = DATA_DIR / CURRENT_SEASON

# Create directories
for directory in [DATA_DIR, PREDICTION_DIR, RESULT_DIR, CURRENT_SEASON_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# League configuration
LEAGUES = {
    "D1": {"name": "Bundesliga", "country": "Germany", "emoji": "🇩🇪"},
    "E0": {"name": "Premier League", "country": "England", "emoji": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "F1": {"name": "Ligue 1", "country": "France", "emoji": "🇫🇷"},
    "I1": {"name": "Serie A", "country": "Italy", "emoji": "🇮🇹"},
    "SP1": {"name": "La Liga", "country": "Spain", "emoji": "🇪🇸"}
}

class PredictionManager:
    """Manages predictions and results with error handling"""
    
    @staticmethod
    def get_latest_predictions():
        """Load the most recent predictions with robust error handling"""
        try:
            # First check for any CSV files
            prediction_files = list(PREDICTION_DIR.glob("*.csv"))
            
            if not prediction_files:
                logger.warning("No prediction files found, creating sample data")
                return PredictionManager._create_sample_predictions()
            
            # Sort by modification time
            prediction_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Try to read the most recent file
            for file in prediction_files:
                try:
                    df = pd.read_csv(file)
                    
                    # Ensure required columns exist
                    required_cols = ['HomeTeam', 'AwayTeam', 'Div', 'Prediction']
                    if not all(col in df.columns for col in required_cols):
                        continue
                    
                    # Parse dates safely
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df = df[df['Date'].notna()]
                    else:
                        df['Date'] = pd.Timestamp.now()
                    
                    # Add default values for missing columns
                    df['Time'] = df.get('Time', '15:30').fillna('15:30')
                    df['High_conf'] = df.get('High_conf', 0).fillna(0)
                    df['Confidence'] = df.get('Confidence', 0.5).fillna(0.5)
                    df['B365H'] = df.get('B365H', 2.0).fillna(2.0)
                    df['B365D'] = df.get('B365D', 3.5).fillna(3.5)
                    df['B365A'] = df.get('B365A', 2.5).fillna(2.5)
                    
                    # Filter future matches
                    now = datetime.now()
                    if 'Time' in df.columns:
                        df['DateTime'] = pd.to_datetime(
                            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'],
                            errors='coerce'
                        )
                        df = df[(df['DateTime'] > now) | df['DateTime'].isna()]
                    
                    return df.sort_values(['Date', 'Time']) if not df.empty else PredictionManager._create_sample_predictions()
                    
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
                    continue
            
            # If all files failed, return sample data
            return PredictionManager._create_sample_predictions()
            
        except Exception as e:
            logger.error(f"Critical error loading predictions: {e}")
            return PredictionManager._create_sample_predictions()
    
    @staticmethod
    def _create_sample_predictions():
        """Create sample predictions for demonstration"""
        tomorrow = datetime.now() + timedelta(days=1)
        sample_data = {
            'Div': ['D1', 'D1', 'E0'],
            'Date': [tomorrow, tomorrow, tomorrow + timedelta(days=2)],
            'Time': ['15:30', '18:30', '20:00'],
            'HomeTeam': ['Bayern Munich', 'Borussia Dortmund', 'Manchester United'],
            'AwayTeam': ['RB Leipzig', 'Wolfsburg', 'Liverpool'],
            'Prediction': ['H', 'H', 'D'],
            'Confidence': [0.68, 0.55, 0.45],
            'High_conf': [1, 0, 0],
            'B365H': [1.80, 2.10, 2.50],
            'B365D': [3.50, 3.40, 3.20],
            'B365A': [4.50, 3.80, 2.90],
            'Prob_H': [0.68, 0.55, 0.35],
            'Prob_D': [0.20, 0.25, 0.45],
            'Prob_A': [0.12, 0.20, 0.20]
        }
        return pd.DataFrame(sample_data)
    
    @staticmethod
    def get_all_results():
        """Load all historical results with error handling"""
        try:
            result_files = list(RESULT_DIR.glob("*.csv"))
            
            if not result_files:
                logger.info("No result files found, creating sample data")
                return PredictionManager._create_sample_results()
            
            dfs = []
            for file in result_files:
                try:
                    df = pd.read_csv(file)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading result file {file}: {e}")
                    continue
            
            if dfs:
                all_results = pd.concat(dfs, ignore_index=True)
                return all_results.sort_values('Date', ascending=False) if 'Date' in all_results.columns else all_results
            
            return PredictionManager._create_sample_results()
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return PredictionManager._create_sample_results()
    
    @staticmethod
    def _create_sample_results():
        """Create sample results for demonstration"""
        past_week = datetime.now() - timedelta(days=7)
        sample_data = {
            'Date': [past_week + timedelta(days=i) for i in range(5)],
            'Time': ['15:30'] * 5,
            'Div': ['D1', 'D1', 'E0', 'D1', 'E0'],
            'HomeTeam': ['Bayern Munich', 'Dortmund', 'Chelsea', 'Leipzig', 'Arsenal'],
            'AwayTeam': ['Augsburg', 'Bremen', 'Fulham', 'Frankfurt', 'Tottenham'],
            'Prediction': ['H', 'H', 'H', 'D', 'A'],
            'Actual_Result': ['H', 'H', 'D', 'D', 'H'],
            'Score': ['3-1', '2-0', '1-1', '2-2', '2-1'],
            'High_conf': [1, 1, 0, 0, 0],
            'B365H': [1.50, 1.80, 2.10, 2.50, 2.20],
            'B365D': [4.00, 3.50, 3.40, 3.20, 3.30],
            'B365A': [6.00, 4.50, 3.50, 2.90, 3.20]
        }
        df = pd.DataFrame(sample_data)
        df['IsCorrect'] = df['Prediction'] == df['Actual_Result']
        df['FinalBet'] = df['Prediction']
        return df
    
    @staticmethod
    def calculate_statistics(df, period=None):
        """Calculate betting statistics with error handling"""
        try:
            if df is None or df.empty:
                return {
                    'total_bets': 0,
                    'correct_bets': 0,
                    'accuracy': 0.0,
                    'roi': 0.0,
                    'best_market': 'N/A',
                    'profit': 0.0
                }
            
            # Filter by period
            if period and 'Date' in df.columns:
                try:
                    if period == 'week':
                        cutoff = datetime.now() - timedelta(days=7)
                        df = df[df['Date'] >= cutoff]
                    elif period == 'month':
                        cutoff = datetime.now() - timedelta(days=30)
                        df = df[df['Date'] >= cutoff]
                except:
                    pass
            
            # Calculate stats
            total = len(df)
            
            # Check for correctness
            if 'IsCorrect' in df.columns:
                correct = df['IsCorrect'].sum()
            elif 'Prediction' in df.columns and 'Actual_Result' in df.columns:
                df['IsCorrect'] = df['Prediction'] == df['Actual_Result']
                correct = df['IsCorrect'].sum()
            else:
                correct = 0
            
            # Calculate ROI
            roi = 0.0
            if total > 0:
                if 'Profit' in df.columns:
                    roi = df['Profit'].sum() / total * 100
                else:
                    # Simple ROI calculation
                    roi = ((correct * 1.8) - total) / total * 100  # Assuming average odds of 1.8
            
            return {
                'total_bets': int(total),
                'correct_bets': int(correct),
                'accuracy': round(correct / total * 100, 2) if total > 0 else 0.0,
                'roi': round(roi, 2),
                'best_market': 'H',  # Default
                'profit': round(roi * total / 100, 2) if total > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {
                'total_bets': 0,
                'correct_bets': 0,
                'accuracy': 0.0,
                'roi': 0.0,
                'best_market': 'N/A',
                'profit': 0.0
            }

# Initialize manager
prediction_manager = PredictionManager()

# Routes
@app.route('/')
def index():
    """Dashboard with statistics and overview"""
    try:
        results = prediction_manager.get_all_results()
        
        # Calculate statistics for different periods
        stats = {
            'all_time': prediction_manager.calculate_statistics(results),
            'last_month': prediction_manager.calculate_statistics(results, 'month'),
            'last_week': prediction_manager.calculate_statistics(results, 'week')
        }
        
        # Get upcoming predictions summary
        predictions = prediction_manager.get_latest_predictions()
        upcoming_count = len(predictions) if predictions is not None and not predictions.empty else 0
        
        # High confidence predictions
        high_conf_count = 0
        if predictions is not None and not predictions.empty and 'High_conf' in predictions.columns:
            high_conf_predictions = predictions[predictions['High_conf'] == 1]
            high_conf_count = len(high_conf_predictions)
        
        return render_template('index.html',
                             stats=stats,
                             upcoming_count=upcoming_count,
                             high_conf_count=high_conf_count,
                             leagues=LEAGUES)
    
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        # Return with default values
        return render_template('index.html',
                             stats={
                                 'all_time': {'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0, 'best_market': 'N/A'},
                                 'last_month': {'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0},
                                 'last_week': {'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0}
                             },
                             upcoming_count=0,
                             high_conf_count=0,
                             leagues=LEAGUES)

@app.route('/predictions')
def predictions():
    """Display upcoming predictions"""
    try:
        # Get filter parameters
        league = request.args.get('league', '')
        confidence = request.args.get('confidence', '')
        date_filter = request.args.get('date', '')
        
        df = prediction_manager.get_latest_predictions()
        
        if df is None or df.empty:
            return render_template('predictions.html',
                                 predictions=[],
                                 leagues=LEAGUES,
                                 dates=[],
                                 filters={})
        
        # Apply filters
        if league and 'Div' in df.columns:
            df = df[df['Div'] == league]
        
        if confidence == 'high' and 'High_conf' in df.columns:
            df = df[df['High_conf'] == 1]
        
        if date_filter and 'Date' in df.columns:
            try:
                filter_date = pd.to_datetime(date_filter)
                df = df[df['Date'].dt.date == filter_date.date()]
            except:
                pass
        
        # Prepare data for template
        predictions_list = df.to_dict(orient='records')
        
        # Convert dates to strings for template
        for pred in predictions_list:
            if 'Date' in pred and pd.notna(pred['Date']):
                if isinstance(pred['Date'], pd.Timestamp):
                    pred['Date'] = pred['Date'].strftime('%Y-%m-%d')
        
        # Get unique dates for filter
        unique_dates = []
        if 'Date' in df.columns:
            try:
                unique_dates = df['Date'].dt.strftime('%Y-%m-%d').unique().tolist()
            except:
                pass
        
        return render_template('predictions.html',
                             predictions=predictions_list,
                             leagues=LEAGUES,
                             dates=sorted(unique_dates),
                             filters={
                                 'league': league,
                                 'confidence': confidence,
                                 'date': date_filter
                             })
    
    except Exception as e:
        logger.error(f"Error in predictions route: {e}")
        return render_template('predictions.html',
                             predictions=[],
                             leagues=LEAGUES,
                             dates=[],
                             filters={})

@app.route('/results')
def results():
    """Display historical results"""
    try:
        # Get filter parameters
        league = request.args.get('league', '')
        period = request.args.get('period', '')
        correct_only = request.args.get('correct', '') == 'true'
        
        df = prediction_manager.get_all_results()
        
        if df is None or df.empty:
            return render_template('results.html',
                                 results=[],
                                 leagues=LEAGUES,
                                 stats={'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0},
                                 filters={})
        
        # Apply filters
        if league and 'Div' in df.columns:
            df = df[df['Div'] == league]
        
        if period == 'week' and 'Date' in df.columns:
            cutoff = datetime.now() - timedelta(days=7)
            df = df[df['Date'] >= cutoff]
        elif period == 'month' and 'Date' in df.columns:
            cutoff = datetime.now() - timedelta(days=30)
            df = df[df['Date'] >= cutoff]
        
        if correct_only and 'IsCorrect' in df.columns:
            df = df[df['IsCorrect'] == True]
        
        # Calculate statistics for filtered data
        stats = prediction_manager.calculate_statistics(df)
        
        # Prepare data for template
        results_list = df.head(100).to_dict(orient='records')
        
        # Convert dates to strings
        for result in results_list:
            if 'Date' in result and pd.notna(result['Date']):
                if isinstance(result['Date'], pd.Timestamp):
                    result['Date'] = result['Date'].strftime('%Y-%m-%d')
        
        return render_template('results.html',
                             results=results_list,
                             leagues=LEAGUES,
                             stats=stats,
                             filters={
                                 'league': league,
                                 'period': period,
                                 'correct_only': correct_only
                             })
    
    except Exception as e:
        logger.error(f"Error in results route: {e}")
        return render_template('results.html',
                             results=[],
                             leagues=LEAGUES,
                             stats={'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0},
                             filters={})

@app.route('/about')
def about():
    """About page with methodology and FAQ"""
    return render_template('about.html', leagues=LEAGUES)

@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/stats')
def api_stats():
    """API endpoint for live statistics"""
    try:
        results = prediction_manager.get_all_results()
        stats = {
            'all_time': prediction_manager.calculate_statistics(results),
            'last_month': prediction_manager.calculate_statistics(results, 'month'),
            'last_week': prediction_manager.calculate_statistics(results, 'week')
        }
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error in API stats: {e}")
        return jsonify({
            'all_time': {'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0},
            'last_month': {'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0},
            'last_week': {'total_bets': 0, 'correct_bets': 0, 'accuracy': 0, 'roi': 0}
        })

@app.route('/api/predictions/latest')
def api_latest_predictions():
    """API endpoint for latest predictions"""
    try:
        df = prediction_manager.get_latest_predictions()
        if df is None or df.empty:
            return jsonify({'predictions': [], 'count': 0})
        
        # Convert to JSON-serializable format
        if 'Date' in df.columns:
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        predictions = df.head(20).to_dict(orient='records')
        
        return jsonify({
            'predictions': predictions,
            'count': len(df),
            'high_confidence_count': len(df[df.get('High_conf', 0) == 1]) if 'High_conf' in df.columns else 0
        })
    
    except Exception as e:
        logger.error(f"Error in API predictions: {e}")
        return jsonify({'predictions': [], 'count': 0, 'error': str(e)})

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Seite nicht gefunden'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Interner Serverfehler'), 500

# Create sample data on startup if needed
def initialize_app():
    """Initialize app with sample data if needed"""
    try:
        # Check if predictions exist
        if not list(PREDICTION_DIR.glob("*.csv")):
            logger.info("Creating initial sample predictions")
            df = PredictionManager._create_sample_predictions()
            df.to_csv(PREDICTION_DIR / "sample_predictions.csv", index=False)
        
        # Check if results exist
        if not list(RESULT_DIR.glob("*.csv")):
            logger.info("Creating initial sample results")
            df = PredictionManager._create_sample_results()
            df.to_csv(RESULT_DIR / "sample_results.csv", index=False)
            
    except Exception as e:
        logger.error(f"Error initializing app: {e}")

# Initialize on startup
initialize_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
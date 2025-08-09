# app.py
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
    "E0": {"name": "Premier League", "country": "England", "emoji": "🏴󐁧󐁢󐁥󐁮󐁧󐁿"},
    "F1": {"name": "Ligue 1", "country": "France", "emoji": "🇫🇷"},
    "I1": {"name": "Serie A", "country": "Italy", "emoji": "🇮🇹"},
    "SP1": {"name": "La Liga", "country": "Spain", "emoji": "🇪🇸"}
}

class PredictionManager:
    """Manages predictions and results"""
    
    @staticmethod
    def get_latest_predictions():
        """Load the most recent predictions"""
        try:
            prediction_files = sorted(
                [f for f in PREDICTION_DIR.glob("predictions_*.csv")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not prediction_files:
                logger.warning("No prediction files found")
                return pd.DataFrame()
            
            df = pd.read_csv(prediction_files[0])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter out past matches
            now = datetime.now()
            if 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(
                    df['Date'].astype(str) + ' ' + df['Time'],
                    errors='coerce'
                )
                df = df[df['DateTime'] > now]
            
            return df.sort_values(['Date', 'Time']) if not df.empty else df
            
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_all_results():
        """Load all historical results"""
        try:
            result_files = list(RESULT_DIR.glob("result_*.csv"))
            
            if not result_files:
                return pd.DataFrame()
            
            dfs = []
            for file in result_files:
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                dfs.append(df)
            
            all_results = pd.concat(dfs, ignore_index=True)
            return all_results.sort_values('Date', ascending=False)
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_statistics(df, period=None):
        """Calculate betting statistics"""
        if df.empty:
            return {
                'total_bets': 0,
                'correct_bets': 0,
                'accuracy': 0.0,
                'roi': 0.0,
                'best_market': 'N/A',
                'profit': 0.0
            }
        
        if period == 'week':
            # Last 7 days
            cutoff = datetime.now() - timedelta(days=7)
            df = df[df['Date'] >= cutoff]
        elif period == 'month':
            # Last 30 days
            cutoff = datetime.now() - timedelta(days=30)
            df = df[df['Date'] >= cutoff]
        
        # Calculate final bets and correctness
        df['FinalBet'] = df.apply(PredictionManager._get_final_bet, axis=1)
        df['IsCorrect'] = df.apply(
            lambda row: PredictionManager._check_bet_correct(
                row.get('Actual_Result'), 
                row.get('FinalBet')
            ),
            axis=1
        )
        
        total = len(df)
        correct = df['IsCorrect'].sum()
        
        # Calculate ROI (simplified)
        df['Profit'] = df.apply(
            lambda row: (row.get('FinalOdds', 1.5) - 1) if row['IsCorrect'] else -1,
            axis=1
        )
        
        return {
            'total_bets': total,
            'correct_bets': correct,
            'accuracy': round(correct / total * 100, 2) if total > 0 else 0.0,
            'roi': round(df['Profit'].sum() / total * 100, 2) if total > 0 else 0.0,
            'best_market': PredictionManager._get_best_market(df),
            'profit': round(df['Profit'].sum(), 2)
        }
    
    @staticmethod
    def _get_final_bet(row):
        """Determine the final bet based on prediction and odds"""
        pred = row.get('Prediction', '')
        prob_h = row.get('Prob_H', 0)
        prob_d = row.get('Prob_D', 0)
        prob_a = row.get('Prob_A', 0)
        odds_h = row.get('B365H', 2.0)
        odds_a = row.get('B365A', 2.0)
        
        if pred == 'H':
            if prob_h > 0.70 and odds_h < 2.0:
                return 'H'
            elif prob_h + prob_d > 0.85:
                return '1X'
            else:
                return '1X'
        elif pred == 'D':
            if odds_h < odds_a:
                return '1X'
            else:
                return 'X2'
        elif pred == 'A':
            if prob_a > 0.70 and odds_a < 2.0:
                return 'A'
            elif prob_a + prob_d > 0.85:
                return 'X2'
            else:
                return 'X2'
        
        return '1X'  # Default safe bet
    
    @staticmethod
    def _check_bet_correct(actual, predicted):
        """Check if a bet was correct"""
        if pd.isna(actual) or pd.isna(predicted):
            return False
        
        actual = str(actual).upper()
        predicted = str(predicted).upper()
        
        if actual == 'H':
            return predicted in ['H', '1X']
        elif actual == 'D':
            return predicted in ['D', '1X', 'X2']
        elif actual == 'A':
            return predicted in ['A', 'X2']
        
        return False
    
    @staticmethod
    def _get_best_market(df):
        """Determine the best performing market"""
        markets = ['H', 'D', 'A', '1X', 'X2']
        best_accuracy = 0
        best_market = 'N/A'
        
        for market in markets:
            market_df = df[df['FinalBet'] == market]
            if len(market_df) > 5:  # Minimum sample size
                accuracy = market_df['IsCorrect'].mean()
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_market = market
        
        return best_market

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
        upcoming_count = len(predictions) if not predictions.empty else 0
        
        # High confidence predictions
        high_conf_predictions = predictions[predictions.get('High_conf', 0) == 1] if not predictions.empty else pd.DataFrame()
        high_conf_count = len(high_conf_predictions)
        
        return render_template('index.html',
                             stats=stats,
                             upcoming_count=upcoming_count,
                             high_conf_count=high_conf_count,
                             leagues=LEAGUES)
    
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/predictions')
def predictions():
    """Display upcoming predictions"""
    try:
        # Get filter parameters
        league = request.args.get('league', '')
        confidence = request.args.get('confidence', '')
        date_filter = request.args.get('date', '')
        
        df = prediction_manager.get_latest_predictions()
        
        if df.empty:
            return render_template('predictions.html',
                                 predictions=[],
                                 leagues=LEAGUES,
                                 filters={})
        
        # Apply filters
        if league:
            df = df[df['Div'] == league]
        
        if confidence == 'high':
            df = df[df.get('High_conf', 0) == 1]
        
        if date_filter:
            filter_date = pd.to_datetime(date_filter)
            df = df[df['Date'].dt.date == filter_date.date()]
        
        # Prepare data for template
        predictions_list = df.to_dict(orient='records')
        
        # Get unique dates for filter
        unique_dates = df['Date'].dt.strftime('%Y-%m-%d').unique().tolist() if not df.empty else []
        
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
        return render_template('error.html', error=str(e)), 500

@app.route('/results')
def results():
    """Display historical results"""
    try:
        # Get filter parameters
        league = request.args.get('league', '')
        period = request.args.get('period', '')
        correct_only = request.args.get('correct', '') == 'true'
        
        df = prediction_manager.get_all_results()
        
        if df.empty:
            return render_template('results.html',
                                 results=[],
                                 leagues=LEAGUES,
                                 stats={})
        
        # Apply filters
        if league:
            df = df[df['Div'] == league]
        
        if period == 'week':
            cutoff = datetime.now() - timedelta(days=7)
            df = df[df['Date'] >= cutoff]
        elif period == 'month':
            cutoff = datetime.now() - timedelta(days=30)
            df = df[df['Date'] >= cutoff]
        
        # Calculate correctness
        df['FinalBet'] = df.apply(prediction_manager._get_final_bet, axis=1)
        df['IsCorrect'] = df.apply(
            lambda row: prediction_manager._check_bet_correct(
                row.get('Actual_Result'),
                row.get('FinalBet')
            ),
            axis=1
        )
        
        if correct_only:
            df = df[df['IsCorrect'] == True]
        
        # Calculate statistics for filtered data
        stats = prediction_manager.calculate_statistics(df)
        
        # Prepare data for template
        results_list = df.head(100).to_dict(orient='records')  # Limit to 100 most recent
        
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
        return render_template('error.html', error=str(e)), 500

@app.route('/about')
def about():
    """About page with methodology and FAQ"""
    return render_template('about.html', leagues=LEAGUES)

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
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/latest')
def api_latest_predictions():
    """API endpoint for latest predictions"""
    try:
        df = prediction_manager.get_latest_predictions()
        if df.empty:
            return jsonify({'predictions': [], 'count': 0})
        
        # Convert to JSON-serializable format
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        predictions = df.head(20).to_dict(orient='records')
        
        return jsonify({
            'predictions': predictions,
            'count': len(df),
            'high_confidence_count': len(df[df.get('High_conf', 0) == 1])
        })
    
    except Exception as e:
        logger.error(f"Error in API predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

## 2. prediction_agent.py - Automated Prediction Runner
```python
# prediction_agent.py
import os
import sys
import time
import schedule
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.predictor import FootballPredictor
from models.comparator import ResultComparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PredictionAgent:
    """Automated agent for running predictions and comparisons"""
    
    def __init__(self):
        self.predictor = FootballPredictor()
        self.comparator = ResultComparator()
        logger.info("Prediction Agent initialized")
    
    def run_predictions(self):
        """Run the prediction pipeline"""
        try:
            logger.info("Starting prediction pipeline...")
            
            # Download latest data
            self.predictor.download_data()
            
            # Train model and generate predictions
            predictions = self.predictor.generate_predictions()
            
            if predictions is not None and not predictions.empty:
                logger.info(f"Generated {len(predictions)} predictions")
                
                # Save predictions
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = Path('predictions') / f'predictions_{timestamp}.csv'
                predictions.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to {output_file}")
            else:
                logger.warning("No predictions generated")
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
    
    def run_comparison(self):
        """Run the results comparison"""
        try:
            logger.info("Starting results comparison...")
            
            # Download and compare results
            comparison_results = self.comparator.compare_results()
            
            if comparison_results is not None and not comparison_results.empty:
                logger.info(f"Compared {len(comparison_results)} results")
                
                # Save comparison results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = Path('results') / f'result_{timestamp}.csv'
                comparison_results.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            else:
                logger.warning("No results to compare")
            
        except Exception as e:
            logger.error(f"Error in comparison pipeline: {e}")
    
    def run_all(self):
        """Run both prediction and comparison"""
        logger.info("Running full pipeline...")
        self.run_predictions()
        time.sleep(5)  # Brief pause
        self.run_comparison()
        logger.info("Full pipeline completed")

def main():
    """Main function to run the agent"""
    agent = PredictionAgent()
    
    # Schedule tasks
    # Run predictions on Tuesday and Friday at 14:00
    schedule.every().tuesday.at("14:00").do(agent.run_predictions)
    schedule.every().friday.at("18:00").do(agent.run_predictions)
    
    # Run comparison daily at 23:00
    schedule.every().day.at("23:00").do(agent.run_comparison)
    
    # Run immediately on startup
    agent.run_all()
    
    logger.info("Prediction Agent started. Waiting for scheduled tasks...")
    
    # Keep the agent running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
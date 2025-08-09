import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ResultComparator:
    """Compare predictions with actual results"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.prediction_dir = self.base_dir / "predictions"
        self.result_dir = self.base_dir / "results"
        self.data_dir = self.base_dir / "data"
        self.current_season = "2025_26"
        
        # Create directories
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ResultComparator initialized")
    
    def download_latest_results(self):
        """Download the latest match results"""
        try:
            season_dir = self.data_dir / self.current_season
            
            # Download current season data
            url = f"https://www.football-data.co.uk/mmz4281/2526/D1.csv"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            results_df = pd.read_csv(pd.io.common.StringIO(response.text))
            
            # Filter completed matches
            results_df = results_df.dropna(subset=['FTR'])
            results_df['Date'] = pd.to_datetime(results_df['Date'], dayfirst=True)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error downloading results: {e}")
            return pd.DataFrame()
    
    def compare_results(self):
        """Compare predictions with actual results"""
        try:
            # Get latest results
            actual_results = self.download_latest_results()
            
            if actual_results.empty:
                logger.warning("No actual results found")
                return pd.DataFrame()
            
            # Get all prediction files
            prediction_files = sorted(
                self.prediction_dir.glob("predictions_*.csv"),
                key=lambda x: x.stat().st_mtime
            )
            
            if not prediction_files:
                logger.warning("No prediction files found")
                return pd.DataFrame()
            
            comparisons = []
            
            for pred_file in prediction_files:
                predictions = pd.read_csv(pred_file)
                predictions['Date'] = pd.to_datetime(predictions['Date'])
                
                # Match predictions with results
                for _, pred in predictions.iterrows():
                    # Find corresponding result
                    match_date = pred['Date'].date()
                    home_team = pred['HomeTeam']
                    away_team = pred['AwayTeam']
                    
                    result = actual_results[
                        (actual_results['Date'].dt.date == match_date) &
                        (actual_results['HomeTeam'] == home_team) &
                        (actual_results['AwayTeam'] == away_team)
                    ]
                    
                    if not result.empty:
                        result_row = result.iloc[0]
                        
                        comparison = {
                            'Date': pred['Date'],
                            'Time': pred.get('Time', '15:30'),
                            'Div': pred['Div'],
                            'HomeTeam': home_team,
                            'AwayTeam': away_team,
                            'Prediction': pred['Prediction'],
                            'Actual_Result': result_row['FTR'],
                            'Score': f"{int(result_row['FTHG'])}-{int(result_row['FTAG'])}",
                            'Prob_H': pred.get('Prob_H', 0),
                            'Prob_D': pred.get('Prob_D', 0),
                            'Prob_A': pred.get('Prob_A', 0),
                            'B365H': pred.get('B365H', 0),
                            'B365D': pred.get('B365D', 0),
                            'B365A': pred.get('B365A', 0),
                            'High_conf': pred.get('High_conf', 0),
                            'Prediction_Correct': 1 if pred['Prediction'] == result_row['FTR'] else 0
                        }
                        
                        comparisons.append(comparison)
            
            if comparisons:
                df = pd.DataFrame(comparisons)
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
                
                # Sort by date
                df = df.sort_values('Date', ascending=False)
                
                logger.info(f"Compared {len(df)} matches")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return pd.DataFrame()
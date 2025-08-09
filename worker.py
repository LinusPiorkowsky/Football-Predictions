# worker.py - Intelligent Hourly Worker
import os
import time
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import logging
import requests
from models.predictor import FootballPredictor
from models.comparator import ResultComparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntelligentWorker:
    """Intelligent worker that checks for new data and only updates when necessary"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.prediction_dir = self.base_dir / "predictions"
        self.result_dir = self.base_dir / "results"
        self.cache_dir = self.base_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files for tracking
        self.fixtures_cache = self.cache_dir / "fixtures_hash.json"
        self.results_cache = self.cache_dir / "results_hash.json"
        self.last_check_file = self.cache_dir / "last_check.json"
        
        # Initialize components
        self.predictor = FootballPredictor()
        self.comparator = ResultComparator()
        
        logger.info("IntelligentWorker initialized")
    
    def get_file_hash(self, content):
        """Generate hash of content to detect changes"""
        if isinstance(content, pd.DataFrame):
            content = content.to_json(orient='records', date_format='iso')
        elif not isinstance(content, str):
            content = str(content)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_cache(self, cache_file):
        """Load cache data"""
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache {cache_file}: {e}")
        return {}
    
    def save_cache(self, cache_file, data):
        """Save cache data"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache {cache_file}: {e}")
    
    def check_for_new_fixtures(self):
        """Check if there are new fixtures available"""
        try:
            logger.info("Checking for new fixtures...")
            
            # Download latest fixtures
            fixtures_url = "https://www.football-data.co.uk/fixtures.csv"
            response = requests.get(fixtures_url, timeout=30)
            
            if response.status_code != 200:
                logger.warning("Could not download fixtures")
                return False
            
            # Calculate hash of new fixtures
            new_hash = self.get_file_hash(response.text)
            
            # Load previous hash
            cache_data = self.load_cache(self.fixtures_cache)
            old_hash = cache_data.get('hash', '')
            
            # Check if fixtures have changed
            if new_hash != old_hash:
                logger.info("✅ New fixtures detected!")
                
                # Save new fixtures
                fixtures_file = self.predictor.season_dir / "fixtures.csv"
                with open(fixtures_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Update cache
                cache_data = {
                    'hash': new_hash,
                    'timestamp': datetime.now().isoformat(),
                    'file': str(fixtures_file)
                }
                self.save_cache(self.fixtures_cache, cache_data)
                
                return True
            else:
                logger.info("No new fixtures found")
                return False
                
        except Exception as e:
            logger.error(f"Error checking fixtures: {e}")
            return False
    
    def check_for_new_results(self):
        """Check if there are new match results"""
        try:
            logger.info("Checking for new results...")
            
            # Download latest results for current season
            results_url = "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
            response = requests.get(results_url, timeout=30)
            
            if response.status_code != 200:
                logger.warning("Could not download results")
                return False
            
            # Parse and filter only completed matches
            try:
                df = pd.read_csv(pd.io.common.StringIO(response.text))
                completed_matches = df[df['FTR'].notna()]
                
                if completed_matches.empty:
                    logger.info("No completed matches yet")
                    return False
                
                # Calculate hash of completed matches
                new_hash = self.get_file_hash(completed_matches)
                
                # Load previous hash
                cache_data = self.load_cache(self.results_cache)
                old_hash = cache_data.get('hash', '')
                
                # Check if results have changed
                if new_hash != old_hash:
                    logger.info(f"✅ New results detected! {len(completed_matches)} completed matches")
                    
                    # Update cache
                    cache_data = {
                        'hash': new_hash,
                        'timestamp': datetime.now().isoformat(),
                        'match_count': len(completed_matches)
                    }
                    self.save_cache(self.results_cache, cache_data)
                    
                    return True
                else:
                    logger.info("No new results found")
                    return False
                    
            except Exception as e:
                logger.error(f"Error parsing results: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking results: {e}")
            return False
    
    def run_predictions_if_needed(self):
        """Run predictions only if there are new fixtures"""
        try:
            if self.check_for_new_fixtures():
                logger.info("🔮 Generating new predictions...")
                
                # Run prediction
                self.predictor.download_data()
                predictions = self.predictor.generate_predictions()
                
                if predictions is not None and not predictions.empty:
                    # Remove duplicates - only keep matches not already predicted
                    existing_predictions = self.get_existing_predictions()
                    
                    if existing_predictions is not None and not existing_predictions.empty:
                        # Create unique key for each match
                        predictions['match_key'] = (predictions['Date'].astype(str) + '_' + 
                                                   predictions['HomeTeam'] + '_' + 
                                                   predictions['AwayTeam'])
                        existing_predictions['match_key'] = (existing_predictions['Date'].astype(str) + '_' + 
                                                            existing_predictions['HomeTeam'] + '_' + 
                                                            existing_predictions['AwayTeam'])
                        
                        # Filter out already predicted matches
                        new_predictions = predictions[~predictions['match_key'].isin(existing_predictions['match_key'])]
                        
                        if not new_predictions.empty:
                            # Save only new predictions
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_file = self.prediction_dir / f'predictions_{timestamp}.csv'
                            new_predictions.drop('match_key', axis=1).to_csv(output_file, index=False)
                            logger.info(f"✅ Saved {len(new_predictions)} NEW predictions to {output_file}")
                            
                            # Also update the "latest" file
                            latest_file = self.prediction_dir / 'latest_predictions.csv'
                            all_predictions = pd.concat([existing_predictions.drop('match_key', axis=1), 
                                                        new_predictions.drop('match_key', axis=1)])
                            all_predictions.to_csv(latest_file, index=False)
                            
                            return True
                        else:
                            logger.info("All matches already predicted")
                    else:
                        # First run - save all predictions
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_file = self.prediction_dir / f'predictions_{timestamp}.csv'
                        predictions.to_csv(output_file, index=False)
                        
                        # Also save as latest
                        latest_file = self.prediction_dir / 'latest_predictions.csv'
                        predictions.to_csv(latest_file, index=False)
                        
                        logger.info(f"✅ Saved {len(predictions)} predictions (first run)")
                        return True
                else:
                    logger.warning("No predictions generated")
                    
        except Exception as e:
            logger.error(f"Error in prediction process: {e}")
        
        return False
    
    def run_comparison_if_needed(self):
        """Run comparison only if there are new results"""
        try:
            if self.check_for_new_results():
                logger.info("📊 Comparing predictions with new results...")
                
                # Run comparison
                comparison_results = self.comparator.compare_results()
                
                if comparison_results is not None and not comparison_results.empty:
                    # Check for new results not already saved
                    existing_results = self.get_existing_results()
                    
                    if existing_results is not None and not existing_results.empty:
                        # Create unique key
                        comparison_results['match_key'] = (comparison_results['Date'].astype(str) + '_' + 
                                                          comparison_results['HomeTeam'] + '_' + 
                                                          comparison_results['AwayTeam'])
                        existing_results['match_key'] = (existing_results['Date'].astype(str) + '_' + 
                                                        existing_results['HomeTeam'] + '_' + 
                                                        existing_results['AwayTeam'])
                        
                        # Filter out already compared matches
                        new_results = comparison_results[~comparison_results['match_key'].isin(existing_results['match_key'])]
                        
                        if not new_results.empty:
                            # Save only new results
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_file = self.result_dir / f'results_{timestamp}.csv'
                            new_results.drop('match_key', axis=1).to_csv(output_file, index=False)
                            logger.info(f"✅ Saved {len(new_results)} NEW results to {output_file}")
                            
                            # Update latest file
                            latest_file = self.result_dir / 'latest_results.csv'
                            all_results = pd.concat([existing_results.drop('match_key', axis=1), 
                                                    new_results.drop('match_key', axis=1)])
                            all_results.to_csv(latest_file, index=False)
                            
                            # Calculate and log accuracy
                            accuracy = (new_results['Prediction_Correct'].sum() / len(new_results) * 100)
                            logger.info(f"📈 Accuracy for new results: {accuracy:.2f}%")
                            
                            return True
                        else:
                            logger.info("All results already compared")
                    else:
                        # First run - save all results
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_file = self.result_dir / f'results_{timestamp}.csv'
                        comparison_results.to_csv(output_file, index=False)
                        
                        # Also save as latest
                        latest_file = self.result_dir / 'latest_results.csv'
                        comparison_results.to_csv(latest_file, index=False)
                        
                        logger.info(f"✅ Saved {len(comparison_results)} results (first run)")
                        return True
                else:
                    logger.warning("No results to compare")
                    
        except Exception as e:
            logger.error(f"Error in comparison process: {e}")
        
        return False
    
    def get_existing_predictions(self):
        """Get all existing predictions"""
        try:
            # First try to load latest file
            latest_file = self.prediction_dir / 'latest_predictions.csv'
            if latest_file.exists():
                return pd.read_csv(latest_file)
            
            # Otherwise combine all prediction files
            prediction_files = list(self.prediction_dir.glob("predictions_*.csv"))
            if prediction_files:
                dfs = []
                for file in prediction_files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except:
                        continue
                
                if dfs:
                    return pd.concat(dfs, ignore_index=True).drop_duplicates()
                    
        except Exception as e:
            logger.error(f"Error loading existing predictions: {e}")
        
        return pd.DataFrame()
    
    def get_existing_results(self):
        """Get all existing results"""
        try:
            # First try to load latest file
            latest_file = self.result_dir / 'latest_results.csv'
            if latest_file.exists():
                return pd.read_csv(latest_file)
            
            # Otherwise combine all result files
            result_files = list(self.result_dir.glob("results_*.csv"))
            if result_files:
                dfs = []
                for file in result_files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except:
                        continue
                
                if dfs:
                    return pd.concat(dfs, ignore_index=True).drop_duplicates()
                    
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
        
        return pd.DataFrame()
    
    def hourly_check(self):
        """Main hourly check function"""
        logger.info("=" * 50)
        logger.info(f"🕐 Starting hourly check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Track if anything was updated
        predictions_updated = self.run_predictions_if_needed()
        results_updated = self.run_comparison_if_needed()
        
        # Save last check time
        check_data = {
            'timestamp': datetime.now().isoformat(),
            'predictions_updated': predictions_updated,
            'results_updated': results_updated
        }
        self.save_cache(self.last_check_file, check_data)
        
        if predictions_updated or results_updated:
            logger.info("✅ Updates completed successfully!")
        else:
            logger.info("✔️ No updates needed - all data is current")
        
        logger.info("=" * 50 + "\n")
    
    def run_continuous(self):
        """Run continuous hourly checks"""
        logger.info("🚀 Starting Intelligent Worker - Hourly Checks")
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Checking every hour for new fixtures and results")
        
        # Run immediately on startup
        self.hourly_check()
        
        # Schedule hourly checks
        schedule.every().hour.at(":00").do(self.hourly_check)
        
        # Also check at specific times for better coverage
        schedule.every().hour.at(":30").do(self.hourly_check)
        
        logger.info("⏰ Scheduler started - checking every 30 minutes")
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for pending tasks

def main():
    """Main entry point"""
    worker = IntelligentWorker()
    
    try:
        worker.run_continuous()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker crashed: {e}")
        # Restart after crash
        time.sleep(60)
        main()

if __name__ == "__main__":
    main()
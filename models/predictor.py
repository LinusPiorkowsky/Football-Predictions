import os
import pandas as pd
import numpy as np
import requests
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

class FootballPredictor:
    """Enhanced football match predictor using ensemble methods"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.current_season = "2025_26"
        self.season_dir = self.data_dir / self.current_season
        
        # Create directories
        self.season_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.scaler = StandardScaler()
        self.model = None
        
        # Data URLs
        self.fixtures_url = "https://www.football-data.co.uk/fixtures.csv"
        self.data_url = "https://www.football-data.co.uk/mmz4281/2526/data.zip"
        
        # Focus on Bundesliga
        self.leagues = ['D1']
        
        logger.info("FootballPredictor initialized")
    
    def download_data(self):
        """Download latest fixtures and results"""
        try:
            # Download fixtures
            logger.info("Downloading fixtures...")
            response = requests.get(self.fixtures_url, timeout=30)
            response.raise_for_status()
            
            fixtures_df = pd.read_csv(pd.io.common.StringIO(response.text))
            fixtures_df = fixtures_df[fixtures_df['Div'].isin(self.leagues)]
            
            fixtures_file = self.season_dir / "fixtures.csv"
            fixtures_df.to_csv(fixtures_file, index=False)
            logger.info(f"Fixtures saved to {fixtures_file}")
            
            # Download current season data
            logger.info("Downloading current season data...")
            response = requests.get(self.data_url, timeout=30)
            response.raise_for_status()
            
            zip_file = self.season_dir / "data.zip"
            with open(zip_file, 'wb') as f:
                f.write(response.content)
            
            # Extract relevant files
            with zipfile.ZipFile(zip_file, 'r') as zf:
                for member in zf.namelist():
                    if any(member.endswith(f"{league}.csv") for league in self.leagues):
                        zf.extract(member, self.season_dir)
            
            zip_file.unlink()  # Remove zip file
            logger.info("Data download completed")
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def load_historical_data(self):
        """Load and prepare historical data"""
        dfs = []
        
        # Load data from multiple seasons
        seasons = ['2023_24', '2024_25', '2025_26']
        for season in seasons:
            season_path = self.data_dir / season
            if season_path.exists():
                for league in self.leagues:
                    file_path = season_path / f"{league}.csv"
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        df['Season'] = season
                        df['League'] = league
                        dfs.append(df)
        
        if not dfs:
            raise ValueError("No historical data found")
        
        # Combine all data
        data = pd.concat(dfs, ignore_index=True)
        
        # Clean data
        data = data[data['FTR'].isin(['H', 'D', 'A'])]
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
        data = data.dropna(subset=['Date', 'FTR'])
        
        logger.info(f"Loaded {len(data)} historical matches")
        return data
    
    def engineer_features(self, df):
        """Create features for the model"""
        features = pd.DataFrame()
        
        # Basic statistics
        for team in df['HomeTeam'].unique():
            team_home = df[df['HomeTeam'] == team]
            team_away = df[df['AwayTeam'] == team]
            
            # Home performance
            features.loc[team, 'home_win_rate'] = (team_home['FTR'] == 'H').mean()
            features.loc[team, 'home_goals_avg'] = team_home['FTHG'].mean()
            features.loc[team, 'home_conceded_avg'] = team_home['FTAG'].mean()
            
            # Away performance
            features.loc[team, 'away_win_rate'] = (team_away['FTR'] == 'A').mean()
            features.loc[team, 'away_goals_avg'] = team_away['FTAG'].mean()
            features.loc[team, 'away_conceded_avg'] = team_away['FTHG'].mean()
            
            # Overall form (last 5 games)
            recent_games = pd.concat([team_home.tail(5), team_away.tail(5)])
            if len(recent_games) > 0:
                points = 0
                for _, game in recent_games.iterrows():
                    if game['HomeTeam'] == team:
                        points += 3 if game['FTR'] == 'H' else (1 if game['FTR'] == 'D' else 0)
                    else:
                        points += 3 if game['FTR'] == 'A' else (1 if game['FTR'] == 'D' else 0)
                features.loc[team, 'recent_form'] = points / len(recent_games)
            else:
                features.loc[team, 'recent_form'] = 1.0
        
        return features.fillna(0)
    
    def prepare_match_features(self, home_team, away_team, team_features):
        """Prepare features for a single match"""
        if home_team not in team_features.index:
            # Default features for new teams
            home_features = pd.Series(0, index=team_features.columns)
        else:
            home_features = team_features.loc[home_team]
        
        if away_team not in team_features.index:
            away_features = pd.Series(0, index=team_features.columns)
        else:
            away_features = team_features.loc[away_team]
        
        # Combine features
        match_features = pd.concat([
            home_features.add_prefix('home_'),
            away_features.add_prefix('away_')
        ])
        
        # Add comparative features
        match_features['form_diff'] = home_features['recent_form'] - away_features['recent_form']
        match_features['home_advantage'] = 0.1  # Small home advantage factor
        
        return match_features
    
    def train_model(self, X, y):
        """Train the prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create ensemble model
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train models
        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)
        
        # Combine predictions (simple voting)
        rf_pred = rf.predict_proba(X_test_scaled)
        gb_pred = gb.predict_proba(X_test_scaled)
        
        ensemble_pred = (rf_pred + gb_pred) / 2
        y_pred = np.argmax(ensemble_pred, axis=1)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.3f}")
        
        # Store the best model
        self.model = rf  # Using RF as primary model
        
        return accuracy
    
    def generate_predictions(self):
        """Generate predictions for upcoming matches"""
        try:
            # Load historical data
            historical_data = self.load_historical_data()
            
            # Engineer features
            team_features = self.engineer_features(historical_data)
            
            # Prepare training data
            X_list = []
            y_list = []
            
            for _, match in historical_data.iterrows():
                try:
                    features = self.prepare_match_features(
                        match['HomeTeam'],
                        match['AwayTeam'],
                        team_features
                    )
                    X_list.append(features)
                    
                    # Target encoding
                    result = match['FTR']
                    y_list.append(['H', 'D', 'A'].index(result))
                except:
                    continue
            
            X = pd.DataFrame(X_list)
            y = np.array(y_list)
            
            # Train model
            self.train_model(X, y)
            
            # Load fixtures
            fixtures_file = self.season_dir / "fixtures.csv"
            if not fixtures_file.exists():
                logger.warning("No fixtures file found")
                return pd.DataFrame()
            
            fixtures = pd.read_csv(fixtures_file)
            fixtures = fixtures[fixtures['Div'].isin(self.leagues)]
            fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
            
            # Filter future matches
            today = datetime.now().date()
            fixtures = fixtures[fixtures['Date'].dt.date >= today]
            
            if fixtures.empty:
                logger.warning("No upcoming fixtures")
                return pd.DataFrame()
            
            # Generate predictions
            predictions = []
            for _, match in fixtures.iterrows():
                try:
                    features = self.prepare_match_features(
                        match['HomeTeam'],
                        match['AwayTeam'],
                        team_features
                    )
                    
                    X_pred = self.scaler.transform([features])
                    
                    # Get prediction and probabilities
                    pred_class = self.model.predict(X_pred)[0]
                    pred_proba = self.model.predict_proba(X_pred)[0]
                    
                    prediction_row = {
                        'Div': match['Div'],
                        'Date': match['Date'],
                        'Time': match.get('Time', '15:30'),
                        'HomeTeam': match['HomeTeam'],
                        'AwayTeam': match['AwayTeam'],
                        'Prediction': ['H', 'D', 'A'][pred_class],
                        'Prob_H': pred_proba[0],
                        'Prob_D': pred_proba[1],
                        'Prob_A': pred_proba[2],
                        'B365H': match.get('B365H', 2.0),
                        'B365D': match.get('B365D', 3.5),
                        'B365A': match.get('B365A', 2.5),
                        'Confidence': max(pred_proba),
                        'High_conf': 1 if max(pred_proba) > 0.65 else 0
                    }
                    
                    # Calculate double chance odds
                    prediction_row['1X_prob'] = pred_proba[0] + pred_proba[1]
                    prediction_row['X2_prob'] = pred_proba[1] + pred_proba[2]
                    prediction_row['1X_odds'] = 1.0 / prediction_row['1X_prob'] if prediction_row['1X_prob'] > 0 else 10.0
                    prediction_row['X2_odds'] = 1.0 / prediction_row['X2_prob'] if prediction_row['X2_prob'] > 0 else 10.0
                    
                    predictions.append(prediction_row)
                    
                except Exception as e:
                    logger.warning(f"Error predicting match {match['HomeTeam']} vs {match['AwayTeam']}: {e}")
                    continue
            
            if predictions:
                df = pd.DataFrame(predictions)
                df['Weekday'] = df['Date'].dt.day_name()
                return df.sort_values(['Date', 'Time'])
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return pd.DataFrame()
"""
Enhanced AI Recommendation Engine with sophisticated machine learning algorithms
"""
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PlayerRecommendation:
    """Enhanced player recommendation with detailed analysis"""
    player_id: int
    web_name: str
    team_name: str
    position: str
    current_price: float
    predicted_points: float
    confidence_score: float
    value_score: float
    form_score: float
    fixture_score: float
    ownership_score: float
    risk_level: str
    reasoning: List[str]
    transfer_priority: int
    expected_roi: float


@dataclass
class TeamRecommendation:
    """Team-level recommendations"""
    recommended_formation: str
    captain_suggestion: PlayerRecommendation
    vice_captain_suggestion: PlayerRecommendation
    transfer_suggestions: List[PlayerRecommendation]
    chip_recommendation: Dict[str, str]
    overall_score: float
    key_insights: List[str]


class AdvancedMLRecommendationEngine:
    """Advanced ML-powered recommendation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.models_trained = False
        
        # Model cache directory
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Feature engineering settings
        self.feature_columns = [
            'total_points', 'form', 'now_cost', 'selected_by_percent',
            'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'own_goals', 'penalties_saved',
            'penalties_missed', 'yellow_cards', 'red_cards',
            'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat',
            'ict_index', 'starts', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded'
        ]
        
        # Weights for different aspects
        self.weights = {
            'points_prediction': 0.35,
            'value_efficiency': 0.25,
            'form_trend': 0.20,
            'fixture_difficulty': 0.15,
            'ownership_factor': 0.05
        }
    
    def prepare_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for ML models"""
        try:
            df = players_df.copy()
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Create advanced features
            
            # 1. Efficiency metrics
            df['points_per_million'] = df['total_points'] / df['now_cost'].replace(0, 1)
            df['points_per_game'] = df['total_points'] / df['starts'].replace(0, 1)
            df['minutes_per_game'] = df['minutes'] / df['starts'].replace(0, 1)
            
            # 2. Goal involvement (for attacking players)
            df['goal_involvement'] = df['goals_scored'] + df['assists']
            df['goal_involvement_per_game'] = df['goal_involvement'] / df['starts'].replace(0, 1)
            
            # 3. Defensive metrics (for defensive players)
            df['clean_sheet_percentage'] = df['clean_sheets'] / df['starts'].replace(0, 1)
            df['defensive_actions'] = df.get('tackles', 0) + df.get('interceptions', 0) + df.get('clearances', 0)
            
            # 4. Expected performance metrics
            if 'expected_goals' in df.columns:
                df['xg_overperformance'] = df['goals_scored'] - df['expected_goals']
                df['xg_per_90'] = df['expected_goals'] / (df['minutes'] / 90).replace(0, 1)
            
            if 'expected_assists' in df.columns:
                df['xa_overperformance'] = df['assists'] - df['expected_assists']
                df['xa_per_90'] = df['expected_assists'] / (df['minutes'] / 90).replace(0, 1)
            
            # 5. Form and consistency metrics
            df['form_numeric'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
            df['bonus_rate'] = df['bonus'] / df['starts'].replace(0, 1)
            df['consistency_score'] = df['bps'] / df['starts'].replace(0, 1)
            
            # 6. Market metrics
            df['ownership_tier'] = pd.cut(df['selected_by_percent'], 
                                        bins=[0, 5, 15, 30, 100], 
                                        labels=['differential', 'low', 'medium', 'high'])
            df['price_tier'] = pd.cut(df['now_cost'], 
                                    bins=[0, 5.5, 8.0, 11.0, 15.0], 
                                    labels=['budget', 'mid', 'premium', 'elite'])
            
            # 7. Position-specific features
            df['position_rank'] = df.groupby('element_type')['total_points'].rank(ascending=False)
            df['position_value_rank'] = df.groupby('element_type')['points_per_million'].rank(ascending=False)
            
            # 8. Team strength features (if available)
            if 'team_strength' in df.columns:
                df['strength_adjusted_points'] = df['total_points'] / df['team_strength'].replace(0, 1)
            
            # Encode categorical variables
            le_team = LabelEncoder()
            le_position = LabelEncoder()
            
            if 'team_name' in df.columns:
                df['team_encoded'] = le_team.fit_transform(df['team_name'].astype(str))
            if 'position_name' in df.columns:
                df['position_encoded'] = le_position.fit_transform(df['position_name'].astype(str))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation: {e}")
            return players_df
    
    def train_models(self, players_df: pd.DataFrame, force_retrain: bool = False) -> bool:
        """Train ML models for different prediction tasks"""
        try:
            model_file = self.model_dir / "recommendation_models.pkl"
            
            # Load existing models if available and not forcing retrain
            if not force_retrain and model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        saved_data = pickle.load(f)
                        self.models = saved_data['models']
                        self.scalers = saved_data['scalers']
                        self.feature_importance = saved_data['feature_importance']
                        self.models_trained = True
                        return True
                except Exception as e:
                    self.logger.warning(f"Could not load saved models: {e}")
            
            # Prepare features
            df = self.prepare_features(players_df)
            
            # Select features for training
            feature_cols = [col for col in self.feature_columns if col in df.columns]
            feature_cols.extend([
                'points_per_million', 'points_per_game', 'goal_involvement',
                'clean_sheet_percentage', 'form_numeric', 'position_rank',
                'team_encoded', 'position_encoded'
            ])
            
            # Remove any remaining non-numeric columns
            feature_cols = [col for col in feature_cols if col in df.columns and df[col].dtype in ['int64', 'float64']]
            
            if len(feature_cols) < 10:
                self.logger.warning("Insufficient features for training")
                return False
            
            X = df[feature_cols].fillna(0)
            
            # Train multiple models for different targets
            targets = {
                'next_gw_points': 'ep_next' if 'ep_next' in df.columns else 'form_numeric',
                'total_points': 'total_points',
                'value_score': 'points_per_million'
            }
            
            for target_name, target_col in targets.items():
                if target_col not in df.columns:
                    continue
                
                y = df[target_col].fillna(0)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train ensemble model
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
                
                # Fit models
                rf_model.fit(X_scaled, y)
                gb_model.fit(X_scaled, y)
                
                # Store models and scalers
                self.models[f'{target_name}_rf'] = rf_model
                self.models[f'{target_name}_gb'] = gb_model
                self.scalers[target_name] = scaler
                
                # Feature importance
                self.feature_importance[target_name] = dict(zip(feature_cols, rf_model.feature_importances_))
                
                # Cross-validation score
                cv_score = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
                self.logger.info(f"Model {target_name} CV MAE: {-cv_score.mean():.3f} ± {cv_score.std():.3f}")
            
            # Save models
            save_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'feature_columns': feature_cols
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.models_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False
    
    def predict_player_performance(self, player_data: Dict, position: str) -> Dict[str, float]:
        """Predict individual player performance using trained models"""
        try:
            if not self.models_trained:
                return {'predicted_points': 0, 'confidence': 0}
            
            # Create feature vector for single player
            features = []
            for col in self.feature_columns:
                features.append(player_data.get(col, 0))
            
            # Add engineered features
            points_per_million = player_data.get('total_points', 0) / max(player_data.get('now_cost', 1), 1)
            features.extend([
                points_per_million,
                player_data.get('total_points', 0) / max(player_data.get('starts', 1), 1),
                player_data.get('goals_scored', 0) + player_data.get('assists', 0),
                player_data.get('clean_sheets', 0) / max(player_data.get('starts', 1), 1),
                float(player_data.get('form', 0)),
                player_data.get('position_rank', 0),
                player_data.get('team_encoded', 0),
                player_data.get('position_encoded', 0)
            ])
            
            features = np.array(features).reshape(1, -1)
            
            # Make predictions
            predictions = {}
            
            if 'next_gw_points' in self.scalers:
                scaler = self.scalers['next_gw_points']
                features_scaled = scaler.transform(features)
                
                rf_pred = self.models['next_gw_points_rf'].predict(features_scaled)[0]
                gb_pred = self.models['next_gw_points_gb'].predict(features_scaled)[0]
                
                # Ensemble prediction
                predictions['predicted_points'] = (rf_pred + gb_pred) / 2
                predictions['confidence'] = 1 - abs(rf_pred - gb_pred) / max(abs(rf_pred), abs(gb_pred), 1)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in player prediction: {e}")
            return {'predicted_points': 0, 'confidence': 0}
    
    def calculate_comprehensive_scores(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive scores for all players"""
        try:
            df = players_df.copy()
            
            # Ensure required columns exist
            required_cols = ['total_points', 'now_cost', 'form', 'selected_by_percent']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # 1. Value Score (Points per Million with form adjustment)
            df['base_value'] = df['total_points'] / df['now_cost'].replace(0, 1)
            df['form_multiplier'] = 1 + (pd.to_numeric(df['form'], errors='coerce').fillna(0) - 5) * 0.1
            df['value_score'] = df['base_value'] * df['form_multiplier']
            
            # 2. Form Score (Recent performance trend)
            df['form_numeric'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
            df['form_score'] = np.clip(df['form_numeric'] / 10, 0, 1)
            
            # 3. Ownership Score (Differential potential)
            df['ownership_score'] = np.where(
                df['selected_by_percent'] < 5, 1.0,  # High differential value
                np.where(df['selected_by_percent'] < 15, 0.8,  # Medium differential
                np.where(df['selected_by_percent'] < 30, 0.6,  # Template player
                0.3))  # Highly owned
            )
            
            # 4. Fixture Score (placeholder - would integrate with fixture difficulty)
            df['fixture_score'] = 0.7  # Default neutral score
            
            # 5. Risk Level Assessment
            df['minutes_reliability'] = np.clip(df.get('minutes', 0) / 2500, 0, 1)  # Based on season minutes
            df['price_risk'] = np.where(df['now_cost'] > 10, 0.3, 0.7)  # Premium players higher risk
            df['rotation_risk'] = 1 - df['minutes_reliability']
            
            df['risk_score'] = (df['minutes_reliability'] * 0.5 + 
                              df['price_risk'] * 0.3 + 
                              (1 - df['rotation_risk']) * 0.2)
            
            df['risk_level'] = pd.cut(df['risk_score'], 
                                    bins=[0, 0.3, 0.6, 1.0], 
                                    labels=['High', 'Medium', 'Low'])
            
            # 6. Overall Recommendation Score
            df['recommendation_score'] = (
                df['value_score'] * self.weights['value_efficiency'] +
                df['form_score'] * self.weights['form_trend'] +
                df['fixture_score'] * self.weights['fixture_difficulty'] +
                df['ownership_score'] * self.weights['ownership_factor'] +
                df['risk_score'] * 0.1
            )
            
            # Normalize to 0-100 scale
            df['recommendation_score'] = np.clip(df['recommendation_score'] * 10, 0, 100)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating scores: {e}")
            return players_df
    
    def generate_player_recommendations(self, players_df: pd.DataFrame, 
                                      position_filter: Optional[str] = None,
                                      budget_max: Optional[float] = None,
                                      top_n: int = 10) -> List[PlayerRecommendation]:
        """Generate sophisticated player recommendations"""
        try:
            # Calculate comprehensive scores
            df = self.calculate_comprehensive_scores(players_df)
            
            # Apply filters
            if position_filter:
                df = df[df['position_name'] == position_filter]
            
            if budget_max:
                df = df[df['now_cost'] <= budget_max]
            
            # Sort by recommendation score
            df = df.sort_values('recommendation_score', ascending=False)
            
            recommendations = []
            
            for _, player in df.head(top_n).iterrows():
                # Generate reasoning
                reasoning = []
                
                if player.get('recommendation_score', 0) > 80:
                    reasoning.append("⭐ Exceptional overall value and performance")
                elif player.get('recommendation_score', 0) > 60:
                    reasoning.append("✅ Strong recommendation with good metrics")
                
                if player.get('value_score', 0) > 15:
                    reasoning.append(f"💰 Excellent value at {player.get('now_cost', 0)}m")
                
                if player.get('form_numeric', 0) > 7:
                    reasoning.append(f"🔥 Excellent recent form ({player.get('form', 0)})")
                elif player.get('form_numeric', 0) > 5:
                    reasoning.append(f"📈 Good recent form ({player.get('form', 0)})")
                
                if player.get('selected_by_percent', 0) < 5:
                    reasoning.append("💎 Strong differential pick")
                elif player.get('selected_by_percent', 0) < 15:
                    reasoning.append("🎯 Good differential opportunity")
                
                if player.get('minutes', 0) > 2000:
                    reasoning.append("🛡️ High playing time reliability")
                
                # Expected ROI calculation
                current_points = player.get('total_points', 0)
                current_cost = player.get('now_cost', 1)
                expected_roi = (current_points / current_cost) * 100 if current_cost > 0 else 0
                
                rec = PlayerRecommendation(
                    player_id=player.get('id', 0),
                    web_name=player.get('web_name', 'Unknown'),
                    team_name=player.get('team_name', 'UNK'),
                    position=player.get('position_name', 'UNK'),
                    current_price=player.get('now_cost', 0) / 10.0,
                    predicted_points=player.get('predicted_points', current_points * 0.1),
                    confidence_score=player.get('confidence', 0.5),
                    value_score=player.get('value_score', 0),
                    form_score=player.get('form_score', 0),
                    fixture_score=player.get('fixture_score', 0.7),
                    ownership_score=player.get('ownership_score', 0.5),
                    risk_level=str(player.get('risk_level', 'Medium')),
                    reasoning=reasoning,
                    transfer_priority=len(recommendations) + 1,
                    expected_roi=expected_roi
                )
                
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def generate_team_recommendations(self, team_data: Dict, players_df: pd.DataFrame) -> TeamRecommendation:
        """Generate comprehensive team-level recommendations"""
        try:
            current_picks = team_data.get('picks', [])
            
            if not current_picks:
                return TeamRecommendation(
                    recommended_formation="3-5-2",
                    captain_suggestion=None,
                    vice_captain_suggestion=None,
                    transfer_suggestions=[],
                    chip_recommendation={'recommendation': 'No team data available'},
                    overall_score=0,
                    key_insights=['Load your team data to get recommendations']
                )
            
            # Analyze current team
            current_player_ids = [p.element for p in current_picks]
            current_players = players_df[players_df['id'].isin(current_player_ids)]
            
            # Calculate team scores
            df_with_scores = self.calculate_comprehensive_scores(current_players)
            
            # Captain recommendations
            captain_candidates = df_with_scores.nlargest(3, 'recommendation_score')
            captain_rec = None
            vice_captain_rec = None
            
            if len(captain_candidates) > 0:
                best_captain = captain_candidates.iloc[0]
                captain_rec = PlayerRecommendation(
                    player_id=best_captain.get('id', 0),
                    web_name=best_captain.get('web_name', 'Unknown'),
                    team_name=best_captain.get('team_name', 'UNK'),
                    position=best_captain.get('position_name', 'UNK'),
                    current_price=best_captain.get('now_cost', 0) / 10.0,
                    predicted_points=best_captain.get('predicted_points', 0),
                    confidence_score=0.8,
                    value_score=best_captain.get('value_score', 0),
                    form_score=best_captain.get('form_score', 0),
                    fixture_score=best_captain.get('fixture_score', 0.7),
                    ownership_score=best_captain.get('ownership_score', 0.5),
                    risk_level='Low',
                    reasoning=['Best performing player in your squad'],
                    transfer_priority=1,
                    expected_roi=0
                )
            
            if len(captain_candidates) > 1:
                vice_best = captain_candidates.iloc[1]
                vice_captain_rec = PlayerRecommendation(
                    player_id=vice_best.get('id', 0),
                    web_name=vice_best.get('web_name', 'Unknown'),
                    team_name=vice_best.get('team_name', 'UNK'),
                    position=vice_best.get('position_name', 'UNK'),
                    current_price=vice_best.get('now_cost', 0) / 10.0,
                    predicted_points=vice_best.get('predicted_points', 0),
                    confidence_score=0.7,
                    value_score=vice_best.get('value_score', 0),
                    form_score=vice_best.get('form_score', 0),
                    fixture_score=vice_best.get('fixture_score', 0.7),
                    ownership_score=vice_best.get('ownership_score', 0.5),
                    risk_level='Low',
                    reasoning=['Second best performer for vice captaincy'],
                    transfer_priority=2,
                    expected_roi=0
                )
            
            # Transfer suggestions (find underperforming players)
            poor_performers = df_with_scores.nsmallest(3, 'recommendation_score')
            transfer_suggestions = []
            
            for _, player in poor_performers.iterrows():
                if player.get('recommendation_score', 0) < 30:  # Poor performance threshold
                    transfer_suggestions.append(PlayerRecommendation(
                        player_id=player.get('id', 0),
                        web_name=player.get('web_name', 'Unknown'),
                        team_name=player.get('team_name', 'UNK'),
                        position=player.get('position_name', 'UNK'),
                        current_price=player.get('now_cost', 0) / 10.0,
                        predicted_points=0,
                        confidence_score=0.3,
                        value_score=player.get('value_score', 0),
                        form_score=player.get('form_score', 0),
                        fixture_score=player.get('fixture_score', 0.7),
                        ownership_score=player.get('ownership_score', 0.5),
                        risk_level='High',
                        reasoning=['Consider transferring due to poor performance'],
                        transfer_priority=1,
                        expected_roi=0
                    ))
            
            # Key insights
            insights = []
            avg_score = df_with_scores['recommendation_score'].mean()
            
            if avg_score > 70:
                insights.append("🏆 Strong squad with high-performing players")
            elif avg_score > 50:
                insights.append("✅ Decent squad with room for improvement")
            else:
                insights.append("⚠️ Squad needs significant improvements")
            
            insights.append(f"📊 Average squad score: {avg_score:.1f}/100")
            
            return TeamRecommendation(
                recommended_formation="3-5-2",  # Could be calculated based on players
                captain_suggestion=captain_rec,
                vice_captain_suggestion=vice_captain_rec,
                transfer_suggestions=transfer_suggestions,
                chip_recommendation={'recommendation': 'Analyze your team performance first'},
                overall_score=avg_score,
                key_insights=insights
            )
            
        except Exception as e:
            self.logger.error(f"Error generating team recommendations: {e}")
            return TeamRecommendation(
                recommended_formation="3-5-2",
                captain_suggestion=None,
                vice_captain_suggestion=None,
                transfer_suggestions=[],
                chip_recommendation={'error': str(e)},
                overall_score=0,
                key_insights=[f"Error generating recommendations: {str(e)}"]
            )


# Global recommendation engine instance
recommendation_engine = AdvancedMLRecommendationEngine()


def get_player_recommendations(players_df: pd.DataFrame, 
                             position: Optional[str] = None,
                             budget: Optional[float] = None,
                             top_n: int = 10) -> List[PlayerRecommendation]:
    """Get sophisticated player recommendations"""
    return recommendation_engine.generate_player_recommendations(
        players_df, position, budget, top_n
    )


def get_team_recommendations(team_data: Dict, players_df: pd.DataFrame) -> TeamRecommendation:
    """Get comprehensive team recommendations"""
    return recommendation_engine.generate_team_recommendations(team_data, players_df)


def train_recommendation_models(players_df: pd.DataFrame, force_retrain: bool = False) -> bool:
    """Train the ML models for recommendations"""
    return recommendation_engine.train_models(players_df, force_retrain)
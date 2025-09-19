"""
Advanced AI and Machine Learning Features
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedAIEngine:
    """Advanced AI engine for sophisticated FPL predictions"""
    
    def __init__(self):
        self.models = {
            'points_predictor': None,
            'price_change_predictor': None,
            'form_predictor': None
        }
        self.scalers = {}
        self.feature_importance = {}
        self.model_accuracy = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare advanced features for ML models"""
        features_df = df.copy()
        
        # Basic features
        numeric_features = [
            'total_points', 'form', 'minutes', 'goals_scored', 
            'assists', 'clean_sheets', 'bonus', 'now_cost'
        ]
        
        # Advanced engineered features
        features_df['points_per_game'] = features_df['total_points'] / np.maximum(features_df.get('games_played', 1), 1)
        features_df['goals_per_90'] = (features_df['goals_scored'] / np.maximum(features_df['minutes'], 1)) * 90
        features_df['assists_per_90'] = (features_df['assists'] / np.maximum(features_df['minutes'], 1)) * 90
        features_df['value_score'] = features_df['total_points'] / (features_df['now_cost'] / 10)
        
        # Team strength features (simplified)
        team_avg_points = features_df.groupby('team')['total_points'].transform('mean')
        features_df['team_strength'] = team_avg_points / team_avg_points.max()
        
        # Position-based features
        position_avg = features_df.groupby('element_type')['total_points'].transform('mean')
        features_df['position_relative_performance'] = features_df['total_points'] / position_avg
        
        # Recent form trends (simulated - would use actual gameweek data)
        features_df['form_trend'] = features_df['form'] * 0.8 + np.random.normal(0, 0.1, len(features_df))
        
        # Fixture difficulty impact (simplified)
        features_df['fixture_impact'] = np.random.uniform(0.8, 1.2, len(features_df))
        
        return features_df
    
    def train_points_predictor(self, df: pd.DataFrame) -> Dict:
        """Train advanced points prediction model"""
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Select features for training
            feature_columns = [
                'form', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
                'bonus', 'points_per_game', 'goals_per_90', 'assists_per_90',
                'value_score', 'team_strength', 'position_relative_performance',
                'form_trend', 'fixture_impact', 'element_type'
            ]
            
            # Handle missing values
            for col in feature_columns:
                if col in features_df.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
            
            X = features_df[feature_columns].fillna(0)
            y = features_df['total_points']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['points_predictor'] = scaler
            
            # Train ensemble model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            rf_model.fit(X_scaled, y)
            gb_model.fit(X_scaled, y)
            
            # Store models
            self.models['points_predictor'] = {
                'random_forest': rf_model,
                'gradient_boost': gb_model,
                'feature_columns': feature_columns
            }
            
            # Calculate feature importance
            self.feature_importance['points_predictor'] = dict(
                zip(feature_columns, rf_model.feature_importances_)
            )
            
            # Cross-validation score
            cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)
            self.model_accuracy['points_predictor'] = cv_scores.mean()
            
            return {
                'success': True,
                'accuracy': cv_scores.mean(),
                'feature_importance': self.feature_importance['points_predictor']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_next_gameweek_points(self, player_data: pd.Series) -> Dict:
        """Predict points for next gameweek"""
        try:
            if 'points_predictor' not in self.models or self.models['points_predictor'] is None:
                return {'error': 'Model not trained'}
            
            model_data = self.models['points_predictor']
            scaler = self.scalers['points_predictor']
            
            # Prepare features for single player
            features = []
            for col in model_data['feature_columns']:
                value = player_data.get(col, 0)
                features.append(pd.to_numeric(value, errors='coerce') if pd.notna(value) else 0)
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Ensemble prediction
            rf_pred = model_data['random_forest'].predict(features_scaled)[0]
            gb_pred = model_data['gradient_boost'].predict(features_scaled)[0]
            
            # Weighted average
            ensemble_pred = (rf_pred * 0.6 + gb_pred * 0.4)
            
            # Apply position and form adjustments
            position_multiplier = {1: 0.8, 2: 0.9, 3: 1.1, 4: 1.2}.get(player_data.get('element_type', 3), 1.0)
            form_multiplier = min(max(player_data.get('form', 5) / 5, 0.5), 1.5)
            
            adjusted_pred = ensemble_pred * position_multiplier * form_multiplier
            
            # Confidence calculation
            pred_std = abs(rf_pred - gb_pred)
            confidence = max(0.5, 1 - (pred_std / max(ensemble_pred, 1)))
            
            return {
                'predicted_points': round(adjusted_pred, 1),
                'confidence': round(confidence, 2),
                'rf_prediction': round(rf_pred, 1),
                'gb_prediction': round(gb_pred, 1),
                'position_factor': position_multiplier,
                'form_factor': round(form_multiplier, 2)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_transfer_recommendations_ml(self, df: pd.DataFrame, budget: float = 100.0, 
                                       position: str = None) -> List[Dict]:
        """Get ML-powered transfer recommendations"""
        try:
            # Filter by position if specified
            if position:
                position_map = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
                df = df[df['element_type'] == position_map.get(position, 3)]
            
            # Filter by budget
            df = df[df['now_cost'] <= budget * 10]  # Convert to FPL cost format
            
            recommendations = []
            
            for _, player in df.head(20).iterrows():
                # Get ML prediction
                prediction = self.predict_next_gameweek_points(player)
                
                if 'predicted_points' in prediction:
                    # Calculate recommendation score
                    predicted_points = prediction['predicted_points']
                    confidence = prediction['confidence']
                    cost = player['now_cost'] / 10
                    ownership = player.get('selected_by_percent', 0)
                    
                    # Multi-factor scoring
                    value_score = predicted_points / cost if cost > 0 else 0
                    form_score = player.get('form', 0)
                    differential_bonus = max(0, (20 - ownership) / 20) * 0.3
                    
                    overall_score = (
                        predicted_points * 0.4 +
                        value_score * 0.25 +
                        form_score * 0.2 +
                        confidence * 0.1 +
                        differential_bonus
                    )
                    
                    recommendations.append({
                        'web_name': player['web_name'],
                        'team_name': player.get('team_short_name', 'Unknown'),
                        'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player['element_type'], 'MID'),
                        'current_price': cost,
                        'predicted_points': predicted_points,
                        'confidence': confidence,
                        'overall_score': round(overall_score, 2),
                        'value_score': round(value_score, 2),
                        'ownership': ownership,
                        'reasoning': self._generate_ml_reasoning(player, prediction)
                    })
            
            # Sort by overall score
            recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
            
            return recommendations[:10]
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def _generate_ml_reasoning(self, player: pd.Series, prediction: Dict) -> List[str]:
        """Generate AI reasoning for recommendations"""
        reasoning = []
        
        if prediction.get('confidence', 0) > 0.8:
            reasoning.append(f"High confidence prediction ({prediction['confidence']:.0%})")
        
        if prediction.get('predicted_points', 0) > 6:
            reasoning.append(f"Predicted to score {prediction['predicted_points']:.1f} points next GW")
        
        if player.get('form', 0) > 6:
            reasoning.append(f"Excellent recent form ({player['form']:.1f})")
        
        if player.get('selected_by_percent', 100) < 15:
            reasoning.append("Low ownership differential opportunity")
        
        points_per_million = player.get('total_points', 0) / (player.get('now_cost', 40) / 10)
        if points_per_million > 15:
            reasoning.append(f"Strong value at {points_per_million:.1f} pts/£m")
        
        return reasoning[:3]  # Limit to top 3 reasons

class PredictiveAnalytics:
    """Advanced predictive analytics for FPL"""
    
    @staticmethod
    def calculate_optimal_formation(df: pd.DataFrame, budget: float = 100.0) -> Dict:
        """Calculate optimal formation based on available players"""
        formations = {
            "3-4-3": (3, 4, 3),
            "4-3-3": (4, 3, 3), 
            "3-5-2": (3, 5, 2),
            "4-4-2": (4, 4, 2),
            "5-3-2": (5, 3, 2)
        }
        
        formation_scores = {}
        
        for formation_name, (def_count, mid_count, fwd_count) in formations.items():
            # Get best players by position within budget
            defenders = df[df['element_type'] == 2].nlargest(def_count + 2, 'total_points')  # +2 for bench
            midfielders = df[df['element_type'] == 3].nlargest(mid_count + 1, 'total_points')  # +1 for bench
            forwards = df[df['element_type'] == 4].nlargest(fwd_count + 1, 'total_points')   # +1 for bench
            goalkeepers = df[df['element_type'] == 1].nlargest(2, 'total_points')           # 2 for rotation
            
            total_cost = (
                defenders['now_cost'].sum() + 
                midfielders['now_cost'].sum() + 
                forwards['now_cost'].sum() + 
                goalkeepers['now_cost'].sum()
            ) / 10
            
            if total_cost <= budget:
                total_points = (
                    defenders['total_points'].sum() + 
                    midfielders['total_points'].sum() + 
                    forwards['total_points'].sum() + 
                    goalkeepers['total_points'].sum()
                )
                
                formation_scores[formation_name] = {
                    'total_cost': round(total_cost, 1),
                    'total_points': total_points,
                    'efficiency': round(total_points / total_cost, 2),
                    'feasible': True
                }
            else:
                formation_scores[formation_name] = {
                    'total_cost': round(total_cost, 1),
                    'feasible': False
                }
        
        # Find best formation
        feasible_formations = {k: v for k, v in formation_scores.items() if v.get('feasible', False)}
        
        if feasible_formations:
            best_formation = max(feasible_formations.keys(), 
                               key=lambda x: feasible_formations[x]['efficiency'])
            
            return {
                'recommended_formation': best_formation,
                'all_formations': formation_scores,
                'best_efficiency': feasible_formations[best_formation]['efficiency']
            }
        
        return {'error': 'No feasible formations within budget'}
    
    @staticmethod
    def predict_price_changes(df: pd.DataFrame) -> List[Dict]:
        """Predict potential price changes based on ownership and form"""
        price_predictions = []
        
        for _, player in df.iterrows():
            ownership = player.get('selected_by_percent', 0)
            form = player.get('form', 0)
            total_points = player.get('total_points', 0)
            
            # Simple price change prediction logic
            rise_probability = 0
            fall_probability = 0
            
            # High ownership + good form = likely rise
            if ownership > 15 and form > 6:
                rise_probability = min(0.8, (ownership / 100) + (form / 10))
            
            # Low ownership + poor form = likely fall
            if ownership < 5 and form < 3:
                fall_probability = min(0.8, (10 - ownership) / 10 + (5 - form) / 10)
            
            # Recent performance impact
            if total_points > 100:  # High performers more likely to rise
                rise_probability *= 1.2
            
            if rise_probability > 0.3 or fall_probability > 0.3:
                price_predictions.append({
                    'web_name': player['web_name'],
                    'current_price': player['now_cost'] / 10,
                    'rise_probability': round(rise_probability, 2),
                    'fall_probability': round(fall_probability, 2),
                    'recommendation': 'BUY' if rise_probability > 0.5 else 'SELL' if fall_probability > 0.5 else 'HOLD'
                })
        
        return sorted(price_predictions, key=lambda x: max(x['rise_probability'], x['fall_probability']), reverse=True)[:20]

def render_advanced_ai_tab(df: pd.DataFrame):
    """Render advanced AI features tab"""
    st.header("🤖 Advanced AI & Machine Learning")
    
    if df.empty:
        st.warning("Please load player data first")
        return
    
    ai_engine = AdvancedAIEngine()
    
    # Initialize model training
    if st.button("🧠 Train AI Models", type="primary"):
        with st.spinner("Training advanced AI models..."):
            results = ai_engine.train_points_predictor(df)
            
            if results.get('success'):
                st.success(f"✅ Models trained successfully! Accuracy: {results['accuracy']:.2%}")
                
                # Show feature importance
                st.subheader("📊 Feature Importance")
                importance_df = pd.DataFrame(
                    list(results['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                st.bar_chart(importance_df.set_index('Feature'))
            else:
                st.error(f"❌ Training failed: {results.get('error')}")
    
    # ML-powered recommendations
    st.subheader("🎯 ML-Powered Transfer Recommendations")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.slider("Budget (£m)", 80.0, 120.0, 100.0, 0.5)
    with col2:
        position = st.selectbox("Position", ["All", "Goalkeeper", "Defender", "Midfielder", "Forward"])
    
    if st.button("🚀 Get ML Recommendations"):
        with st.spinner("Generating AI recommendations..."):
            recommendations = ai_engine.get_transfer_recommendations_ml(
                df, budget, position if position != "All" else None
            )
            
            if recommendations and 'error' not in recommendations[0]:
                for i, rec in enumerate(recommendations[:5], 1):
                    with st.expander(f"{i}. {rec['web_name']} ({rec['position']}) - {rec['overall_score']:.1f}/10"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Points", f"{rec['predicted_points']:.1f}")
                            st.metric("Confidence", f"{rec['confidence']:.0%}")
                        
                        with col2:
                            st.metric("Current Price", f"£{rec['current_price']:.1f}m")
                            st.metric("Value Score", f"{rec['value_score']:.1f}")
                        
                        with col3:
                            st.metric("Ownership", f"{rec['ownership']:.1f}%")
                            st.metric("Overall Score", f"{rec['overall_score']:.1f}/10")
                        
                        st.write("**AI Reasoning:**")
                        for reason in rec['reasoning']:
                            st.write(f"• {reason}")
            else:
                st.error("Failed to generate recommendations")
    
    # Predictive analytics
    st.subheader("📈 Predictive Analytics")
    
    analytics = PredictiveAnalytics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**💰 Optimal Formation Analysis**")
        if st.button("Calculate Optimal Formation"):
            formation_analysis = analytics.calculate_optimal_formation(df)
            
            if 'recommended_formation' in formation_analysis:
                st.success(f"🏆 Recommended: {formation_analysis['recommended_formation']}")
                st.metric("Efficiency", f"{formation_analysis['best_efficiency']:.1f} pts/£m")
                
                # Show all formations
                formations_df = pd.DataFrame(formation_analysis['all_formations']).T
                st.dataframe(formations_df)
            else:
                st.error("Could not calculate optimal formation")
    
    with col2:
        st.write("**📊 Price Change Predictions**")
        if st.button("Predict Price Changes"):
            price_predictions = analytics.predict_price_changes(df)
            
            if price_predictions:
                for pred in price_predictions[:5]:
                    recommendation_color = {
                        'BUY': '🟢',
                        'SELL': '🔴', 
                        'HOLD': '🟡'
                    }.get(pred['recommendation'], '⚪')
                    
                    st.write(f"{recommendation_color} **{pred['web_name']}** - {pred['recommendation']}")
                    st.write(f"   Rise: {pred['rise_probability']:.0%} | Fall: {pred['fall_probability']:.0%}")
            else:
                st.info("No significant price changes predicted")
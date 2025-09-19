"""
Automated Iteration Recommendation Service
Continuously improves recommendations based on user feedback and performance tracking
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

from services.ai_recommendation_engine import PlayerRecommendation, get_player_recommendations
from services.personalization_service import track_user_interaction, get_personalized_suggestions
from services.data_export_import import UserPreferenceManager
from utils.error_handling import handle_errors, logger


@dataclass
class FeedbackEntry:
    """User feedback on recommendations"""
    recommendation_id: str
    player_id: int
    player_name: str
    feedback_type: str  # 'positive', 'negative', 'neutral'
    feedback_score: float  # 1-5 rating
    actual_outcome: Optional[float]  # actual points scored if available
    user_action: str  # 'transferred_in', 'transferred_out', 'ignored', 'considered'
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class RecommendationPerformance:
    """Track recommendation performance over time"""
    recommendation_id: str
    predicted_points: float
    actual_points: Optional[float]
    accuracy_score: Optional[float]
    user_satisfaction: Optional[float]
    conversion_rate: float  # did user follow recommendation
    timestamp: datetime


class AutomatedIterationEngine:
    """Automated system for improving recommendations through continuous learning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("iteration_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Feedback storage
        self.feedback_file = self.data_dir / "user_feedback.json"
        self.performance_file = self.data_dir / "recommendation_performance.json"
        self.model_file = self.data_dir / "feedback_prediction_model.pkl"
        
        # Learning models
        self.feedback_predictor = None
        self.recommendation_weights = {
            'value_score': 0.25,
            'form_score': 0.20,
            'fixture_score': 0.15,
            'ownership_score': 0.10,
            'predicted_points': 0.30
        }
        
        self.iteration_history = []
        self.performance_metrics = {
            'recommendation_accuracy': [],
            'user_satisfaction': [],
            'conversion_rate': [],
            'prediction_error': []
        }
        
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical feedback and performance data"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                    self.feedback_history = [
                        FeedbackEntry(**entry) for entry in feedback_data
                    ]
            else:
                self.feedback_history = []
            
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    perf_data = json.load(f)
                    self.performance_history = [
                        RecommendationPerformance(**entry) for entry in perf_data
                    ]
            else:
                self.performance_history = []
                
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            self.feedback_history = []
            self.performance_history = []
    
    def collect_user_feedback(self, recommendation: PlayerRecommendation, 
                            feedback_type: str, feedback_score: float,
                            user_action: str, context: Dict = None) -> str:
        """Collect user feedback on a recommendation"""
        try:
            feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{recommendation.player_id}"
            
            feedback = FeedbackEntry(
                recommendation_id=feedback_id,
                player_id=recommendation.player_id,
                player_name=recommendation.web_name,
                feedback_type=feedback_type,
                feedback_score=feedback_score,
                actual_outcome=None,  # Will be updated later
                user_action=user_action,
                timestamp=datetime.now(),
                context=context or {}
            )
            
            self.feedback_history.append(feedback)
            self._save_feedback_data()
            
            # Track interaction for personalization
            track_user_interaction('recommendation_feedback', {
                'player_id': recommendation.player_id,
                'feedback_type': feedback_type,
                'feedback_score': feedback_score,
                'user_action': user_action,
                'recommendation_type': 'automated'
            })
            
            # Trigger learning update
            self._update_recommendation_weights(feedback)
            
            return feedback_id
            
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {e}")
            return ""
    
    def track_recommendation_performance(self, recommendation: PlayerRecommendation,
                                       actual_points: float, user_followed: bool):
        """Track actual performance vs predicted performance"""
        try:
            accuracy = 1.0 - abs(recommendation.predicted_points - actual_points) / max(recommendation.predicted_points, 1)
            
            performance = RecommendationPerformance(
                recommendation_id=f"perf_{recommendation.player_id}_{datetime.now().strftime('%Y%m%d')}",
                predicted_points=recommendation.predicted_points,
                actual_points=actual_points,
                accuracy_score=accuracy,
                user_satisfaction=None,  # Will be updated from feedback
                conversion_rate=1.0 if user_followed else 0.0,
                timestamp=datetime.now()
            )
            
            self.performance_history.append(performance)
            self._save_performance_data()
            
            # Update performance metrics
            self.performance_metrics['recommendation_accuracy'].append(accuracy)
            self.performance_metrics['conversion_rate'].append(performance.conversion_rate)
            
            prediction_error = abs(recommendation.predicted_points - actual_points)
            self.performance_metrics['prediction_error'].append(prediction_error)
            
            # Keep only recent metrics (last 100)
            for key in self.performance_metrics:
                if len(self.performance_metrics[key]) > 100:
                    self.performance_metrics[key] = self.performance_metrics[key][-100:]
            
        except Exception as e:
            self.logger.error(f"Error tracking performance: {e}")
    
    def _update_recommendation_weights(self, feedback: FeedbackEntry):
        """Update recommendation weights based on feedback"""
        try:
            # Simple adaptive weighting based on feedback
            weight_adjustment = 0.01  # Small incremental changes
            
            if feedback.feedback_type == 'positive':
                # Increase weights for factors that led to positive feedback
                if feedback.feedback_score >= 4:
                    self.recommendation_weights['predicted_points'] += weight_adjustment
                    self.recommendation_weights['value_score'] += weight_adjustment * 0.5
            
            elif feedback.feedback_type == 'negative':
                # Decrease weights for factors that led to negative feedback
                if feedback.feedback_score <= 2:
                    self.recommendation_weights['predicted_points'] -= weight_adjustment
                    if self.recommendation_weights['predicted_points'] < 0.1:
                        self.recommendation_weights['predicted_points'] = 0.1
            
            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(self.recommendation_weights.values())
            for key in self.recommendation_weights:
                self.recommendation_weights[key] /= total_weight
            
            self.logger.info(f"Updated recommendation weights: {self.recommendation_weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}")
    
    def generate_automated_recommendations(self, players_df: pd.DataFrame,
                                         user_context: Dict = None) -> List[PlayerRecommendation]:
        """Generate recommendations with automated improvements"""
        try:
            # Get personalized suggestions
            personalized_prefs = get_personalized_suggestions(user_context or {})
            
            # Apply learned weights to recommendation engine
            base_recommendations = get_player_recommendations(
                players_df,
                position=user_context.get('position_filter'),
                budget=user_context.get('budget_max'),
                top_n=15  # Get more to filter down
            )
            
            # Apply iteration improvements
            improved_recommendations = self._apply_iteration_improvements(
                base_recommendations, personalized_prefs, user_context
            )
            
            # Track that recommendations were generated
            track_user_interaction('automated_recommendations_generated', {
                'count': len(improved_recommendations),
                'context': user_context or {},
                'weights_used': self.recommendation_weights.copy()
            })
            
            return improved_recommendations[:10]  # Return top 10
            
        except Exception as e:
            self.logger.error(f"Error generating automated recommendations: {e}")
            return []
    
    def _apply_iteration_improvements(self, recommendations: List[PlayerRecommendation],
                                    personalized_prefs: Dict, context: Dict) -> List[PlayerRecommendation]:
        """Apply learned improvements to recommendations"""
        try:
            improved_recs = []
            
            for rec in recommendations:
                # Calculate improved score based on learned weights
                improved_score = (
                    rec.value_score * self.recommendation_weights['value_score'] +
                    rec.form_score * self.recommendation_weights['form_score'] +
                    rec.fixture_score * self.recommendation_weights['fixture_score'] +
                    rec.ownership_score * self.recommendation_weights['ownership_score'] +
                    (rec.predicted_points / 10) * self.recommendation_weights['predicted_points']
                )
                
                # Apply personalization adjustments
                if personalized_prefs.get('risk_level'):
                    risk_pref = personalized_prefs['risk_level']['suggestion']
                    if risk_pref == 'low' and rec.risk_level == 'High':
                        improved_score *= 0.7  # Penalize high-risk picks for risk-averse users
                    elif risk_pref == 'high' and rec.risk_level == 'Low':
                        improved_score *= 1.2  # Boost low-risk picks for high-risk users
                
                # Apply price range preferences
                if personalized_prefs.get('price_ranges'):
                    position_lower = rec.position.lower()
                    if position_lower in personalized_prefs['price_ranges']:
                        target_price = personalized_prefs['price_ranges'][position_lower]['target_price']
                        price_diff = abs(rec.current_price - target_price)
                        if price_diff <= 1.0:  # Within £1m of preferred price
                            improved_score *= 1.1
                
                # Create improved recommendation
                improved_rec = PlayerRecommendation(
                    player_id=rec.player_id,
                    web_name=rec.web_name,
                    team_name=rec.team_name,
                    position=rec.position,
                    current_price=rec.current_price,
                    predicted_points=rec.predicted_points,
                    confidence_score=min(1.0, rec.confidence_score * 1.1),  # Boost confidence
                    value_score=improved_score,  # Use improved score
                    form_score=rec.form_score,
                    fixture_score=rec.fixture_score,
                    ownership_score=rec.ownership_score,
                    risk_level=rec.risk_level,
                    reasoning=rec.reasoning + [f"🤖 AI-enhanced based on your preferences"],
                    transfer_priority=rec.transfer_priority,
                    expected_roi=rec.expected_roi
                )
                
                improved_recs.append((improved_rec, improved_score))
            
            # Sort by improved score
            improved_recs.sort(key=lambda x: x[1], reverse=True)
            return [rec[0] for rec in improved_recs]
            
        except Exception as e:
            self.logger.error(f"Error applying improvements: {e}")
            return recommendations
    
    def get_iteration_insights(self) -> Dict[str, Any]:
        """Get insights about the iteration learning process"""
        try:
            if not self.performance_metrics['recommendation_accuracy']:
                return {'status': 'No data available yet'}
            
            recent_accuracy = np.mean(self.performance_metrics['recommendation_accuracy'][-20:])
            recent_conversion = np.mean(self.performance_metrics['conversion_rate'][-20:])
            
            # Calculate trends
            if len(self.performance_metrics['recommendation_accuracy']) >= 10:
                early_accuracy = np.mean(self.performance_metrics['recommendation_accuracy'][:10])
                accuracy_trend = recent_accuracy - early_accuracy
            else:
                accuracy_trend = 0
            
            # Feedback analysis
            positive_feedback = sum(1 for f in self.feedback_history[-50:] if f.feedback_type == 'positive')
            total_feedback = len(self.feedback_history[-50:])
            satisfaction_rate = positive_feedback / max(total_feedback, 1)
            
            insights = {
                'recommendation_accuracy': recent_accuracy,
                'accuracy_trend': accuracy_trend,
                'user_conversion_rate': recent_conversion,
                'user_satisfaction_rate': satisfaction_rate,
                'total_feedback_entries': len(self.feedback_history),
                'current_weights': self.recommendation_weights.copy(),
                'learning_status': 'Active' if total_feedback > 10 else 'Collecting Data',
                'recommendations_count': len(self.performance_history)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return {'status': 'Error generating insights'}
    
    def _save_feedback_data(self):
        """Save feedback data to file"""
        try:
            feedback_data = [asdict(f) for f in self.feedback_history]
            # Convert datetime objects to strings
            for entry in feedback_data:
                entry['timestamp'] = entry['timestamp'].isoformat()
            
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving feedback data: {e}")
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            perf_data = [asdict(p) for p in self.performance_history]
            # Convert datetime objects to strings
            for entry in perf_data:
                entry['timestamp'] = entry['timestamp'].isoformat()
            
            with open(self.performance_file, 'w') as f:
                json.dump(perf_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    def export_learning_data(self) -> str:
        """Export all learning data for analysis"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = self.data_dir / f"learning_export_{timestamp}.json"
            
            export_data = {
                'feedback_history': [asdict(f) for f in self.feedback_history],
                'performance_history': [asdict(p) for p in self.performance_history],
                'current_weights': self.recommendation_weights,
                'performance_metrics': self.performance_metrics,
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            for entry in export_data['feedback_history']:
                entry['timestamp'] = entry['timestamp'].isoformat() if isinstance(entry['timestamp'], datetime) else entry['timestamp']
            
            for entry in export_data['performance_history']:
                entry['timestamp'] = entry['timestamp'].isoformat() if isinstance(entry['timestamp'], datetime) else entry['timestamp']
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return str(export_file)
            
        except Exception as e:
            self.logger.error(f"Error exporting learning data: {e}")
            return ""


# Global iteration engine instance
iteration_engine = AutomatedIterationEngine()


def get_automated_recommendations(players_df: pd.DataFrame, 
                                user_context: Dict = None) -> List[PlayerRecommendation]:
    """Get AI-enhanced recommendations with continuous learning"""
    return iteration_engine.generate_automated_recommendations(players_df, user_context)


def collect_recommendation_feedback(recommendation: PlayerRecommendation,
                                  feedback_type: str, feedback_score: float,
                                  user_action: str, context: Dict = None) -> str:
    """Collect user feedback to improve future recommendations"""
    return iteration_engine.collect_user_feedback(
        recommendation, feedback_type, feedback_score, user_action, context
    )


def track_prediction_accuracy(recommendation: PlayerRecommendation,
                            actual_points: float, user_followed: bool):
    """Track how accurate our predictions were"""
    iteration_engine.track_recommendation_performance(
        recommendation, actual_points, user_followed
    )


def get_learning_insights() -> Dict[str, Any]:
    """Get insights about the learning process"""
    return iteration_engine.get_iteration_insights()
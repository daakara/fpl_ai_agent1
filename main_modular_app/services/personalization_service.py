"""
Advanced User Personalization and Preference Learning System
"""
import streamlit as st
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path
import hashlib

@dataclass
class UserPreference:
    """User preference data structure"""
    preference_type: str  # 'player_selection', 'risk_appetite', 'formation', etc.
    value: Any
    confidence: float  # 0-1, how confident we are in this preference
    last_updated: datetime
    source: str  # 'explicit', 'inferred', 'ml_prediction'

@dataclass
class UserProfile:
    """Complete user profile"""
    user_id: str
    preferences: Dict[str, UserPreference]
    interaction_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    created_at: datetime
    last_active: datetime

class PersonalizationEngine:
    """Advanced personalization engine for FPL recommendations"""
    
    def __init__(self, data_dir: str = "user_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.current_profile: Optional[UserProfile] = None
    
    def get_or_create_user_profile(self, user_id: str = None) -> UserProfile:
        """Get existing user profile or create new one"""
        if not user_id:
            user_id = self._generate_user_id()
        
        profile_path = self.data_dir / f"{user_id}.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    profile = self._deserialize_profile(data)
                    profile.last_active = datetime.now()
                    self.current_profile = profile
                    return profile
            except Exception as e:
                st.warning(f"Error loading user profile: {e}")
        
        # Create new profile
        profile = UserProfile(
            user_id=user_id,
            preferences={},
            interaction_history=[],
            performance_metrics={},
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        self.current_profile = profile
        self.save_profile(profile)
        return profile
    
    def _generate_user_id(self) -> str:
        """Generate anonymous user ID based on session"""
        if 'user_id' not in st.session_state:
            # Create stable ID based on session info
            session_info = str(datetime.now().date()) + str(hash(str(st.session_state)))
            st.session_state.user_id = hashlib.md5(session_info.encode()).hexdigest()[:12]
        return st.session_state.user_id
    
    def learn_from_interaction(self, interaction_type: str, data: Dict[str, Any]):
        """Learn from user interactions to improve personalization"""
        if not self.current_profile:
            return
        
        # Record interaction
        interaction = {
            'type': interaction_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.current_profile.interaction_history.append(interaction)
        
        # Infer preferences from interactions
        self._infer_preferences_from_interaction(interaction_type, data)
        
        # Keep only recent interactions (last 100)
        if len(self.current_profile.interaction_history) > 100:
            self.current_profile.interaction_history = self.current_profile.interaction_history[-100:]
        
        self.save_profile(self.current_profile)
    
    def _infer_preferences_from_interaction(self, interaction_type: str, data: Dict[str, Any]):
        """Infer user preferences from their interactions"""
        confidence = 0.3  # Base confidence for inferred preferences
        
        if interaction_type == 'player_selection':
            # Learn from player selections
            if 'position' in data:
                self._update_preference(
                    f'preferred_{data["position"]}_price_range',
                    data.get('price', 0),
                    confidence,
                    'inferred'
                )
            
            if 'team' in data:
                self._update_preference(
                    'team_diversification',
                    data['team'],
                    confidence * 0.5,
                    'inferred'
                )
        
        elif interaction_type == 'transfer_suggestion':
            if data.get('accepted', False):
                # User accepted suggestion - high confidence in this type of recommendation
                self._update_preference(
                    'transfer_risk_appetite',
                    data.get('risk_level', 'medium'),
                    confidence * 2,
                    'inferred'
                )
        
        elif interaction_type == 'formation_selection':
            self._update_preference(
                'preferred_formation',
                data.get('formation'),
                confidence * 1.5,
                'inferred'
            )
    
    def _update_preference(self, pref_type: str, value: Any, confidence: float, source: str):
        """Update or create a user preference"""
        if pref_type in self.current_profile.preferences:
            # Update existing preference with weighted average
            existing = self.current_profile.preferences[pref_type]
            new_confidence = min(1.0, (existing.confidence + confidence) / 2)
            
            # For numerical values, use weighted average
            if isinstance(value, (int, float)) and isinstance(existing.value, (int, float)):
                new_value = (existing.value * existing.confidence + value * confidence) / (existing.confidence + confidence)
            else:
                new_value = value  # For categorical values, use latest
            
            self.current_profile.preferences[pref_type] = UserPreference(
                preference_type=pref_type,
                value=new_value,
                confidence=new_confidence,
                last_updated=datetime.now(),
                source=source
            )
        else:
            # Create new preference
            self.current_profile.preferences[pref_type] = UserPreference(
                preference_type=pref_type,
                value=value,
                confidence=min(1.0, confidence),
                last_updated=datetime.now(),
                source=source
            )
    
    def get_personalized_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized recommendations based on user profile"""
        if not self.current_profile:
            return {}
        
        recommendations = {}
        
        # Formation recommendations
        if 'preferred_formation' in self.current_profile.preferences:
            pref = self.current_profile.preferences['preferred_formation']
            recommendations['formation'] = {
                'suggestion': pref.value,
                'confidence': pref.confidence,
                'reason': f"Based on your previous selections (confidence: {pref.confidence:.1%})"
            }
        
        # Risk appetite recommendations
        if 'transfer_risk_appetite' in self.current_profile.preferences:
            pref = self.current_profile.preferences['transfer_risk_appetite']
            recommendations['risk_level'] = {
                'suggestion': pref.value,
                'confidence': pref.confidence,
                'reason': f"Matches your typical risk preference"
            }
        
        # Price range recommendations
        price_prefs = {k: v for k, v in self.current_profile.preferences.items() 
                      if k.startswith('preferred_') and k.endswith('_price_range')}
        
        if price_prefs:
            recommendations['price_ranges'] = {}
            for pref_name, pref in price_prefs.items():
                position = pref_name.replace('preferred_', '').replace('_price_range', '')
                recommendations['price_ranges'][position] = {
                    'target_price': pref.value,
                    'confidence': pref.confidence
                }
        
        return recommendations
    
    def save_profile(self, profile: UserProfile):
        """Save user profile to disk"""
        try:
            profile_path = self.data_dir / f"{profile.user_id}.json"
            data = self._serialize_profile(profile)
            with open(profile_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            st.warning(f"Failed to save user profile: {e}")
    
    def _serialize_profile(self, profile: UserProfile) -> Dict[str, Any]:
        """Convert profile to JSON-serializable format"""
        data = asdict(profile)
        
        # Convert datetime objects to ISO strings
        data['created_at'] = profile.created_at.isoformat()
        data['last_active'] = profile.last_active.isoformat()
        
        # Convert preferences
        prefs = {}
        for key, pref in profile.preferences.items():
            prefs[key] = {
                'preference_type': pref.preference_type,
                'value': pref.value,
                'confidence': pref.confidence,
                'last_updated': pref.last_updated.isoformat(),
                'source': pref.source
            }
        data['preferences'] = prefs
        
        return data
    
    def _deserialize_profile(self, data: Dict[str, Any]) -> UserProfile:
        """Convert JSON data back to UserProfile"""
        # Convert datetime strings back
        created_at = datetime.fromisoformat(data['created_at'])
        last_active = datetime.fromisoformat(data['last_active'])
        
        # Convert preferences
        preferences = {}
        for key, pref_data in data.get('preferences', {}).items():
            preferences[key] = UserPreference(
                preference_type=pref_data['preference_type'],
                value=pref_data['value'],
                confidence=pref_data['confidence'],
                last_updated=datetime.fromisoformat(pref_data['last_updated']),
                source=pref_data['source']
            )
        
        return UserProfile(
            user_id=data['user_id'],
            preferences=preferences,
            interaction_history=data.get('interaction_history', []),
            performance_metrics=data.get('performance_metrics', {}),
            created_at=created_at,
            last_active=last_active
        )

# Global personalization engine
personalization_engine = PersonalizationEngine()

def initialize_personalization():
    """Initialize personalization for the current session"""
    if 'personalization_initialized' not in st.session_state:
        profile = personalization_engine.get_or_create_user_profile()
        st.session_state.personalization_initialized = True
        st.session_state.user_profile = profile
        return profile
    return st.session_state.get('user_profile')

def track_user_interaction(interaction_type: str, data: Dict[str, Any]):
    """Track a user interaction for learning"""
    personalization_engine.learn_from_interaction(interaction_type, data)

def get_personalized_suggestions(context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get personalized suggestions for the current user"""
    return personalization_engine.get_personalized_recommendations(context or {})

def display_personalization_panel():
    """Display user personalization settings and insights"""
    st.subheader("🎯 Personalization Settings")
    
    profile = st.session_state.get('user_profile')
    if not profile:
        st.info("Personalization data will be collected as you use the app.")
        return
    
    # Display current preferences
    if profile.preferences:
        st.write("**Your Learned Preferences:**")
        
        for pref_name, pref in profile.preferences.items():
            if pref.confidence > 0.3:  # Only show confident preferences
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"• {pref_name.replace('_', ' ').title()}: {pref.value}")
                with col2:
                    st.write(f"Confidence: {pref.confidence:.1%}")
                with col3:
                    st.write(f"Source: {pref.source}")
    
    # Manual preference setting
    st.write("**Manual Preferences:**")
    
    risk_appetite = st.selectbox(
        "Risk Appetite",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="How much risk you're willing to take with transfers"
    )
    
    if st.button("Save Manual Preferences"):
        track_user_interaction('manual_preference', {
            'risk_appetite': risk_appetite.lower(),
            'source': 'manual'
        })
        st.success("Preferences saved!")
    
    # Usage statistics
    if profile.interaction_history:
        st.write("**Usage Statistics:**")
        interactions_df = pd.DataFrame([
            {
                'Type': interaction['type'],
                'Date': interaction['timestamp'][:10]
            } for interaction in profile.interaction_history[-20:]
        ])
        
        interaction_counts = interactions_df['Type'].value_counts()
        st.bar_chart(interaction_counts)
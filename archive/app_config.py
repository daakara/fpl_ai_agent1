from dataclasses import dataclass
import streamlit as st

@dataclass
class AppConfig:
    """Configuration settings for the FPL app"""
    app_title: str = "FPL AI Agent"
    page_icon: str = "⚽"
    layout: str = "wide"
    max_players_display: int = 100

class SessionStateManager:
    """Manages Streamlit session state"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        if 'user_settings' not in st.session_state:
            st.session_state.user_settings = {}
    
    def reset_data(self):
        """Reset data-related session state"""
        st.session_state.data_loaded = False
        if 'data_manager' in st.session_state:
            del st.session_state.data_manager
    
    def mark_data_loaded(self):
        """Mark data as loaded"""
        st.session_state.data_loaded = True
        st.session_state.last_refresh = st.session_state.get('last_refresh', None)
    
    def clear_chat(self):
        """Clear chat messages"""
        st.session_state.chat_messages = []
        st.session_state.chat_history = []
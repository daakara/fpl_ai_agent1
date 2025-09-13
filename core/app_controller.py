import pandas as pd
import streamlit as st
"""
Main Application Controller - Coordinates all components
"""
from services.fpl_data_service import FPLDataService
from core.page_router import PageRouter


class FPLAppController:
    """Main application controller coordinating all components"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.data_service = FPLDataService()
        self.page_router = PageRouter()
        
        # Initialize page components
        self.pages = {}
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="FPL Analytics Dashboard",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'players_df' not in st.session_state:
            st.session_state.players_df = pd.DataFrame()
        if 'teams_df' not in st.session_state:
            st.session_state.teams_df = pd.DataFrame()
    
    def run(self):
        """Main application runner"""
        try:
            # Render sidebar and get selected page
            selected_page = self.page_router.render_sidebar_navigation()
            
            # Route to appropriate page
            if selected_page == "Dashboard":
                self.render_dashboard()
            elif selected_page == "Transfer Recommendations":
                st.write("Transfer Recommendations Page")
            elif selected_page == "Captain Picks":
                st.write("Captain Picks Page")
            elif selected_page == "Team Planning":
                st.write("Team Planning Page")
            elif selected_page == "Settings":
                st.write("Settings Page")
            else:
                self.render_dashboard() # Default to dashboard
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.write("Please try refreshing the page or loading data again.")
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.title("⚽ FPL Analytics Dashboard")
        st.markdown("### Welcome to your Fantasy Premier League Analytics Hub!")
        
        if not st.session_state.data_loaded:
            st.info("👋 Welcome! Click '🔄 Refresh Data' in the sidebar to get started.")
            return
        
        # Dashboard content here
        st.success("Dashboard loaded successfully!")

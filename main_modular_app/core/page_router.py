"""
Page Router - Handles navigation and sidebar
"""
import streamlit as st
import pandas as pd


class PageRouter:
    """Handles page routing and navigation"""
    
    def __init__(self):
        self.pages = {
            "🏠 Dashboard": "dashboard",
            "👥 Player Analysis": "players", 
            "🎯 Fixture Difficulty": "fixtures",
            "👤 My FPL Team": "my_team",
            "🤖 AI Recommendations": "ai_recommendations",
            "🔄 Smart Iteration AI": "automated_iteration",
            "⚽ Team Builder": "team_builder",
        }
    
    def render_sidebar(self):
        """Render sidebar navigation and data controls"""
        st.sidebar.title("⚽ FPL Analytics")
        st.sidebar.markdown("---")
        
        # Navigation
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            list(self.pages.keys()),
            index=0
        )
        
        st.sidebar.markdown("---")
        
        # Data status
        self._render_data_status()
        
        # Data controls
        self._render_data_controls()
        
        return self.pages[selected_page]
    
    def _render_data_status(self):
        """Render data loading status"""
        if st.session_state.get('data_loaded', False):
            st.sidebar.success("✅ Data Loaded")
            if not st.session_state.get('players_df', pd.DataFrame()).empty:
                player_count = len(st.session_state.players_df)
                st.sidebar.info(f"📊 {player_count} players loaded")
        else:
            st.sidebar.warning("⚠️ No data loaded")
        
        # My FPL Team status
        if st.session_state.get('my_team_loaded', False):
            st.sidebar.success("✅ My Team Loaded")
            if 'my_team_data' in st.session_state:
                team_name = st.session_state.my_team_data.get('entry_name', 'Team')
                st.sidebar.info(f"👤 {team_name}")
    
    def _render_data_controls(self):
        """Render data loading controls"""
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            self._load_data()
        
        # Additional controls can be added here
        if st.sidebar.button("🧹 Clear Cache"):
            self._clear_cache()
    
    def _load_data(self):
        """Load FPL data"""
        with st.spinner("Loading FPL data..."):
            try:
                from services.fpl_data_service import FPLDataService
                data_service = FPLDataService()
                players_df, teams_df = data_service.load_fpl_data()
                
                if not players_df.empty:
                    st.session_state.players_df = players_df
                    st.session_state.teams_df = teams_df
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Data refreshed successfully!")
                else:
                    st.sidebar.error("❌ Failed to load data")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    def _clear_cache(self):
        """Clear all cached data"""
        keys_to_clear = [
            'data_loaded', 'players_df', 'teams_df', 
            'fdr_data_loaded', 'fixtures_df',
            'my_team_loaded', 'my_team_data'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.sidebar.success("🧹 Cache cleared!")


"""
Page Router - Handles navigation and page routing for the FPL Analytics App
"""
import streamlit as st
from typing import Dict, Any, Optional


class PageRouter:
    """Handles page navigation and routing for the FPL Analytics App"""
    
    def __init__(self):
        """Initialize the page router"""
        self.pages = {
            "Dashboard": "🏠",
            "Player Analysis": "👤", 
            "Fixture Analysis": "📅",
            "My FPL Team": "⚽",
            "Transfer Recommendations": "🔄",
            "Captain Picks": "👑",
            "Team Planning": "📋",
            "Settings": "⚙️"
        }
    
    def render_sidebar_navigation(self):
        """Render the sidebar navigation menu"""
        st.sidebar.markdown("## 🚀 FPL Analytics")
        st.sidebar.markdown("---")
        
        # Navigation menu
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            list(self.pages.keys()),
            index=0
        )
        
        st.sidebar.markdown("---")
        
        # Data status indicator
        self._render_data_status()
        
        return selected_page
    
    def _render_data_status(self):
        """Render data loading status in sidebar"""
        if 'data_loaded' in st.session_state and st.session_state.data_loaded:
            st.sidebar.success("✅ Data Loaded")
        else:
            st.sidebar.warning("⚠️ Data Not Loaded")
            
            if st.sidebar.button("🔄 Load FPL Data"):
                self._trigger_data_load()
    
    def _trigger_data_load(self):
        """Trigger FPL data loading"""
        try:
            # Import here to avoid circular imports
            from services.fpl_data_service import FPLDataService
            
            data_service = FPLDataService()
            players_df, teams_df, fixtures_df, gameweek_info = data_service.load_fpl_data()
            
            if not players_df.empty:
                data_service.cache_data(players_df, teams_df, fixtures_df, gameweek_info)
                st.sidebar.success("✅ Data loaded successfully!")
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to load data")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    def route_to_page(self, page_name: str):
        """Route to the specified page"""
        if page_name not in self.pages:
            st.error(f"Page '{page_name}' not found")
            return
        
        # Set the current page in session state
        st.session_state.current_page = page_name
        
        # Page-specific routing logic can be added here
        return page_name


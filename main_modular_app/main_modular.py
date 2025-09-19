"""
Refactored FPL Analytics Application - Main Entry Point
Uses modular architecture with separate components for better maintainability
"""
import streamlit as st
import pandas as pd
import numpy as np
from core.app_controller import FPLAppController


@st.cache_resource
def get_app_controller():
    """Create and cache the application controller instance"""
    try:
        return FPLAppController()
    except Exception as e:
        st.error(f"Error initializing app controller: {str(e)}")
        return None


@st.cache_resource
def load_fpl_static_data():
    """Cache static FPL data like teams, positions, etc."""
    try:
        # Load data that rarely changes
        # This would be implemented with actual static data loading
        pass
    except Exception as e:
        st.error(f"Error loading static data: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_fpl_gameweek_data(gameweek):
    """Cache gameweek data with TTL"""
    try:
        # Load gameweek-specific data
        # This would be implemented with actual gameweek data loading
        pass
    except Exception as e:
        st.error(f"Error loading gameweek data: {str(e)}")
        return None


@st.cache_data(ttl=300)   # Cache for 5 minutes
def load_live_fpl_data():
    """Cache live data with shorter TTL"""
    try:
        # Load frequently changing data
        # This would be implemented with actual live data loading
        pass
    except Exception as e:
        st.error(f"Error loading live data: {str(e)}")
        return None


def clean_numeric_data(df, numeric_columns):
    """Helper function to clean numeric data and handle type conversion"""
    if df is None or df.empty:
        return df
        
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN with 0 or appropriate default
            df[col] = df[col].fillna(0)
    
    return df


def main():
    """Main application entry point"""
    try:
        # Get cached application controller
        app = get_app_controller()
        
        if app is None:
            st.error("Failed to initialize application. Please refresh the page.")
            return
        
        # Clean any loaded data before processing
        if hasattr(st.session_state, 'players_df') and st.session_state.players_df is not None:
            numeric_cols = ['total_points', 'now_cost', 'form', 'selected_by_percent', 
                          'minutes', 'goals_scored', 'assists', 'clean_sheets',
                          'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index']
            st.session_state.players_df = clean_numeric_data(st.session_state.players_df, numeric_cols)
        
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("This may be related to data type issues in the Transfer Target recommendations.")
        
        # Add debug information for transfer target errors
        if "multiply sequence" in str(e):
            st.warning("⚠️ **Transfer Target Error Detected**")
            st.write("This error occurs when the AI recommendation engine tries to perform mathematical operations on mixed data types.")
            st.write("**Possible solutions:**")
            st.write("1. Refresh the page and reload data")
            st.write("2. Check that all numeric columns contain valid numbers")
            st.write("3. Try clearing the browser cache")
        
        # Add debug information
        if st.checkbox("Show debug information"):
            st.write("Error details:")
            st.exception(e)


if __name__ == "__main__":
    main()
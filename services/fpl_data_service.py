"""
FPL Data Service - Handles all FPL API interactions and data management
"""
import requests
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime


class FPLDataService:
    """Service for handling FPL API data loading and caching"""
    
    def __init__(self):
        """Initialize the FPL data service"""
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def load_fpl_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all FPL data including players, teams, fixtures, and gameweek info
        
        Returns:
            Tuple of (players_df, teams_df, fixtures_df, gameweek_info)
        """
        try:
            # Load bootstrap data (contains players, teams, and gameweek info)
            bootstrap_data = self._fetch_bootstrap_data()
            
            # Process players data
            players_df = self._process_players_data(bootstrap_data['elements'])
            
            # Process teams data
            teams_df = self._process_teams_data(bootstrap_data['teams'])
            
            # Process gameweek info
            gameweek_info = self._process_gameweek_info(bootstrap_data['events'])
            
            # Load fixtures
            fixtures_df = self._load_fixtures()
            
            return players_df, teams_df, fixtures_df, gameweek_info
            
        except Exception as e:
            st.error(f"Error loading FPL data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    def _fetch_bootstrap_data(self) -> Dict:
        """Fetch the main bootstrap data from FPL API"""
        try:
            response = self.session.get(f"{self.base_url}/bootstrap-static/", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch bootstrap data: {str(e)}")
    
    def _process_players_data(self, players_data: List[Dict]) -> pd.DataFrame:
        """Process raw players data into a pandas DataFrame"""
        try:
            players_df = pd.DataFrame(players_data)
            
            # Convert price from tenths to actual value
            if 'now_cost' in players_df.columns:
                players_df['now_cost'] = players_df['now_cost'] / 10.0
            
            # Add calculated fields
            if 'total_points' in players_df.columns and 'now_cost' in players_df.columns:
                players_df['points_per_million'] = players_df['total_points'] / players_df['now_cost']
            
            # Add position names
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            if 'element_type' in players_df.columns:
                players_df['position'] = players_df['element_type'].map(position_map)
            
            # Convert percentage strings to floats
            if 'selected_by_percent' in players_df.columns:
                players_df['selected_by_percent'] = pd.to_numeric(
                    players_df['selected_by_percent'], errors='coerce'
                ).fillna(0)
            
            # Handle expected stats
            for col in ['expected_goals', 'expected_assists', 'expected_goal_involvements']:
                if col in players_df.columns:
                    players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
            
            return players_df
            
        except Exception as e:
            raise Exception(f"Failed to process players data: {str(e)}")
    
    def _process_teams_data(self, teams_data: List[Dict]) -> pd.DataFrame:
        """Process raw teams data into a pandas DataFrame"""
        try:
            teams_df = pd.DataFrame(teams_data)
            
            # Add team strength calculations
            if 'strength_overall_home' in teams_df.columns and 'strength_overall_away' in teams_df.columns:
                teams_df['strength_overall'] = (
                    teams_df['strength_overall_home'] + teams_df['strength_overall_away']
                ) / 2
            
            return teams_df
            
        except Exception as e:
            raise Exception(f"Failed to process teams data: {str(e)}")
    
    def _process_gameweek_info(self, events_data: List[Dict]) -> Dict:
        """Process gameweek information"""
        try:
            # Find current gameweek
            current_gw = None
            next_gw = None
            
            for event in events_data:
                if event.get('is_current', False):
                    current_gw = event
                elif event.get('is_next', False):
                    next_gw = event
            
            return {
                'current_gameweek': current_gw,
                'next_gameweek': next_gw,
                'all_gameweeks': events_data
            }
            
        except Exception as e:
            return {}
    
    def _load_fixtures(self) -> pd.DataFrame:
        """Load fixtures data from FPL API"""
        try:
            response = self.session.get(f"{self.base_url}/fixtures/", timeout=30)
            response.raise_for_status()
            fixtures_data = response.json()
            
            fixtures_df = pd.DataFrame(fixtures_data)
            
            # Convert kickoff times
            if 'kickoff_time' in fixtures_df.columns:
                fixtures_df['kickoff_time'] = pd.to_datetime(
                    fixtures_df['kickoff_time'], errors='coerce'
                )
            
            return fixtures_df
            
        except Exception as e:
            st.warning(f"Could not load fixtures: {str(e)}")
            return pd.DataFrame()
    
    def get_cached_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Get cached FPL data from session state"""
        if (
            'players_df' in st.session_state and 
            'teams_df' in st.session_state and
            'fixtures_df' in st.session_state and
            'gameweek_info' in st.session_state
        ):
            return (
                st.session_state.players_df,
                st.session_state.teams_df,
                st.session_state.fixtures_df,
                st.session_state.gameweek_info
            )
        else:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    def cache_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame, 
                   fixtures_df: pd.DataFrame, gameweek_info: Dict):
        """Cache FPL data in session state"""
        st.session_state.players_df = players_df
        st.session_state.teams_df = teams_df
        st.session_state.fixtures_df = fixtures_df
        st.session_state.gameweek_info = gameweek_info
        st.session_state.data_loaded = True
        st.session_state.data_load_time = datetime.now()


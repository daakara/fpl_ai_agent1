"""
FPL Data Service - Handles all data loading and processing
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
import urllib3

# Suppress warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FPLDataService:
    """Service for loading and processing FPL data"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
    
    def load_fpl_data(self):
        """Load comprehensive FPL data from API"""
        try:
            url = f"{self.base_url}/bootstrap-static/"
            response = requests.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            data = response.json()
            
            # Process data
            players_df = self._process_players_data(data)
            teams_df = self._process_teams_data(data)
            
            return players_df, teams_df
            
        except Exception as e:
            st.error(f"Error loading FPL data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _process_players_data(self, data):
        """Process and enhance players data"""
        players_df = pd.DataFrame(data['elements'])
        teams_df = pd.DataFrame(data['teams'])
        element_types_df = pd.DataFrame(data['element_types'])
        
        # Create lookup dictionaries
        team_lookup = dict(zip(teams_df['id'], teams_df['name']))
        team_short_lookup = dict(zip(teams_df['id'], teams_df['short_name']))
        position_lookup = dict(zip(element_types_df['id'], element_types_df['singular_name']))
        
        # Add team and position information
        players_df['team_name'] = players_df['team'].map(team_lookup)
        players_df['team_short_name'] = players_df['team'].map(team_short_lookup)
        players_df['position_name'] = players_df['element_type'].map(position_lookup)
        
        # Fill missing values
        players_df['team_name'] = players_df['team_name'].fillna('Unknown Team')
        players_df['team_short_name'] = players_df['team_short_name'].fillna('UNK')
        players_df['position_name'] = players_df['position_name'].fillna('Unknown Position')
        
        # Calculate derived metrics
        players_df['cost_millions'] = players_df['now_cost'] / 10
        players_df['points_per_million'] = np.where(
            players_df['cost_millions'] > 0,
            players_df['total_points'] / players_df['cost_millions'],
            0
        ).round(2)
        
        # Ensure numeric columns
        players_df['form'] = pd.to_numeric(players_df.get('form', 0), errors='coerce').fillna(0.0)
        players_df['selected_by_percent'] = pd.to_numeric(players_df.get('selected_by_percent', 0), errors='coerce').fillna(0.0)
        
        return players_df
    
    def _process_teams_data(self, data):
        """Process teams data"""
        teams_df = pd.DataFrame(data['teams'])
        return teams_df
    
    def get_current_gameweek(self):
        """Get current gameweek from FPL API"""
        try:
            url = f"{self.base_url}/bootstrap-static/"
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()
            
            events = data.get('events', [])
            current_event = next((event for event in events if event['is_current']), None)
            return current_event['id'] if current_event else 1
        except:
            return 1
    
    def load_team_data(self, team_id, gameweek=None):
        """Load specific FPL team data"""
        try:
            if gameweek is None:
                gameweek = self.get_current_gameweek()
            
            # FPL API endpoints
            entry_url = f"{self.base_url}/entry/{team_id}/"
            picks_url = f"{self.base_url}/entry/{team_id}/event/{gameweek}/picks/"
            
            # Load entry data
            entry_response = requests.get(entry_url, timeout=10, verify=False)
            entry_response.raise_for_status()
            entry_data = entry_response.json()
            
            # Load picks data
            try:
                picks_response = requests.get(picks_url, timeout=10, verify=False)
                picks_response.raise_for_status()
                picks_data = picks_response.json()
                entry_data['picks'] = picks_data.get('picks', [])
                entry_data['gameweek'] = gameweek
            except:
                entry_data['picks'] = []
                entry_data['gameweek'] = gameweek
            
            return entry_data
            
        except Exception as e:
            st.error(f"Error loading team data: {str(e)}")
            return None


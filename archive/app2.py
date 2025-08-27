from typing import Tuple, List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import logging
import requests
import json
import time
import traceback
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import sys
import os
from collections import Counter
from dotenv import load_dotenv
from st_aggrid import AgGrid, GridOptionsBuilder

# Import our modular components
from app_config import AppConfig, SessionStateManager
from fixture_manager import FPLAPIFixtureSource, FixtureDifficultyCalculator
from data_processor import PlayerDataProcessor, InjuryDataProcessor, TeamMappingProcessor
from ai_chat_manager import ChatManager, AIResponseGenerator
from data_loader import FPLDataLoader, DataFetcher, FPLDataProcessor, SeleniumSetup, cache
from fpl_official import get_players_data, get_teams_data, get_fixtures_data_async  # Add get_fixtures_data_async here
from fpl_myteam import load_my_fpl_team  # This should use the updated version from fpl_myteam.py

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing from the root team_recommender.py file
try:
    import team_recommender as tr
    TeamOptimizer = tr.TeamOptimizer
    get_latest_team_recommendations = tr.get_latest_team_recommendations
    valid_team_formation = tr.valid_team_formation
except ImportError as e:
    st.error(f"Could not import team_recommender: {e}")
    # Create dummy implementations to prevent app crash
    class TeamOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def optimize_team(self):
            return None
    
    def get_latest_team_recommendations(*args, **kwargs):
        return {
            'team': pd.DataFrame(columns=['web_name', 'element_type', 'team_name', 'now_cost']),
            'formation': (3, 4, 3),
            'captain': 1,
            'vice_captain': 2,
            'total_expected_points': 0,
            'total_cost': 1000,
            'rationale': {'message': 'Team recommender not available'}
        }
    
    def valid_team_formation(*args, **kwargs):
        return False

from llm_integration import test_cohere_connection

load_dotenv()

# ---------------------- MOVE THE StyleManager CLASS HERE ----------------------
class StyleManager:
    """Manages custom styling for the Streamlit app"""
    
    def __init__(self):
        pass
    
    def load_styles(self):
        """Load custom CSS styles"""
        try:
            custom_css = """
            <style>
            /* Custom styles for FPL App */
            .main {
                padding-top: 1rem;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #f0f2f6;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-left: 20px;
                padding-right: 20px;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #ffffff;
            }
            
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #e9ecef;
                margin: 0.5rem 0;
            }
            
            .player-card {
                background-color: #ffffff;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #dee2e6;
                margin: 0.5rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .transfer-suggestion {
                background-color: #e8f5e8;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #28a745;
                margin: 0.5rem 0;
            }
            
            .warning-box {
                background-color: #fff3cd;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #ffc107;
                margin: 0.5rem 0;
            }
            
            .error-box {
                background-color: #f8d7da;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #dc3545;
                margin: 0.5rem 0;
            }
            </style>
            """
            st.markdown(custom_css, unsafe_allow_html=True)
            
        except Exception as e:
            # If styling fails, log the error but don't break the app
            logging.warning(f"Failed to load custom styles: {e}")
    
    def apply_metric_style(self, container):
        """Apply metric card styling to a container"""
        with container:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    def close_metric_style(self, container):
        """Close metric card styling"""
        with container:
            st.markdown('</div>', unsafe_allow_html=True)
    
    def apply_player_card_style(self):
        """Apply player card styling"""
        return st.markdown('<div class="player-card">', unsafe_allow_html=True)
    
    def apply_transfer_suggestion_style(self):
        """Apply transfer suggestion styling"""
        return st.markdown('<div class="transfer-suggestion">', unsafe_allow_html=True)
    
    def apply_warning_style(self):
        """Apply warning box styling"""
        return st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    
    def apply_error_style(self):
        """Apply error box styling"""
        return st.markdown('<div class="error-box">', unsafe_allow_html=True)
# ---------------------- END OF StyleManager CLASS ----------------------

class DataManager:
    """Manages data loading and processing for FPL data"""
    
    def __init__(self, fpl_team_id: str = None):
        self.fpl_team_id = fpl_team_id
        
        # Initialize data containers
        self.df_players = None
        self.teams = []
        self.info = {}
        self.picks = []
        self.transfers = []
        self.chips = []
        self.predictions = []
        self.live_points = []
        self.team_id_to_name = {}
        
        # Initialize processors
        from data_processor import PlayerProcessor, TeamMappingProcessor
        self.player_processor = PlayerProcessor()
        self.team_mapping_processor = TeamMappingProcessor()
        
        # Initialize data loaders
        from data_loader import FPLDataLoader
        self.fpl_data_loader = FPLDataLoader()

    async def load_fpl_data(self):
        """Load all FPL data from API"""
        try:
            import requests
            import asyncio
            
            # Load bootstrap data
            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30)
            if response.status_code != 200:
                raise Exception(f"FPL API returned status {response.status_code}")
            
            data = response.json()
            
            players = data.get('elements', [])
            teams = data.get('teams', [])
            
            # Initialize other data as empty for now
            info = {}
            picks = []
            transfers = []
            chips = []
            predictions = []
            live_points = []
            
            # If team ID is provided, try to load team data
            if self.fpl_team_id:
                try:
                    team_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{self.fpl_team_id}/", timeout=10)
                    if team_response.status_code == 200:
                        info = team_response.json()
                        
                        # Try to get current picks
                        current_gw = info.get('current_event', 1)
                        if current_gw:
                            picks_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{self.fpl_team_id}/event/{current_gw}/picks/", timeout=10)
                            if picks_response.status_code == 200:
                                picks_data = picks_response.json()
                                picks = picks_data.get('picks', [])
                
                except Exception as team_error:
                    st.warning(f"Could not load team data: {team_error}")
            
            return players, teams, info, picks, transfers, chips, predictions, live_points
            
        except Exception as e:
            st.error(f"Error loading FPL data: {e}")
            return [], [], {}, [], [], [], [], []

    async def load_and_process_data(self):
        """Main data loading and processing pipeline"""
        try:
            st.info("🔄 Loading FPL data...")
            
            # Load FPL data
            players, teams, info, picks, transfers, chips, predictions, live_points = await self.load_fpl_data()
            
            if not players:
                st.error("❌ Failed to load player data")
                return False
            
            if not teams:
                st.error("❌ Failed to load team data")
                return False
            
            st.success(f"✅ Loaded {len(players)} players and {len(teams)} teams")
            
            # Store data
            self.teams = teams
            self.info = info
            self.picks = picks
            self.transfers = transfers
            self.chips = chips
            self.predictions = predictions
            self.live_points = live_points
            
            # Convert players to DataFrame
            self.df_players = pd.DataFrame(players)
            
            # Create team mapping
            self.team_id_to_name = {team['id']: team['name'] for team in teams}
            
            # Add team names to players
            if 'team' in self.df_players.columns:
                self.df_players['team_name'] = self.df_players['team'].map(self.team_id_to_name)
            
            # Add basic calculated fields
            if 'now_cost' in self.df_players.columns:
                self.df_players['cost_millions'] = self.df_players['now_cost'] / 10
            
            # Add position names
            position_map = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
            if 'element_type' in self.df_players.columns:
                self.df_players['position_name'] = self.df_players['element_type'].map(position_map)
            
            st.success("✅ Data processing completed!")
            return True
            
        except Exception as e:
            st.error(f"❌ Critical error in data loading: {e}")
            st.code(traceback.format_exc())
            
            # Initialize with empty data to prevent app crash
            self.df_players = pd.DataFrame()
            self.teams = []
            self.info = {}
            return False

    def get_player_summary(self):
        """Get a summary of loaded player data"""
        if self.df_players is None or self.df_players.empty:
            return "No player data loaded"
        
        summary = f"""
        📊 **Data Summary:**
        - Total Players: {len(self.df_players)}
        - Teams: {len(self.teams)}
        - Columns: {len(self.df_players.columns)}
        """
        
        if 'total_points' in self.df_players.columns:
            top_scorer = self.df_players.loc[self.df_players['total_points'].idxmax()]
            summary += f"\n- Top Scorer: {top_scorer.get('web_name', 'Unknown')} ({top_scorer.get('total_points', 0)} points)"
        
        return summary
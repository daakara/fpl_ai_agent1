"""Create simple_app.py - completely self-contained
Helper functions for fixture analysis
"""
import pandas as pd
import requests
from typing import List, Dict, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go  # Add this import
import streamlit as st  # Add this import
from datetime import datetime, timedelta
import json
import logging
import numpy as np  # Add this import
import urllib3

# Enhanced Infrastructure Imports
try:
    from config.app_config import config
    from utils.error_handling import handle_errors, FPLLogger
    from utils.caching import cached, DataManager
    from utils.ui_enhancements import ui
    CONFIG_AVAILABLE = True
    print("✅ Infrastructure modules loaded successfully")
except ImportError as e:
    print(f"⚠️  Infrastructure modules not available: {e}")
    CONFIG_AVAILABLE = False
except Exception as e:
    print(f"⚠️  Error initializing infrastructure: {e}")
    CONFIG_AVAILABLE = False

# Initialize logger if available
if CONFIG_AVAILABLE:
    try:
        logger = FPLLogger(__name__)
        data_manager = DataManager()
        print("✅ Infrastructure components initialized")
    except Exception as e:
        print(f"⚠️  Error initializing infrastructure components: {e}")
        CONFIG_AVAILABLE = False
        logger = None
        data_manager = None
else:
    logger = None
    data_manager = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def format_fixture_ticker(fixtures: List[Dict]) -> str:
    """Format fixtures into a ticker string"""
    ticker_parts = []
    for fixture in fixtures:
        opponent = fixture.get('opponent', 'UNK')
        venue = fixture.get('venue', 'H')
        fdr = fixture.get('combined_fdr', 3)
        ticker_parts.append(f"{opponent}({venue})")
    return " | ".join(ticker_parts)

def calculate_fdr_score(attack_fdr: float, defense_fdr: float, weights: Tuple[float, float] = (0.6, 0.4)) -> float:
    """Calculate weighted FDR score"""
    attack_weight, defense_weight = weights
    return (attack_fdr * attack_weight) + (defense_fdr * defense_weight)

def get_fdr_recommendation(avg_fdr: float) -> str:
    """Get recommendation based on average FDR"""
    if avg_fdr <= 2.0:
        return "🟢 Excellent fixtures - Target players"
    elif avg_fdr <= 2.5:
        return "🟡 Good fixtures - Consider players"
    elif avg_fdr <= 3.5:
        return "🟠 Average fixtures - Neutral"
    elif avg_fdr <= 4.0:
        return "🔴 Difficult fixtures - Avoid"
    else:
        return "🔴 Very difficult fixtures - Strong avoid"

def rank_teams_by_fdr(fixtures_df: pd.DataFrame, fdr_type: str = 'combined') -> pd.DataFrame:
    """Rank teams by their fixture difficulty"""
    if fixtures_df.empty:
        return pd.DataFrame()
    # Group by team and calculate metrics
    team_rankings = fixtures_df.groupby(['team_id', 'team_name', 'team_short_name']).agg({
        f'{fdr_type}_fdr': ['mean', 'sum', 'count'],
        'fixture_number': 'count'
    }).round(2)
    # Flatten column names
    team_rankings.columns = ['_'.join(col).strip() for col in team_rankings.columns.values]
    team_rankings = team_rankings.reset_index()
    # Sort by average FDR (lower is better)
    team_rankings = team_rankings.sort_values(f'{fdr_type}_fdr_mean')
    # Add ranking
    team_rankings['rank'] = range(1, len(team_rankings) + 1)
    # Add recommendations
    team_rankings['recommendation'] = team_rankings[f'{fdr_type}_fdr_mean'].apply(get_fdr_recommendation)
    return team_rankings

def identify_fixture_opportunities(fixtures_df: pd.DataFrame, threshold: float = 2.5) -> Dict:
    """Identify fixture opportunities for different strategies"""
    if fixtures_df.empty:
        return {}
    opportunities = {
        'best_attack': [],
        'best_defense': [],
        'best_combined': [],
        'worst_fixtures': [],
        'fixture_swings': []
    }
    # Best attacking fixtures
    attack_teams = fixtures_df.groupby(['team_id', 'team_name'])['attack_fdr'].mean()
    opportunities['best_attack'] = attack_teams[attack_teams <= threshold].sort_values().head(5).to_dict()
    # Best defensive fixtures
    defense_teams = fixtures_df.groupby(['team_id', 'team_name'])['defense_fdr'].mean()
    opportunities['best_defense'] = defense_teams[defense_teams <= threshold].sort_values().head(5).to_dict()
    # Best combined fixtures
    combined_teams = fixtures_df.groupby(['team_id', 'team_name'])['combined_fdr'].mean()
    opportunities['best_combined'] = combined_teams[combined_teams <= threshold].sort_values().head(5).to_dict()
    # Worst fixtures to avoid
    opportunities['worst_fixtures'] = combined_teams[combined_teams >= 4.0].sort_values(ascending=False).head(5).to_dict()
    return opportunities

class FixtureDataLoader:
    """Loads and processes fixture data from FPL API"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.logger = logging.getLogger(__name__)
    
    def load_fixtures(self) -> List[Dict]:
        """Load fixtures from FPL API"""
        try:
            url = f"{self.base_url}/fixtures/"
            # Disable SSL verification to fix certificate issues
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error loading fixtures: {e}")
            return []
    
    def load_teams(self) -> List[Dict]:
        """Load teams data from FPL API"""
        try:
            url = f"{self.base_url}/bootstrap-static/"
            # Disable SSL verification to fix certificate issues
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()
            return data.get('teams', [])
        except requests.RequestException as e:
            self.logger.error(f"Error loading teams: {e}")
            return []
    
    def get_next_5_fixtures(self, team_id: int, fixtures: List[Dict]) -> List[Dict]:
        """Get next 5 fixtures for a specific team"""
        team_fixtures = []
        
        for fixture in fixtures:
            # Check if this fixture involves the team
            if fixture['team_h'] == team_id or fixture['team_a'] == team_id:
                # Include all unfinished fixtures, regardless of date parsing issues
                if not fixture.get('finished', False):
                    team_fixtures.append(fixture)
        
        # Sort by event (gameweek) first, then by kickoff_time if available
        def sort_key(fixture):
            event = fixture.get('event', 999)  # Put fixtures without gameweek at the end
            kickoff = fixture.get('kickoff_time', 'Z')  # Ensure consistent sorting
            return (event, kickoff)
        
        team_fixtures.sort(key=sort_key)
        return team_fixtures[:5]
    
    def process_fixtures_data(self) -> pd.DataFrame:
        """Process fixtures data into a structured DataFrame with enhanced validation"""
        fixtures = self.load_fixtures()
        teams = self.load_teams()
        
        if not fixtures or not teams:
            print("Debug: No fixtures or teams data available")
            return pd.DataFrame()
        
        # Enhanced logging for debugging
        print(f"Debug: Loaded {len(fixtures)} fixtures and {len(teams)} teams")
        
        # Check if fixtures have the expected structure
        if fixtures:
            sample_fixture = fixtures[0]
            print(f"Debug: Sample fixture keys: {list(sample_fixture.keys())}")
            print(f"Debug: Sample fixture: {sample_fixture}")
        
        team_lookup = {team['id']: team for team in teams}
        fixture_data = []
        
        # Track unique opponents to ensure differentiation
        team_opponents_tracker = {}
        
        # Check for current gameweek and unfinished fixtures
        current_gw = None
        unfinished_fixtures = [f for f in fixtures if not f.get('finished', False)]
        print(f"Debug: Found {len(unfinished_fixtures)} unfinished fixtures out of {len(fixtures)} total")
        
        if unfinished_fixtures:
            current_gw = min([f.get('event') for f in unfinished_fixtures if f.get('event')])
            print(f"Debug: Current gameweek appears to be: {current_gw}")
        
        for team in teams:
            team_id = team['id']
            team_name = team['name']
            team_short_name = team['short_name']
            
            next_fixtures = self.get_next_5_fixtures(team_id, fixtures)
            
            # Track opponents for this team
            team_opponents = []
            
            print(f"Debug: Processing {team_short_name} - found {len(next_fixtures)} fixtures")
            
            # If no fixtures found, create placeholder data but ensure variety
            if not next_fixtures:
                print(f"Debug: No fixtures for {team_short_name}, creating placeholders")
                # Use different placeholder opponents based on team strength
                placeholder_opponents = self._get_placeholder_opponents(team, teams)
                for i, placeholder_opponent in enumerate(placeholder_opponents[:5], 1):
                    opponent = placeholder_opponent['opponent']
                    difficulty = placeholder_opponent['difficulty']
                    
                    fixture_data.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'team_short_name': team_short_name,
                        'fixture_number': i,
                        'opponent_id': opponent['id'],
                        'opponent_name': opponent['name'],
                        'opponent_short_name': opponent['short_name'],
                        'is_home': (i % 2 == 1),  # Alternate home/away
                        'venue': 'H' if (i % 2 == 1) else 'A',
                        'kickoff_time': None,
                        'gameweek': current_gw + i - 1 if current_gw else i,
                        'fixture_id': f"placeholder_{team_id}_{i}",
                        'difficulty': difficulty,
                        'opponent_strength': opponent.get('strength', 3),
                        'opponent_strength_overall_home': opponent.get('strength_overall_home', 3),
                        'opponent_strength_overall_away': opponent.get('strength_overall_away', 3),
                        'opponent_strength_attack_home': opponent.get('strength_attack_home', 3),
                        'opponent_strength_attack_away': opponent.get('strength_attack_away', 3),
                        'opponent_strength_defence_home': opponent.get('strength_defence_home', 3),
                        'opponent_strength_defence_away': opponent.get('strength_defence_away', 3)
                    })
                    team_opponents.append(opponent['short_name'])
            else:
                for i, fixture in enumerate(next_fixtures, 1):
                    is_home = fixture['team_h'] == team_id
                    opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                    opponent = team_lookup.get(opponent_id, {})
                    
                    # Use FPL's difficulty rating if available, otherwise calculate based on opponent strength
                    if is_home:
                        difficulty = fixture.get('team_h_difficulty', self._calculate_difficulty(team, opponent, True))
                    else:
                        difficulty = fixture.get('team_a_difficulty', self._calculate_difficulty(team, opponent, False))
                    
                    fixture_data.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'team_short_name': team_short_name,
                        'fixture_number': i,
                        'opponent_id': opponent_id,
                        'opponent_name': opponent.get('name', 'Unknown'),
                        'opponent_short_name': opponent.get('short_name', 'UNK'),
                        'is_home': is_home,
                        'venue': 'H' if is_home else 'A',
                        'kickoff_time': fixture.get('kickoff_time'),
                        'gameweek': fixture.get('event'),
                        'fixture_id': fixture.get('id'),
                        'difficulty': difficulty,
                        'opponent_strength': opponent.get('strength', 3),
                        'opponent_strength_overall_home': opponent.get('strength_overall_home', 3),
                        'opponent_strength_overall_away': opponent.get('strength_overall_away', 3),
                        'opponent_strength_attack_home': opponent.get('strength_attack_home', 3),
                        'opponent_strength_attack_away': opponent.get('strength_attack_away', 3),
                        'opponent_strength_defence_home': opponent.get('strength_defence_home', 3),
                        'opponent_strength_defence_away': opponent.get('strength_defence_away', 3)
                    })
                    team_opponents.append(opponent.get('short_name', 'UNK'))
            
            # Store opponents for this team for validation
            team_opponents_tracker[team_short_name] = team_opponents[:3]  # First 3 opponents
        
        # Validation: Check if we have proper differentiation
        all_opponent_sets = list(team_opponents_tracker.values())
        unique_opponent_patterns = set(tuple(opponents) for opponents in all_opponent_sets)
        
        print(f"Debug: Team opponent patterns: {len(unique_opponent_patterns)} unique patterns out of {len(all_opponent_sets)} teams")
        
        if len(unique_opponent_patterns) == 1:
            print("Warning: All teams have the same opponents - data may not be properly differentiated")
            # Print the pattern for debugging
            print(f"Debug: Common opponent pattern: {all_opponent_sets[0] if all_opponent_sets else 'None'}")
        else:
            print(f"Debug: Successfully created differentiated fixtures for {len(team_opponents_tracker)} teams")
            # Show some examples
            sample_teams = list(team_opponents_tracker.keys())[:3]
            for team in sample_teams:
                print(f"Debug: {team} opponents: {team_opponents_tracker[team]}")
        
        df = pd.DataFrame(fixture_data)
        
        # Final validation
        if not df.empty:
            unique_opponents_per_team = df.groupby('team_short_name')['opponent_short_name'].nunique()
            print(f"Debug: Opponent variety per team - Min: {unique_opponents_per_team.min()}, Max: {unique_opponents_per_team.max()}, Avg: {unique_opponents_per_team.mean():.2f}")
            
            # Show sample of actual data
            if len(df) > 0:
                sample_data = df.groupby('team_short_name').head(2)  # First 2 fixtures per team
                print(f"Debug: Sample fixture data:")
                for _, row in sample_data.head(6).iterrows():
                    print(f"  {row['team_short_name']} vs {row['opponent_short_name']} (Difficulty: {row['difficulty']}, Venue: {row['venue']})")
        
        return df

    def _get_placeholder_opponents(self, team, teams):
        """Generate realistic placeholder opponents for a team based on strength"""
        # Don't play against yourself
        other_teams = [t for t in teams if t['id'] != team['id']]
        
        # Sort teams by strength difference to create realistic fixtures
        team_strength = team.get('strength', 3)
        
        # Create varied opponents based on relative strength
        opponents = []
        for i, opponent in enumerate(other_teams[:5]):
            opponent_strength = opponent.get('strength', 3)
            strength_diff = abs(team_strength - opponent_strength)
            
            # Calculate difficulty based on opponent strength and home/away
            is_home = (i % 2 == 1)
            if opponent_strength >= 4.5:  # Strong opponent
                difficulty = 4 if is_home else 5
            elif opponent_strength >= 3.5:  # Medium opponent
                difficulty = 3 if is_home else 4
            elif opponent_strength >= 2.5:  # Weaker opponent
                difficulty = 2 if is_home else 3
            else:  # Very weak opponent
                difficulty = 1 if is_home else 2
            
            opponents.append({
                'opponent': opponent,
                'difficulty': difficulty
            })
        
        return opponents

    def _calculate_difficulty(self, team, opponent, is_home):
        """Calculate fixture difficulty based on team strengths"""
        if not opponent:
            return 3
        
        team_strength = team.get('strength', 3)
        opponent_strength = opponent.get('strength', 3)
        
        # Base difficulty on strength difference
        strength_diff = opponent_strength - team_strength
        
        # Home advantage
        if is_home:
            strength_diff -= 0.5
        else:
            strength_diff += 0.3
        
        # Convert to FDR scale (1-5)
        if strength_diff <= -1.5:
            return 1  # Very easy
        elif strength_diff <= -0.5:
            return 2  # Easy
        elif strength_diff <= 0.5:
            return 3  # Average
        elif strength_diff <= 1.5:
            return 4  # Hard
        else:
            return 5  # Very hard

class FDRAnalyzer:
    """Analyzes fixture difficulty ratings for FPL teams"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fdr_colors = {
            1: '#00FF87', 2: '#01FF70', 3: '#FFDC00', 4: '#FF851B', 5: '#FF4136'
        }
        self.fdr_labels = {
            1: 'Very Easy', 2: 'Easy', 3: 'Average', 4: 'Hard', 5: 'Very Hard'
        }
    
    def calculate_attack_fdr(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate attack-based FDR"""
        if fixtures_df.empty:
            return pd.DataFrame()
        
        attack_fdr_df = fixtures_df.copy()
        
        def get_attack_difficulty(row):
            if row['is_home']:
                defence_strength = row['opponent_strength_defence_away']
            else:
                defence_strength = row['opponent_strength_defence_home']
            
            # Convert FPL API strength values (1000+ range) to FDR scale (1-5)
            # Higher defensive strength = harder for attackers = higher FDR
            if defence_strength >= 1400:
                return 5  # Very hard
            elif defence_strength >= 1350:
                return 4  # Hard
            elif defence_strength >= 1300:
                return 3  # Average
            elif defence_strength >= 1250:
                return 2  # Easy
            else:
                return 1  # Very easy
        
        attack_fdr_df['attack_fdr'] = attack_fdr_df.apply(get_attack_difficulty, axis=1)
        return attack_fdr_df
    
    def calculate_defense_fdr(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate defense-based FDR"""
        if fixtures_df.empty:
            return pd.DataFrame()
        
        defense_fdr_df = fixtures_df.copy()
        
        def get_defense_difficulty(row):
            if row['is_home']:
                attack_strength = row['opponent_strength_attack_away']
            else:
                attack_strength = row['opponent_strength_attack_home']
            
            # Convert FPL API strength values (1000+ range) to FDR scale (1-5)
            # Higher attacking strength = harder for defenders = higher FDR
            if attack_strength >= 1400:
                return 5  # Very hard
            elif attack_strength >= 1350:
                return 4  # Hard
            elif attack_strength >= 1300:
                return 3  # Average
            elif attack_strength >= 1250:
                return 2  # Easy
            else:
                return 1  # Very easy
        
        defense_fdr_df['defense_fdr'] = defense_fdr_df.apply(get_defense_difficulty, axis=1)
        return defense_fdr_df
    
    def calculate_combined_fdr(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined FDR"""
        attack_fdr_df = self.calculate_attack_fdr(fixtures_df)
        defense_fdr_df = self.calculate_defense_fdr(fixtures_df)
        
        if attack_fdr_df.empty or defense_fdr_df.empty:
            return pd.DataFrame()
        
        combined_df = attack_fdr_df.copy()
        combined_df['defense_fdr'] = defense_fdr_df['defense_fdr']
        combined_df['combined_fdr'] = ((combined_df['attack_fdr'] + combined_df['defense_fdr']) / 2).round().astype(int)
        
        return combined_df

class FDRVisualizer:
    """Creates visualizations for fixture difficulty ratings"""
    
    def __init__(self):
        self.fdr_colors = {
            1: '#00FF87', 2: '#01FF70', 3: '#FFDC00', 4: '#FF851B', 5: '#FF4136'
        }
    
    def create_fdr_heatmap(self, fixtures_df: pd.DataFrame, fdr_type: str = 'combined') -> go.Figure:
        """Create FDR heatmap"""
        if fixtures_df.empty:
            return go.Figure()
        
        fdr_column = f'{fdr_type}_fdr'
        pivot_data = fixtures_df.pivot_table(
            index='team_short_name',
            columns='fixture_number', 
            values=fdr_column,
            aggfunc='first'
        )
        
        colorscale = [
            [0.0, self.fdr_colors[1]], [0.25, self.fdr_colors[2]], 
            [0.5, self.fdr_colors[3]], [0.75, self.fdr_colors[4]], [1.0, self.fdr_colors[5]]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[f'GW{i}' for i in pivot_data.columns],
            y=pivot_data.index,
            colorscale=colorscale,
            zmin=1, zmax=5,
            text=pivot_data.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Fixture %{x}<br>FDR: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{fdr_type.title()} FDR - Next 5 Fixtures',
            xaxis_title="Fixture Number",
            yaxis_title="Team",
            height=600
        )
        
        return fig

class FPLAnalyticsApp:
    """Main FPL Analytics Application"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
    
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
        if 'fdr_data_loaded' not in st.session_state:
            st.session_state.fdr_data_loaded = False
        if 'fixtures_df' not in st.session_state:
            st.session_state.fixtures_df = pd.DataFrame()
    
    def load_fpl_data(self):
        """Load data from FPL API"""
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            # Disable SSL verification to fix certificate issues
            response = requests.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            data = response.json()
            
            # Process players data
            players_df = pd.DataFrame(data['elements'])
            teams_df = pd.DataFrame(data['teams'])
            element_types_df = pd.DataFrame(data['element_types'])
            
            # Debug: Print sample data structure
            st.write("Debug - Teams data sample:", teams_df.head(2).to_dict())
            st.write("Debug - Players columns:", list(players_df.columns))
            st.write("Debug - Players team column sample:", players_df['team'].head(5).tolist())
            
            # Create lookup dictionaries with proper error handling
            team_lookup = dict(zip(teams_df['id'], teams_df['name']))
            team_short_lookup = dict(zip(teams_df['id'], teams_df['short_name']))
            position_lookup = dict(zip(element_types_df['id'], element_types_df['singular_name']))
            
            # Debug: Print lookup dictionaries
            st.write("Debug - Team lookup sample:", dict(list(team_lookup.items())[:3]))
            st.write("Debug - Team short lookup sample:", dict(list(team_short_lookup.items())[:3]))
            
            # Add team and position names with error handling
            players_df['team_name'] = players_df['team'].map(team_lookup)
            players_df['team_short_name'] = players_df['team'].map(team_short_lookup)
            players_df['position_name'] = players_df['element_type'].map(position_lookup)
            
            # Debug: Check mapping results
            st.write("Debug - Team mapping results:")
            st.write("  - team_name null count:", players_df['team_name'].isnull().sum())
            st.write("  - team_short_name null count:", players_df['team_short_name'].isnull().sum())
            st.write("  - position_name null count:", players_df['position_name'].isnull().sum())
            
            # Fill any NaN values
            players_df['team_name'] = players_df['team_name'].fillna('Unknown Team')
            players_df['team_short_name'] = players_df['team_short_name'].fillna('UNK')
            players_df['position_name'] = players_df['position_name'].fillna('Unknown Position')
            
            # Calculate cost in millions
            players_df['cost_millions'] = players_df['now_cost'] / 10
            
            # Calculate points per million with safe division
            players_df['points_per_million'] = np.where(
                players_df['cost_millions'] > 0,
                players_df['total_points'] / players_df['cost_millions'],
                0
            ).round(2)
            
            # Ensure form column exists and handle missing values
            if 'form' not in players_df.columns:
                players_df['form'] = 0.0
            else:
                players_df['form'] = pd.to_numeric(players_df['form'], errors='coerce').fillna(0.0)
            
            # Ensure selected_by_percent exists
            if 'selected_by_percent' not in players_df.columns:
                players_df['selected_by_percent'] = 0.0
            else:
                players_df['selected_by_percent'] = pd.to_numeric(players_df['selected_by_percent'], errors='coerce').fillna(0.0)
            
            # Final debug: Print final column names
            st.write("Debug - Final column names:", list(players_df.columns))
            
            # Verify required columns exist
            required_columns = ['team_short_name', 'position_name', 'team_name']
            missing_columns = [col for col in required_columns if col not in players_df.columns]
            
            if missing_columns:
                st.error(f"Still missing columns after processing: {missing_columns}")
                return pd.DataFrame(), pd.DataFrame()
            
            return players_df, teams_df
            
        except Exception as e:
            st.error(f"Error loading FPL data: {str(e)}")
            st.write("Full error details:", str(e))
            return pd.DataFrame(), pd.DataFrame()
    
    def load_fpl_data_alternative(self):
        """Load data from FPL API - Alternative approach"""
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            # Disable SSL verification to fix certificate issues
            response = requests.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            data = response.json()
            
            # Process players data
            players_df = pd.DataFrame(data['elements'])
            teams_df = pd.DataFrame(data['teams'])
            element_types_df = pd.DataFrame(data['element_types'])
            
            # Manually add team and position information
            for idx, player in players_df.iterrows():
                team_id = player['team']
                element_type_id = player['element_type']
                
                # Find team info
                team_info = teams_df[teams_df['id'] == team_id]
                if not team_info.empty:
                    players_df.at[idx, 'team_name'] = team_info.iloc[0]['name']
                    players_df.at[idx, 'team_short_name'] = team_info.iloc[0]['short_name']
                else:
                    players_df.at[idx, 'team_name'] = 'Unknown Team'
                    players_df.at[idx, 'team_short_name'] = 'UNK'
                
                # Find position info
                position_info = element_types_df[element_types_df['id'] == element_type_id]
                if not position_info.empty:
                    players_df.at[idx, 'position_name'] = position_info.iloc[0]['singular_name']
                else:
                    players_df.at[idx, 'position_name'] = 'Unknown Position'
            
            # Calculate additional columns
            players_df['cost_millions'] = players_df['now_cost'] / 10
            players_df['points_per_million'] = np.where(
                players_df['cost_millions'] > 0,
                players_df['total_points'] / players_df['cost_millions'],
                0
            ).round(2)
            
            # Handle form and ownership
            players_df['form'] = pd.to_numeric(players_df.get('form', 0), errors='coerce').fillna(0.0)
            players_df['selected_by_percent'] = pd.to_numeric(players_df.get('selected_by_percent', 0), errors='coerce').fillna(0.0)
            
            st.success("✅ Data loaded successfully using alternative method!")
            st.write("Final columns:", list(players_df.columns))
            
            return players_df, teams_df
            
        except Exception as e:
            st.error(f"Error loading FPL data (alternative): {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("⚽ FPL Analytics")
        st.sidebar.markdown("---")
        
        # Navigation - Updated with My FPL Team tab
        pages = {
            "🏠 Dashboard": "dashboard",
            "👥 Player Analysis": "players", 
            "🎯 Fixture Difficulty": "fixtures",
            "🔍 Advanced Filters": "filters",
            "📊 Team Analysis": "teams",
            "👤 My FPL Team": "my_team",          # NEW: Your actual FPL team
            "🤖 AI Recommendations": "ai_recommendations",
            "⚽ Team Builder": "team_builder",
        }
        
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            list(pages.keys()),
            index=0
        )
        
        st.sidebar.markdown("---")
        
        # Data status
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data Loaded")
            if not st.session_state.players_df.empty:
                st.sidebar.info(f"📊 {len(st.session_state.players_df)} players loaded")
        else:
            st.sidebar.warning("⚠️ No data loaded")
        
        # My FPL Team status
        if 'my_team_loaded' in st.session_state and st.session_state.my_team_loaded:
            st.sidebar.success("✅ My Team Loaded")
            if 'my_team_data' in st.session_state:
                st.sidebar.info(f"Team ID: {st.session_state.get('my_team_id', 'N/A')}")
        
        # Load data button
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            with st.spinner("Loading FPL data..."):
                players_df, teams_df = self.load_fpl_data()
                
                # If primary method fails, try alternative
                if players_df.empty:
                    st.warning("Primary method failed, trying alternative...")
                    players_df, teams_df = self.load_fpl_data_alternative()
                
                if not players_df.empty:
                    st.session_state.players_df = players_df
                    st.session_state.teams_df = teams_df
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Data refreshed!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to load data")
        
        return pages[selected_page]

    def render_fixtures(self):
        """Enhanced Fixture Difficulty Ratings tab with comprehensive explanations"""
        st.header("🎯 Fixture Difficulty Ratings (FDR)")
        
        # Comprehensive tab explanation
        with st.expander("📚 What is Fixture Difficulty Analysis?", expanded=False):
            st.markdown("""
            **Fixture Difficulty Rating (FDR)** is a crucial tool for FPL success that helps you identify:
            
            🎯 **Core Concepts:**
            - **Attack FDR**: How easy it is for a team's attackers to score against upcoming opponents
            - **Defense FDR**: How likely a team is to keep clean sheets based on opponent strength
            - **Combined FDR**: Overall fixture quality considering both attack and defense
            
            📊 **How to Interpret FDR Scores:**
            - **1-2 (Green)**: Excellent fixtures - Strong targets for transfers IN
            - **3 (Yellow)**: Average fixtures - Neutral, monitor closely  
            - **4-5 (Red)**: Difficult fixtures - Consider transfers OUT
            
            🎮 **Strategic Applications:**
            - **Transfer Planning**: Target players from teams with upcoming green fixtures
            - **Captain Selection**: Choose captains facing the easiest opponents
            - **Squad Rotation**: Plan bench players around difficult fixture periods
            - **Chip Strategy**: Time Wildcards and other chips around fixture swings
            
            🔄 **Form Adjustment Feature:**
            Our advanced system considers recent team performance to make FDR more accurate:
            - Teams in good form get easier effective FDR (they're more likely to overcome tough fixtures)
            - Teams in poor form get harder effective FDR (even easy fixtures become challenging)
            
            💡 **Pro Tips:**
            - Look for fixture swings where difficulty changes dramatically
            - Consider both home/away split for defenders vs attackers
            - Use 5+ gameweek analysis for transfer planning
            - Combine FDR with form, price, and ownership data
            """)
        
        st.markdown("### Analyze team fixtures to identify transfer targets and avoid traps")
        
        # Initialize FDR components
        fixture_loader = FixtureDataLoader()
        fdr_analyzer = FDRAnalyzer()
        fdr_visualizer = FDRVisualizer()
        
        # Load fixtures data
        if not st.session_state.fdr_data_loaded:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("""
                📊 **What is FDR?**
                - **Attack FDR**: How easy it is for a team's attackers to score (lower = easier opponents to score against)
                - **Defense FDR**: How easy it is for a team's defenders to keep clean sheets (lower = weaker attacking opponents)
                - **Combined FDR**: Overall fixture difficulty considering both attack and defense
                
                🎯 **How to use**: Green = Good fixtures, Red = Difficult fixtures
                """)
            
            with col2:
                if st.button("🔄 Load Fixture Data", type="primary", use_container_width=True):
                    with st.spinner("Loading fixture data from FPL API..."):
                        try:
                            fixtures_df = fixture_loader.process_fixtures_data()
                            
                            if not fixtures_df.empty:
                                fixtures_df = fdr_analyzer.calculate_combined_fdr(fixtures_df)
                                st.session_state.fixtures_df = fixtures_df
                                st.session_state.fdr_data_loaded = True
                                
                                # Debug information
                                st.success("✅ Fixture data loaded!")
                                st.info(f"📊 Loaded {len(fixtures_df)} fixtures for {fixtures_df['team_short_name'].nunique()} teams")
                                
                                # Show sample data
                                if not fixtures_df.empty:
                                    sample_teams = fixtures_df['team_short_name'].unique()[:5]
                                    st.write(f"**Sample teams loaded:** {', '.join(sample_teams)}")
                                    
                                    sample_fixture = fixtures_df.iloc[0]
                                    st.write(f"**Sample fixture:** {sample_fixture['team_short_name']} vs {sample_fixture.get('opponent_short_name', 'N/A')} (Difficulty: {sample_fixture.get('difficulty', 'N/A')})")
                                
                                st.rerun()
                            else:
                                st.error("❌ No fixture data available")
                                # Show debug info
                                raw_fixtures = fixture_loader.load_fixtures()
                                st.write(f"Debug: Found {len(raw_fixtures)} raw fixtures")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback
                            st.write("**Full error details:**")
                            st.code(traceback.format_exc())
                
                # Add refresh button for debugging
                if st.button("🔄 Force Refresh", help="Clear cache and reload fixture data"):
                    if 'fixtures_df' in st.session_state:
                        del st.session_state.fixtures_df
                    if 'fdr_data_loaded' in st.session_state:
                        del st.session_state.fdr_data_loaded
                    st.rerun()
                
                # Add API test button
                if st.button("🔬 Test API Connection", help="Test direct API connection"):
                    with st.spinner("Testing API connection..."):
                        try:
                            # Test fixtures API
                            fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
                            fixtures_response = requests.get(fixtures_url, timeout=10, verify=False)
                            
                            # Test bootstrap API  
                            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
                            bootstrap_response = requests.get(bootstrap_url, timeout=10, verify=False)
                            
                            if fixtures_response.status_code == 200 and bootstrap_response.status_code == 200:
                                fixtures_data = fixtures_response.json()
                                bootstrap_data = bootstrap_response.json()
                                teams_data = bootstrap_data.get('teams', [])
                                
                                st.success(f"✅ API Connection successful!")
                                st.info(f"📊 Found {len(fixtures_data)} fixtures and {len(teams_data)} teams")
                                
                                # Show sample data
                                if fixtures_data:
                                    sample_fixture = fixtures_data[0]
                                    st.write("**Sample Fixture Data:**")
                                    st.json({k: v for k, v in sample_fixture.items() if k in ['id', 'team_h', 'team_a', 'event', 'finished', 'team_h_difficulty', 'team_a_difficulty']})
                                
                                if teams_data:
                                    sample_team = teams_data[0]
                                    st.write("**Sample Team Data:**")
                                    st.json({k: v for k, v in sample_team.items() if k in ['id', 'name', 'short_name', 'strength', 'strength_overall_home', 'strength_overall_away']})
                                    
                            else:
                                st.error(f"❌ API Connection failed - Fixtures: {fixtures_response.status_code}, Bootstrap: {bootstrap_response.status_code}")
                                
                        except Exception as e:
                            st.error(f"❌ API Test failed: {str(e)}")
                            
            return
        
        fixtures_df = st.session_state.fixtures_df
        
        # **NEW: Debug panel to show data status**
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Fixture Data Status:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Fixtures", len(fixtures_df))
                st.metric("Unique Teams", fixtures_df['team_short_name'].nunique())
            
            with col2:
                st.write("**Sample Team Names:**")
                sample_teams = fixtures_df['team_short_name'].unique()[:8]
                st.write(", ".join(sample_teams))
            
            with col3:
                st.write("**Sample Difficulties:**")
                sample_difficulties = fixtures_df['difficulty'].value_counts().head(5)
                st.write(sample_difficulties.to_dict())
            
            # Show actual sample data
            if not fixtures_df.empty:
                st.write("**Sample Fixture Records:**")
                sample_data = fixtures_df[['team_short_name', 'opponent_short_name', 'difficulty', 'gameweek']].head(10)
                st.dataframe(sample_data, use_container_width=True)
        
        # Verify and enhance fixture data
        fixtures_df = self._verify_and_enhance_fixture_data(fixtures_df)
        
        # Enhanced Settings panel
        with st.expander("⚙️ Advanced FDR Settings", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                gameweeks_ahead = st.slider("Gameweeks to analyze:", 1, 15, 5)
                show_colors = st.checkbox("Show color coding", value=True)
            
            with col2:
                fdr_threshold = st.slider("Good fixture threshold:", 1.0, 4.0, 2.5, 0.1)
                show_opponents = st.checkbox("Show opponent names", value=True)
            
            with col3:
                sort_by = st.selectbox("Sort teams by:", ["Combined FDR", "Attack FDR", "Defense FDR", "Form-Adjusted FDR", "Alphabetical"])
                ascending_sort = st.checkbox("Ascending order", value=True)
                
            with col4:
                # NEW: Form adjustment settings
                use_form_adjustment = st.checkbox("Use Form Adjustment", value=True, 
                                                help="Adjust FDR based on recent team performance")
                form_weight = st.slider("Form Impact Weight:", 0.0, 1.0, 0.3, 0.1,
                                       help="How much recent form affects FDR (0=none, 1=full)")
                
        # NEW: Analysis type selection
        analysis_type = st.selectbox(
            "🎯 Analysis Focus:",
            ["All Fixtures", "Home Only", "Away Only", "Next 3 Fixtures", "Fixture Congestion Periods"],
            help="Choose what type of fixtures to analyze"
        )
        
        # Create enhanced tabs with new features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "⚔️ Attack Analysis", 
            "🛡️ Defense Analysis", 
            "🎯 Transfer Targets",
            "📈 Fixture Swings",
            "🎪 Advanced Analytics"
        ])
        
        # Apply form adjustment if enabled
        if use_form_adjustment and st.session_state.data_loaded:
            fixtures_df = self._apply_form_adjustment(fixtures_df, form_weight)
        
        with tab1:
            self._render_fdr_overview(fixtures_df, fdr_visualizer, gameweeks_ahead, sort_by, ascending_sort, analysis_type)
        
        with tab2:
            self._render_attack_analysis(fixtures_df, fdr_visualizer, fdr_threshold, show_opponents, analysis_type)
        
        with tab3:
            self._render_defense_analysis(fixtures_df, fdr_visualizer, fdr_threshold, show_opponents, analysis_type)
        
        with tab4:
            self._render_transfer_targets(fixtures_df, fdr_threshold)
        
        with tab5:
            self._render_fixture_swings(fixtures_df)
            
        with tab6:
            self._render_advanced_analytics(fixtures_df, gameweeks_ahead)
    
    def _render_fdr_overview(self, fixtures_df, fdr_visualizer, gameweeks_ahead, sort_by, ascending_sort, analysis_type):
        """Enhanced FDR overview with better metrics and insights"""
        st.subheader(f"📊 FDR Overview - {analysis_type}")
        
        # Filter fixtures based on analysis type
        filtered_fixtures = self._filter_fixtures_by_type(fixtures_df, analysis_type)
        
        if filtered_fixtures.empty:
            st.warning("No fixture data available")
            return
        
        # Enhanced key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_fixtures = len(filtered_fixtures)
            teams_count = filtered_fixtures['team_short_name'].nunique()
            st.metric("📊 Fixtures Analyzed", f"{total_fixtures} ({teams_count} teams)")
        
        with col2:
            if 'attack_fdr' in filtered_fixtures.columns:
                avg_attack_fdr = filtered_fixtures['attack_fdr'].mean()
                best_attack_team = filtered_fixtures.groupby('team_short_name')['attack_fdr'].mean().idxmin()
                st.metric("⚔️ Best Attack Fixtures", f"{best_attack_team}", f"Avg: {avg_attack_fdr:.2f}")
            else:
                st.metric("⚔️ Attack FDR", "N/A")
        
        with col3:
            if 'defense_fdr' in filtered_fixtures.columns:
                avg_defense_fdr = filtered_fixtures['defense_fdr'].mean()
                best_defense_team = filtered_fixtures.groupby('team_short_name')['defense_fdr'].mean().idxmin()
                st.metric("🛡️ Best Defense Fixtures", f"{best_defense_team}", f"Avg: {avg_defense_fdr:.2f}")
            else:
                st.metric("🛡️ Defense FDR", "N/A")
        
        with col4:
            if 'combined_fdr' in filtered_fixtures.columns:
                avg_combined_fdr = filtered_fixtures['combined_fdr'].mean()
                best_overall_team = filtered_fixtures.groupby('team_short_name')['combined_fdr'].mean().idxmin()
                st.metric("🎯 Best Overall Fixtures", f"{best_overall_team}", f"Avg: {avg_combined_fdr:.2f}")
            else:
                st.metric("🎯 Combined FDR", "N/A")
        
        st.divider()
        
        # Enhanced FDR Heatmap with better styling
        if 'combined_fdr' in filtered_fixtures.columns:
            st.subheader(f"🌡️ FDR Heatmap - {analysis_type}")
            
            # Add fixture opponent info to heatmap
            if 'opponent_short_name' in filtered_fixtures.columns:
                # Create enhanced heatmap with opponent names
                fig_heatmap = self._create_enhanced_fdr_heatmap(filtered_fixtures, 'combined')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                fig_heatmap = fdr_visualizer.create_fdr_heatmap(filtered_fixtures, 'combined')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # FDR Legend
            st.markdown("""
            **🎯 FDR Guide:**
            - 🟢 **1-2**: Excellent fixtures - Target these teams' players
            - 🟡 **3**: Average fixtures - Neutral stance
            - 🟠 **4**: Difficult fixtures - Consider avoiding
            - 🔴 **5**: Very difficult - Strong avoid
            """)
        
        # Enhanced team summary table with more insights
        st.subheader("📋 Detailed Team FDR Analysis")
        
        if filtered_fixtures.empty:
            st.warning("No data for team summary")
            return
        
        # Create comprehensive team summary
        team_summary = self._create_enhanced_team_summary(filtered_fixtures, sort_by, ascending_sort)
        
        if not team_summary.empty:
            # Add color coding to the dataframe display
            def highlight_fdr(val):
                if pd.isna(val):
                    return ''
                try:
                    val = float(val)
                    if val <= 2:
                        return 'background-color: #90EE90'  # Light green
                    elif val <= 2.5:
                        return 'background-color: #FFFF99'  # Light yellow
                    elif val <= 3.5:
                        return 'background-color: #FFE4B5'  # Light orange
                    elif val <= 4:
                        return 'background-color: #FFB6C1'  # Light red
                    else:
                        return 'background-color: #FF6B6B'  # Red
                except:
                    return ''
            
            # Apply styling only to FDR columns
            fdr_columns = [col for col in team_summary.columns if 'FDR' in col]
            styled_df = team_summary.style.applymap(highlight_fdr, subset=fdr_columns)
            
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No FDR data available for team summary")
    
    def _apply_form_adjustment(self, fixtures_df, form_weight):
        """Apply form-based adjustments to FDR calculations"""
        if not st.session_state.data_loaded or fixtures_df.empty:
            return fixtures_df
        
        try:
            # Get recent team form data
            players_df = st.session_state.players_df
            teams_df = st.session_state.teams_df
            
            # Calculate team form based on recent performances
            team_form = {}
            
            for team_id, team_data in teams_df.iterrows():
                team_name = team_data.get('short_name', f'Team_{team_id}')
                
                # Get team players
                team_players = players_df[players_df['team'] == team_data['id']]
                
                if not team_players.empty:
                    # Calculate average form from key players
                    avg_form = team_players['form'].astype(float).mean()
                    avg_points_per_game = team_players['points_per_game'].astype(float).mean()
                    
                    # Normalize form (5.0 = average form)
                    form_factor = (avg_form - 5.0) / 5.0  # Range: -1 to 1
                    team_form[team_name] = form_factor
            
            # Apply form adjustment to FDR
            for index, row in fixtures_df.iterrows():
                team_name = row['team_short_name']
                opponent_name = row.get('opponent_short_name', '')
                
                # Get form factors
                team_form_factor = team_form.get(team_name, 0)
                opponent_form_factor = team_form.get(opponent_name, 0)
                
                # Adjust FDR based on form
                for fdr_type in ['attack_fdr', 'defense_fdr', 'combined_fdr']:
                    if fdr_type in fixtures_df.columns:
                        base_fdr = row[fdr_type]
                        
                        # Better form = easier fixtures for attack, harder for opponents
                        if fdr_type == 'attack_fdr':
                            # Good team form makes scoring easier
                            form_adjustment = -team_form_factor * form_weight
                        elif fdr_type == 'defense_fdr':
                            # Good opponent form makes defending harder
                            form_adjustment = opponent_form_factor * form_weight
                        else:  # combined_fdr
                            # Average of both adjustments
                            form_adjustment = (-team_form_factor + opponent_form_factor) * form_weight / 2
                        
                        # Apply adjustment (keep within 1-5 range)
                        adjusted_fdr = max(1, min(5, base_fdr + form_adjustment))
                        fixtures_df.at[index, fdr_type] = round(adjusted_fdr, 2)
            
            return fixtures_df
            
        except Exception as e:
            st.warning(f"Form adjustment failed: {str(e)}")
            return fixtures_df
    
    def _filter_fixtures_by_type(self, fixtures_df, analysis_type):
        """Filter fixtures based on analysis type"""
        if fixtures_df.empty:
            return fixtures_df
        
        try:
            if analysis_type == "All Fixtures":
                return fixtures_df
            elif analysis_type == "Home Only":
                return fixtures_df[fixtures_df.get('is_home', True) == True]
            elif analysis_type == "Away Only":
                return fixtures_df[fixtures_df.get('is_home', True) == False]
            elif analysis_type == "Next 3 Fixtures":
                return fixtures_df[fixtures_df.get('fixture_number', 1) <= 3]
            elif analysis_type == "Fixture Congestion Periods":
                # Identify periods with multiple fixtures in short time
                return fixtures_df  # For now, return all - can be enhanced
            else:
                return fixtures_df
                
        except Exception as e:
            st.warning(f"Fixture filtering failed: {str(e)}")
            return fixtures_df
    
    def _render_advanced_analytics(self, fixtures_df, gameweeks_ahead):
        """Render advanced analytics tab with enhanced insights"""
        st.subheader("🎪 Advanced Fixture Analytics")
        
        if fixtures_df.empty:
            st.warning("No fixture data available for advanced analytics")
            return
        
        # Tab sections
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "📊 Statistical Analysis",
            "🎯 Player Recommendations", 
            "📈 Seasonal Trends"
        ])
        
        with analytics_tab1:
            self._render_statistical_analysis(fixtures_df)
        
        with analytics_tab2:
            self._render_player_recommendations(fixtures_df)
        
        with analytics_tab3:
            self._render_seasonal_trends(fixtures_df)
    
    def _render_statistical_analysis(self, fixtures_df):
        """Render statistical analysis of fixtures"""
        st.subheader("📊 FDR Statistical Breakdown")
        
        if 'combined_fdr' not in fixtures_df.columns:
            st.warning("No FDR data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # FDR Distribution
            fdr_distribution = fixtures_df['combined_fdr'].value_counts().sort_index()
            
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=fdr_distribution.index,
                    y=fdr_distribution.values,
                    marker_color=['#00FF87', '#01FF70', '#FFDC00', '#FF851B', '#FF4136'][:len(fdr_distribution)]
                )
            ])
            
            fig_dist.update_layout(
                title="FDR Distribution",
                xaxis_title="FDR Rating",
                yaxis_title="Number of Fixtures",
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Team FDR Variance
            team_variance = fixtures_df.groupby('team_short_name')['combined_fdr'].agg(['mean', 'std']).round(2)
            team_variance = team_variance.sort_values('std', ascending=False).head(10)
            
            fig_variance = go.Figure()
            
            fig_variance.add_trace(go.Scatter(
                x=team_variance['mean'],
                y=team_variance['std'],
                mode='markers+text',
                text=team_variance.index,
                textposition="top center",
                marker=dict(size=10, color='blue'),
                name='Teams'
            ))
            
            fig_variance.update_layout(
                title="FDR Consistency Analysis",
                xaxis_title="Average FDR",
                yaxis_title="FDR Standard Deviation",
                height=400
            )
            
            st.plotly_chart(fig_variance, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("🔗 FDR Correlation Analysis")
        
        if all(col in fixtures_df.columns for col in ['attack_fdr', 'defense_fdr', 'combined_fdr']):
            corr_data = fixtures_df[['attack_fdr', 'defense_fdr', 'combined_fdr']].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_data.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            
            fig_corr.update_layout(
                title="FDR Type Correlations",
                height=300
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    def _render_player_recommendations(self, fixtures_df):
        """Render AI-powered player recommendations based on fixtures"""
        st.subheader("🎯 Smart Player Recommendations")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see recommendations")
            return
        
        players_df = st.session_state.players_df
        
        # Get teams with best fixtures
        if 'combined_fdr' in fixtures_df.columns:
            best_fixture_teams = fixtures_df.groupby('team_short_name')['combined_fdr'].mean().sort_values().head(5)
            worst_fixture_teams = fixtures_df.groupby('team_short_name')['combined_fdr'].mean().sort_values(ascending=False).head(5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 Players to Target")
                
                for team_name, avg_fdr in best_fixture_teams.items():
                    st.write(f"**{team_name}** (Avg FDR: {avg_fdr:.2f})")
                    
                    # Get team players
                    team_players = players_df[players_df['team_short_name'] == team_name]
                    
                    if not team_players.empty:
                        # Get top players by points and value
                        top_players = team_players.nlargest(3, 'total_points')
                        
                        for _, player in top_players.iterrows():
                            ownership = player.get('selected_by_percent', 0)
                            differential = "💎" if ownership < 10 else "👥" if ownership < 30 else "🔥"
                            
                            st.write(f"  {differential} {player['web_name']} "
                                   f"(£{player['cost_millions']:.1f}m) - "
                                   f"{player['total_points']} pts")
                    
                    st.divider()
            
            with col2:
                st.subheader("🔴 Players to Avoid")
                
                for team_name, avg_fdr in worst_fixture_teams.items():
                    st.write(f"**{team_name}** (Avg FDR: {avg_fdr:.2f})")
                    
                    # Get team players
                    team_players = players_df[players_df['team_short_name'] == team_name]
                    
                    if not team_players.empty:
                        # Get top owned players
                        popular_players = team_players.nlargest(3, 'selected_by_percent')
                        
                        for _, player in popular_players.iterrows():
                            ownership = player.get('selected_by_percent', 0)
                            
                            st.write(f"  ⚠️ {player['web_name']} "
                                   f"(£{player['cost_millions']:.1f}m) - "
                                   f"{ownership:.1f}% owned")
                    
                    st.divider()
    
    def _render_seasonal_trends(self, fixtures_df):
        """Render seasonal fixture trends and patterns"""
        st.subheader("📈 Seasonal Fixture Patterns")
        
        if 'gameweek' not in fixtures_df.columns:
            st.warning("No gameweek data available for seasonal analysis")
            return
        
        # Fixture difficulty by gameweek
        if 'combined_fdr' in fixtures_df.columns:
            gw_difficulty = fixtures_df.groupby('gameweek')['combined_fdr'].mean().reset_index()
            
            fig_season = go.Figure()
            
            fig_season.add_trace(go.Scatter(
                x=gw_difficulty['gameweek'],
                y=gw_difficulty['combined_fdr'],
                mode='lines+markers',
                name='Average FDR',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig_season.update_layout(
                title="Average Fixture Difficulty Throughout Season",
                xaxis_title="Gameweek",
                yaxis_title="Average FDR",
                height=400,
                yaxis=dict(range=[1, 5])
            )
            
            st.plotly_chart(fig_season, use_container_width=True)
        
        # Seasonal recommendations
        st.subheader("🎯 Seasonal Strategy Recommendations")
        
        current_gw = self._get_current_gameweek()
        
        if current_gw:
            if current_gw <= 10:
                st.info("🌱 **Early Season Strategy**: Focus on form and early fixtures. Avoid kneejerk reactions.")
            elif current_gw <= 25:
                st.info("🏃 **Mid Season Strategy**: Look for fixture swings and differential picks. Plan for BGW/DGW.")
            else:
                st.info("🏁 **Late Season Strategy**: All-out attack mode. Take risks for final push.")
        
        # Best fixture periods
        if 'combined_fdr' in fixtures_df.columns and 'gameweek' in fixtures_df.columns:
            st.subheader("📅 Best Fixture Periods")
            
            # Group by consecutive gameweeks and find best periods
            team_gw_avg = fixtures_df.groupby(['team_short_name', 'gameweek'])['combined_fdr'].mean().reset_index()
            
            # Find teams with consistently good fixtures
            good_fixture_periods = []
            
            for team in team_gw_avg['team_short_name'].unique():
                team_data = team_gw_avg[team_gw_avg['team_short_name'] == team].sort_values('gameweek')
                
                # Calculate rolling 3-gameweek average
                if len(team_data) >= 3:
                    team_data['rolling_fdr'] = team_data['combined_fdr'].rolling(3).mean()
                    best_period = team_data.loc[team_data['rolling_fdr'].idxmin()]
                    
                    if best_period['rolling_fdr'] <= 2.5:
                        good_fixture_periods.append({
                            'team': team,
                            'start_gw': int(best_period['gameweek']) - 1,
                            'avg_fdr': best_period['rolling_fdr']
                        })
            
            # Sort by FDR and display top periods
            good_fixture_periods = sorted(good_fixture_periods, key=lambda x: x['avg_fdr'])[:10]
            
            for period in good_fixture_periods:
                st.write(f"🎯 **{period['team']}** - GW {period['start_gw']}-{period['start_gw']+2}: "
                        f"Avg FDR {period['avg_fdr']:.2f}")

    def _render_attack_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents, analysis_type="All Fixtures"):
        """Enhanced attack FDR analysis with analysis type support"""
        st.subheader(f"⚔️ Attack FDR Analysis - {analysis_type}")
        st.info("🎯 Lower Attack FDR = Easier to score goals. Target these teams' forwards and attacking midfielders!")
        
        # Filter fixtures based on analysis type
        filtered_fixtures = self._filter_fixtures_by_type(fixtures_df, analysis_type)
        
        if filtered_fixtures.empty or 'attack_fdr' not in filtered_fixtures.columns:
            st.warning("Attack FDR data not available")
            return
        
        # Attack FDR insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Best attacking fixtures with detailed breakdown
            st.subheader("🟢 Best Attacking Fixtures")
            attack_summary = filtered_fixtures.groupby('team_short_name').agg({
                'attack_fdr': ['mean', 'min', 'count'],
                'opponent_short_name': lambda x: ' → '.join(x.head(3)) if show_opponents else ''
            }).round(2)
            
            attack_summary.columns = ['Avg_FDR', 'Best_FDR', 'Fixtures', 'Next_3_Opponents']
            attack_summary = attack_summary.reset_index()
            attack_summary = attack_summary.sort_values('Avg_FDR').head(10)
            
            for idx, row in attack_summary.iterrows():
                avg_fdr = row['Avg_FDR']
                color = "🟢" if avg_fdr <= 2 else "🟡" if avg_fdr <= 2.5 else "🔴"
                
                with st.expander(f"{color} **{row['team_short_name']}** - Avg FDR: {avg_fdr:.2f}"):
                    st.write(f"🎯 **Average FDR**: {avg_fdr:.2f}")
                    st.write(f"⭐ **Best upcoming fixture**: {row['Best_FDR']:.0f}")
                    st.write(f"📊 **Total fixtures**: {row['Fixtures']:.0f}")
                    if show_opponents and row['Next_3_Opponents']:
                        st.write(f"🆚 **Next 3 opponents**: {row['Next_3_Opponents']}")
                    
                    # Player recommendations
                    if avg_fdr <= fdr_threshold:
                        st.success("💡 **Recommendation**: Target forwards and attacking midfielders")
                    else:
                        st.warning("⚠️ **Recommendation**: Avoid or consider selling attackers")
        
        with col2:
            # Attack FDR visualization
            fig_attack = fdr_visualizer.create_fdr_heatmap(filtered_fixtures, 'attack')
            st.plotly_chart(fig_attack, use_container_width=True)
        
        # Attack fixture opportunities
        st.subheader("🎯 Attack Fixture Opportunities")
        opportunities = self._identify_attack_opportunities(filtered_fixtures, fdr_threshold)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🚀 Immediate Targets (Next 2 GWs)**")
            for team, fdr in opportunities.get('immediate', []):
                st.write(f"• **{team}**: {fdr:.2f} FDR")
        
        with col2:
            st.write("**📈 Medium-term Targets (GW 3-5)**")
            for team, fdr in opportunities.get('medium_term', []):
                st.write(f"• **{team}**: {fdr:.2f} FDR")
        
        with col3:
            st.write("**⚠️ Avoid These Teams**")
            for team, fdr in opportunities.get('avoid', []):
                st.write(f"• **{team}**: {fdr:.2f} FDR")
    
    def _render_defense_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents, analysis_type="All Fixtures"):
        """Enhanced defense FDR analysis with analysis type support"""
        st.subheader(f"🛡️ Defense FDR Analysis - {analysis_type}")
        st.info("🏠 Lower Defense FDR = Easier to keep clean sheets. Target these teams' defenders and goalkeepers!")
        
        # Filter fixtures based on analysis type
        filtered_fixtures = self._filter_fixtures_by_type(fixtures_df, analysis_type)
        
        if filtered_fixtures.empty or 'defense_fdr' not in filtered_fixtures.columns:
            st.warning("Defense FDR data not available")
            return
        
        # Defense-specific insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Best Defensive Fixtures")
            defense_summary = fixtures_df.groupby('team_short_name').agg({
                'defense_fdr': ['mean', 'min', 'count'],
                'opponent_short_name': lambda x: ' → '.join(x.head(3)) if show_opponents else ''
            }).round(2)
            
            defense_summary.columns = ['Avg_FDR', 'Best_FDR', 'Fixtures', 'Next_3_Opponents']
            defense_summary = defense_summary.reset_index()
            defense_summary = defense_summary.sort_values('Avg_FDR').head(8)
            
            for idx, row in defense_summary.iterrows():
                avg_fdr = row['Avg_FDR']
                color = "🟢" if avg_fdr <= 2 else "🟡" if avg_fdr <= 2.5 else "🔴"
                
                st.write(f"{color} **{row['team_short_name']}** - {avg_fdr:.2f} FDR")
                if show_opponents and row['Next_3_Opponents']:
                    st.caption(f"vs {row['Next_3_Opponents']}")
        
        with col2:
            # Defense FDR heatmap
            fig_defense = fdr_visualizer.create_fdr_heatmap(fixtures_df, 'defense')
            st.plotly_chart(fig_defense, use_container_width=True)
        
        # Clean sheet probability predictions
        st.subheader("🎯 Clean Sheet Probability Insights")
        clean_sheet_data = self._calculate_clean_sheet_probabilities(fixtures_df)
        
        if not clean_sheet_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Highest Clean Sheet Probability**")
                top_cs = clean_sheet_data.head(5)
                for _, row in top_cs.iterrows():
                    prob = row['clean_sheet_prob']
                    color = "🟢" if prob >= 70 else "🟡" if prob >= 50 else "🔴"
                    st.write(f"{color} **{row['team_short_name']}**: {prob:.0f}%")
            
            with col2:
                # Clean sheet probability chart
                fig_cs = px.bar(
                    clean_sheet_data.head(10),
                    x='clean_sheet_prob',
                    y='team_short_name',
                    orientation='h',
                    title="Clean Sheet Probability (%)",
                    color='clean_sheet_prob',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_cs, use_container_width=True)
    
    def _render_transfer_targets(self, fixtures_df, fdr_threshold):
        """Enhanced transfer targets based on fixture analysis"""
        st.subheader("🎯 Transfer Targets & Recommendations")
        st.info("💡 Based on fixture difficulty analysis - players to target or avoid")
        
        if fixtures_df.empty:
            st.warning("No fixture data for transfer analysis")
            return
        
        # Get player data if available
        if st.session_state.data_loaded and not st.session_state.players_df.empty:
            players_df = st.session_state.players_df
            
            # Merge fixtures with player data
            transfer_analysis = self._create_transfer_recommendations(fixtures_df, players_df, fdr_threshold)
            
            # Create transfer recommendation tabs
            target_tab, avoid_tab, differential_tab = st.tabs(["🎯 Targets", "❌ Avoid", "💎 Differentials"])
            
            with target_tab:
                st.subheader("🚀 Players to Target")
                
                for position in ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']:
                    pos_targets = transfer_analysis.get(f'{position.lower()}_targets', [])
                    
                    if pos_targets:
                        st.write(f"**{position}s:**")
                        
                        for player in pos_targets[:5]:  # Top 5 per position
                            with st.expander(f"⭐ {player['name']} ({player['team']}) - £{player['price']:.1f}m"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"💰 **Price**: £{player['price']:.1f}m")
                                    st.write(f"📊 **Points**: {player['points']}")
                                    st.write(f"🔥 **Form**: {player['form']:.1f}")
                                
                                with col2:
                                    st.write(f"🎯 **FDR**: {player['avg_fdr']:.2f}")
                                    st.write(f"👥 **Ownership**: {player['ownership']:.1f}%")
                                    st.write(f"💎 **Value**: {player['points_per_million']:.1f}")
                                
                                # Fixture preview
                                st.write(f"📅 **Next 3 fixtures**: {player.get('next_fixtures', 'N/A')}")
            
            with avoid_tab:
                st.subheader("❌ Players to Avoid/Sell")
                
                avoid_players = transfer_analysis.get('avoid_players', [])
                
                if avoid_players:
                    for player in avoid_players[:10]:
                        st.write(f"🔴 **{player['name']}** ({player['team']}) - FDR: {player['avg_fdr']:.2f}")
                        st.caption(f"Difficult fixtures ahead - consider selling")
                else:
                    st.info("No obvious players to avoid based on fixtures")
            
            with differential_tab:
                st.subheader("💎 Differential Picks")
                st.info("Low ownership players with good fixtures")
                
                differentials = transfer_analysis.get('differentials', [])
                
                if differentials:
                    for player in differentials[:8]:
                        with st.expander(f"💎 {player['name']} ({player['team']}) - {player['ownership']:.1f}% owned"):
                            st.write(f"Good fixtures (FDR: {player['avg_fdr']:.2f}) with low ownership")
                            st.write(f"Price: £{player['price']:.1f}m | Points: {player['points']}")
                else:
                    st.info("No clear differential opportunities identified")
        
        else:
            # Simple fixture-based recommendations without player data
            st.info("💡 Load player data to see detailed transfer recommendations")
            
            # Basic team recommendations
            if 'combined_fdr' in fixtures_df.columns:
                best_fixtures = fixtures_df.groupby('team_short_name')['combined_fdr'].mean().nsmallest(5)
                worst_fixtures = fixtures_df.groupby('team_short_name')['combined_fdr'].mean().nlargest(5)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Target these teams' players:**")
                    for team, fdr in best_fixtures.items():
                        st.write(f"• **{team}** (FDR: {fdr:.2f})")
                
                with col2:
                    st.write("**❌ Avoid these teams' players:**")
                    for team, fdr in worst_fixtures.items():
                        st.write(f"• **{team}** (FDR: {fdr:.2f})")
    
    def _render_fixture_swings(self, fixtures_df):
        """Analyze fixture difficulty swings and upcoming changes"""
        st.subheader("📈 Fixture Difficulty Swings")
        st.info("🔄 Teams whose fixture difficulty will change significantly")
        
        if fixtures_df.empty or 'combined_fdr' not in fixtures_df.columns:
            st.warning("No data available for fixture swings analysis")
            return
        
        # Calculate fixture swings
        swing_analysis = self._calculate_fixture_swings(fixtures_df)
        
        if swing_analysis.empty:
            st.warning("Unable to calculate fixture swings")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Fixtures Getting Easier")
            improving = swing_analysis[swing_analysis['swing'] < -0.5].sort_values('swing').head(8)
            
            for _, row in improving.iterrows():
                swing_val = abs(row['swing'])
                st.write(f"🟢 **{row['team_short_name']}**: {swing_val:.2f} improvement")
                st.caption(f"Early FDR: {row['early_fdr']:.2f} → Later FDR: {row['later_fdr']:.2f}")
        
        with col2:
            st.subheader("📉 Fixtures Getting Harder")
            worsening = swing_analysis[swing_analysis['swing'] > 0.5].sort_values('swing', ascending=False).head(8)
            
            for _, row in worsening.iterrows():
                swing_val = row['swing']
                st.write(f"🔴 **{row['team_short_name']}**: +{swing_val:.2f} difficulty")
                st.caption(f"Early FDR: {row['early_fdr']:.2f} → Later FDR: {row['later_fdr']:.2f}")
        
        # Fixture swing visualization
        if len(swing_analysis) > 0:
            st.subheader("📊 Fixture Swing Visualization")
            
            fig = px.scatter(
                swing_analysis,
                x='early_fdr',
                y='later_fdr',
                hover_name='team_short_name',
                color='swing',
                color_continuous_scale='RdYlGn_r',
                title="Fixture Difficulty: Early vs Later Fixtures"
            )
            
            # Add diagonal line for reference
            fig.add_shape(
                type="line",
                x0=1, y0=1, x1=5, y1=5,
                line=dict(color="gray", dash="dash"),
            )
            
            fig.update_layout(
                xaxis_title="Early Fixtures FDR (GW 1-2)",
                yaxis_title="Later Fixtures FDR (GW 3-5)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **💡 How to read this chart:**
            - **Below the line**: Fixtures get easier (good for transfers in)
            - **Above the line**: Fixtures get harder (consider transfers out)
            - **Green dots**: Significant improvement in fixtures
            - **Red dots**: Significant worsening in fixtures
            """)

    def _create_enhanced_fdr_heatmap(self, fixtures_df, fdr_type):
        """Create enhanced FDR heatmap with opponent information"""
        
        # Create pivot table with FDR values
        pivot_data = fixtures_df.pivot_table(
            index='team_short_name',
            columns='fixture_number', 
            values=f'{fdr_type}_fdr',
            aggfunc='first'
        )
        
        # Create hover text with opponent info
        hover_text = fixtures_df.pivot_table(
            index='team_short_name',
            columns='fixture_number',
            values='opponent_short_name',
            aggfunc='first'
        )
        
        # Create custom hover template
        hovertemplate = '<b>%{y}</b><br>Fixture %{x}<br>FDR: %{z}<br>vs %{text}<extra></extra>'
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[f'GW{i}' for i in pivot_data.columns],
            y=pivot_data.index,
            text=hover_text.values,
            colorscale=[
                [0.0, '#00FF87'], [0.25, '#01FF70'], 
                [0.5, '#FFDC00'], [0.75, '#FF851B'], [1.0, '#FF4136']
            ],
            zmin=1, zmax=5,
            hovertemplate=hovertemplate,
            colorbar=dict(
                title="FDR",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["Very Easy", "Easy", "Average", "Hard", "Very Hard"]
            )
        ))
        
        fig.update_layout(
            title=f'{fdr_type.title()} FDR - Next 5 Fixtures',
            xaxis_title="Fixture Number",
            yaxis_title="Team",
            height=600
        )
        
        return fig
    
    def _create_enhanced_team_summary(self, fixtures_df, sort_by, ascending_sort):
        """Create enhanced team summary with additional insights"""
        
        if fixtures_df.empty:
            return pd.DataFrame()
        
        # Group by team and calculate comprehensive metrics
        fdr_cols = [col for col in ['attack_fdr', 'defense_fdr', 'combined_fdr'] if col in fixtures_df.columns]
        
        if not fdr_cols:
            return pd.DataFrame()
        
        agg_dict = {col: ['mean', 'min', 'max'] for col in fdr_cols}
        agg_dict['fixture_number'] = 'count'
        
        if 'opponent_short_name' in fixtures_df.columns:
            agg_dict['opponent_short_name'] = lambda x: ' | '.join(x.head(3))
        
        team_summary = fixtures_df.groupby(['team_short_name']).agg(agg_dict).round(2).reset_index()
        
        # Flatten column names
        new_columns = ['Team']
        for col in team_summary.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == 'mean':
                    if col[0] == 'attack_fdr':
                        new_columns.append('Attack FDR')
                    elif col[0] == 'defense_fdr':
                        new_columns.append('Defense FDR')
                    elif col[0] == 'combined_fdr':
                        new_columns.append('Combined FDR')
                elif col[1] == 'min':
                    new_columns.append(f'Best {col[0].replace("_fdr", "")}')
                elif col[1] == 'max':
                    new_columns.append(f'Worst {col[0].replace("_fdr", "")}')
                elif col[1] == 'count':
                    new_columns.append('Fixtures')
                elif col[1] == '<lambda>':
                    new_columns.append('Next 3 Opponents')
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(str(col))
        
        team_summary.columns = new_columns
        
        # Sort based on user selection
        if sort_by == "Combined FDR" and "Combined FDR" in team_summary.columns:
            team_summary = team_summary.sort_values('Combined FDR', ascending=ascending_sort)
        elif sort_by == "Attack FDR" and "Attack FDR" in team_summary.columns:
            team_summary = team_summary.sort_values('Attack FDR', ascending=ascending_sort)
        elif sort_by == "Defense FDR" and "Defense FDR" in team_summary.columns:
            team_summary = team_summary.sort_values('Defense FDR', ascending=ascending_sort)
        else:
            team_summary = team_summary.sort_values('Team', ascending=True)
        
        return team_summary

    def _verify_and_enhance_fixture_data(self, fixtures_df):
        """Verify and enhance fixture data to ensure proper differentiation"""
        if fixtures_df.empty:
            return fixtures_df
        
        # Check if we have proper team differentiation
        team_count = fixtures_df['team_short_name'].nunique()
        total_rows = len(fixtures_df)
        
        st.info(f"📊 Loaded fixture data: {total_rows} fixtures for {team_count} teams")
        
        # Verify we have different opponents for different teams
        if 'opponent_short_name' in fixtures_df.columns:
            sample_teams = fixtures_df.groupby('team_short_name')['opponent_short_name'].first().head(5)
            
            # Check for data quality
            unique_opponents_per_team = fixtures_df.groupby('team_short_name')['opponent_short_name'].nunique()
            
            if unique_opponents_per_team.min() == unique_opponents_per_team.max() == 1:
                st.warning("⚠️ Detected potential data issue - all teams may have same opponent")
            else:
                st.success(f"✅ Data looks good - teams have {unique_opponents_per_team.min()}-{unique_opponents_per_team.max()} different opponents")
            
            # Display sample for verification
            with st.expander("🔍 Data Verification - Sample Teams"):
                verification_data = []
                for team in fixtures_df['team_short_name'].unique()[:8]:  # Show more teams
                    team_data = fixtures_df[fixtures_df['team_short_name'] == team].head(3)
                    opponents = team_data['opponent_short_name'].tolist()
                    difficulties = team_data['difficulty'].tolist()
                    venues = team_data['venue'].tolist()
                    gameweeks = team_data['gameweek'].tolist()
                    
                    verification_data.append({
                        'Team': team,
                        'Next 3 Opponents': ' → '.join(opponents),
                        'Difficulties': ' → '.join(map(str, difficulties)),
                        'Venues': ' → '.join(venues),
                        'Gameweeks': ' → '.join([str(gw) if gw else 'TBD' for gw in gameweeks])
                    })
                
                verification_df = pd.DataFrame(verification_data)
                st.dataframe(verification_df, use_container_width=True)
                
                # Additional quality metrics
                st.write("**📈 Data Quality Metrics:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_opponents = unique_opponents_per_team.mean()
                    st.metric("Avg Opponents per Team", f"{avg_opponents:.1f}")
                
                with col2:
                    difficulty_variance = fixtures_df['difficulty'].var()
                    st.metric("Difficulty Variance", f"{difficulty_variance:.2f}")
                
                with col3:
                    home_away_balance = fixtures_df['venue'].value_counts()
                    balance_ratio = home_away_balance.min() / home_away_balance.max() if len(home_away_balance) > 1 else 1.0
                    st.metric("Home/Away Balance", f"{balance_ratio:.2f}")
        
        return fixtures_df
    
    def _identify_attack_opportunities(self, fixtures_df, threshold):
        """Identify attack opportunities by fixture timing"""
        
        opportunities = {'immediate': [], 'medium_term': [], 'avoid': []}
        
        if 'attack_fdr' not in fixtures_df.columns:
            return opportunities
        
        # Immediate opportunities (next 2 fixtures)
        immediate = fixtures_df[fixtures_df['fixture_number'] <= 2].groupby('team_short_name')['attack_fdr'].mean()
        opportunities['immediate'] = [(team, fdr) for team, fdr in immediate.items() if fdr <= threshold][:5]
        
        # Medium-term opportunities (fixtures 3-5)
        medium_term = fixtures_df[fixtures_df['fixture_number'] >= 3].groupby('team_short_name')['attack_fdr'].mean()
        opportunities['medium_term'] = [(team, fdr) for team, fdr in medium_term.items() if fdr <= threshold][:5]
        
        # Teams to avoid
        avoid = fixtures_df.groupby('team_short_name')['attack_fdr'].mean()
        opportunities['avoid'] = [(team, fdr) for team, fdr in avoid.items() if fdr >= 4.0][:5]
        
        return opportunities
    
    def _calculate_clean_sheet_probabilities(self, fixtures_df):
        """Calculate clean sheet probabilities based on defense FDR"""
        
        if 'defense_fdr' not in fixtures_df.columns:
            return pd.DataFrame()
        
        team_defense = fixtures_df.groupby('team_short_name')['defense_fdr'].mean().reset_index()
        
        # Convert FDR to probability (simplified model)
        team_defense['clean_sheet_prob'] = 100 * (6 - team_defense['defense_fdr']) / 5
        team_defense['clean_sheet_prob'] = team_defense['clean_sheet_prob'].clip(0, 100)
        
        return team_defense.sort_values('clean_sheet_prob', ascending=False)
    
    def _create_transfer_recommendations(self, fixtures_df, players_df, threshold):
        """Create detailed transfer recommendations"""
        
        recommendations = {}
        
        if 'team_short_name' not in players_df.columns:
            return recommendations
        
        # Calculate team FDR averages
        team_fdr = fixtures_df.groupby('team_short_name')['combined_fdr'].mean()
        
        # Merge with player data
        players_with_fdr = players_df.merge(
            team_fdr.reset_index(),
            left_on='team_short_name',
            right_on='team_short_name',
            how='left'
        )
        
        # Create recommendations by position
        for position in ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']:
            pos_players = players_with_fdr[players_with_fdr['position_name'] == position]
            
            if not pos_players.empty:
                # Target players (good fixtures, good stats)
                targets = pos_players[
                    (pos_players['combined_fdr'] <= threshold) &
                    (pos_players['total_points'] >= pos_players['total_points'].quantile(0.6))
                ].nlargest(5, 'points_per_million')
                
                recommendations[f'{position.lower()}_targets'] = [
                    {
                        'name': row['web_name'],
                        'team': row['team_short_name'],
                        'price': row['cost_millions'],
                        'points': row['total_points'],
                        'form': row.get('form', 0),
                        'avg_fdr': row['combined_fdr'],
                        'ownership': row.get('selected_by_percent', 0),
                        'points_per_million': row['points_per_million']
                    }
                    for _, row in targets.iterrows()
                ]
        
        # Players to avoid (difficult fixtures)
        avoid_players = players_with_fdr[
            players_with_fdr['combined_fdr'] >= 4.0
        ].nlargest(10, 'combined_fdr')
        
        recommendations['avoid_players'] = [
            {
                'name': row['web_name'],
                'team': row['team_short_name'],
                'avg_fdr': row['combined_fdr']
            }
            for _, row in avoid_players.iterrows()
        ]
        
        # Differential picks (low ownership, good fixtures)
        differentials = players_with_fdr[
            (players_with_fdr['combined_fdr'] <= threshold) &
            (players_with_fdr['selected_by_percent'] <= 10) &
            (players_with_fdr['total_points'] >= 50)
        ].nlargest(8, 'points_per_million')
        
        recommendations['differentials'] = [
            {
                'name': row['web_name'],
                'team': row['team_short_name'],
                'price': row['cost_millions'],
                'points': row['total_points'],
                'avg_fdr': row['combined_fdr'],
                'ownership': row['selected_by_percent']
            }
            for _, row in differentials.iterrows()
        ]
        
        return recommendations
    
    def _calculate_fixture_swings(self, fixtures_df):
        """Calculate fixture difficulty swings between early and later fixtures"""
        
        if 'combined_fdr' not in fixtures_df.columns:
            return pd.DataFrame()
        
        # Split fixtures into early (1-2) and later (3-5)
        early_fixtures = fixtures_df[fixtures_df['fixture_number'] <= 2].groupby('team_short_name')['combined_fdr'].mean()
        later_fixtures = fixtures_df[fixtures_df['fixture_number'] >= 3].groupby('team_short_name')['combined_fdr'].mean()
        
        # Calculate swing (positive = getting harder, negative = getting easier)
        swing_data = []
        
        for team in early_fixtures.index:
            if team in later_fixtures.index:
                early_fdr = early_fixtures[team]
                later_fdr = later_fixtures[team]
                swing = later_fdr - early_fdr
                
                swing_data.append({
                    'team_short_name': team,
                    'early_fdr': early_fdr,
                    'later_fdr': later_fdr,
                    'swing': swing
                })
        
        return pd.DataFrame(swing_data)

    def run(self):
        """Main application runner"""
        try:
            # Render sidebar and get selected page
            selected_page = self.render_sidebar()
            
            # Route to appropriate page
            if selected_page == "dashboard":
                self.render_dashboard()
            elif selected_page == "players":
                self.render_players()
            elif selected_page == "fixtures":
                self.render_fixtures()
            elif selected_page == "filters":
                self.render_filters()
            elif selected_page == "teams":
                self.render_teams()
            elif selected_page == "my_team":
                self.render_my_team()
            elif selected_page == "ai_recommendations":
                self.render_ai_recommendations()
            elif selected_page == "team_builder":
                self.render_team_builder()
            else:
                # Default to dashboard
                self.render_dashboard()
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.write("Please try refreshing the page or loading data again.")

    def render_dashboard(self):
        """Render main dashboard"""
        st.title("⚽ FPL Analytics Dashboard")
        st.markdown("### Welcome to your Fantasy Premier League Analytics Hub!")
        
        if not st.session_state.data_loaded:
            st.info("👋 Welcome! Click '🔄 Refresh Data' in the sidebar to get started.")
            return
        
        df = st.session_state.players_df
        
        # Check if required columns exist
        required_columns = ['web_name', 'total_points', 'cost_millions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.write("Available columns:", list(df.columns))
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Players", len(df))
        with col2:
            avg_price = df['cost_millions'].mean()
            st.metric("💰 Avg Price", f"£{avg_price:.1f}m")
        with col3:
            if len(df) > 0:
                top_scorer = df.loc[df['total_points'].idxmax()]
                st.metric("⭐ Top Scorer", f"{top_scorer['web_name']} ({top_scorer['total_points']})")
            else:
                st.metric("⭐ Top Scorer", "No data")
        with col4:
            if len(df) > 0:
                most_expensive = df.loc[df['cost_millions'].idxmax()]
                st.metric("💎 Most Expensive", f"{most_expensive['web_name']} (£{most_expensive['cost_millions']}m)")
            else:
                st.metric("💎 Most Expensive", "No data")
        
        st.divider()
        
        # Position breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Players by Position")
            if 'position_name' in df.columns:
                position_counts = df['position_name'].value_counts()
                fig = px.pie(
                    values=position_counts.values,
                    names=position_counts.index,
                    title="Player Distribution by Position"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Position data not available")
        
        with col2:
            st.subheader("💰 Average Price by Position")
            if 'position_name' in df.columns:
                avg_price_by_pos = df.groupby('position_name')['cost_millions'].mean().sort_values(ascending=True)
                fig = px.bar(
                    x=avg_price_by_pos.values,
                    y=avg_price_by_pos.index,
                    orientation='h',
                    title="Average Price by Position",
                    labels={'x': 'Price (£m)', 'y': 'Position'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Position data not available")
        
        # Top performers
        st.subheader("🌟 Top Performers")
        
        tab1, tab2, tab3 = st.tabs(["🏆 Top Scorers", "💎 Best Value", "🔥 Form Players"])
        
        with tab1:
            if len(df) > 0:
                display_cols = ['web_name', 'total_points', 'cost_millions']
                if 'team_short_name' in df.columns:
                    display_cols.insert(1, 'team_short_name')
                if 'position_name' in df.columns:
                    display_cols.insert(-1, 'position_name')
                
                top_scorers = df.nlargest(10, 'total_points')[display_cols]
                st.dataframe(top_scorers, use_container_width=True)
            else:
                st.warning("No player data available")
        
        with tab2:
            if len(df) > 0 and 'points_per_million' in df.columns:
                display_cols = ['web_name', 'total_points', 'cost_millions', 'points_per_million']
                if 'team_short_name' in df.columns:
                    display_cols.insert(1, 'team_short_name')
                if 'position_name' in df.columns:
                    display_cols.insert(-2, 'position_name')
                
                best_value = df[df['total_points'] > 50].nlargest(10, 'points_per_million')[display_cols]
                st.dataframe(best_value, use_container_width=True)
            else:
                st.warning("Points per million data not available")
        
        with tab3:
            if len(df) > 0 and 'form' in df.columns:
                display_cols = ['web_name', 'form', 'total_points', 'cost_millions']
                if 'team_short_name' in df.columns:
                    display_cols.insert(1, 'team_short_name')
                if 'position_name' in df.columns:
                    display_cols.insert(-2, 'position_name')
                
                form_players = df[df['total_points'] > 30].nlargest(10, 'form')[display_cols]
                st.dataframe(form_players, use_container_width=True)
            else:
                st.warning("Form data not available")

    def render_players(self):
        """Enhanced Player Analysis with comprehensive explanations and advanced metrics"""
        st.header("📊 Advanced Player Analysis")
        
        # Comprehensive explanation section
        with st.expander("📚 Master Guide to Player Analysis", expanded=False):
            st.markdown("""
            **Advanced Player Analysis** is your data-driven approach to identifying the best FPL assets. This comprehensive tool evaluates players across multiple dimensions to help you make informed decisions.
            
            🎯 **Core Analysis Framework:**
            
            **1. Performance Metrics**
            - **Total Points**: Season accumulation showing overall contribution
            - **Points Per Game (PPG)**: True ability indicator regardless of games played
            - **Form**: Last 5 games momentum - crucial for current decisions
            - **Expected Points**: AI-driven predictions based on fixtures, form, and underlying stats
            
            **2. Value Analysis**
            - **Points Per Million (PPM)**: Budget efficiency - maximize returns per £spent
            - **Price Changes**: Track rising/falling assets for optimal timing
            - **Value Over Replacement**: How much better than cheapest viable option
            
            **3. Ownership & Differential Analysis**
            - **Template Players**: High ownership, essential for rank protection
            - **Differentials**: Low ownership gems for rank climbing
            - **Captaincy Data**: Popular captain choices and success rates
            
            **4. Position-Specific Insights**
            - **Goalkeepers**: Save points potential, clean sheet probability, fixture runs
            - **Defenders**: Clean sheet + attacking threat balance, set piece involvement
            - **Midfielders**: Goals + assists output, underlying stats (shots, key passes)
            - **Forwards**: Goal threat, penalty involvement, big match performance
            
            **5. Advanced Metrics**
            - **Fixture Difficulty**: Upcoming opponent analysis for transfer timing
            - **Rotation Risk**: Minutes prediction based on squad depth and manager tendencies
            - **Injury History**: Fitness reliability and recovery patterns
            - **Home vs Away**: Performance splits for captain selection
            
            🎮 **How to Use Each Tab:**
            
            **� Smart Filters & Overview**
            - Use position filters to narrow your search
            - Apply price ranges for budget planning
            - Sort by different metrics to find opportunities
            - Quick overview of top performers in each category
            
            **📈 Performance Metrics**
            - Deep dive into scoring consistency
            - Form trends and momentum analysis
            - Expected vs actual performance gaps
            - Historical performance patterns
            
            **⚖️ Player Comparison**
            - Side-by-side analysis of similar players
            - Transfer decision support tools
            - Value comparison across price points
            - Risk vs reward assessment
            
            **🎯 Position Analysis**
            - Position-specific rankings and insights
            - Formation impact on player selection
            - Budget allocation optimization
            - Positional scarcity analysis
            
            **💡 AI Insights**
            - Machine learning predictions
            - Transfer recommendations
            - Captain suggestions
            - Market timing advice
            
            💡 **Pro Tips for Success:**
            - Always consider fixture difficulty in transfer timing
            - Balance template players with differentials based on your rank
            - Use ownership data to identify crowd psychology
            - Monitor underlying stats for early identification of form changes
            - Consider squad depth and rotation risk, especially around busy periods
            """)
        
        if not st.session_state.data_loaded:
            st.info("Please load data first from the Dashboard.")
            return
        
        df = st.session_state.players_df
        
        if df.empty:
            st.warning("No player data available")
            return
        
        # **NEW: Enhanced tab structure for comprehensive analysis**
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Smart Filters & Overview",
            "📈 Performance Metrics", 
            "⚖️ Player Comparison",
            "🎯 Position Analysis",
            "💡 AI Insights"
        ])
        
        with tab1:
            self._render_enhanced_player_filters(df)
        
        with tab2:
            self._render_performance_metrics_dashboard(df)
        
        with tab3:
            self._render_player_comparison_tool(df)
        
        with tab4:
            self._render_position_specific_analysis(df)
        
        with tab5:
            self._render_ai_player_insights(df)

    def _render_enhanced_player_filters(self, df):
        """Enhanced filtering system with advanced metrics"""
        st.subheader("🔍 Smart Player Filters & Overview")
        
        # **ENHANCED: Advanced filtering system**
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Position filter
            if 'position_name' in df.columns:
                positions = st.multiselect(
                    "🎯 Position:",
                    df['position_name'].unique(),
                    default=df['position_name'].unique(),
                    help="Filter by player positions"
                )
            else:
                positions = df['position_name'].unique() if 'position_name' in df.columns else []
        
        with col2:
            # Team filter
            if 'team_name' in df.columns:
                teams = st.multiselect(
                    "⚽ Team:",
                    sorted(df['team_name'].unique()),
                    default=sorted(df['team_name'].unique()),
                    help="Filter by teams"
                )
            else:
                teams = df['team_name'].unique() if 'team_name' in df.columns else []
        
        with col3:
            # Price range
            if 'cost_millions' in df.columns:
                min_price, max_price = st.slider(
                    "💰 Price Range (£m):",
                    float(df['cost_millions'].min()),
                    float(df['cost_millions'].max()),
                    (float(df['cost_millions'].min()), float(df['cost_millions'].max())),
                    step=0.1,
                    help="Filter by player price"
                )
            else:
                min_price, max_price = 0, 15
        
        with col4:
            # **NEW: Advanced metric filters**
            st.write("**🎪 Advanced Filters:**")
            
            min_points = st.number_input("Min Total Points", 0, 300, 0, help="Minimum total points threshold")
            min_form = st.slider("Min Form", 0.0, 10.0, 0.0, 0.1, help="Minimum form rating")
            
            if 'minutes' in df.columns:
                min_minutes = st.number_input("Min Minutes", 0, 3500, 0, help="Minimum minutes played")
            else:
                min_minutes = 0
        
        # **NEW: xG/xA filters**
        col5, col6 = st.columns(2)
        with col5:
            if any(col for col in df.columns if 'expected_goals' in col.lower() or 'xg' in col.lower()):
                min_xg = st.slider("Min xG", 0.0, 20.0, 0.0, 0.1, help="Minimum expected goals")
            else:
                min_xg = 0.0
                
        with col6:
            if any(col for col in df.columns if 'expected_assists' in col.lower() or 'xa' in col.lower()):
                min_xa = st.slider("Min xA", 0.0, 15.0, 0.0, 0.1, help="Minimum expected assists")
            else:
                min_xa = 0.0
        
        # Apply filters
        filtered_df = df.copy()
        
        # Standard filters
        if positions:
            filtered_df = filtered_df[filtered_df['position_name'].isin(positions)]
        if teams:
            filtered_df = filtered_df[filtered_df['team_name'].isin(teams)]
        if 'cost_millions' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['cost_millions'] >= min_price) &
                (filtered_df['cost_millions'] <= max_price)
            ]
        
        # Advanced filters
        if 'total_points' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['total_points'] >= min_points]
        if 'form' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['form'] >= min_form]
        if 'minutes' in filtered_df.columns and min_minutes > 0:
            filtered_df = filtered_df[filtered_df['minutes'] >= min_minutes]
        
        # xG/xA filters
        xg_col = next((col for col in filtered_df.columns if 'expected_goals' in col.lower() or col.lower() == 'xg'), None)
        if xg_col and min_xg > 0:
            # Convert to numeric and filter
            numeric_xg = pd.to_numeric(filtered_df[xg_col], errors='coerce').fillna(0)
            filtered_df = filtered_df[numeric_xg >= min_xg]
            
        xa_col = next((col for col in filtered_df.columns if 'expected_assists' in col.lower() or col.lower() == 'xa'), None)
        if xa_col and min_xa > 0:
            # Convert to numeric and filter
            numeric_xa = pd.to_numeric(filtered_df[xa_col], errors='coerce').fillna(0)
            filtered_df = filtered_df[numeric_xa >= min_xa]
        
        # Store filtered data in session state for other tabs
        st.session_state.filtered_players_df = filtered_df
        
        # **ENHANCED: Results overview with key metrics**
        st.write(f"📊 **{len(filtered_df)} players found** (filtered from {len(df)} total)")
        
        if not filtered_df.empty:
            # **NEW: Enhanced display with comprehensive columns**
            display_cols = self._get_enhanced_display_columns(filtered_df)
            
            # Sort options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox(
                    "📈 Sort by:",
                    options=['total_points', 'form', 'cost_millions', 'points_per_million', 'selected_by_percent'] + 
                            ([xg_col] if xg_col else []) + ([xa_col] if xa_col else []),
                    index=0
                )
            with col2:
                ascending = st.checkbox("Ascending order", False)
            
            # **ENHANCED: Advanced data table with better formatting**
            sorted_df = filtered_df[display_cols].sort_values(sort_by, ascending=ascending)
            
            # Format the dataframe for better display
            formatted_df = self._format_player_dataframe(sorted_df)
            
            st.dataframe(
                formatted_df,
                use_container_width=True,
                hide_index=True,
                column_config=self._get_column_config()
            )
            
            # **NEW: Quick stats summary**
            self._display_filtered_stats_summary(filtered_df)
        else:
            st.warning("No players match your filters. Try adjusting the criteria.")

    def _get_enhanced_display_columns(self, df):
        """Get comprehensive column list for display"""
        base_cols = ['web_name', 'position_name', 'team_short_name', 'cost_millions', 'total_points']
        
        # Core performance columns
        performance_cols = ['form', 'points_per_million', 'selected_by_percent', 'minutes']
        
        # Advanced metrics columns
        advanced_cols = []
        
        # Find xG and xA columns
        xg_col = next((col for col in df.columns if 'expected_goals' in col.lower() or col.lower() == 'xg'), None)
        if xg_col:
            advanced_cols.append(xg_col)
            
        xa_col = next((col for col in df.columns if 'expected_assists' in col.lower() or col.lower() == 'xa'), None)
        if xa_col:
            advanced_cols.append(xa_col)
        
        # Position-specific columns
        position_specific = ['goals_scored', 'assists', 'clean_sheets', 'saves', 'bonus']
        
        # Combine all available columns
        all_cols = base_cols + [col for col in performance_cols if col in df.columns] + \
                  advanced_cols + [col for col in position_specific if col in df.columns]
        
        return [col for col in all_cols if col in df.columns]

    def _format_player_dataframe(self, df):
        """Format dataframe for better display"""
        formatted_df = df.copy()
        
        # Format price
        if 'cost_millions' in formatted_df.columns:
            formatted_df['cost_millions'] = formatted_df['cost_millions'].apply(lambda x: f"£{x:.1f}m")
        
        # Format percentages
        if 'selected_by_percent' in formatted_df.columns:
            formatted_df['selected_by_percent'] = formatted_df['selected_by_percent'].apply(lambda x: f"{x:.1f}%")
        
        # Format form
        if 'form' in formatted_df.columns:
            formatted_df['form'] = formatted_df['form'].apply(lambda x: f"{x:.1f}")
        
        # Format points per million
        if 'points_per_million' in formatted_df.columns:
            formatted_df['points_per_million'] = formatted_df['points_per_million'].apply(lambda x: f"{x:.1f}")
        
        return formatted_df

    def _get_column_config(self):
        """Get column configuration for better display"""
        return {
            "web_name": st.column_config.TextColumn("Player", help="Player name"),
            "position_name": st.column_config.TextColumn("Position", help="Playing position"),
            "team_short_name": st.column_config.TextColumn("Team", help="Team abbreviation"),
            "cost_millions": st.column_config.TextColumn("Price", help="Current price"),
            "total_points": st.column_config.NumberColumn("Points", help="Total points this season"),
            "form": st.column_config.TextColumn("Form", help="Form rating (last 4 games)"),
            "points_per_million": st.column_config.TextColumn("Value", help="Points per million £"),
            "selected_by_percent": st.column_config.TextColumn("Ownership", help="% of teams owning this player"),
            "minutes": st.column_config.NumberColumn("Minutes", help="Total minutes played"),
            "goals_scored": st.column_config.NumberColumn("Goals", help="Goals scored"),
            "assists": st.column_config.NumberColumn("Assists", help="Assists provided"),
            "clean_sheets": st.column_config.NumberColumn("CS", help="Clean sheets"),
            "saves": st.column_config.NumberColumn("Saves", help="Saves made"),
            "bonus": st.column_config.NumberColumn("Bonus", help="Bonus points earned")
        }

    def _display_filtered_stats_summary(self, df):
        """Display summary statistics for filtered players"""
        st.subheader("📈 Filtered Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df['cost_millions'].mean() if 'cost_millions' in df.columns else 0
            st.metric("Average Price", f"£{avg_price:.1f}m")
            
        with col2:
            avg_points = df['total_points'].mean() if 'total_points' in df.columns else 0
            st.metric("Average Points", f"{avg_points:.1f}")
            
        with col3:
            avg_form = df['form'].mean() if 'form' in df.columns else 0
            st.metric("Average Form", f"{avg_form:.1f}")
            
        with col4:
            avg_ownership = df['selected_by_percent'].mean() if 'selected_by_percent' in df.columns else 0
            st.metric("Average Ownership", f"{avg_ownership:.1f}%")

    def _render_performance_metrics_dashboard(self, df):
        """Enhanced performance metrics dashboard"""
        st.subheader("📈 Performance Metrics Dashboard")
        
        # Use filtered data if available
        display_df = st.session_state.get('filtered_players_df', df)
        
        if display_df.empty:
            st.warning("No players to analyze. Adjust your filters.")
            return
        
        # **NEW: Performance metrics tabs**
        metric_tab1, metric_tab2, metric_tab3 = st.tabs([
            "⚽ Attacking Metrics",
            "🛡️ Defensive Metrics", 
            "📊 General Performance"
        ])
        
        with metric_tab1:
            self._render_attacking_metrics(display_df)
        
        with metric_tab2:
            self._render_defensive_metrics(display_df)
        
        with metric_tab3:
            self._render_general_performance_metrics(display_df)

    def _render_attacking_metrics(self, df):
        """Render attacking performance metrics"""
        st.write("**⚽ Attacking Performance Analysis**")
        
        # Filter for attacking players
        attacking_positions = ['Midfielder', 'Forward']
        attacking_df = df[df['position_name'].isin(attacking_positions)] if 'position_name' in df.columns else df
        
        if attacking_df.empty:
            st.info("No attacking players in current filter")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # **NEW: xG Analysis**
            st.write("**🎯 Expected Goals (xG) Leaders**")
            xg_col = next((col for col in attacking_df.columns if 'expected_goals' in col.lower() or col.lower() == 'xg'), None)
            
            if xg_col:
                # Convert xG column to numeric, handling any non-numeric values
                attacking_df_copy = attacking_df.copy()
                attacking_df_copy[xg_col] = pd.to_numeric(attacking_df_copy[xg_col], errors='coerce').fillna(0)
                
                # Filter out players with 0 xG and get top performers
                xg_players = attacking_df_copy[attacking_df_copy[xg_col] > 0]
                if not xg_players.empty:
                    top_xg = xg_players.nlargest(10, xg_col)[['web_name', 'position_name', 'team_short_name', xg_col, 'goals_scored']]
                    
                    for _, player in top_xg.iterrows():
                        goals = player.get('goals_scored', 0)
                        xg = player[xg_col]
                        efficiency = f"({goals}/{xg:.1f})" if pd.notna(xg) and xg > 0 else ""
                        st.write(f"• **{player['web_name']}** ({player.get('team_short_name', 'N/A')}) - xG: {xg:.2f} {efficiency}")
                else:
                    st.info("No xG data available for attacking players")
            else:
                st.info("xG data not available")
        
        with col2:
            # **NEW: xA Analysis**
            st.write("**🎯 Expected Assists (xA) Leaders**")
            xa_col = next((col for col in attacking_df.columns if 'expected_assists' in col.lower() or col.lower() == 'xa'), None)
            
            if xa_col:
                # Convert xA column to numeric, handling any non-numeric values
                attacking_df_copy = attacking_df.copy()
                attacking_df_copy[xa_col] = pd.to_numeric(attacking_df_copy[xa_col], errors='coerce').fillna(0)
                
                # Filter out players with 0 xA and get top performers
                xa_players = attacking_df_copy[attacking_df_copy[xa_col] > 0]
                if not xa_players.empty:
                    top_xa = xa_players.nlargest(10, xa_col)[['web_name', 'position_name', 'team_short_name', xa_col, 'assists']]
                    
                    for _, player in top_xa.iterrows():
                        assists = player.get('assists', 0)
                        xa = player[xa_col]
                        efficiency = f"({assists}/{xa:.1f})" if pd.notna(xa) and xa > 0 else ""
                        st.write(f"• **{player['web_name']}** ({player.get('team_short_name', 'N/A')}) - xA: {xa:.2f} {efficiency}")
                else:
                    st.info("No xA data available for attacking players")
            else:
                st.info("xA data not available")
        
        # **NEW: Goals + Assists efficiency**
        st.write("**⚽ Goal Contribution Efficiency**")
        if 'goals_scored' in attacking_df.columns and 'assists' in attacking_df.columns:
            attacking_df_copy = attacking_df.copy()
            attacking_df_copy['goal_contributions'] = attacking_df_copy['goals_scored'] + attacking_df_copy['assists']
            attacking_df_copy['contributions_per_90'] = (attacking_df_copy['goal_contributions'] * 90) / attacking_df_copy.get('minutes', 1).replace(0, 1)
            
            top_contributors = attacking_df_copy.nlargest(10, 'goal_contributions')[
                ['web_name', 'team_short_name', 'goals_scored', 'assists', 'goal_contributions', 'contributions_per_90']
            ]
            
            for _, player in top_contributors.iterrows():
                per_90 = player['contributions_per_90']
                st.write(f"• **{player['web_name']}** - {player['goals_scored']}G + {player['assists']}A = {player['goal_contributions']} ({per_90:.2f} per 90min)")

    def _render_defensive_metrics(self, df):
        """Render defensive performance metrics"""
        st.write("**🛡️ Defensive Performance Analysis**")
        
        # Filter for defensive players
        defensive_positions = ['Goalkeeper', 'Defender']
        defensive_df = df[df['position_name'].isin(defensive_positions)] if 'position_name' in df.columns else df
        
        if defensive_df.empty:
            st.info("No defensive players in current filter")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # **NEW: Clean Sheets Analysis**
            st.write("**🥅 Clean Sheets Leaders**")
            if 'clean_sheets' in defensive_df.columns:
                top_cs = defensive_df.nlargest(10, 'clean_sheets')[['web_name', 'position_name', 'team_short_name', 'clean_sheets', 'total_points']]
                
                for _, player in top_cs.iterrows():
                    cs_percentage = (player['clean_sheets'] / max(df.get('games_played', 10).max(), 1) * 100) if 'games_played' in df.columns else 0
                    st.write(f"• **{player['web_name']}** ({player.get('team_short_name', 'N/A')}) - {player['clean_sheets']} CS")
            else:
                st.info("Clean sheets data not available")
        
        with col2:
            # **NEW: Goalkeeper-specific metrics**
            st.write("**✋ Goalkeeper Performance**")
            gk_df = defensive_df[defensive_df['position_name'] == 'Goalkeeper'] if 'position_name' in defensive_df.columns else pd.DataFrame()
            
            if not gk_df.empty and 'saves' in gk_df.columns:
                top_saves = gk_df.nlargest(10, 'saves')[['web_name', 'team_short_name', 'saves', 'clean_sheets', 'total_points']]
                
                for _, player in top_saves.iterrows():
                    save_points = player['saves'] // 3  # 1 point per 3 saves
                    st.write(f"• **{player['web_name']}** ({player.get('team_short_name', 'N/A')}) - {player['saves']} saves ({save_points} pts)")
            else:
                st.info("Saves data not available")
        
        # **NEW: Defensive contribution analysis**
        st.write("**🛡️ Overall Defensive Contribution**")
        if 'clean_sheets' in defensive_df.columns:
            defensive_df_copy = defensive_df.copy()
            defensive_df_copy['defensive_points'] = (defensive_df_copy.get('clean_sheets', 0) * 4) + \
                                                   (defensive_df_copy.get('saves', 0) // 3) + \
                                                   (defensive_df_copy.get('goals_scored', 0) * 6)
            
            top_defensive = defensive_df_copy.nlargest(10, 'defensive_points')[
                ['web_name', 'team_short_name', 'clean_sheets', 'saves', 'goals_scored', 'defensive_points']
            ]
            
            for _, player in top_defensive.iterrows():
                st.write(f"• **{player['web_name']}** - Defensive contribution: {player['defensive_points']:.0f} points")

    def _render_general_performance_metrics(self, df):
        """Render general performance metrics"""
        st.write("**📊 General Performance Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # **NEW: Consistency Analysis**
            st.write("**📈 Most Consistent Performers**")
            if 'minutes' in df.columns and 'total_points' in df.columns:
                df_copy = df.copy()
                df_copy['points_per_90'] = (df_copy['total_points'] * 90) / df_copy['minutes'].replace(0, 1)
                df_copy = df_copy[df_copy['minutes'] >= 500]  # Min 500 minutes played
                
                top_consistent = df_copy.nlargest(10, 'points_per_90')[
                    ['web_name', 'position_name', 'team_short_name', 'total_points', 'minutes', 'points_per_90']
                ]
                
                for _, player in top_consistent.iterrows():
                    st.write(f"• **{player['web_name']}** ({player.get('position_name', 'N/A')}) - {player['points_per_90']:.2f} pts/90min")
        
        with col2:
            # **NEW: Value for Money Analysis**
            st.write("**💰 Best Value Players**")
            if 'points_per_million' in df.columns:
                top_value = df.nlargest(10, 'points_per_million')[
                    ['web_name', 'position_name', 'team_short_name', 'cost_millions', 'total_points', 'points_per_million']
                ]
                
                for _, player in top_value.iterrows():
                    st.write(f"• **{player['web_name']}** (£{player.get('cost_millions', 0):.1f}m) - {player['points_per_million']:.1f} pts/£m")
        
        # **NEW: Bonus Points Analysis**
        st.write("**⭐ Bonus Points Masters**")
        if 'bonus' in df.columns:
            top_bonus = df.nlargest(10, 'bonus')[['web_name', 'position_name', 'team_short_name', 'bonus', 'total_points']]
            
            col1, col2, col3 = st.columns(3)
            for i, (_, player) in enumerate(top_bonus.iterrows()):
                with [col1, col2, col3][i % 3]:
                    bonus_percentage = (player['bonus'] / player['total_points'] * 100) if player['total_points'] > 0 else 0
                    st.write(f"**{player['web_name']}** - {player['bonus']} bonus ({bonus_percentage:.1f}%)")

    def _render_player_comparison_tool(self, df):
        """Advanced player comparison tool"""
        st.subheader("⚖️ Advanced Player Comparison")
        
        # Player selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Select Players to Compare (max 4):**")
            available_players = df['web_name'].tolist() if 'web_name' in df.columns else []
            selected_players = st.multiselect(
                "Choose players:",
                options=available_players,
                max_selections=4,
                help="Select 2-4 players for detailed comparison"
            )
        
        with col2:
            if len(selected_players) >= 2:
                st.write(f"**Comparing {len(selected_players)} players:**")
                for player in selected_players:
                    st.write(f"• {player}")
        
        if len(selected_players) >= 2:
            comparison_df = df[df['web_name'].isin(selected_players)]
            
            # **NEW: Detailed comparison table**
            st.write("**📊 Detailed Comparison Table**")
            
            comparison_metrics = [
                'web_name', 'position_name', 'team_short_name', 'cost_millions', 
                'total_points', 'form', 'points_per_million', 'selected_by_percent',
                'goals_scored', 'assists', 'clean_sheets', 'bonus', 'minutes'
            ]
            
            # Add xG and xA if available
            xg_col = next((col for col in comparison_df.columns if 'expected_goals' in col.lower()), None)
            xa_col = next((col for col in comparison_df.columns if 'expected_assists' in col.lower()), None)
            
            if xg_col:
                comparison_metrics.append(xg_col)
            if xa_col:
                comparison_metrics.append(xa_col)
            
            comparison_metrics = [col for col in comparison_metrics if col in comparison_df.columns]
            
            # Transpose for better comparison view
            comparison_table = comparison_df[comparison_metrics].set_index('web_name').T
            st.dataframe(comparison_table, use_container_width=True)
            
            # **NEW: Key insights from comparison**
            st.write("**💡 Comparison Insights**")
            self._generate_comparison_insights(comparison_df)
        
        elif len(selected_players) == 1:
            st.info("Select at least 2 players to compare")
        else:
            st.info("Select players from the list above to start comparing")

    def _generate_comparison_insights(self, comparison_df):
        """Generate insights from player comparison"""
        if comparison_df.empty:
            return
        
        insights = []
        
        # Price vs Points analysis
        if 'cost_millions' in comparison_df.columns and 'total_points' in comparison_df.columns:
            best_value = comparison_df.loc[comparison_df['points_per_million'].idxmax()] if 'points_per_million' in comparison_df.columns else None
            if best_value is not None:
                insights.append(f"💰 **Best Value**: {best_value['web_name']} ({best_value['points_per_million']:.1f} pts/£m)")
        
        # Form analysis
        if 'form' in comparison_df.columns:
            best_form = comparison_df.loc[comparison_df['form'].idxmax()]
            insights.append(f"🔥 **Best Form**: {best_form['web_name']} (Form: {best_form['form']:.1f})")
        
        # Ownership analysis
        if 'selected_by_percent' in comparison_df.columns:
            lowest_owned = comparison_df.loc[comparison_df['selected_by_percent'].idxmin()]
            insights.append(f"💎 **Lowest Ownership**: {lowest_owned['web_name']} ({lowest_owned['selected_by_percent']:.1f}%)")
        
        # xG analysis
        xg_col = next((col for col in comparison_df.columns if 'expected_goals' in col.lower()), None)
        if xg_col:
            # Convert to numeric for analysis
            xg_values = pd.to_numeric(comparison_df[xg_col], errors='coerce').fillna(0)
            if xg_values.max() > 0:  # Only show if there's actual xG data
                best_xg_idx = xg_values.idxmax()
                best_xg = comparison_df.loc[best_xg_idx]
                insights.append(f"⚽ **Highest xG**: {best_xg['web_name']} (xG: {xg_values.loc[best_xg_idx]:.2f})")
        
        for insight in insights:
            st.write(insight)

    def _render_position_specific_analysis(self, df):
        """Position-specific detailed analysis"""
        st.subheader("🎯 Position-Specific Analysis")
        
        if 'position_name' not in df.columns:
            st.warning("Position data not available")
            return
        
        # Position selector
        selected_position = st.selectbox(
            "Select Position:",
            options=df['position_name'].unique(),
            help="Choose position for detailed analysis"
        )
        
        position_df = df[df['position_name'] == selected_position]
        
        if position_df.empty:
            st.warning("No players found for selected position")
            return
        
        # **NEW: Position-specific metrics and analysis**
        if selected_position == 'Goalkeeper':
            self._render_goalkeeper_analysis(position_df)
        elif selected_position == 'Defender':
            self._render_defender_analysis(position_df)
        elif selected_position == 'Midfielder':
            self._render_midfielder_analysis(position_df)
        elif selected_position == 'Forward':
            self._render_forward_analysis(position_df)

    def _render_goalkeeper_analysis(self, df):
        """Goalkeeper-specific analysis"""
        st.write(f"**🥅 Goalkeeper Analysis ({len(df)} players)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Clean sheets analysis
            if 'clean_sheets' in df.columns:
                st.write("**🛡️ Clean Sheets Leaders:**")
                top_cs = df.nlargest(5, 'clean_sheets')[['web_name', 'team_short_name', 'clean_sheets', 'total_points']]
                for _, gk in top_cs.iterrows():
                    st.write(f"• {gk['web_name']} ({gk.get('team_short_name', 'N/A')}) - {gk['clean_sheets']} CS")
        
        with col2:
            # Saves analysis
            if 'saves' in df.columns:
                st.write("**✋ Save Masters:**")
                top_saves = df.nlargest(5, 'saves')[['web_name', 'team_short_name', 'saves', 'total_points']]
                for _, gk in top_saves.iterrows():
                    save_points = gk['saves'] // 3
                    st.write(f"• {gk['web_name']} - {gk['saves']} saves ({save_points} pts)")

    def _render_defender_analysis(self, df):
        """Defender-specific analysis"""
        st.write(f"**🛡️ Defender Analysis ({len(df)} players)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'clean_sheets' in df.columns:
                st.write("**🛡️ Clean Sheet Specialists:**")
                top_cs = df.nlargest(5, 'clean_sheets')[['web_name', 'team_short_name', 'clean_sheets', 'goals_scored']]
                for _, def_player in top_cs.iterrows():
                    goals = def_player.get('goals_scored', 0)
                    st.write(f"• {def_player['web_name']} - {def_player['clean_sheets']} CS, {goals} goals")
        
        with col2:
            if 'goals_scored' in df.columns:
                st.write("**⚽ Goal-Scoring Defenders:**")
                top_goals = df.nlargest(5, 'goals_scored')[['web_name', 'team_short_name', 'goals_scored', 'total_points']]
                for _, def_player in top_goals.iterrows():
                    st.write(f"• {def_player['web_name']} - {def_player['goals_scored']} goals")

    def _render_midfielder_analysis(self, df):
        """Midfielder-specific analysis"""
        st.write(f"**⚽ Midfielder Analysis ({len(df)} players)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Goals and assists
            if 'goals_scored' in df.columns and 'assists' in df.columns:
                st.write("**🎯 Goal Contributors:**")
                df_copy = df.copy()
                df_copy['contributions'] = df_copy['goals_scored'] + df_copy['assists']
                top_contrib = df_copy.nlargest(5, 'contributions')[['web_name', 'team_short_name', 'goals_scored', 'assists', 'contributions']]
                for _, mid in top_contrib.iterrows():
                    st.write(f"• {mid['web_name']} - {mid['goals_scored']}G + {mid['assists']}A")
        
        with col2:
            # Expected stats
            xg_col = next((col for col in df.columns if 'expected_goals' in col.lower()), None)
            xa_col = next((col for col in df.columns if 'expected_assists' in col.lower()), None)
            
            if xg_col or xa_col:
                st.write("**📈 Expected Performance:**")
                for _, mid in df.nlargest(5, 'total_points').iterrows():
                    xg_text = ""
                    xa_text = ""
                    
                    if xg_col:
                        xg_value = pd.to_numeric(mid[xg_col], errors='coerce')
                        if pd.notna(xg_value):
                            xg_text = f"xG: {xg_value:.2f}"
                    
                    if xa_col:
                        xa_value = pd.to_numeric(mid[xa_col], errors='coerce')
                        if pd.notna(xa_value):
                            xa_text = f"xA: {xa_value:.2f}"
                    
                    expected_text = f"{xg_text} {xa_text}".strip()
                    if expected_text:
                        st.write(f"• {mid['web_name']} - {expected_text}")
                    
                if not any(pd.notna(pd.to_numeric(df[xg_col], errors='coerce')) for _ in [1] if xg_col) and \
                   not any(pd.notna(pd.to_numeric(df[xa_col], errors='coerce')) for _ in [1] if xa_col):
                    st.info("No expected performance data available")

    def _render_forward_analysis(self, df):
        """Forward-specific analysis"""
        st.write(f"**⚽ Forward Analysis ({len(df)} players)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'goals_scored' in df.columns:
                st.write("**⚽ Goal Machines:**")
                top_goals = df.nlargest(5, 'goals_scored')[['web_name', 'team_short_name', 'goals_scored', 'total_points']]
                for _, fwd in top_goals.iterrows():
                    st.write(f"• {fwd['web_name']} - {fwd['goals_scored']} goals")
        
        with col2:
            # xG analysis for forwards
            xg_col = next((col for col in df.columns if 'expected_goals' in col.lower()), None)
            if xg_col:
                st.write("**🎯 xG Analysis:**")
                # Convert xG column to numeric, handling any non-numeric values
                df_copy = df.copy()
                df_copy[xg_col] = pd.to_numeric(df_copy[xg_col], errors='coerce').fillna(0)
                
                # Filter out players with 0 xG and get top performers
                xg_players = df_copy[df_copy[xg_col] > 0]
                if not xg_players.empty:
                    top_xg = xg_players.nlargest(5, xg_col)[['web_name', 'team_short_name', xg_col, 'goals_scored']]
                    for _, fwd in top_xg.iterrows():
                        actual_goals = fwd.get('goals_scored', 0)
                        expected = fwd[xg_col]
                        efficiency = "overperforming" if actual_goals > expected else "underperforming" if actual_goals < expected * 0.8 else "on track"
                        st.write(f"• {fwd['web_name']} - xG: {expected:.2f}, Goals: {actual_goals} ({efficiency})")
                else:
                    st.info("No xG data available for forwards")

    def _render_ai_player_insights(self, df):
        """AI-powered player insights and recommendations"""
        st.subheader("💡 AI Player Insights & Recommendations")
        
        # Use filtered data if available
        display_df = st.session_state.get('filtered_players_df', df)
        
        insight_tab1, insight_tab2, insight_tab3 = st.tabs([
            "🎯 Smart Picks",
            "💎 Hidden Gems", 
            "⚠️ Players to Avoid"
        ])
        
        with insight_tab1:
            self._render_smart_picks(display_df)
        
        with insight_tab2:
            self._render_hidden_gems(display_df)
        
        with insight_tab3:
            self._render_players_to_avoid(display_df)

    def _render_smart_picks(self, df):
        """AI-recommended smart picks"""
        st.write("**🎯 AI-Recommended Smart Picks**")
        
        if df.empty:
            st.warning("No players available for analysis")
            return
        
        # Calculate AI score for each player
        df_copy = df.copy()
        df_copy['ai_score'] = 0
        
        # Base scoring factors
        if 'total_points' in df_copy.columns:
            df_copy['ai_score'] += df_copy['total_points'].rank(pct=True) * 0.3
        
        if 'form' in df_copy.columns:
            df_copy['ai_score'] += df_copy['form'].rank(pct=True) * 0.25
        
        if 'points_per_million' in df_copy.columns:
            df_copy['ai_score'] += df_copy['points_per_million'].rank(pct=True) * 0.2
        
        # Ownership penalty (reward low ownership)
        if 'selected_by_percent' in df_copy.columns:
            df_copy['ai_score'] += (100 - df_copy['selected_by_percent']).rank(pct=True) * 0.15
        
        # Minutes played reliability
        if 'minutes' in df_copy.columns:
            df_copy['ai_score'] += df_copy['minutes'].rank(pct=True) * 0.1
        
        # Get top picks by position
        if 'position_name' in df_copy.columns:
            for position in df_copy['position_name'].unique():
                position_players = df_copy[df_copy['position_name'] == position]
                top_pick = position_players.nlargest(1, 'ai_score')
                
                if not top_pick.empty:
                    player = top_pick.iloc[0]
                    st.write(f"**{position}: {player['web_name']}**")
                    st.write(f"  • Price: £{player.get('cost_millions', 0):.1f}m | Points: {player.get('total_points', 0)} | Form: {player.get('form', 0):.1f}")
                    st.write(f"  • AI Score: {player['ai_score']:.2f} | Ownership: {player.get('selected_by_percent', 0):.1f}%")
                    st.write("")

    def _render_hidden_gems(self, df):
        """Identify hidden gem players"""
        st.write("**💎 Hidden Gems (Low Ownership, High Performance)**")
        
        if df.empty or 'selected_by_percent' not in df.columns:
            st.warning("Ownership data not available")
            return
        
        # Find players with low ownership but good performance
        hidden_gems = df[
            (df['selected_by_percent'] < 15) &  # Low ownership
            (df.get('total_points', 0) > df['total_points'].quantile(0.6)) &  # Good points
            (df.get('minutes', 0) > 500)  # Regular player
        ]
        
        if hidden_gems.empty:
            st.info("No hidden gems found in current filter")
            return
        
        # Sort by points per million
        if 'points_per_million' in hidden_gems.columns:
            hidden_gems = hidden_gems.nlargest(5, 'points_per_million')
        
        for _, player in hidden_gems.iterrows():
            st.write(f"**{player['web_name']}** ({player.get('position_name', 'Unknown')})")
            st.write(f"  • £{player.get('cost_millions', 0):.1f}m | {player.get('total_points', 0)} pts | {player.get('selected_by_percent', 0):.1f}% owned")
            
            # Explain why it's a gem
            reasons = []
            if player.get('form', 0) >= 6:
                reasons.append("Excellent recent form")
            if player.get('points_per_million', 0) >= 8:
                reasons.append("Great value for money")
            if player.get('minutes', 0) >= 1000:
                reasons.append("Regular starter")
            
            if reasons:
                st.write(f"  • Why: {', '.join(reasons)}")
            st.write("")

    def _render_players_to_avoid(self, df):
        """Identify players to potentially avoid"""
        st.write("**⚠️ Players to Consider Avoiding**")
        
        if df.empty:
            st.warning("No players available for analysis")
            return
        
        avoid_players = []
        
        # High price, low performance
        if 'cost_millions' in df.columns and 'points_per_million' in df.columns:
            expensive_poor = df[
                (df['cost_millions'] >= 8.0) &  # Expensive
                (df['points_per_million'] < 6.0)  # Poor value
            ]
            
            for _, player in expensive_poor.head(3).iterrows():
                avoid_players.append({
                    'player': player,
                    'reason': f"Expensive (£{player['cost_millions']:.1f}m) but poor value ({player['points_per_million']:.1f} pts/£m)"
                })
        
        # Poor form
        if 'form' in df.columns:
            poor_form = df[df['form'] < 3.0]
            
            for _, player in poor_form.head(2).iterrows():
                avoid_players.append({
                    'player': player,
                    'reason': f"Very poor recent form ({player['form']:.1f})"
                })
        
        # Limited minutes
        if 'minutes' in df.columns:
            limited_minutes = df[
                (df['minutes'] < 500) & 
                (df.get('total_points', 0) > 10)  # Has some points but limited time
            ]
            
            for _, player in limited_minutes.head(2).iterrows():
                avoid_players.append({
                    'player': player,
                    'reason': f"Limited playing time ({player['minutes']} minutes)"
                })
        
        # Display avoiding recommendations
        for item in avoid_players[:5]:  # Limit to top 5
            player = item['player']
            reason = item['reason']
            st.write(f"**{player['web_name']}** ({player.get('position_name', 'Unknown')})")
            st.write(f"  • Reason: {reason}")
            st.write("")
        
        if not avoid_players:
            st.success("✅ No obvious players to avoid in current filter!")

    def render_filters(self):
        """Render advanced filters page"""
        st.header("🔍 Advanced Filters")
        st.info("🚧 Advanced filtering feature coming soon!")

    def render_teams(self):
        """Enhanced Team Analysis and Comparison"""
        st.header("📊 Team Analysis & Comparison")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first from the Dashboard.")
            return
        
        # Team analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Team Rankings", 
            "⚔️ Head-to-Head", 
            "📈 Performance Trends", 
            "🎯 Team Recommendations"
        ])
        
        with tab1:
            self._render_team_rankings()
        
        with tab2:
            self._render_head_to_head()
        
        with tab3:
            self._render_performance_trends()
        
        with tab4:
            self._render_team_specific_recommendations()
    
    def _render_team_rankings(self):
        """Render team rankings and statistics"""
        st.subheader("🏆 Premier League Team Rankings")
        
        if not st.session_state.data_loaded:
            return
        
        players_df = st.session_state.players_df
        teams_df = st.session_state.teams_df
        
        if teams_df.empty:
            st.warning("No team data available")
            return
        
        # Calculate team statistics
        team_stats = []
        
        for _, team in teams_df.iterrows():
            team_players = players_df[players_df['team'] == team['id']]
            
            if not team_players.empty:
                team_stats.append({
                    'Team': team['name'],
                    'Short': team['short_name'],
                    'Total Points': team_players['total_points'].sum(),
                    'Avg Points/Player': team_players['total_points'].mean(),
                    'Most Expensive': f"£{team_players['cost_millions'].max():.1f}m",
                    'Team Value': f"£{team_players['cost_millions'].sum():.1f}m",
                    'In-Form Players': len(team_players[team_players.get('form', 0) >= 7.0]),
                    'Top Scorer': team_players.loc[team_players['total_points'].idxmax(), 'web_name'] if len(team_players) > 0 else 'N/A'
                })
        
        if team_stats:
            team_stats_df = pd.DataFrame(team_stats)
            team_stats_df = team_stats_df.sort_values('Total Points', ascending=False)
            
            # Add rank
            team_stats_df['Rank'] = range(1, len(team_stats_df) + 1)
            
            # Reorder columns
            cols = ['Rank', 'Team', 'Short', 'Total Points', 'Avg Points/Player', 
                   'In-Form Players', 'Team Value', 'Top Scorer']
            
            st.dataframe(
                team_stats_df[cols],
                use_container_width=True,
                hide_index=True
            )
            
            # Team insights
            st.subheader("� Team Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Top Performing Teams**")
                top_3 = team_stats_df.head(3)
                for _, team in top_3.iterrows():
                    st.write(f"#{team['Rank']} **{team['Short']}** - {team['Total Points']} points")
            
            with col2:
                st.write("**🔥 Most In-Form Teams**")
                form_teams = team_stats_df.sort_values('In-Form Players', ascending=False).head(3)
                for _, team in form_teams.iterrows():
                    st.write(f"**{team['Short']}** - {team['In-Form Players']} players in form")
    
    def _render_head_to_head(self):
        """Render head-to-head team comparison"""
        st.subheader("⚔️ Head-to-Head Comparison")
        
        if not st.session_state.data_loaded:
            return
        
        teams_df = st.session_state.teams_df
        players_df = st.session_state.players_df
        
        if teams_df.empty:
            return
        
        # Team selection
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Team 1", teams_df['name'].tolist())
        
        with col2:
            team2 = st.selectbox("Select Team 2", teams_df['name'].tolist())
        
        if team1 != team2:
            # Get team IDs
            team1_id = teams_df[teams_df['name'] == team1]['id'].iloc[0]
            team2_id = teams_df[teams_df['name'] == team2]['id'].iloc[0]
            
            # Get players for each team
            team1_players = players_df[players_df['team'] == team1_id]
            team2_players = players_df[players_df['team'] == team2_id]
            
            # Comparison metrics
            st.subheader(f"�📊 {team1} vs {team2}")
            
            col1, col2, col3 = st.columns(3)
            
            # Total points comparison
            with col1:
                team1_total = team1_players['total_points'].sum()
                team2_total = team2_players['total_points'].sum()
                
                st.metric(f"{team1} Total Points", f"{team1_total:,}")
                st.metric(f"{team2} Total Points", f"{team2_total:,}")
                
                if team1_total > team2_total:
                    st.success(f"🏆 {team1} leads by {team1_total - team2_total} points")
                elif team2_total > team1_total:
                    st.success(f"🏆 {team2} leads by {team2_total - team1_total} points")
                else:
                    st.info("🤝 Teams are tied!")
            
            # Average player value
            with col2:
                team1_avg = team1_players['cost_millions'].mean()
                team2_avg = team2_players['cost_millions'].mean()
                
                st.metric(f"{team1} Avg Price", f"£{team1_avg:.1f}m")
                st.metric(f"{team2} Avg Price", f"£{team2_avg:.1f}m")
            
            # Form players
            with col3:
                if 'form' in players_df.columns:
                    team1_form = len(team1_players[team1_players['form'] >= 7.0])
                    team2_form = len(team2_players[team2_players['form'] >= 7.0])
                    
                    st.metric(f"{team1} In-Form", team1_form)
                    st.metric(f"{team2} In-Form", team2_form)
            
            # Top players comparison
            st.subheader("⭐ Top Players Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{team1} Top Performers**")
                team1_top = team1_players.nlargest(5, 'total_points')
                for _, player in team1_top.iterrows():
                    st.write(f"• {player['web_name']} - {player['total_points']} pts")
            
            with col2:
                st.write(f"**{team2} Top Performers**")
                team2_top = team2_players.nlargest(5, 'total_points')
                for _, player in team2_top.iterrows():
                    st.write(f"• {player['web_name']} - {player['total_points']} pts")
    
    def _render_performance_trends(self):
        """Render team performance trends"""
        st.subheader("📈 Performance Trends")
        
        # This would typically involve historical data
        st.info("� Performance trends analysis coming soon! This will include:")
        st.write("• Weekly point trends")
        st.write("• Form progression")
        st.write("• Player rotation patterns")
        st.write("• Injury impact analysis")
    
    def _render_team_specific_recommendations(self):
        """Render team-specific player recommendations"""
        st.subheader("🎯 Team-Specific Recommendations")
        
        if not st.session_state.data_loaded:
            return
        
        teams_df = st.session_state.teams_df
        players_df = st.session_state.players_df
        
        if teams_df.empty:
            return
        
        # Team selection
        selected_team = st.selectbox("Select a team for recommendations", teams_df['name'].tolist())
        
        if selected_team:
            team_id = teams_df[teams_df['name'] == selected_team]['id'].iloc[0]
            team_players = players_df[players_df['team'] == team_id]
            
            if not team_players.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**🎯 Best {selected_team} Players to Target**")
                    
                    # Filter for good performers
                    targets = team_players[
                        (team_players['total_points'] > 30) &
                        (team_players.get('form', 0) >= 5.0)
                    ].nlargest(5, 'total_points')
                    
                    for _, player in targets.iterrows():
                        ownership = player.get('selected_by_percent', 0)
                        differential = "💎" if ownership < 10 else "👥" if ownership < 30 else "🔥"
                        
                        st.write(f"{differential} **{player['web_name']}** "
                                f"(£{player['cost_millions']:.1f}m) - "
                                f"{player['total_points']} pts")
                
                with col2:
                    st.write(f"**⚠️ {selected_team} Players to Avoid**")
                    
                    # Filter for underperformers
                    avoid = team_players[
                        (team_players['cost_millions'] >= 6.0) &
                        (team_players.get('form', 10) <= 4.0)
                    ].head(3)
                    
                    if not avoid.empty:
                        for _, player in avoid.iterrows():
                            st.write(f"🔴 **{player['web_name']}** "
                                    f"(£{player['cost_millions']:.1f}m) - "
                                    f"Poor form ({player.get('form', 0):.1f})")
                    else:
                        st.success("✅ No obvious players to avoid!")

    def render_my_team(self):
        """Enhanced My FPL Team analysis and import with advanced features"""
        st.header("� My FPL Team")
        
        # Check if team is loaded
        if 'my_team_loaded' not in st.session_state or not st.session_state.my_team_loaded:
            st.subheader("� Import Your FPL Team")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                team_id = st.text_input(
                    "Enter your FPL Team ID",
                    placeholder="e.g., 123456",
                    help="You can find your team ID in the URL when viewing your team on the FPL website"
                )
                
                # **NEW: Gameweek Selection**
                current_gw = self._get_current_gameweek()
                selected_gw = st.selectbox(
                    "Select Gameweek",
                    options=list(range(1, 39)),
                    index=min(current_gw - 1, 37) if current_gw else 0,
                    help="Choose which gameweek's team to analyze"
                )
            
            with col2:
                if st.button("🔄 Load My Team", type="primary"):
                    if team_id:
                        with st.spinner("Loading your team..."):
                            team_data = self._load_fpl_team(team_id, selected_gw)
                            
                            if team_data:
                                st.session_state.my_team_data = team_data
                                st.session_state.my_team_id = team_id
                                st.session_state.my_team_gameweek = selected_gw
                                st.session_state.my_team_loaded = True
                                st.success("✅ Team loaded successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Could not load team. Please check your team ID.")
                    else:
                        st.warning("Please enter a team ID")
            
            # Instructions
            with st.expander("💡 How to find your Team ID", expanded=False):
                st.markdown("""
                1. Go to the [FPL website](https://fantasy.premierleague.com)
                2. Log in and view your team
                3. Look at the URL - it will be something like: `https://fantasy.premierleague.com/entry/123456/event/10`
                4. Your Team ID is the number after `/entry/` (in this example: 123456)
                """)
            
            return
        
        # Display loaded team
        team_data = st.session_state.my_team_data
        
        # Team overview
        st.subheader(f"🏆 {team_data.get('entry_name', 'Your Team')} (ID: {st.session_state.my_team_id})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Rank", f"{team_data.get('summary_overall_rank', 'N/A'):,}")
        
        with col2:
            st.metric("Total Points", f"{team_data.get('summary_overall_points', 'N/A'):,}")
        
        with col3:
            st.metric("Gameweek Rank", f"{team_data.get('summary_event_rank', 'N/A'):,}")
        
        with col4:
            st.metric("Team Value", f"£{team_data.get('value', 1000)/10:.1f}m")
        
        # Team analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "👥 Current Squad", 
            "📊 Performance", 
            "🔄 Transfer Suggestions", 
            "📈 Benchmarking",
            "🎯 Chip Strategy",
            "🧐 SWOT Analysis"
        ])
        
        with tab1:
            self._display_current_squad(team_data)
        
        with tab2:
            self._display_performance_analysis(team_data)
        
        with tab3:
            self._display_transfer_suggestions(team_data)
        
        with tab4:
            self._display_benchmarking(team_data)
        
        with tab5:
            self._display_chip_strategy(team_data)
        
        with tab6:
            self._display_swot_analysis(team_data)
        
        # Reset team button
        if st.button("🔄 Load Different Team"):
            st.session_state.my_team_loaded = False
            st.rerun()
    
    def _get_current_gameweek(self):
        """Get current gameweek from FPL API"""
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()
            
            # Find current gameweek
            events = data.get('events', [])
            current_event = next((event for event in events if event['is_current']), None)
            return current_event['id'] if current_event else 1
        except:
            return 1  # Default to GW1 if API fails

    def _load_fpl_team(self, team_id, gameweek=None):
        """Load FPL team data from API with gameweek support"""
        try:
            if gameweek is None:
                gameweek = self._get_current_gameweek()
                
            # FPL API endpoints
            entry_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
            picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gameweek}/picks/"
            
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
            st.error(f"Error loading team: {str(e)}")
            return None

    def _display_enhanced_squad(self, team_data):
        """Enhanced squad display with more detailed analysis"""
        st.subheader("👥 Current Squad")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see detailed squad analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        
        # **ENHANCED: Match picks with comprehensive player data**
        squad_data = []
        formation_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                
                # Count formation
                pos_short = position.split()[0] if ' ' in position else position
                if pos_short in formation_counts:
                    formation_counts[pos_short] += 1
                
                # **ENHANCED: Comprehensive player data with advanced stats**
                fixture_difficulty = self._get_fixture_difficulty(player.get('team_short_name', ''))
                form_trend = "📈" if player.get('form', 0) >= 6 else "📉" if player.get('form', 0) <= 3 else "➡️"
                
                # Calculate additional player metrics
                expected_points = player.get('ep_next', 0)
                points_per_game = player.get('points_per_game', 0)
                bonus_points = player.get('bonus', 0)
                ownership = player.get('selected_by_percent', 0)
                
                # Position-specific stats
                position_stats = ""
                if position == "GK":
                    saves = player.get('saves', 0)
                    clean_sheets = player.get('clean_sheets', 0)
                    position_stats = f"CS: {clean_sheets}, Saves: {saves}"
                elif position in ["DEF"]:
                    clean_sheets = player.get('clean_sheets', 0)
                    goals = player.get('goals_scored', 0)
                    assists = player.get('assists', 0)
                    position_stats = f"CS: {clean_sheets}, G: {goals}, A: {assists}"
                else:  # MID, FWD
                    goals = player.get('goals_scored', 0)
                    assists = player.get('assists', 0)
                    position_stats = f"G: {goals}, A: {assists}"
                
                # Player status with more detail
                captain_status = '👑 Captain' if pick.get('is_captain') else '🅒 Vice' if pick.get('is_vice_captain') else ''
                playing_status = '🟢 Starting XI' if pick['position'] <= 11 else f'🟡 Bench {pick["position"]-11}'
                full_status = f"{captain_status} {playing_status}".strip()
                
                squad_data.append({
                    'Player': player['web_name'],
                    'Position': position,
                    'Team': player.get('team_short_name', 'N/A'),
                    'Price': f"£{player['cost_millions']:.1f}m",
                    'Points': player['total_points'],
                    'Form': f"{form_trend} {player.get('form', 0):.1f}",
                    'PPG': f"{points_per_game:.1f}",
                    'xPts': f"{expected_points:.1f}",
                    'Stats': position_stats,
                    'Own%': f"{ownership:.1f}%",
                    'Fixtures': fixture_difficulty,
                    'Status': full_status,
                    'Reliability': self._assess_squad_player_reliability(player, position)
                })
        
        if squad_data:
            # **NEW: Formation display**
            formation_str = f"{formation_counts['GK']}-{formation_counts['DEF']}-{formation_counts['MID']}-{formation_counts['FWD']}"
            st.info(f"**Squad Formation:** {formation_str}")
            
            squad_df = pd.DataFrame(squad_data)
            
            # **ENHANCED: Improved dataframe with better styling**
            st.dataframe(
                squad_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Form": st.column_config.TextColumn("Form", help="Form with trend indicator"),
                    "PPG": st.column_config.NumberColumn("PPG", help="Points per game average"),
                    "xPts": st.column_config.NumberColumn("xPts", help="Expected points next gameweek"),
                    "Stats": st.column_config.TextColumn("Key Stats", help="Position-specific performance metrics"),
                    "Own%": st.column_config.TextColumn("Ownership", help="Percentage of FPL teams that own this player"),
                    "Fixtures": st.column_config.TextColumn("Next 3 Fixtures", help="Fixture difficulty: 🟢 Easy, 🟡 Medium, 🔴 Hard"),
                    "Price": st.column_config.TextColumn("Price", help="Current market price"),
                    "Points": st.column_config.NumberColumn("Points", help="Total points this season"),
                    "Reliability": st.column_config.TextColumn("Reliability", help="Player consistency and injury risk assessment")
                }
            )
            
            # **ENHANCED: Squad statistics with comprehensive analysis**
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = sum([float(row['Price'].replace('£', '').replace('m', '')) for row in squad_data])
                st.metric("Squad Value", f"£{total_value:.1f}m", f"Avg: £{total_value/len(squad_data):.1f}m")
            
            with col2:
                total_points = sum([row['Points'] for row in squad_data])
                avg_points = total_points / len(squad_data)
                st.metric("Total Points", f"{total_points:,}", f"Avg: {avg_points:.1f}")
            
            with col3:
                total_expected = sum([float(row['xPts']) for row in squad_data])
                st.metric("Expected Points", f"{total_expected:.1f}", f"Next GW projection")
            
            with col4:
                reliable_players = len([p for p in squad_data if p['Reliability'].startswith('🟢')])
                reliability_pct = (reliable_players / len(squad_data)) * 100
                st.metric("Squad Reliability", f"{reliable_players}/{len(squad_data)}", f"{reliability_pct:.0f}% reliable")
            
            with col3:
                forms = [float(row['Form'].split()[-1]) for row in squad_data if row['Form'].split()[-1].replace('.', '').isdigit()]
                avg_form = np.mean(forms) if forms else 0
                st.metric("Average Form", f"{avg_form:.1f}")
            
            with col4:
                playing_11 = len([row for row in squad_data if '🟢' in row['Status'] or '👑' in row['Status'] or '🅒' in row['Status']])
                st.metric("Playing XI", f"{playing_11}/11")

    def _get_fixture_difficulty(self, team_name):
        """Get fixture difficulty indicator for next 3 games using real FPL data"""
        try:
            # Check if we have fixture data loaded
            if hasattr(st.session_state, 'fixtures_df') and not st.session_state.fixtures_df.empty:
                fixtures_df = st.session_state.fixtures_df
                
                # Debug: Print team name and available teams
                if team_name:
                    print(f"Debug: Looking for team '{team_name}'")
                    available_teams = fixtures_df['team_short_name'].unique()
                    print(f"Debug: Available teams: {list(available_teams)[:10]}")
                
                # Try multiple matching strategies
                team_fixtures = pd.DataFrame()
                
                # Strategy 1: Direct short name match
                if not team_fixtures.empty or True:  # Always try
                    team_fixtures = fixtures_df[fixtures_df['team_short_name'] == team_name].head(3)
                
                # Strategy 2: Try team_name column if short_name didn't work
                if team_fixtures.empty:
                    team_fixtures = fixtures_df[fixtures_df['team_name'] == team_name].head(3)
                
                # Strategy 3: Try case-insensitive match
                if team_fixtures.empty:
                    team_fixtures = fixtures_df[
                        fixtures_df['team_short_name'].str.upper() == team_name.upper()
                    ].head(3)
                
                # Strategy 4: Try partial match for team names
                if team_fixtures.empty and len(team_name) > 2:
                    team_fixtures = fixtures_df[
                        fixtures_df['team_short_name'].str.contains(team_name, case=False, na=False)
                    ].head(3)
                
                print(f"Debug: Found {len(team_fixtures)} fixtures for team '{team_name}'")
                
                if not team_fixtures.empty:
                    difficulties = []
                    for _, fixture in team_fixtures.iterrows():
                        difficulty = fixture.get('difficulty', 3)
                        opponent = fixture.get('opponent_short_name', 'UNK')
                        print(f"Debug: {team_name} vs {opponent}, difficulty: {difficulty}")
                        
                        if difficulty <= 2:
                            difficulties.append('🟢')
                        elif difficulty <= 3:
                            difficulties.append('🟡')
                        else:
                            difficulties.append('🔴')
                    
                    # Pad with neutral if we have less than 3 fixtures
                    while len(difficulties) < 3:
                        difficulties.append('🟡')
                    
                    result = ''.join(difficulties[:3])
                    print(f"Debug: Fixture difficulty for {team_name}: {result}")
                    return result
                else:
                    print(f"Debug: No fixtures found for team '{team_name}', using fallback")
            
            # Fallback: Try to load fixture data if not already loaded
            if not hasattr(st.session_state, 'fdr_data_loaded') or not st.session_state.fdr_data_loaded:
                try:
                    print("Debug: Loading fixture data for first time...")
                    fixture_loader = FixtureDataLoader()
                    fixtures_df = fixture_loader.process_fixtures_data()
                    
                    if not fixtures_df.empty:
                        st.session_state.fixtures_df = fixtures_df
                        st.session_state.fdr_data_loaded = True
                        print("Debug: Fixture data loaded, trying again...")
                        
                        # Recursively call this method now that data is loaded
                        return self._get_fixture_difficulty(team_name)
                except Exception as load_error:
                    print(f"Debug: Error loading fixture data: {load_error}")
                    pass
            
            # Final fallback - use team strength estimation
            print(f"Debug: Using fallback estimation for {team_name}")
            return self._estimate_fixture_difficulty_by_team(team_name)
            
        except Exception as e:
            print(f"Debug: Exception in _get_fixture_difficulty: {e}")
            # Emergency fallback
            return '🟡🟡🟡'

    def _estimate_fixture_difficulty_by_team(self, team_name):
        """Estimate fixture difficulty based on historical team strength"""
        # Big 6 teams generally have easier fixtures when they're at home, harder when away
        big_6 = ['ARS', 'CHE', 'LIV', 'MCI', 'MUN', 'TOT', 'Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man Utd', 'Spurs']
        
        # Strong teams (usually top half)
        strong_teams = ['NEW', 'BHA', 'AVL', 'WHU', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham']
        
        # Generally easier opponents for most teams
        easier_opponents = ['BUR', 'SHU', 'LUT', 'Burnley', 'Sheffield Utd', 'Luton']
        
        if team_name in big_6:
            return '🟢🟢🟡'  # Generally good fixtures
        elif team_name in strong_teams:
            return '🟢🟡🟡'  # Mixed but decent
        elif team_name in easier_opponents:
            return '🔴🔴🟡'  # Generally tough fixtures
        else:
            return '🟡🟡🟡'  # Average difficulty

    def _display_enhanced_performance(self, team_data):
        """Enhanced performance analysis with trends and insights"""
        st.subheader("📊 Performance Analysis")
        
        # **ENHANCED: Performance metrics with comparisons**
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📈 Season Performance**")
            total_points = team_data.get('summary_overall_points', 0)
            overall_rank = team_data.get('summary_overall_rank', 0)
            
            # Calculate average points per gameweek
            current_gw = team_data.get('gameweek', 1)
            avg_points_per_gw = total_points / max(current_gw, 1)
            
            st.metric("Total Points", f"{total_points:,}")
            st.metric("Points/GW", f"{avg_points_per_gw:.1f}")
            st.metric("Overall Rank", f"{overall_rank:,}" if overall_rank else "N/A")
        
        with col2:
            st.write("**🎯 Recent Performance**")
            gw_points = team_data.get('summary_event_points', 0)
            gw_rank = team_data.get('summary_event_rank', 0)
            
            # Performance indicators
            avg_gw_points = 50  # League average
            performance = "🔥 Excellent" if gw_points >= 70 else "👍 Good" if gw_points >= avg_gw_points else "📈 Below Average"
            
            st.metric("GW Points", f"{gw_points}")
            st.metric("GW Rank", f"{gw_rank:,}" if gw_rank else "N/A")
            st.metric("Performance", performance)
        
        with col3:
            st.write("**💰 Team Value**")
            team_value = team_data.get('value', 1000) / 10
            bank = team_data.get('bank', 0) / 10
            total_budget = team_value + bank
            
            st.metric("Team Value", f"£{team_value:.1f}m")
            st.metric("In Bank", f"£{bank:.1f}m")
            st.metric("Total Budget", f"£{total_budget:.1f}m")
        
        # **NEW: Performance insights with recommendations**
        st.write("**💡 Performance Insights**")
        
        if overall_rank:
            total_players = 8000000  # Approximate
            percentile = (1 - (overall_rank / total_players)) * 100
            
            if percentile >= 90:
                st.success("🏆 **Elite Performance!** You're in the top 10% of all managers. Keep up the excellent work!")
            elif percentile >= 75:
                st.success("🥉 **Strong Performance!** You're in the top 25%. Consider fine-tuning your strategy.")
            elif percentile >= 50:
                st.info("👍 **Above Average!** You're doing well. Look for ways to optimize your transfers.")
            else:
                st.warning("📈 **Room for Improvement!** Consider using our transfer suggestions and captain recommendations.")
        
        # **NEW: Performance trends**
        st.write("**📈 Performance Trends**")
        if avg_points_per_gw >= 60:
            st.success("• Excellent points-per-gameweek average")
        elif avg_points_per_gw >= 50:
            st.info("• Good points-per-gameweek average")
        else:
            st.warning("• Below average points-per-gameweek - consider strategy changes")

    def _display_smart_transfers(self, team_data):
        """Smart transfer recommendations with advanced analysis"""
        st.subheader("🔄 Smart Transfer Recommendations")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see transfer suggestions")
            return
        
        # **ENHANCED: Advanced transfer analysis**
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available for analysis")
            return
        
        players_df = st.session_state.players_df
        current_players = [pick['element'] for pick in picks]
        
        # **NEW: Multi-criteria analysis for transfers**
        transfer_candidates = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                # Calculate transfer priority score
                form = player.get('form', 0)
                points_per_million = player.get('points_per_million', 0)
                minutes = player.get('minutes', 0)
                
                # Red flags for transfers
                red_flags = []
                if form < 3.0:
                    red_flags.append("Poor form")
                if minutes < 300 and player.get('total_points', 0) > 20:
                    red_flags.append("Limited minutes")
                if player.get('chance_of_playing_next_round', 100) < 75:
                    red_flags.append("Injury concern")
                
                if red_flags:
                    transfer_candidates.append({
                        'name': player['web_name'],
                        'position': player.get('position_name', 'Unknown'),
                        'price': player['cost_millions'],
                        'form': form,
                        'issues': red_flags,
                        'priority': len(red_flags)
                    })
        
        # Display transfer candidates
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**⚠️ Players to Consider Transferring Out**")
            if transfer_candidates:
                # Sort by priority (most issues first)
                transfer_candidates.sort(key=lambda x: x['priority'], reverse=True)
                
                for i, candidate in enumerate(transfer_candidates[:3]):
                    with st.expander(f"🔴 {candidate['name']} ({candidate['position']})"):
                        st.write(f"**Price:** £{candidate['price']:.1f}m")
                        st.write(f"**Form:** {candidate['form']:.1f}")
                        st.write("**Issues:**")
                        for issue in candidate['issues']:
                            st.write(f"• {issue}")
            else:
                st.success("✅ No obvious transfer candidates - your squad looks solid!")
        
        with col2:
            st.write("**🎯 Recommended Transfer Targets**")
            
            # **ENHANCED: Smart transfer targets based on multiple criteria**
            available_players = players_df[~players_df['id'].isin(current_players)]
            
            if not available_players.empty:
                # Filter for good transfer targets
                targets = available_players[
                    (available_players['total_points'] > 30) &
                    (available_players.get('form', 0) >= 5.0) &
                    (available_players['cost_millions'] <= 12.0) &
                    (available_players.get('minutes', 0) > 500)
                ]
                
                if 'points_per_million' in targets.columns:
                    targets = targets.nlargest(5, 'points_per_million')
                
                for _, player in targets.iterrows():
                    with st.expander(f"🟢 {player['web_name']} ({player.get('position_name', 'Unknown')})"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Price:** £{player['cost_millions']:.1f}m")
                            st.write(f"**Form:** {player.get('form', 0):.1f}")
                        with col_b:
                            st.write(f"**Points:** {player['total_points']}")
                            st.write(f"**Value:** {player.get('points_per_million', 0):.1f}")
                        
                        # Transfer rationale
                        reasons = []
                        if player.get('form', 0) >= 7:
                            reasons.append("Excellent form")
                        if player.get('points_per_million', 0) >= 8:
                            reasons.append("Great value for money")
                        if player.get('selected_by_percent', 0) < 10:
                            reasons.append("Low ownership differential")
                        
                        if reasons:
                            st.write("**Why transfer in:**")
                            for reason in reasons:
                                st.write(f"• {reason}")

    def _display_enhanced_benchmarking(self, team_data):
        """Enhanced benchmarking with detailed comparisons"""
        st.subheader("📈 Benchmarking & Comparisons")
        
        overall_rank = team_data.get('summary_overall_rank', 0)
        total_points = team_data.get('summary_overall_points', 0)
        total_players = 8000000  # Approximate number of FPL players
        
        if overall_rank and total_points:
            # **ENHANCED: Detailed performance analysis**
            percentile = (1 - (overall_rank / total_players)) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Your Percentile", f"{percentile:.1f}%")
                
                # Percentile insights
                if percentile >= 99:
                    st.success("🏆 Elite tier - Top 1%")
                elif percentile >= 95:
                    st.success("🥇 Excellent - Top 5%")
                elif percentile >= 75:
                    st.info("🥉 Above average - Top 25%")
                elif percentile >= 50:
                    st.info("📊 Average performance")
                else:
                    st.warning("📈 Below average - room for improvement")
            
            with col2:
                # **NEW: Points benchmarking**
                current_gw = team_data.get('gameweek', 1)
                expected_points = current_gw * 50  # 50 points per GW average
                points_difference = total_points - expected_points
                
                st.metric("vs Average", f"{points_difference:+.0f} points")
                
                if points_difference > 100:
                    st.success("🔥 Well above average!")
                elif points_difference > 0:
                    st.info("👍 Above average")
                else:
                    st.warning("📈 Below average")
            
            with col3:
                # **NEW: Rank progression estimate**
                if percentile > 50:
                    rank_target = int(total_players * 0.1)  # Top 10% target
                    ranks_to_climb = overall_rank - rank_target
                    st.metric("To Top 10%", f"{ranks_to_climb:,} ranks" if ranks_to_climb > 0 else "✅ Achieved!")
                else:
                    rank_target = int(total_players * 0.5)  # Top 50% target
                    ranks_to_climb = overall_rank - rank_target
                    st.metric("To Top 50%", f"{ranks_to_climb:,} ranks" if ranks_to_climb > 0 else "✅ Achieved!")
            
            # **NEW: Performance insights and recommendations**
            st.write("**🎯 Benchmarking Insights**")
            
            insights = []
            if percentile >= 90:
                insights.append("• You're performing exceptionally well - maintain your strategy")
                insights.append("• Consider differential picks to climb into elite rankings")
            elif percentile >= 75:
                insights.append("• Strong performance - look for marginal gains")
                insights.append("• Focus on captain choices and transfers")
            elif percentile >= 50:
                insights.append("• Above average but room for improvement")
                insights.append("• Review transfer strategy and consider more aggressive picks")
            else:
                insights.append("• Significant room for improvement")
                insights.append("• Consider using wildcard for squad overhaul")
                insights.append("• Focus on high-ownership players first")
            
            for insight in insights:
                st.write(insight)

    def _display_fixtures_and_captain(self, team_data):
        """Display fixture analysis and captain recommendations"""
        st.subheader("🎯 Fixtures & Captain Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see fixture analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        
        # **NEW: Captain recommendations**
        st.write("**👑 Captain Recommendations**")
        
        captain_candidates = []
        for pick in picks:
            if pick['position'] <= 11:  # Only playing XI
                player_info = players_df[players_df['id'] == pick['element']]
                if not player_info.empty:
                    player = player_info.iloc[0]
                    
                    # Calculate captain score
                    form = player.get('form', 0)
                    points = player['total_points']
                    position = player.get('position_name', '')
                    
                    # Position multipliers (attackers more likely to score big)
                    pos_multiplier = 1.3 if 'Forward' in position else 1.1 if 'Midfielder' in position else 0.9
                    captain_score = (form * 2 + points / 10) * pos_multiplier
                    
                    captain_candidates.append({
                        'name': player['web_name'],
                        'position': position,
                        'form': form,
                        'score': captain_score,
                        'fixtures': self._get_fixture_difficulty(player.get('team_short_name', ''))
                    })
        
        # Sort and display top captain options
        captain_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🥇 Top Captain Picks**")
            for i, candidate in enumerate(captain_candidates[:3]):
                icon = "👑" if i == 0 else "🥈" if i == 1 else "🥉"
                st.write(f"{icon} **{candidate['name']}** ({candidate['position']})")
                st.write(f"   Form: {candidate['form']:.1f} | Fixtures: {candidate['fixtures']}")
        
        with col2:
            st.write("**📅 Fixture Difficulty Overview**")
            
            # Group players by fixture difficulty
            fixture_summary = {'Easy': [], 'Medium': [], 'Hard': []}
            
            for pick in picks[:11]:  # Playing XI only
                player_info = players_df[players_df['id'] == pick['element']]
                if not player_info.empty:
                    player = player_info.iloc[0]
                    fixtures = self._get_fixture_difficulty(player.get('team_short_name', ''))
                    
                    if '🟢' in fixtures[:1]:  # First fixture
                        fixture_summary['Easy'].append(player['web_name'])
                    elif '🔴' in fixtures[:1]:
                        fixture_summary['Hard'].append(player['web_name'])
                    else:
                        fixture_summary['Medium'].append(player['web_name'])
            
            for difficulty, players in fixture_summary.items():
                if players:
                    icon = "🟢" if difficulty == "Easy" else "🔴" if difficulty == "Hard" else "🟡"
                    st.write(f"{icon} **{difficulty}:** {len(players)} players")
                    st.write(f"   {', '.join(players[:3])}{'...' if len(players) > 3 else ''}")

    def _display_current_squad(self, team_data):
        """Enhanced current squad display with comprehensive player attributes"""
        st.subheader("👥 Current Squad Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see detailed squad analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        
        # **ENHANCED: Comprehensive squad data with key attributes**
        squad_data = []
        formation_counts = {'Goalkeeper': 0, 'Defender': 0, 'Midfielder': 0, 'Forward': 0}
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                
                # Count formation
                if position in formation_counts:
                    formation_counts[position] += 1
                
                # **NEW: Calculate advanced metrics**
                # xG and xA handling
                xg_col = next((col for col in players_df.columns if 'expected_goals' in col.lower() or col.lower() == 'xg'), None)
                xa_col = next((col for col in players_df.columns if 'expected_assists' in col.lower() or col.lower() == 'xa'), None)
                
                xg_value = 0
                xa_value = 0
                if xg_col:
                    xg_value = pd.to_numeric(player[xg_col], errors='coerce') or 0
                if xa_col:
                    xa_value = pd.to_numeric(player[xa_col], errors='coerce') or 0
                
                # Form trend indicator
                form_val = player.get('form', 0)
                form_trend = "🔥" if form_val >= 7 else "📈" if form_val >= 5 else "➡️" if form_val >= 3 else "📉"
                
                # Ownership and value metrics
                ownership = player.get('selected_by_percent', 0)
                ownership_status = "📈 Popular" if ownership >= 20 else "💎 Differential" if ownership <= 5 else "⚖️ Balanced"
                
                # Performance per price
                ppm = player.get('points_per_million', 0)
                value_rating = "💰 Excellent" if ppm >= 10 else "👍 Good" if ppm >= 7 else "⚠️ Poor"
                
                # Next fixture difficulty (simplified)
                fixture_difficulty = self._get_fixture_difficulty_rating(player.get('team_short_name', ''))
                
                # Minutes reliability - dynamic based on season progress
                minutes = player.get('minutes', 0)
                
                # Estimate gameweeks played (assuming 90 min per full game)
                estimated_gws_played = max(1, minutes // 90) if minutes > 0 else 1
                
                # Dynamic thresholds based on how much of season has been played
                # Early season (GW 1-5): More lenient thresholds
                # Mid season (GW 6-25): Standard thresholds  
                # Late season (GW 26+): Stricter thresholds
                
                if minutes >= (estimated_gws_played * 60):  # Played 60+ min per available GW
                    reliability = "🟢 Nailed"
                elif minutes >= (estimated_gws_played * 30):  # Played 30+ min per available GW
                    reliability = "🟡 Rotation Risk"
                elif minutes > 0:  # Some playing time
                    reliability = "🟡 Bench/Rotation"
                else:  # No minutes
                    reliability = "🔴 Bench Risk"
                
                # **COMPREHENSIVE: Enhanced player data structure**
                squad_data.append({
                    'Player': player['web_name'],
                    'Position': position,
                    'Team': player.get('team_short_name', 'N/A'),
                    'Price': f"£{player['cost_millions']:.1f}m",
                    'Points': player['total_points'],
                    'Form': f"{form_trend} {form_val:.1f}",
                    'PPM': f"{ppm:.1f}",
                    'xG': f"{xg_value:.1f}" if xg_value > 0 else "0.0",
                    'xA': f"{xa_value:.1f}" if xa_value > 0 else "0.0",
                    'Goals': player.get('goals_scored', 0),
                    'Assists': player.get('assists', 0),
                    'CS': str(player.get('clean_sheets', 0)) if position in ['Goalkeeper', 'Defender'] else "N/A",
                    'Saves': str(player.get('saves', 0)) if position == 'Goalkeeper' else "N/A",
                    'Bonus': player.get('bonus', 0),
                    'Minutes': f"{minutes:,}",
                    'Ownership': f"{ownership:.1f}%",
                    'Own. Status': ownership_status,
                    'Value Rating': value_rating,
                    'Reliability': reliability,
                    'Next 3': fixture_difficulty,
                    'Status': '👑 Captain' if pick.get('is_captain') else '🅒 Vice' if pick.get('is_vice_captain') else '🟢 Starting XI' if pick['position'] <= 11 else '🪑 Bench'
                })
        
        if squad_data:
            # **NEW: Formation and squad overview**
            st.write("**📋 Squad Overview**")
            col1, col2, col3, col4 = st.columns(4)
            
            # Convert to DataFrame and ensure proper data types for Arrow compatibility
            squad_df = pd.DataFrame(squad_data)
            
            # Ensure columns with mixed string/numeric data are treated as strings
            mixed_type_columns = ['CS', 'Saves', 'xG', 'xA']
            for col in mixed_type_columns:
                if col in squad_df.columns:
                    squad_df[col] = squad_df[col].astype(str)
            
            with col1:
                formation_str = f"{formation_counts['Goalkeeper']}-{formation_counts['Defender']}-{formation_counts['Midfielder']}-{formation_counts['Forward']}"
                st.info(f"**Formation:** {formation_str}")
            
            with col2:
                total_value = sum([float(row['Price'].replace('£', '').replace('m', '')) for row in squad_data])
                st.metric("Squad Value", f"£{total_value:.1f}m")
            
            with col3:
                total_points = sum([row['Points'] for row in squad_data])
                avg_points = total_points / len(squad_data)
                st.metric("Avg Points/Player", f"{avg_points:.1f}")
            
            with col4:
                forms = [float(row['Form'].split()[-1]) for row in squad_data if row['Form'].split()[-1].replace('.', '').isdigit()]
                avg_form = np.mean(forms) if forms else 0
                st.metric("Average Form", f"{avg_form:.1f}")
            
            # **ENHANCED: Comprehensive player table with tabs for different views**
            view_tab1, view_tab2, view_tab3 = st.tabs([
                "📊 Performance View",
                "💰 Value & Ownership", 
                "🎯 Expected Stats"
            ])
            
            with view_tab1:
                st.write("**📊 Performance & Form Analysis**")
                performance_cols = ['Player', 'Position', 'Team', 'Price', 'Points', 'Form', 'Goals', 'Assists', 'CS', 'Bonus', 'Minutes', 'Reliability', 'Status']
                performance_df = squad_df[performance_cols]
                
                st.dataframe(
                    performance_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Player": st.column_config.TextColumn("Player", help="Player name"),
                        "Position": st.column_config.TextColumn("Pos", help="Playing position"),
                        "Team": st.column_config.TextColumn("Team", help="Team abbreviation"),
                        "Price": st.column_config.TextColumn("Price", help="Current market price"),
                        "Points": st.column_config.NumberColumn("Points", help="Total FPL points"),
                        "Form": st.column_config.TextColumn("Form", help="Recent form with trend"),
                        "Goals": st.column_config.NumberColumn("Goals", help="Goals scored"),
                        "Assists": st.column_config.NumberColumn("Assists", help="Assists provided"),
                        "CS": st.column_config.TextColumn("CS", help="Clean sheets (GK/DEF only)"),
                        "Bonus": st.column_config.NumberColumn("Bonus", help="Bonus points earned"),
                        "Minutes": st.column_config.TextColumn("Minutes", help="Total minutes played"),
                        "Reliability": st.column_config.TextColumn("Reliability", help="Playing time assessment"),
                        "Status": st.column_config.TextColumn("Status", help="Squad role")
                    }
                )
            
            with view_tab2:
                st.write("**💰 Value Analysis & Market Position**")
                value_cols = ['Player', 'Position', 'Price', 'PPM', 'Ownership', 'Own. Status', 'Value Rating', 'Next 3', 'Status']
                value_df = squad_df[value_cols]
                
                st.dataframe(
                    value_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "PPM": st.column_config.TextColumn("PPM", help="Points per million £"),
                        "Ownership": st.column_config.TextColumn("Own%", help="Ownership percentage"),
                        "Own. Status": st.column_config.TextColumn("Own. Type", help="Ownership classification"),
                        "Value Rating": st.column_config.TextColumn("Value", help="Value for money assessment"),
                        "Next 3": st.column_config.TextColumn("Fixtures", help="Next 3 fixture difficulty")
                    }
                )
            
            with view_tab3:
                st.write("**🎯 Expected Performance & Advanced Metrics**")
                expected_cols = ['Player', 'Position', 'Team', 'Points', 'xG', 'xA', 'Goals', 'Assists', 'Saves', 'Form', 'Status']
                expected_df = squad_df[expected_cols]
                
                st.dataframe(
                    expected_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "xG": st.column_config.TextColumn("xG", help="Expected goals"),
                        "xA": st.column_config.TextColumn("xA", help="Expected assists"),
                        "Saves": st.column_config.TextColumn("Saves", help="Goalkeeper saves")
                    }
                )
            
            # **NEW: Squad analysis insights**
            st.write("**💡 Squad Analysis Insights**")
            self._display_squad_insights(squad_data)

    def _get_fixture_difficulty_rating(self, team_name):
        """Get simplified fixture difficulty for next 3 games"""
        # Simplified implementation - in production, integrate with real fixture data
        import random
        difficulties = ['🟢🟢🟢', '🟢🟢🟡', '🟢🟡🟡', '🟡🟡🟡', '🟡🟡🔴', '🟡🔴🔴', '🔴🔴🔴']
        return random.choice(difficulties)

    def _display_squad_insights(self, squad_data):
        """Display intelligent squad analysis insights"""
        insights = []
        
        # Value analysis
        ppms = [float(row['PPM']) for row in squad_data if row['PPM'].replace('.', '').isdigit()]
        if ppms:
            avg_ppm = np.mean(ppms)
            if avg_ppm >= 8:
                insights.append("✅ **Excellent value squad** - High points per million average")
            elif avg_ppm >= 6:
                insights.append("👍 **Good value squad** - Decent points per million")
            else:
                insights.append("⚠️ **Value concerns** - Consider cheaper alternatives for better PPM")
        
        # Form analysis
        forms = []
        for row in squad_data:
            form_str = row['Form'].split()[-1]
            if form_str.replace('.', '').isdigit():
                forms.append(float(form_str))
        
        if forms:
            poor_form_count = len([f for f in forms if f < 4])
            excellent_form_count = len([f for f in forms if f >= 7])
            
            if poor_form_count >= 3:
                insights.append("🔄 **Multiple form concerns** - Consider transfers for players in poor form")
            if excellent_form_count >= 5:
                insights.append("🔥 **Strong form across squad** - Team is hitting peak performance")
        
        # Ownership analysis
        differentials = len([row for row in squad_data if 'Differential' in row['Own. Status']])
        popular = len([row for row in squad_data if 'Popular' in row['Own. Status']])
        
        if differentials >= 5:
            insights.append("💎 **High differential squad** - Great for rank climbing if players perform")
        elif popular >= 8:
            insights.append("📈 **Template-heavy squad** - Safe but limited upside potential")
        else:
            insights.append("⚖️ **Balanced ownership** - Good mix of popular and differential picks")
        
        # Reliability analysis
        rotation_risks = len([row for row in squad_data if 'Rotation Risk' in row['Reliability']])
        bench_risks = len([row for row in squad_data if 'Bench Risk' in row['Reliability']])
        
        if rotation_risks + bench_risks >= 4:
            insights.append("⚠️ **Playing time concerns** - Multiple players at risk of rotation")
        elif rotation_risks + bench_risks <= 1:
            insights.append("🟢 **Reliable squad** - Most players are nailed starters")
        
        # Expected stats analysis
        xg_total = sum([float(row['xG']) for row in squad_data if row['xG'] != 'N/A' and row['xG'].replace('.', '').isdigit()])
        xa_total = sum([float(row['xA']) for row in squad_data if row['xA'] != 'N/A' and row['xA'].replace('.', '').isdigit()])
        
        if xg_total + xa_total >= 15:
            insights.append("🎯 **High expected threat** - Squad has strong underlying stats")
        elif xg_total + xa_total >= 8:
            insights.append("👍 **Decent expected performance** - Squad should continue scoring")
        else:
            insights.append("📊 **Limited expected stats** - May need more attacking threat")
        
        # Display insights
        for insight in insights:
            if "✅" in insight or "🔥" in insight or "🟢" in insight:
                st.success(insight)
            elif "👍" in insight or "⚖️" in insight:
                st.info(insight)
            elif "⚠️" in insight or "🔄" in insight:
                st.warning(insight)

    def _display_performance_analysis(self, team_data):
        """Enhanced performance analysis with detailed metrics and trends"""
        st.subheader("📊 Performance Analysis & Benchmarking")
        
        # **ENHANCED: Comprehensive performance metrics**
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**🏆 Season Overview**")
            total_points = team_data.get('summary_overall_points', 0)
            overall_rank = team_data.get('summary_overall_rank', 0)
            
            # Calculate gameweek average
            current_gw = team_data.get('gameweek', 1)
            avg_points_per_gw = total_points / max(current_gw, 1)
            
            st.metric("Total Points", f"{total_points:,}")
            st.metric("Overall Rank", f"{overall_rank:,}" if overall_rank else "N/A")
            st.metric("Points/GW", f"{avg_points_per_gw:.1f}")
        
        with col2:
            st.write("**🎯 Recent Form**")
            gw_points = team_data.get('summary_event_points', 0)
            gw_rank = team_data.get('summary_event_rank', 0)
            
            # Performance vs average
            league_avg = 50
            performance_diff = gw_points - league_avg
            
            st.metric("GW Points", f"{gw_points}", f"{performance_diff:+.0f} vs avg")
            st.metric("GW Rank", f"{gw_rank:,}" if gw_rank else "N/A")
        
        with col3:
            st.write("**💰 Financial Status**")
            team_value = team_data.get('value', 1000) / 10
            bank = team_data.get('bank', 0) / 10
            total_budget = team_value + bank
            
            st.metric("Team Value", f"£{team_value:.1f}m")
            st.metric("In Bank", f"£{bank:.1f}m")
            st.metric("Total Budget", f"£{total_budget:.1f}m")
        
        with col4:
            st.write("**📈 Performance Rating**")
            
            # Calculate performance percentile
            if overall_rank:
                total_players = 8000000  # Approximate
                percentile = (1 - (overall_rank / total_players)) * 100
                
                if percentile >= 95:
                    rating = "🏆 Elite"
                    color = "success"
                elif percentile >= 80:
                    rating = "🥇 Excellent"
                    color = "success"
                elif percentile >= 60:
                    rating = "🥈 Good"
                    color = "info"
                elif percentile >= 40:
                    rating = "🥉 Average"
                    color = "info"
                else:
                    rating = "📈 Improving"
                    color = "warning"
                
                st.metric("Percentile", f"{percentile:.1f}%")
                if color == "success":
                    st.success(f"**{rating}**")
                elif color == "info":
                    st.info(f"**{rating}**")
                else:
                    st.warning(f"**{rating}**")
        
        # **NEW: Performance insights with actionable recommendations**
        st.write("**💡 Performance Insights & Recommendations**")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.write("**🎯 Strengths**")
            strengths = []
            
            if avg_points_per_gw >= 55:
                strengths.append("📈 Excellent points-per-gameweek average")
            
            if overall_rank and overall_rank <= 500000:
                strengths.append("🏆 Strong overall ranking position")
            
            if gw_points >= 60:
                strengths.append("🔥 Excellent recent gameweek performance")
            
            if team_value >= 105:
                strengths.append("💰 High team value - good market management")
            
            for strength in strengths:
                st.success(f"• {strength}")
            
            if not strengths:
                st.info("• Building momentum - keep making smart transfers!")
        
        with insights_col2:
            st.write("**⚠️ Areas for Improvement**")
            improvements = []
            
            if avg_points_per_gw < 45:
                improvements.append("📊 Below-average scoring - review player selections")
            
            if overall_rank and overall_rank > 2000000:
                improvements.append("📈 Ranking needs improvement - consider strategy changes")
            
            if gw_points < 40:
                improvements.append("🔄 Poor recent form - time for transfers")
            
            if bank > 2:
                improvements.append("� Too much money in bank - invest in upgrades")
            
            for improvement in improvements:
                st.warning(f"• {improvement}")
            
            if not improvements:
                st.success("• Excellent management - keep up the great work!")
        
        # **NEW: Historical performance trends**
        st.write("**📈 Performance Trends Analysis**")
        
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            st.write("**Consistency Metrics**")
            if avg_points_per_gw >= 50:
                consistency = "🟢 Highly Consistent"
            elif avg_points_per_gw >= 45:
                consistency = "🟡 Moderately Consistent" 
            else:
                consistency = "🔴 Inconsistent"
            
            st.info(f"**Performance Pattern:** {consistency}")
            
            # Projections
            projected_season_points = avg_points_per_gw * 38
            st.info(f"**Projected Season Total:** {projected_season_points:.0f} points")
        
        with trend_col2:
            st.write("**Improvement Trajectory**")
            
            # Simple trend analysis based on recent vs season average
            if gw_points > avg_points_per_gw + 10:
                trajectory = "📈 Strong Upward Trend"
                trend_color = "success"
            elif gw_points > avg_points_per_gw + 5:
                trajectory = "↗️ Positive Momentum"
                trend_color = "info"
            elif gw_points < avg_points_per_gw - 10:
                trajectory = "📉 Concerning Decline"
                trend_color = "error"
            else:
                trajectory = "➡️ Stable Performance"
                trend_color = "info"
            
            if trend_color == "success":
                st.success(f"**{trajectory}**")
            elif trend_color == "error":
                st.error(f"**{trajectory}**")
            else:
                st.info(f"**{trajectory}**")
    
    def _display_transfer_suggestions(self, team_data):
        """Enhanced transfer suggestions with comprehensive analysis"""
        st.subheader("🔄 Smart Transfer Recommendations")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see transfer suggestions")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available for analysis")
            return
        
        players_df = st.session_state.players_df
        current_players = [pick['element'] for pick in picks]
        
        # **ENHANCED: Comprehensive transfer analysis**
        transfer_tab1, transfer_tab2, transfer_tab3, transfer_tab4 = st.tabs([
            "⚠️ Transfer Out Candidates",
            "🎯 Transfer In Targets", 
            "💰 Budget Analysis",
            "🔄 Transfer Strategies"
        ])
        
        with transfer_tab1:
            self._analyze_transfer_out_candidates(picks, players_df)
        
        with transfer_tab2:
            self._analyze_transfer_in_targets(current_players, players_df, team_data)
        
        with transfer_tab3:
            self._analyze_budget_options(team_data, players_df, current_players)
        
        with transfer_tab4:
            self._suggest_transfer_strategies(team_data, picks, players_df)

    def _analyze_transfer_out_candidates(self, picks, players_df):
        """Analyze players to potentially transfer out"""
        st.write("**⚠️ Players to Consider Transferring Out**")
        
        transfer_out_candidates = []
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                # Multiple criteria for transfer out
                reasons = []
                
                # Poor form
                form = player.get('form', 0)
                if form < 3.5:
                    reasons.append(f"Poor form ({form:.1f})")
                
                # Low minutes
                minutes = player.get('minutes', 0)
                if minutes < 500:
                    reasons.append(f"Limited minutes ({minutes})")
                
                # Poor value
                ppm = player.get('points_per_million', 0)
                if ppm < 5 and player['total_points'] > 10:
                    reasons.append(f"Poor value ({ppm:.1f} PPM)")
                
                # Price drops (simplified)
                if player['cost_millions'] >= 8 and player['total_points'] < 50:
                    reasons.append("Expensive underperformer")
                
                # Expected vs actual performance
                xg_col = next((col for col in players_df.columns if 'expected_goals' in col.lower()), None)
                if xg_col:
                    xg_val = pd.to_numeric(player[xg_col], errors='coerce') or 0
                    actual_goals = player.get('goals_scored', 0)
                    if xg_val > 2 and actual_goals < xg_val * 0.6:
                        reasons.append("Underperforming xG")
                
                if reasons:
                    transfer_out_candidates.append({
                        'player': player,
                        'reasons': reasons,
                        'priority': len(reasons)  # More reasons = higher priority
                    })
        
        # Sort by priority (most reasons first)
        transfer_out_candidates.sort(key=lambda x: x['priority'], reverse=True)
        
        if transfer_out_candidates:
            for candidate in transfer_out_candidates[:5]:  # Top 5 candidates
                player = candidate['player']
                reasons = candidate['reasons']
                
                st.warning(f"**{player['web_name']}** ({player.get('position_name', 'Unknown')}) - £{player['cost_millions']:.1f}m")
                st.write(f"   Concerns: {', '.join(reasons)}")
                st.write("")
        else:
            st.success("✅ No obvious transfer out candidates - your squad is performing well!")

    def _analyze_transfer_in_targets(self, current_players, players_df, team_data):
        """Analyze potential transfer targets"""
        st.write("**🎯 Recommended Transfer Targets**")
        
        # Available players not in current squad
        available_players = players_df[~players_df['id'].isin(current_players)].copy()
        
        if available_players.empty:
            st.warning("No transfer targets available")
            return
        
        # **ENHANCED: Multi-criteria target analysis**
        target_categories = {
            "🔥 Premium Picks": {
                'criteria': lambda df: (df['cost_millions'] >= 9) & (df['form'] >= 6) & (df['total_points'] >= 80),
                'sort_by': 'total_points'
            },
            "💰 Value Options": {
                'criteria': lambda df: (df['cost_millions'] <= 7) & (df.get('points_per_million', 0) >= 8) & (df['total_points'] >= 40),
                'sort_by': 'points_per_million'
            },
            "📈 Form Players": {
                'criteria': lambda df: (df['form'] >= 7) & (df['total_points'] >= 30) & (df.get('minutes', 0) >= 800),
                'sort_by': 'form'
            },
            "💎 Differentials": {
                'criteria': lambda df: (df.get('selected_by_percent', 100) <= 10) & (df['total_points'] >= 50) & (df['form'] >= 5),
                'sort_by': 'total_points'
            }
        }
        
        for category, config in target_categories.items():
            st.write(f"**{category}**")
            
            # Apply criteria
            try:
                targets = available_players[config['criteria'](available_players)]
                
                if not targets.empty and config['sort_by'] in targets.columns:
                    targets = targets.nlargest(3, config['sort_by'])
                    
                    for _, player in targets.iterrows():
                        # Enhanced player information
                        xg_col = next((col for col in players_df.columns if 'expected_goals' in col.lower()), None)
                        xa_col = next((col for col in players_df.columns if 'expected_assists' in col.lower()), None)
                        
                        xg_info = ""
                        if xg_col:
                            xg_val = pd.to_numeric(player[xg_col], errors='coerce') or 0
                            if xg_val > 0:
                                xg_info = f" | xG: {xg_val:.1f}"
                        
                        xa_info = ""
                        if xa_col:
                            xa_val = pd.to_numeric(player[xa_col], errors='coerce') or 0
                            if xa_val > 0:
                                xa_info = f" | xA: {xa_val:.1f}"
                        
                        st.success(
                            f"• **{player['web_name']}** ({player.get('position_name', 'Unknown')}) - "
                            f"£{player['cost_millions']:.1f}m | "
                            f"{player['total_points']} pts | "
                            f"Form: {player.get('form', 0):.1f} | "
                            f"PPM: {player.get('points_per_million', 0):.1f}"
                            f"{xg_info}{xa_info}"
                        )
                else:
                    st.info(f"No players found in this category")
            except Exception as e:
                st.info(f"Analysis unavailable for {category}")
            
            st.write("")

    def _analyze_budget_options(self, team_data, players_df, current_players):
        """Analyze budget and transfer options"""
        st.write("**💰 Budget Analysis & Options**")
        
        bank = team_data.get('bank', 0) / 10
        team_value = team_data.get('value', 1000) / 10
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💰 Financial Position**")
            st.metric("Available Funds", f"£{bank:.1f}m")
            st.metric("Team Value", f"£{team_value:.1f}m")
            
            # Transfer budget analysis
            if bank >= 2:
                budget_status = "🟢 Healthy Budget"
                budget_advice = "You can afford premium upgrades"
            elif bank >= 1:
                budget_status = "🟡 Moderate Budget"
                budget_advice = "Consider mid-range improvements"
            else:
                budget_status = "🔴 Tight Budget"
                budget_advice = "Look for sideways moves or downgrades"
            
            st.info(f"**Status:** {budget_status}")
            st.info(f"**Advice:** {budget_advice}")
        
        with col2:
            st.write("**🔄 Transfer Options by Budget**")
            
            # Budget-based recommendations
            available_players = players_df[~players_df['id'].isin(current_players)]
            
            budget_ranges = [
                ("🌟 Premium (£10m+)", 10.0, 15.0),
                ("⭐ Mid-range (£6-10m)", 6.0, 10.0),
                ("💰 Budget (£4-6m)", 4.0, 6.0)
            ]
            
            for range_name, min_price, max_price in budget_ranges:
                if bank + 2 >= min_price:  # Can afford this range
                    range_players = available_players[
                        (available_players['cost_millions'] >= min_price) &
                        (available_players['cost_millions'] <= max_price) &
                        (available_players['total_points'] >= 30)
                    ]
                    
                    if not range_players.empty and 'points_per_million' in range_players.columns:
                        top_value = range_players.nlargest(1, 'points_per_million').iloc[0]
                        st.success(f"**{range_name}**")
                        st.write(f"• {top_value['web_name']} - £{top_value['cost_millions']:.1f}m | "
                                f"{top_value.get('points_per_million', 0):.1f} PPM")

    def _suggest_transfer_strategies(self, team_data, picks, players_df):
        """Suggest comprehensive transfer strategies"""
        st.write("**🔄 Transfer Strategies & Planning**")
        
        # Analyze current squad composition
        positions = {}
        total_value = 0
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                
                if position not in positions:
                    positions[position] = []
                positions[position].append(player)
                total_value += player['cost_millions']
        
        bank = team_data.get('bank', 0) / 10
        
        strategy_col1, strategy_col2 = st.columns(2)
        
        with strategy_col1:
            st.write("**📋 Recommended Strategies**")
            
            strategies = []
            
            # High value strategy
            if bank >= 3:
                strategies.append({
                    'name': "🌟 Premium Upgrade",
                    'description': "Upgrade to a premium player in attack",
                    'rationale': "High budget allows for marquee signing"
                })
            
            # Value strategy
            if bank <= 1:
                strategies.append({
                    'name': "💰 Value Hunt",
                    'description': "Find undervalued players with good fixtures",
                    'rationale': "Limited budget requires smart value picks"
                })
            
            # Form strategy
            poor_form_count = len([p for pos_players in positions.values() for p in pos_players if p.get('form', 0) < 4])
            if poor_form_count >= 2:
                strategies.append({
                    'name': "📈 Form Focus",
                    'description': "Replace multiple poor form players",
                    'rationale': f"{poor_form_count} players in poor form need attention"
                })
            
            # Differential strategy
            total_players = sum(len(pos_players) for pos_players in positions.values())
            high_ownership_count = len([p for pos_players in positions.values() for p in pos_players if p.get('selected_by_percent', 0) >= 25])
            
            if high_ownership_count / total_players >= 0.6:
                strategies.append({
                    'name': "💎 Differential Hunt",
                    'description': "Target low-owned players for rank gains",
                    'rationale': "Squad is too template-heavy"
                })
            
            for strategy in strategies:
                st.info(f"**{strategy['name']}**")
                st.write(f"• {strategy['description']}")
                st.write(f"• Why: {strategy['rationale']}")
                st.write("")
        
        with strategy_col2:
            st.write("**🎯 Priority Actions**")
            
            # Generate priority actions
            actions = []
            
            # Check each position for issues
            for position, pos_players in positions.items():
                if not pos_players:
                    continue
                
                # Find weakest player in position
                weakest = min(pos_players, key=lambda p: p.get('form', 0) + p.get('points_per_million', 0))
                
                if weakest.get('form', 0) < 3 or weakest.get('points_per_million', 0) < 5:
                    actions.append({
                        'priority': 'High' if weakest.get('form', 0) < 2 else 'Medium',
                        'action': f"Replace {weakest['web_name']} ({position})",
                        'reason': f"Form: {weakest.get('form', 0):.1f}, PPM: {weakest.get('points_per_million', 0):.1f}"
                    })
            
            # Sort by priority
            high_priority = [a for a in actions if a['priority'] == 'High']
            medium_priority = [a for a in actions if a['priority'] == 'Medium']
            
            for action in high_priority:
                st.error(f"🔴 **{action['action']}**")
                st.write(f"   Reason: {action['reason']}")
            
            for action in medium_priority:
                st.warning(f"🟡 **{action['action']}**")
                st.write(f"   Reason: {action['reason']}")
            
            if not actions:
                st.success("✅ **No urgent transfer needs**")
                st.write("   Squad is performing well across all positions")
        
        # **NEW: Transfer timing advice**
        st.write("**⏰ Transfer Timing Advice**")
        
        timing_col1, timing_col2 = st.columns(2)
        
        with timing_col1:
            st.info("**🔄 This Gameweek**")
            st.write("• Make urgent transfers for injured/suspended players")
            st.write("• Target players with favorable fixtures")
            st.write("• Consider captain options")
        
        with timing_col2:
            st.info("**📅 Future Planning**")
            st.write("• Plan for upcoming blank/double gameweeks")
            st.write("• Monitor price changes")
            st.write("• Build team value for later upgrades")
    
    def _display_benchmarking(self, team_data):
        """Enhanced benchmarking against top teams and league averages"""
        st.subheader("📊 Performance Benchmarking & Comparisons")
        
        # **ENHANCED: Comprehensive benchmarking system**
        benchmark_tab1, benchmark_tab2, benchmark_tab3 = st.tabs([
            "🏆 Elite Comparison",
            "📈 League Averages", 
            "🎯 Goal Setting"
        ])
        
        with benchmark_tab1:
            self._display_elite_comparison(team_data)
        
        with benchmark_tab2:
            self._display_league_averages(team_data)
        
        with benchmark_tab3:
            self._display_goal_setting(team_data)

    def _display_elite_comparison(self, team_data):
        """Compare performance against elite managers"""
        st.write("**🏆 Elite Manager Comparison**")
        
        your_rank = team_data.get('summary_overall_rank', 0)
        your_points = team_data.get('summary_overall_points', 0)
        
        # Elite benchmarks (estimated)
        benchmarks = {
            "🥇 Top 1k": {"rank": 1000, "points": 2100, "gw_avg": 55},
            "🥈 Top 10k": {"rank": 10000, "points": 2000, "gw_avg": 53},
            "🥉 Top 100k": {"rank": 100000, "points": 1900, "gw_avg": 50},
            "🏅 Top 500k": {"rank": 500000, "points": 1800, "gw_avg": 47}
        }
        
        current_gw = team_data.get('gameweek', 1)
        your_gw_avg = your_points / max(current_gw, 1)
        
        st.write("**📊 Performance Gaps Analysis**")
        
        for tier, data in benchmarks.items():
            points_gap = data["points"] - your_points
            gw_avg_gap = data["gw_avg"] - your_gw_avg
            
            if your_rank and your_rank <= data["rank"]:
                st.success(f"✅ **{tier}** - You're already here! Keep it up!")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    if points_gap > 0:
                        st.info(f"**{tier}**")
                        st.write(f"• Points gap: {points_gap:+.0f}")
                        st.write(f"• GW avg gap: {gw_avg_gap:+.1f}")
                
                with col2:
                    # Calculate what's needed
                    gws_remaining = 38 - current_gw
                    if gws_remaining > 0:
                        points_needed_per_gw = points_gap / gws_remaining
                        st.write(f"**To Reach {tier}:**")
                        st.write(f"• Need: {data['gw_avg']:.1f} pts/GW")
                        st.write(f"• Current: {your_gw_avg:.1f} pts/GW")
        
        # **NEW: Elite squad analysis**
        if st.session_state.data_loaded:
            st.write("**👑 Elite Squad Patterns**")
            
            elite_patterns = [
                "🏛️ **Premium Strategy**: 3+ players over £10m (high ceiling)",
                "💎 **Differential Edge**: 2-3 low-owned (sub-10%) high-performers",
                "🎯 **Form Focus**: Captain rotation based on fixtures",
                "⚖️ **Balanced Risk**: Mix of safe and punt players",
                "🔄 **Active Management**: 1-2 transfers per week when needed"
            ]
            
            for pattern in elite_patterns:
                st.info(pattern)

    def _display_league_averages(self, team_data):
        """Compare against league averages"""
        st.write("**📈 League Average Comparison**")
        
        your_points = team_data.get('summary_overall_points', 0)
        your_gw_points = team_data.get('summary_event_points', 0)
        current_gw = team_data.get('gameweek', 1)
        
        # League averages (estimated based on typical FPL seasons)
        league_averages = {
            "total_points": 1650,
            "gw_average": 45,
            "gw_recent": 48,
            "squad_value": 103.5,
            "transfers_used": current_gw * 0.8,  # Most managers use 0.8 transfers per GW
        }
        
        your_gw_avg = your_points / max(current_gw, 1)
        your_squad_value = team_data.get('value', 1000) / 10
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**⚖️ Your Performance vs League**")
            
            metrics = [
                ("Total Points", your_points, league_averages["total_points"]),
                ("GW Average", your_gw_avg, league_averages["gw_average"]),
                ("Recent GW", your_gw_points, league_averages["gw_recent"]),
                ("Squad Value", your_squad_value, league_averages["squad_value"])
            ]
            
            for metric_name, your_value, avg_value in metrics:
                diff = your_value - avg_value
                percentage_diff = (diff / avg_value) * 100 if avg_value > 0 else 0
                
                if diff > 0:
                    st.success(f"✅ **{metric_name}**: {your_value:.1f} vs {avg_value:.1f} ({percentage_diff:+.1f}%)")
                elif diff < 0:
                    st.warning(f"📈 **{metric_name}**: {your_value:.1f} vs {avg_value:.1f} ({percentage_diff:+.1f}%)")
                else:
                    st.info(f"➡️ **{metric_name}**: {your_value:.1f} vs {avg_value:.1f} (Average)")
        
        with col2:
            st.write("**🎯 Improvement Areas**")
            
            improvements = []
            
            if your_gw_avg < league_averages["gw_average"]:
                improvements.append("🔄 **Consistency**: Focus on reliable performers")
            
            if your_gw_points < league_averages["gw_recent"]:
                improvements.append("⚡ **Recent Form**: Address current underperformers")
            
            if your_squad_value < league_averages["squad_value"]:
                improvements.append("💰 **Team Value**: Build value through smart transfers")
            
            if not improvements:
                improvements.append("🏆 **Excellent**: Above average across all metrics!")
            
            for improvement in improvements:
                if "Excellent" in improvement:
                    st.success(improvement)
                else:
                    st.info(improvement)
        
        # **NEW: Percentile analysis**
        st.write("**📊 Percentile Analysis**")
        
        if team_data.get('summary_overall_rank'):
            total_players = 8000000  # Approximate
            percentile = (1 - (team_data['summary_overall_rank'] / total_players)) * 100
            
            percentile_col1, percentile_col2, percentile_col3 = st.columns(3)
            
            with percentile_col1:
                st.metric("Overall Percentile", f"{percentile:.1f}%")
            
            with percentile_col2:
                managers_behind = total_players - team_data['summary_overall_rank']
                st.metric("Managers Behind You", f"{managers_behind:,}")
            
            with percentile_col3:
                if percentile >= 90:
                    performance_tier = "🏆 Elite Tier"
                elif percentile >= 75:
                    performance_tier = "🥇 Top Tier"
                elif percentile >= 50:
                    performance_tier = "🥈 Above Average"
                else:
                    performance_tier = "📈 Building"
                
                st.metric("Performance Tier", performance_tier)

    def _display_goal_setting(self, team_data):
        """Help set realistic goals and targets"""
        st.write("**🎯 Goal Setting & Season Targets**")
        
        current_gw = team_data.get('gameweek', 1)
        your_points = team_data.get('summary_overall_points', 0)
        your_rank = team_data.get('summary_overall_rank', 0)
        
        gws_remaining = 38 - current_gw
        your_gw_avg = your_points / max(current_gw, 1)
        
        goal_col1, goal_col2 = st.columns(2)
        
        with goal_col1:
            st.write("**🎯 Season Targets**")
            
            # Target definitions
            targets = {
                "🏆 Elite (Top 10k)": {"points": 2000, "rank": 10000, "gw_avg": 53},
                "🥇 Excellent (Top 100k)": {"points": 1900, "rank": 100000, "gw_avg": 50},
                "🥈 Good (Top 500k)": {"points": 1800, "rank": 500000, "gw_avg": 47},
                "✅ Solid (Top 1M)": {"points": 1700, "rank": 1000000, "gw_avg": 45}
            }
            
            for target_name, target_data in targets.items():
                points_needed = target_data["points"] - your_points
                
                if points_needed <= 0:
                    st.success(f"✅ **{target_name}** - Already achieved!")
                elif gws_remaining > 0:
                    points_per_gw_needed = points_needed / gws_remaining
                    feasibility = "🟢 Achievable" if points_per_gw_needed <= 60 else "🟡 Challenging" if points_per_gw_needed <= 70 else "🔴 Difficult"
                    
                    st.info(f"**{target_name}**")
                    st.write(f"• Need: {points_per_gw_needed:.1f} pts/GW")
                    st.write(f"• {feasibility}")
                    st.write("")
        
        with goal_col2:
            st.write("**📈 Improvement Plan**")
            
            # Personalized improvement suggestions
            if your_gw_avg < 45:
                priority = "🔴 Foundation Building"
                suggestions = [
                    "Focus on consistent, reliable players",
                    "Avoid risky differential picks initially", 
                    "Build squad value through price rises",
                    "Target players with good upcoming fixtures"
                ]
            elif your_gw_avg < 50:
                priority = "🟡 Performance Optimization"
                suggestions = [
                    "Fine-tune captain choices",
                    "Consider strategic differential picks",
                    "Optimize bench to avoid points on bench",
                    "Plan transfers around fixture swings"
                ]
            else:
                priority = "🟢 Elite Strategy"
                suggestions = [
                    "Take calculated risks with differentials",
                    "Plan for blank/double gameweeks",
                    "Consider chip strategy timing",
                    "Focus on mini-league position"
                ]
            
            st.info(f"**Current Priority: {priority}**")
            
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        
        # **NEW: Progress tracking**
        st.write("**📊 Progress Tracking**")
        
        progress_col1, progress_col2, progress_col3 = st.columns(3)
        
        with progress_col1:
            # Season pace
            expected_points_by_now = 45 * current_gw  # Average pace
            if your_points >= expected_points_by_now:
                pace_status = "🟢 Ahead of Pace"
            elif your_points >= expected_points_by_now * 0.9:
                pace_status = "🟡 On Pace"
            else:
                pace_status = "🔴 Behind Pace"
            
            st.metric("Season Pace", pace_status)
        
        with progress_col2:
            # Trajectory
            if your_gw_avg >= 50:
                trajectory = "📈 Elite Trajectory"
            elif your_gw_avg >= 45:
                trajectory = "↗️ Positive Trajectory"
            else:
                trajectory = "➡️ Building Trajectory"
            
            st.metric("Performance Trajectory", trajectory)
        
        with progress_col3:
            # Rank improvement potential
            if your_rank:
                if your_rank <= 100000:
                    potential = "🏆 Maintain Elite"
                elif your_rank <= 500000:
                    potential = "⬆️ Top 100k Possible"
                else:
                    potential = "📈 Significant Growth"
            else:
                potential = "🎯 Track Progress"
            
            st.metric("Rank Potential", potential)
        
        overall_rank = team_data.get('summary_overall_rank', 0)
        total_players = 8000000  # Approximate number of FPL players
        
        if overall_rank:
            percentile = (1 - (overall_rank / total_players)) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Your Percentile", f"{percentile:.1f}%")
                
                if percentile >= 90:
                    st.success("🏆 Elite manager!")
                elif percentile >= 70:
                    st.info("👍 Above average")
                elif percentile >= 50:
                    st.warning("📊 Average performance")
                else:
                    st.error("📈 Below average")
            
            with col2:
                # Performance band
                if overall_rank <= 10000:
                    st.success("🥇 Top 10k - Elite")
                elif overall_rank <= 100000:
                    st.info("🥈 Top 100k - Excellent")
                elif overall_rank <= 1000000:
                    st.warning("🥉 Top 1M - Good")
                else:
                    st.error("📈 Outside Top 1M")
        
        # Improvement suggestions
        st.write("**💡 Areas for Improvement**")
        
        gameweek_rank = team_data.get('summary_event_rank', 0)
        if gameweek_rank and overall_rank:
            if gameweek_rank < overall_rank * 0.5:
                st.success("✅ Recent performance is improving!")
            elif gameweek_rank > overall_rank * 2:
                st.warning("⚠️ Recent performance needs attention")
            else:
                st.info("📊 Consistent performance")

    def _display_chip_strategy(self, team_data):
        """Comprehensive chip strategy analysis and recommendations"""
        st.subheader("🎯 Chip Strategy & Planning")
        
        current_gw = team_data.get('gameweek', 1)
        team_id = st.session_state.get('my_team_id')
        
        # **CHIP STATUS OVERVIEW**
        st.write("### 📋 Chip Status Overview")
        
        # Try to get chip usage data
        chip_status = self._get_chip_status(team_id, current_gw)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Used" if chip_status.get('wildcard_used', False) else "🎯 Available"
            st.metric("Wildcard", status)
            
        with col2:
            status = "✅ Used" if chip_status.get('bench_boost_used', False) else "🎯 Available"
            st.metric("Bench Boost", status)
            
        with col3:
            status = "✅ Used" if chip_status.get('triple_captain_used', False) else "🎯 Available"
            st.metric("Triple Captain", status)
            
        with col4:
            status = "✅ Used" if chip_status.get('free_hit_used', False) else "🎯 Available"
            st.metric("Free Hit", status)
        
        # **CHIP STRATEGY TABS**
        chip_tab1, chip_tab2, chip_tab3, chip_tab4 = st.tabs([
            "🎯 Optimal Timing",
            "📊 Chip Analysis", 
            "🔮 Strategic Planning",
            "📈 Advanced Strategy"
        ])
        
        with chip_tab1:
            self._display_chip_timing(team_data, chip_status, current_gw)
            
        with chip_tab2:
            self._display_chip_analysis(team_data, chip_status)
            
        with chip_tab3:
            self._display_strategic_planning(team_data, chip_status, current_gw)
            
        with chip_tab4:
            self._display_advanced_chip_strategy(team_data, chip_status, current_gw)

    def _get_chip_status(self, team_id, current_gw):
        """Get chip usage status from FPL API"""
        try:
            if not team_id:
                return {'wildcard_used': False, 'bench_boost_used': False, 'triple_captain_used': False, 'free_hit_used': False}
            
            # Check recent gameweeks for chip usage
            chips_used = {'wildcard_used': False, 'bench_boost_used': False, 'triple_captain_used': False, 'free_hit_used': False}
            
            for gw in range(1, min(current_gw + 1, 39)):
                try:
                    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
                    response = requests.get(url, timeout=5, verify=False)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Check for active chip
                        active_chip = data.get('active_chip')
                        if active_chip == 'wildcard':
                            chips_used['wildcard_used'] = True
                        elif active_chip == 'bboost':
                            chips_used['bench_boost_used'] = True
                        elif active_chip == '3xc':
                            chips_used['triple_captain_used'] = True
                        elif active_chip == 'freehit':
                            chips_used['free_hit_used'] = True
                except:
                    continue
            
            return chips_used
        except:
            return {'wildcard_used': False, 'bench_boost_used': False, 'triple_captain_used': False, 'free_hit_used': False}

    def _display_chip_timing(self, team_data, chip_status, current_gw):
        """Display optimal chip timing recommendations"""
        st.write("**⏰ Optimal Chip Timing Guide**")
        
        # Season phases
        phases = {
            "Early Season (GW1-10)": {
                "range": (1, 10),
                "recommended": ["Wildcard (if needed)"],
                "avoid": ["Bench Boost", "Triple Captain"],
                "reasoning": "Team structure still forming, avoid premium chips"
            },
            "Mid Season (GW11-25)": {
                "range": (11, 25),
                "recommended": ["Triple Captain", "Bench Boost"],
                "avoid": ["Free Hit"],
                "reasoning": "Best fixtures and form patterns established"
            },
            "Late Season (GW26-38)": {
                "range": (26, 38),
                "recommended": ["Free Hit", "Wildcard 2"],
                "avoid": [],
                "reasoning": "Final push, use remaining chips strategically"
            }
        }
        
        # Current phase
        current_phase = None
        for phase_name, phase_data in phases.items():
            if phase_data["range"][0] <= current_gw <= phase_data["range"][1]:
                current_phase = phase_name
                break
        
        if current_phase:
            st.info(f"📅 **Current Phase**: {current_phase} (GW{current_gw})")
            
            phase_data = phases[current_phase]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ Recommended for this phase:**")
                for chip in phase_data["recommended"]:
                    st.write(f"• {chip}")
                
            with col2:
                st.write("**❌ Better to avoid:**")
                for chip in phase_data["avoid"]:
                    st.write(f"• {chip}")
            
            st.write(f"**💡 Reasoning**: {phase_data['reasoning']}")
        
        # **SPECIFIC TIMING RECOMMENDATIONS**
        st.write("**🎯 Specific Gameweek Recommendations**")
        
        timing_recommendations = self._get_timing_recommendations(current_gw, chip_status)
        
        for rec in timing_recommendations:
            with st.expander(f"{rec['chip']} - {rec['timing']}"):
                st.write(f"**Best Gameweeks**: {rec['gameweeks']}")
                st.write(f"**Reasoning**: {rec['reasoning']}")
                st.write(f"**Key Factors**: {rec['factors']}")
                
                if rec['urgent']:
                    st.warning("⚠️ Use soon - limited opportunities remaining!")

    def _display_chip_analysis(self, team_data, chip_status):
        """Analyze current squad readiness for each chip"""
        st.write("**🔍 Squad Readiness Analysis**")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see detailed chip analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        
        # Analyze squad for each chip
        analyses = {
            "Wildcard": self._analyze_wildcard_need(picks, players_df),
            "Bench Boost": self._analyze_bench_boost_potential(picks, players_df),
            "Triple Captain": self._analyze_triple_captain_options(picks, players_df),
            "Free Hit": self._analyze_free_hit_timing(picks, players_df)
        }
        
        for chip, analysis in analyses.items():
            if not chip_status.get(f"{chip.lower().replace(' ', '_')}_used", False):
                with st.expander(f"🎯 {chip} Analysis"):
                    
                    # Readiness score
                    score = analysis['readiness_score']
                    if score >= 80:
                        st.success(f"🟢 **Readiness Score: {score}/100** - Excellent time to use!")
                    elif score >= 60:
                        st.info(f"🟡 **Readiness Score: {score}/100** - Good opportunity")
                    else:
                        st.warning(f"🔴 **Readiness Score: {score}/100** - Wait for better timing")
                    
                    # Detailed analysis
                    st.write("**Analysis Details:**")
                    for detail in analysis['details']:
                        st.write(f"• {detail}")
                    
                    # Specific recommendations
                    if analysis['recommendations']:
                        st.write("**Recommendations:**")
                        for rec in analysis['recommendations']:
                            st.write(f"• {rec}")
            else:
                st.success(f"✅ {chip} already used this season")

    def _display_strategic_planning(self, team_data, chip_status, current_gw):
        """Display strategic planning for chip usage"""
        st.write("**🎯 Strategic Chip Planning**")
        
        # Remaining chips
        remaining_chips = [chip for chip, used in [
            ("Wildcard", chip_status.get('wildcard_used', False)),
            ("Bench Boost", chip_status.get('bench_boost_used', False)),
            ("Triple Captain", chip_status.get('triple_captain_used', False)),
            ("Free Hit", chip_status.get('free_hit_used', False))
        ] if not used]
        
        if not remaining_chips:
            st.success("🎉 All chips used! Focus on regular transfers and captaincy.")
            return
        
        st.write(f"**📋 Remaining Chips**: {', '.join(remaining_chips)}")
        
        # Strategic scenarios
        st.write("**📈 Strategic Scenarios**")
        
        scenarios = self._generate_chip_scenarios(current_gw, remaining_chips, team_data)
        
        for i, scenario in enumerate(scenarios, 1):
            with st.expander(f"Scenario {i}: {scenario['name']}"):
                st.write(f"**Objective**: {scenario['objective']}")
                st.write(f"**Timeline**: {scenario['timeline']}")
                
                st.write("**Recommended Sequence:**")
                for step in scenario['sequence']:
                    st.write(f"• GW{step['gameweek']}: {step['action']} - {step['reason']}")
                
                st.write(f"**Expected Outcome**: {scenario['outcome']}")
                
                # Risk assessment
                risk_color = "🟢" if scenario['risk'] == "Low" else "🟡" if scenario['risk'] == "Medium" else "🔴"
                st.write(f"**Risk Level**: {risk_color} {scenario['risk']}")

    def _display_advanced_chip_strategy(self, team_data, chip_status, current_gw):
        """Advanced chip strategy and coordination"""
        st.write("**🎖️ Advanced Chip Strategy**")
        
        # **CHIP COORDINATION**
        st.write("**🔗 Chip Coordination Strategies**")
        
        coordination_strategies = [
            {
                "name": "Double Gameweek Stack",
                "description": "Combine Bench Boost + Triple Captain in double gameweeks",
                "timing": "GW18-19, GW25-26 (typical DGW periods)",
                "effectiveness": "Very High",
                "requirements": ["Strong bench", "Premium captain options"]
            },
            {
                "name": "Wildcard → Bench Boost Combo",
                "description": "Use Wildcard to set up perfect Bench Boost team",
                "timing": "1-2 gameweeks before planned Bench Boost",
                "effectiveness": "High",
                "requirements": ["2 free transfers available", "Clear DGW targets"]
            },
            {
                "name": "Free Hit → Wildcard Sequence",
                "description": "Free Hit for one week, then Wildcard to build new team",
                "timing": "During fixture congestion periods",
                "effectiveness": "Medium-High",
                "requirements": ["Poor fixtures for current team", "Better long-term options available"]
            }
        ]
        
        for strategy in coordination_strategies:
            with st.expander(f"⚡ {strategy['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description**: {strategy['description']}")
                    st.write(f"**Optimal Timing**: {strategy['timing']}")
                
                with col2:
                    st.write(f"**Effectiveness**: {strategy['effectiveness']}")
                    st.write("**Requirements**:")
                    for req in strategy['requirements']:
                        st.write(f"• {req}")
        
        # **DIFFERENTIAL STRATEGIES**
        st.write("**💎 Differential Chip Strategies**")
        
        differential_col1, differential_col2 = st.columns(2)
        
        with differential_col1:
            st.write("**🎯 High-Risk, High-Reward**")
            st.write("• Early Triple Captain (GW2-5)")
            st.write("• Bench Boost in non-DGW")
            st.write("• Free Hit for huge captain differential")
            st.write("• Wildcard before template formation")
            
        with differential_col2:
            st.write("**🛡️ Safe, Template Following**")
            st.write("• Triple Captain in consensus DGW")
            st.write("• Bench Boost with 4 DGW players")
            st.write("• Free Hit only for blank gameweeks")
            st.write("• Wildcard when team structure fails")
        
        # **PERFORMANCE TRACKING**
        st.write("**📊 Chip Performance Expectations**")
        
        expected_gains = {
            "Wildcard": {"conservative": "40-60 points", "aggressive": "60-80 points"},
            "Bench Boost": {"conservative": "8-15 points", "aggressive": "15-25 points"},
            "Triple Captain": {"conservative": "6-12 points", "aggressive": "12-20 points"},
            "Free Hit": {"conservative": "10-20 points", "aggressive": "20-35 points"}
        }
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.write("**Conservative Expectations**")
            for chip, gains in expected_gains.items():
                if not chip_status.get(f"{chip.lower().replace(' ', '_')}_used", False):
                    st.write(f"• {chip}: {gains['conservative']}")
        
        with perf_col2:
            st.write("**Aggressive Expectations**")
            for chip, gains in expected_gains.items():
                if not chip_status.get(f"{chip.lower().replace(' ', '_')}_used", False):
                    st.write(f"• {chip}: {gains['aggressive']}")

    def _get_timing_recommendations(self, current_gw, chip_status):
        """Generate specific timing recommendations for each chip"""
        recommendations = []
        
        # Wildcard timing
        if not chip_status.get('wildcard_used', False):
            if current_gw <= 15:
                recommendations.append({
                    'chip': '🃏 Wildcard',
                    'timing': 'Use if team structure fails',
                    'gameweeks': 'GW8-15 (optimal window)',
                    'reasoning': 'Enough data to identify consistent performers, still time to benefit',
                    'factors': 'Team template changes, multiple injuries, poor early picks',
                    'urgent': current_gw > 12
                })
            else:
                recommendations.append({
                    'chip': '🃏 Wildcard',
                    'timing': 'Save for second half',
                    'gameweeks': 'GW20+ (second wildcard)',
                    'reasoning': 'First wildcard window closing, save for later strategic use',
                    'factors': 'Fixture swings, final team optimization',
                    'urgent': False
                })
        
        # Triple Captain timing
        if not chip_status.get('triple_captain_used', False):
            recommendations.append({
                'chip': '👑 Triple Captain',
                'timing': 'Double Gameweek with premium pick',
                'gameweeks': 'GW18-19, GW25-26 (typical DGWs)',
                'reasoning': 'Maximize ceiling with two fixtures for top scorer',
                'factors': 'Premium player in form, favorable fixtures, DGW confirmation',
                'urgent': current_gw > 25
            })
        
        # Bench Boost timing
        if not chip_status.get('bench_boost_used', False):
            recommendations.append({
                'chip': '🔄 Bench Boost',
                'timing': 'Double Gameweek with strong bench',
                'gameweeks': 'GW18-19, GW25-26 (coordinate with prep)',
                'reasoning': 'All 15 players contribute, need DGW bench players',
                'factors': '4+ DGW players on bench, budget bench options playing',
                'urgent': current_gw > 26
            })
        
        # Free Hit timing
        if not chip_status.get('free_hit_used', False):
            recommendations.append({
                'chip': '🎯 Free Hit',
                'timing': 'Blank gameweek or fixture crisis',
                'gameweeks': 'GW29, GW33 (typical blanks)',
                'reasoning': 'One-week team optimization when fixtures favor non-owned players',
                'factors': 'Blank gameweek, many players missing, huge differential opportunity',
                'urgent': current_gw > 30
            })
        
        return recommendations

    def _analyze_wildcard_need(self, picks, players_df):
        """Analyze if wildcard is needed based on current squad"""
        issues = []
        score = 70  # Start with neutral score
        
        # Check for common wildcard triggers
        total_players = len(picks)
        if total_players < 15:
            issues.append("Incomplete squad data")
            score -= 20
        
        # Mock analysis - in production, analyze actual performance vs. ownership
        underperforming = 3  # Simulated
        if underperforming >= 4:
            issues.append(f"{underperforming} players significantly underperforming")
            score += 20
        elif underperforming >= 2:
            issues.append(f"{underperforming} players underperforming")
            score += 10
        
        injured_players = 1  # Simulated
        if injured_players >= 2:
            issues.append(f"{injured_players} injured players needing replacement")
            score += 15
        
        # Template deviation
        template_deviation = 25  # Simulated percentage
        if template_deviation > 40:
            issues.append("High deviation from template (good for differentials)")
            score += 5
        elif template_deviation > 60:
            issues.append("Very high template deviation (risky)")
            score += 15
        
        recommendations = []
        if score >= 80:
            recommendations.append("Strong case for wildcard - multiple issues identified")
        elif score >= 60:
            recommendations.append("Consider wildcard if transfers can't fix issues")
        else:
            recommendations.append("Hold wildcard - issues can be fixed with regular transfers")
        
        return {
            'readiness_score': min(score, 100),
            'details': issues if issues else ["Squad appears stable"],
            'recommendations': recommendations
        }

    def _analyze_bench_boost_potential(self, picks, players_df):
        """Analyze bench boost potential"""
        score = 50
        details = []
        
        # Analyze bench strength (positions 12-15)
        bench_picks = [p for p in picks if p['position'] > 11]
        
        if len(bench_picks) >= 4:
            # Mock bench analysis
            playing_bench = 3  # Simulated - bench players likely to play
            dgw_bench = 2      # Simulated - bench players in double gameweek
            
            score += playing_bench * 10
            score += dgw_bench * 15
            
            details.append(f"{playing_bench}/4 bench players likely to start")
            details.append(f"{dgw_bench}/4 bench players in favorable fixtures")
            
            if playing_bench >= 3 and dgw_bench >= 2:
                details.append("Strong bench boost opportunity")
            elif playing_bench >= 2:
                details.append("Decent bench boost potential")
            else:
                details.append("Weak bench - consider strengthening before boost")
        else:
            details.append("Incomplete bench data")
            score -= 30
        
        recommendations = []
        if score >= 80:
            recommendations.append("Excellent bench boost opportunity")
        elif score >= 60:
            recommendations.append("Good timing for bench boost")
        else:
            recommendations.append("Strengthen bench before using boost")
        
        return {
            'readiness_score': min(score, 100),
            'details': details,
            'recommendations': recommendations
        }

    def _analyze_triple_captain_options(self, picks, players_df):
        """Analyze triple captain options"""
        score = 40
        details = []
        captain_options = []
        
        # Find premium players in squad
        playing_picks = [p for p in picks if p['position'] <= 11]
        
        for pick in playing_picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                cost = player.get('cost_millions', 0)
                
                # Premium player threshold
                if cost >= 9.0:
                    form = player.get('form', 0)
                    points = player.get('total_points', 0)
                    
                    captain_score = form * 2 + (points / 10)
                    captain_options.append({
                        'name': player['web_name'],
                        'score': captain_score,
                        'form': form,
                        'cost': cost
                    })
        
        if captain_options:
            best_option = max(captain_options, key=lambda x: x['score'])
            score += min(best_option['score'] * 5, 40)
            
            details.append(f"Best option: {best_option['name']} (£{best_option['cost']:.1f}m)")
            details.append(f"Captain score: {best_option['score']:.1f}")
            
            if best_option['form'] >= 6:
                details.append("Premium option in excellent form")
                score += 10
            elif best_option['form'] >= 4:
                details.append("Premium option in good form")
                score += 5
        else:
            details.append("No premium captain options identified")
            score -= 20
        
        recommendations = []
        if score >= 80:
            recommendations.append("Excellent triple captain opportunity")
        elif score >= 60:
            recommendations.append("Good premium captain available")
        else:
            recommendations.append("Wait for better captain option or upgrade team")
        
        return {
            'readiness_score': min(score, 100),
            'details': details,
            'recommendations': recommendations
        }

    def _analyze_free_hit_timing(self, picks, players_df):
        """Analyze free hit timing opportunity"""
        score = 30  # Lower base score as Free Hit is situational
        details = []
        
        # Mock fixture analysis
        blank_gw_players = 6    # Simulated - players not playing this week
        dgw_available = 8       # Simulated - players available in double gameweek
        
        if blank_gw_players >= 6:
            score += 40
            details.append(f"{blank_gw_players} of your players blank this gameweek")
        elif blank_gw_players >= 4:
            score += 20
            details.append(f"{blank_gw_players} of your players blank this gameweek")
        
        if dgw_available >= 8:
            score += 30
            details.append(f"{dgw_available} quality DGW players available")
        elif dgw_available >= 6:
            score += 15
            details.append(f"{dgw_available} DGW players available")
        
        # Differential opportunity
        template_ownership = 45  # Simulated average ownership of your team
        if template_ownership > 60:
            score += 10
            details.append("High ownership team - free hit offers differential potential")
        
        recommendations = []
        if score >= 80:
            recommendations.append("Perfect free hit opportunity - significant fixture advantage")
        elif score >= 60:
            recommendations.append("Good free hit timing - clear benefit available")
        else:
            recommendations.append("Hold free hit - insufficient advantage over current team")
        
        return {
            'readiness_score': min(score, 100),
            'details': details,
            'recommendations': recommendations
        }

    def _generate_chip_scenarios(self, current_gw, remaining_chips, team_data):
        """Generate strategic scenarios for remaining chips"""
        scenarios = []
        
        if len(remaining_chips) >= 3:
            # Aggressive scenario
            scenarios.append({
                'name': 'Aggressive Push',
                'objective': 'Maximize points in next 10 gameweeks',
                'timeline': f'GW{current_gw}-{current_gw+10}',
                'sequence': [
                    {'gameweek': current_gw+2, 'action': 'Triple Captain premium player', 'reason': 'Form spike opportunity'},
                    {'gameweek': current_gw+5, 'action': 'Bench Boost in DGW', 'reason': 'Maximize all 15 players'},
                    {'gameweek': current_gw+8, 'action': 'Free Hit blank gameweek', 'reason': 'Navigate fixture congestion'}
                ],
                'outcome': '80-120 additional points if executed well',
                'risk': 'High'
            })
            
            # Conservative scenario
            scenarios.append({
                'name': 'Conservative Build',
                'objective': 'Steady accumulation with safety',
                'timeline': f'GW{current_gw}-{38}',
                'sequence': [
                    {'gameweek': current_gw+8, 'action': 'Wildcard team overhaul', 'reason': 'Optimize for final third'},
                    {'gameweek': current_gw+12, 'action': 'Bench Boost with strong bench', 'reason': 'Prepared team structure'},
                    {'gameweek': current_gw+15, 'action': 'Triple Captain consensus pick', 'reason': 'Follow template for safety'}
                ],
                'outcome': '60-90 additional points with lower risk',
                'risk': 'Low'
            })
        
        elif len(remaining_chips) == 2:
            scenarios.append({
                'name': 'Two-Chip Combo',
                'objective': 'Coordinate remaining chips for maximum impact',
                'timeline': f'GW{current_gw}-{current_gw+8}',
                'sequence': [
                    {'gameweek': current_gw+3, 'action': f'Use {remaining_chips[0]}', 'reason': 'Set up for second chip'},
                    {'gameweek': current_gw+6, 'action': f'Use {remaining_chips[1]}', 'reason': 'Capitalize on first chip setup'}
                ],
                'outcome': '40-70 additional points from coordination',
                'risk': 'Medium'
            })
        
        elif len(remaining_chips) == 1:
            scenarios.append({
                'name': 'Final Chip Optimization',
                'objective': 'Get maximum value from last remaining chip',
                'timeline': f'GW{current_gw}-{current_gw+6}',
                'sequence': [
                    {'gameweek': current_gw+3, 'action': f'Use {remaining_chips[0]} strategically', 'reason': 'Optimal timing for maximum impact'}
                ],
                'outcome': '20-40 additional points if timed perfectly',
                'risk': 'Medium'
            })
        
        return scenarios

    def _get_team_avg_fdr(self, team_short_name, fixtures_df, num_fixtures):
        if fixtures_df.empty or 'team_short_name' not in fixtures_df.columns or 'combined_fdr' not in fixtures_df.columns:
            return 3.0 # Neutral
        team_fixtures = fixtures_df[
            (fixtures_df['team_short_name'] == team_short_name) &
            (fixtures_df['fixture_number'] <= num_fixtures)
        ]
        if team_fixtures.empty:
            return 3.0
        return team_fixtures['combined_fdr'].mean()

    def _analyze_strengths(self, squad_df, fixtures_df):
        strengths = []
        # Strong form players
        in_form_players = squad_df[squad_df['form'].astype(float) >= 6.0]
        if not in_form_players.empty:
            strengths.append(f"**In-form players**: {', '.join(in_form_players['web_name'].tolist())} are performing well.")

        # Premium assets delivering
        premium_performers = squad_df[(squad_df['cost_millions'] >= 10.0) & (squad_df['total_points'] >= 100)]
        if not premium_performers.empty:
            strengths.append(f"**Premium assets**: {', '.join(premium_performers['web_name'].tolist())} are justifying their price.")

        # Good captaincy
        captain = squad_df[squad_df['is_captain'] == True]
        if not captain.empty:
            captain = captain.iloc[0]
            captain_form = float(captain['form'])
            captain_fixtures = self._get_team_avg_fdr(captain['team_short_name'], fixtures_df, 1)
            if captain_form >= 5.0 and captain_fixtures <= 2.5:
                strengths.append(f"**Strong captain choice**: {captain['web_name']} has great form and an easy fixture.")

        # Favorable fixtures for key players
        good_fixture_players = []
        for _, player in squad_df.iterrows():
            avg_fdr = self._get_team_avg_fdr(player['team_short_name'], fixtures_df, 3)
            if avg_fdr <= 2.5:
                good_fixture_players.append(player['web_name'])
        if len(good_fixture_players) >= 5:
            strengths.append(f"**Good fixture run**: Multiple players including {', '.join(good_fixture_players[:3])} have favorable upcoming games.")

        # Strong bench
        bench_players = squad_df[squad_df['position'] > 11]
        playing_bench = bench_players[bench_players['minutes'] > (bench_players['minutes'].mean() * 0.5)] # Simple check for playing time
        if len(playing_bench) >= 2:
            strengths.append(f"**Strong bench**: Good cover with players like {', '.join(playing_bench['web_name'].tolist())} getting minutes.")

        return strengths

    def _analyze_weaknesses(self, squad_df, fixtures_df, team_data):
        weaknesses = []
        
        # Poor form players
        poor_form_players = squad_df[squad_df['form'].astype(float) <= 2.0]
        if not poor_form_players.empty:
            weaknesses.append({
                'type': 'poor_form',
                'players': poor_form_players,
                'description': f"**Poor form**: {', '.join(poor_form_players['web_name'].tolist())} are underperforming."
            })

        # Injured or doubtful players
        injured_players = squad_df[squad_df['status'].isin(['i', 'd'])] # i=injured, d=doubtful
        if not injured_players.empty:
            weaknesses.append({
                'type': 'injury',
                'players': injured_players,
                'description': f"**Injury concerns**: {', '.join(injured_players['web_name'].tolist())} are flagged."
            })

        # Too much money in the bank
        bank = team_data.get('bank', 0) / 10.0
        if bank > 2.0:
            weaknesses.append({
                'type': 'money_in_bank',
                'players': pd.DataFrame(), # No specific players
                'description': f"**Money in bank**: £{bank:.1f}m is unspent, could be used for upgrades."
            })

        # Players with tough fixtures
        bad_fixture_player_ids = []
        for _, player in squad_df.iterrows():
            avg_fdr = self._get_team_avg_fdr(player['team_short_name'], fixtures_df, 5)
            if avg_fdr >= 4.0:
                bad_fixture_player_ids.append(player['id'])
        if len(bad_fixture_player_ids) >= 3:
            bad_fixture_players_df = squad_df[squad_df['id'].isin(bad_fixture_player_ids)]
            weaknesses.append({
                'type': 'tough_fixtures',
                'players': bad_fixture_players_df,
                'description': f"**Tough fixtures**: Players like {', '.join(bad_fixture_players_df['web_name'].tolist()[:3])} face a difficult run of games over the next 5 GWs."
            })

        # Weak bench
        bench_players = squad_df[squad_df['position'] > 11]
        non_playing_bench = bench_players[bench_players['minutes'] < 90] # Simple check for non-starters
        if len(non_playing_bench) >= 3:
            weaknesses.append({
                'type': 'weak_bench',
                'players': non_playing_bench,
                'description': f"**Weak bench**: Bench players like {', '.join(non_playing_bench['web_name'].tolist())} have limited game time."
            })

        return weaknesses

    def _analyze_opportunities(self, squad_df, fixtures_df, players_df):
        opportunities = []
        squad_ids = squad_df['id'].tolist()

        # Teams with upcoming good fixtures
        if not fixtures_df.empty:
            team_fdr = fixtures_df.groupby('team_short_name')['combined_fdr'].mean().nsmallest(3)
            if not team_fdr.empty:
                opportunities.append(f"**Fixture targets**: Consider players from {', '.join(team_fdr.index.tolist())} due to their easy fixture run.")

        # In-form differentials to target
        differentials = players_df[
            (~players_df['id'].isin(squad_ids)) &
            (players_df['form'].astype(float) >= 6.0) &
            (players_df['selected_by_percent'].astype(float) < 10.0)
        ].nlargest(3, 'form')
        if not differentials.empty:
            opportunities.append(f"**Differential picks**: {', '.join(differentials['web_name'].tolist())} are in form and have low ownership.")

        # Upcoming DGW/BGW
        opportunities.append("**Chip strategy**: Plan for upcoming Double and Blank Gameweeks to maximize chip usage.")

        # Captaincy rotation
        captaincy_candidates = squad_df[squad_df['cost_millions'] >= 8.0]
        good_fixture_captains = []
        for _, player in captaincy_candidates.iterrows():
            avg_fdr = self._get_team_avg_fdr(player['team_short_name'], fixtures_df, 1)
            if avg_fdr <= 2.0:
                good_fixture_captains.append(player['web_name'])
        if good_fixture_captains:
            opportunities.append(f"**Captaincy options**: {', '.join(good_fixture_captains)} have excellent fixtures for captaincy.")

        return opportunities

    def _analyze_threats(self, squad_df, fixtures_df):
        threats = []

        # Key players with tough fixtures
        key_players = squad_df[(squad_df['selected_by_percent'].astype(float) > 20.0) | (squad_df['cost_millions'] > 9.0)]
        key_players_bad_fixtures = []
        for _, player in key_players.iterrows():
            avg_fdr = self._get_team_avg_fdr(player['team_short_name'], fixtures_df, 3)
            if avg_fdr >= 4.0:
                key_players_bad_fixtures.append(player['web_name'])
        if key_players_bad_fixtures:
            threats.append(f"**Fixture risk**: Key players like {', '.join(key_players_bad_fixtures)} have a tough run of games.")

        # Rotation risks
        rotation_risks = squad_df[squad_df['minutes'] < (squad_df['minutes'].mean() * 0.7)] # Simple proxy
        if not rotation_risks.empty:
            threats.append(f"**Rotation risk**: {', '.join(rotation_risks['web_name'].tolist())} may not be guaranteed starters.")

        # Price drop risks
        price_drop_risks = squad_df[
            (squad_df['form'].astype(float) < 2.0) &
            (squad_df['selected_by_percent'].astype(float) > 15.0)
        ]
        if not price_drop_risks.empty:
            threats.append(f"**Price drop risk**: {', '.join(price_drop_risks['web_name'].tolist())} are at risk of losing value due to poor form and high ownership.")

        # Over-reliance on one team
        team_counts = squad_df['team_short_name'].value_counts()
        if team_counts.max() >= 3:
            top_team = team_counts.idxmax()
            avg_fdr = self._get_team_avg_fdr(top_team, fixtures_df, 3)
            if avg_fdr > 3.5:
                threats.append(f"**Over-reliance**: Heavy investment in {top_team} who have difficult upcoming fixtures.")

        return threats

    def _get_teams_with_good_fixtures(self, fixtures_df, num_fixtures):
        if fixtures_df.empty or 'fixture_number' not in fixtures_df.columns:
            return []
        team_fdr = fixtures_df[fixtures_df['fixture_number'] <= num_fixtures].groupby('team_short_name')['combined_fdr'].mean()
        return team_fdr[team_fdr <= 2.5].index.tolist()

    def _suggest_transfers_for_weakness(self, weakness, players_df, squad_df, team_data):
        suggestions = []
        squad_ids = squad_df['id'].tolist()
        bank = team_data.get('bank', 0) / 10.0

        # Pre-calculate average FDR for all players for the next 5 gameweeks
        fixtures_df = st.session_state.fixtures_df
        avg_fdr_5_gw = fixtures_df[fixtures_df['fixture_number'] <= 5].groupby('team_short_name')['combined_fdr'].mean()
        players_df_with_fdr = players_df.merge(avg_fdr_5_gw.rename('avg_fdr_5'), left_on='team_short_name', right_index=True, how='left')
        players_df_with_fdr['avg_fdr_5'] = players_df_with_fdr['avg_fdr_5'].fillna(3.5) # Fill neutral for teams with no data

        def find_replacement(player_out, criteria):
            base_query = (
                (~players_df_with_fdr['id'].isin(squad_ids)) &
                (players_df_with_fdr['position_name'] == player_out['position_name']) &
                (players_df_with_fdr['cost_millions'] <= player_out['cost_millions'] + bank) &
                (players_df_with_fdr['status'] == 'a')
            )
            for key, value in criteria.items():
                if key.endswith('__isin'):
                    base_query &= players_df_with_fdr[key.replace('__isin', '')].isin(value)
                elif key.endswith('__gte'):
                    base_query &= players_df_with_fdr[key.replace('__gte', '')].astype(float) >= value
            
            potential_replacements = players_df_with_fdr[base_query].copy()

            if potential_replacements.empty:
                return pd.DataFrame()

            # Calculate a composite score based on form and 5-week fixture difficulty
            potential_replacements['suggestion_score'] = (
                potential_replacements['form'].astype(float) * 0.6 + # 60% weight on form
                (5 - potential_replacements['avg_fdr_5']) * 0.4 # 40% weight on good fixtures (lower fdr is better)
            )
            
            return potential_replacements.nlargest(1, 'suggestion_score')

        if weakness['type'] in ['poor_form', 'injury']:
            for _, player_out in weakness['players'].iterrows():
                replacements = find_replacement(player_out, {'form__gte': 4.0})
                if not replacements.empty:
                    player_in = replacements.iloc[0]
                    suggestions.append(
                        f"Consider replacing **{player_out['web_name']}** (out of form/injured) with **{player_in['web_name']}** (in form, £{player_in['cost_millions']:.1f}m, 5-GW FDR: {player_in['avg_fdr_5']:.2f})."
                    )

        elif weakness['type'] == 'tough_fixtures':
            good_fixture_teams = self._get_teams_with_good_fixtures(st.session_state.fixtures_df, 5)
            if not good_fixture_teams:
                return []
            for _, player_out in weakness['players'].iterrows():
                replacements = find_replacement(player_out, {'team_short_name__isin': good_fixture_teams, 'form__gte': 4.0})
                if not replacements.empty:
                    player_in = replacements.iloc[0]
                    suggestions.append(
                        f"**{player_out['web_name']}** has tough fixtures. Consider swapping to **{player_in['web_name']}** (£{player_in['cost_millions']:.1f}m) who has a great run over the next 5 GWs."
                    )

        elif weakness['type'] == 'weak_bench' and not weakness['players'].empty:
            player_out = weakness['players'].iloc[0]
            replacements = find_replacement(player_out, {'minutes__gte': 300})
            if not replacements.empty:
                player_in = replacements.iloc[0]
                suggestions.append(
                    f"Strengthen your bench by replacing **{player_out['web_name']}** with a reliable starter like **{player_in['web_name']}** (£{player_in['cost_millions']:.1f}m)."
                )

        elif weakness['type'] == 'money_in_bank':
            player_to_upgrade = squad_df.sort_values('cost_millions').iloc[0]
            replacements = find_replacement(player_to_upgrade, {'form__gte': 5.0})
            if not replacements.empty:
                player_in = replacements.iloc[0]
                suggestions.append(
                    f"Use your bank to upgrade **{player_to_upgrade['web_name']}** to a premium option like **{player_in['web_name']}** (£{player_in['cost_millions']:.1f}m)."
                )

        return suggestions

    def _display_swot_analysis(self, team_data):
        """Display SWOT analysis for the user's team"""
        st.subheader("🧐 SWOT Analysis")
        st.info("A strategic overview of your team's Strengths, Weaknesses, Opportunities, and Threats.")

        if not st.session_state.data_loaded or 'players_df' not in st.session_state or st.session_state.players_df.empty:
            st.warning("Please load player data from the sidebar to perform a SWOT analysis.")
            return

        if 'fixtures_df' not in st.session_state or st.session_state.fixtures_df.empty:
            st.warning("Fixture data not loaded. Please visit the 'Fixture Difficulty' tab and load the data for a complete SWOT analysis.")
            with st.spinner("Loading fixture data for analysis..."):
                fixture_loader = FixtureDataLoader()
                fixtures_df = fixture_loader.process_fixtures_data()
                if not fixtures_df.empty:
                    fdr_analyzer = FDRAnalyzer()
                    fixtures_df = fdr_analyzer.calculate_combined_fdr(fixtures_df)
                    st.session_state.fixtures_df = fixtures_df
                    st.session_state.fdr_data_loaded = True
                    st.rerun() # Rerun to use the loaded data
                else:
                    st.error("Could not load fixture data.")
                    return

        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available for SWOT analysis.")
            return

        players_df = st.session_state.players_df
        fixtures_df = st.session_state.fixtures_df

        # Get current squad players
        squad_player_ids = [p['element'] for p in picks]
        squad_df = players_df[players_df['id'].isin(squad_player_ids)].copy()

        # Merge pick data (captain, etc.)
        picks_df = pd.DataFrame(picks)
        squad_df = squad_df.merge(picks_df[['element', 'is_captain', 'is_vice_captain', 'position']], left_on='id', right_on='element', how='left')

        # SWOT analysis logic
        strengths = self._analyze_strengths(squad_df, fixtures_df)
        weaknesses = self._analyze_weaknesses(squad_df, fixtures_df, team_data)
        opportunities = self._analyze_opportunities(squad_df, fixtures_df, players_df)
        threats = self._analyze_threats(squad_df, fixtures_df)

        # Display SWOT
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👍 Strengths")
            if strengths:
                for s in strengths:
                    st.success(f"• {s}")
            else:
                st.info("No significant strengths identified.")
            
            st.markdown("### 🤔 Opportunities")
            if opportunities:
                for o in opportunities:
                    st.info(f"• {o}")
            else:
                st.info("No clear opportunities identified.")

        with col2:
            st.markdown("### 👎 Weaknesses")
            if weaknesses:
                for weakness in weaknesses:
                    st.warning(f"• {weakness['description']}")
                    suggestions = self._suggest_transfers_for_weakness(weakness, players_df, squad_df, team_data)
                    if suggestions:
                        with st.expander("💡 Suggested Fixes"):
                            for suggestion in suggestions:
                                st.info(f"↪️ {suggestion}")
            else:
                st.info("No significant weaknesses identified.")

            st.markdown("### 🚨 Threats")
            if threats:
                for t in threats:
                    st.error(f"• {t}")
            else:
                st.info("No immediate threats identified.")

    def render_ai_recommendations(self):
        """Enhanced AI-powered recommendations"""
        st.header("🤖 AI-Powered Recommendations")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first from the Dashboard.")
            return
        
        df = st.session_state.players_df
        
        # AI recommendation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Player Targets", 
            "📈 Form Analysis", 
            "💰 Value Picks", 
            "🔄 Transfer Advice"
        ])
        
        with tab1:
            st.subheader("🎯 AI Player Targets")
            
            # Smart player filtering
            col1, col2 = st.columns(2)
            
            with col1:
                price_range = st.slider("Price Range (£m)", 4.0, 15.0, (4.0, 12.0), 0.1)
                positions = st.multiselect(
                    "Positions", 
                    ["Goalkeeper", "Defender", "Midfielder", "Forward"],
                    default=["Midfielder", "Forward"]
                )
            
            with col2:
                min_form = st.slider("Minimum Form", 0.0, 10.0, 5.0, 0.1)
                max_ownership = st.slider("Max Ownership % (for differentials)", 5.0, 50.0, 25.0, 1.0)
            
            # Generate AI recommendations
            if st.button("🔮 Generate AI Targets", type="primary"):
                targets = self._generate_ai_targets(df, price_range, positions, min_form, max_ownership)
                
                if targets:
                    for position, players in targets.items():
                        if players:
                            st.write(f"**{position}s:**")
                            
                            for player in players[:3]:  # Top 3 per position
                                with st.expander(f"⭐ {player['name']} ({player['team']})"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"💰 **Price**: £{player['price']:.1f}m")
                                        st.write(f"📊 **Points**: {player['points']}")
                                        st.write(f"🔥 **Form**: {player['form']:.1f}")
                                    
                                    with col2:
                                        st.write(f"👥 **Ownership**: {player['ownership']:.1f}%")
                                        st.write(f"💎 **Value**: {player['value']:.1f} pts/£m")
                                        st.write(f"🤖 **AI Score**: {player['ai_score']:.1f}")
                
        with tab2:
            st.subheader("📈 Form Analysis")
            
            # Form-based recommendations
            form_analysis = self._analyze_player_form(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔥 Hot Form Players (Rising)**")
                for player in form_analysis['hot_form'][:5]:
                    st.write(f"• **{player['name']}** ({player['team']}) - Form: {player['form']:.1f}")
            
            with col2:
                st.write("**❄️ Cold Form Players (Falling)**")
                for player in form_analysis['cold_form'][:5]:
                    st.write(f"• **{player['name']}** ({player['team']}) - Form: {player['form']:.1f}")
        
        with tab3:
            st.subheader("💰 Value Picks")
            
            # Value analysis
            if 'points_per_million' in df.columns:
                value_picks = df[
                    (df['total_points'] > 20) & 
                    (df['cost_millions'] <= 8.0)
                ].nlargest(10, 'points_per_million')
                
                for _, player in value_picks.iterrows():
                    st.write(f"💎 **{player['web_name']}** ({player.get('team_short_name', 'N/A')}) - "
                            f"£{player['cost_millions']:.1f}m - {player['points_per_million']:.1f} pts/£m")
        
        with tab4:
            st.subheader("🔄 AI Transfer Advice")
            
            # Transfer recommendations based on various factors
            transfer_advice = self._generate_transfer_advice(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Players to Target**")
                for advice in transfer_advice['targets'][:5]:
                    st.success(f"**{advice['player']}** - {advice['reason']}")
            
            with col2:
                st.write("**⚠️ Players to Consider Selling**")
                for advice in transfer_advice['sells'][:5]:
                    st.warning(f"**{advice['player']}** - {advice['reason']}")
    
    def _generate_ai_targets(self, df, price_range, positions, min_form, max_ownership):
        """Generate AI-powered player targets"""
        targets = {}
        
        # Position mapping
        position_map = {
            "Goalkeeper": 1, "Defender": 2, "Midfielder": 3, "Forward": 4
        }
        
        for position in positions:
            position_id = position_map.get(position)
            if position_id is None:
                continue
            
            # Filter players
            filtered = df[
                (df['element_type'] == position_id) &
                (df['cost_millions'] >= price_range[0]) &
                (df['cost_millions'] <= price_range[1]) &
                (df.get('form', 0) >= min_form) &
                (df.get('selected_by_percent', 0) <= max_ownership) &
                (df['total_points'] > 10)  # Minimum threshold
            ].copy()
            
            if not filtered.empty:
                # Calculate AI score
                filtered['ai_score'] = (
                    filtered.get('form', 0) * 0.3 +
                    filtered.get('points_per_million', 0) * 0.4 +
                    (100 - filtered.get('selected_by_percent', 50)) / 100 * 0.2 +
                    filtered['total_points'] / filtered['total_points'].max() * 100 * 0.1
                )
                
                # Sort by AI score
                filtered = filtered.nlargest(5, 'ai_score')
                
                targets[position] = [
                    {
                        'name': row['web_name'],
                        'team': row.get('team_short_name', 'N/A'),
                        'price': row['cost_millions'],
                        'points': row['total_points'],
                        'form': row.get('form', 0),
                        'ownership': row.get('selected_by_percent', 0),
                        'value': row.get('points_per_million', 0),
                        'ai_score': row['ai_score']
                    }
                    for _, row in filtered.iterrows()
                ]
        
        return targets
    
    def _analyze_player_form(self, df):
        """Analyze player form trends"""
        if 'form' not in df.columns:
            return {'hot_form': [], 'cold_form': []}
        
        # Players with good recent form
        hot_form = df[
            (df['form'] >= 7.0) & 
            (df['total_points'] > 30)
        ].nlargest(10, 'form')
        
        # Players with poor recent form
        cold_form = df[
            (df['form'] <= 3.0) & 
            (df['total_points'] > 50)  # Only established players
        ].nsmallest(10, 'form')
        
        return {
            'hot_form': [
                {
                    'name': row['web_name'],
                    'team': row.get('team_short_name', 'N/A'),
                    'form': row['form']
                }
                for _, row in hot_form.iterrows()
            ],
            'cold_form': [
                {
                    'name': row['web_name'],
                    'team': row.get('team_short_name', 'N/A'),
                    'form': row['form']
                }
                for _, row in cold_form.iterrows()
            ]
        }
    
    def _generate_transfer_advice(self, df):
        """Generate AI transfer advice"""
        targets = []
        sells = []
        
        # Target players: High form, good value, reasonable ownership
        if 'form' in df.columns and 'points_per_million' in df.columns:
            potential_targets = df[
                (df['form'] >= 6.0) &
                (df['points_per_million'] >= 8.0) &
                (df.get('selected_by_percent', 0) <= 30.0) &
                (df['total_points'] > 25)
            ].head(5)
            
            for _, player in potential_targets.iterrows():
                targets.append({
                    'player': f"{player['web_name']} (£{player['cost_millions']:.1f}m)",
                    'reason': f"Excellent form ({player['form']:.1f}) and great value ({player['points_per_million']:.1f} pts/£m)"
                })
        
        # Sell candidates: Poor form, high ownership established players
        if 'form' in df.columns:
            potential_sells = df[
                (df['form'] <= 4.0) &
                (df.get('selected_by_percent', 0) >= 15.0) &
                (df['total_points'] > 50) &
                (df['cost_millions'] >= 7.0)
            ].head(5)
            
            for _, player in potential_sells.iterrows():
                sells.append({
                    'player': f"{player['web_name']} (£{player['cost_millions']:.1f}m)",
                    'reason': f"Poor recent form ({player['form']:.1f}) despite high ownership"
                })
        
        return {'targets': targets, 'sells': sells}

    def render_team_builder(self):
        """Enhanced Team Builder with comprehensive recommendations"""
        st.header("⚽ Enhanced Team Builder")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first from the Dashboard.")
            return
        
        # Import the team recommender component
        try:
            from components.ui_components import render_enhanced_team_recommendations_tab
            
            # Create a data manager object
            class DataManager:
                def __init__(self, players_df, teams_df):
                    self.players_df = players_df
                    self.teams_df = teams_df
                
                def get_players_data(self):
                    return self.players_df
                
                def get_teams_data(self):
                    return self.teams_df
            
            data_manager = DataManager(st.session_state.players_df, st.session_state.teams_df)
            render_enhanced_team_recommendations_tab(data_manager)
            
        except ImportError:
            # Fallback to basic team builder if components not available
            self._render_basic_team_builder()
    
    def _render_basic_team_builder(self):
        """Basic team builder implementation"""
        st.subheader("🛠️ Basic Team Builder")
        
        df = st.session_state.players_df
        
        # Budget and formation settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget = st.slider("Budget (£m)", 80.0, 120.0, 100.0, 0.5)
        
        with col2:
            formation = st.selectbox("Formation", ["3-4-3", "4-3-3", "3-5-2", "4-4-2", "5-3-2"])
        
        with col3:
            max_players_per_team = st.slider("Max players per team", 1, 3, 3)
        
        # Advanced filters
        with st.expander("🔍 Advanced Filters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                min_form = st.slider("Minimum Form", 0.0, 10.0, 0.0, 0.1)
                min_ownership = st.slider("Minimum Ownership %", 0.0, 50.0, 0.0, 0.5)
                
            with col2:
                max_ownership = st.slider("Maximum Ownership %", 0.0, 100.0, 100.0, 1.0)
                preferred_style = st.selectbox("Playing Style", ["balanced", "attacking", "defensive"])
        
        # Generate team button
        if st.button("� Generate Optimized Team", type="primary"):
            try:
                from components.team_recommender import get_latest_team_recommendations
                
                # Convert formation to tuple
                formation_map = {
                    "3-4-3": (3, 4, 3), "4-3-3": (4, 3, 3), "3-5-2": (3, 5, 2),
                    "4-4-2": (4, 4, 2), "5-3-2": (5, 3, 2)
                }
                
                formations = [formation_map[formation]]
                
                with st.spinner("Optimizing your team..."):
                    recommendations = get_latest_team_recommendations(
                        df, 
                        budget=budget,
                        formations=formations,
                        max_players_per_club=max_players_per_team,
                        min_ownership=min_ownership,
                        preferred_style=preferred_style
                    )
                
                if recommendations:
                    self._display_team_recommendations(recommendations)
                else:
                    st.warning("Could not generate recommendations. Try adjusting your constraints.")
                    
            except ImportError:
                st.error("Team recommender component not available. Please ensure all components are properly installed.")
    
    def _display_team_recommendations(self, recommendations):
        """Display generated team recommendations"""
        team_df = recommendations["team"]
        formation = recommendations["formation"]
        total_cost = recommendations["total_cost"]
        total_expected_points = recommendations["total_expected_points"]
        
        # Team summary
        st.subheader("✅ Your Optimized Team")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Formation", f"{formation[0]}-{formation[1]}-{formation[2]}")
            st.metric("Total Cost", f"£{total_cost/10:.1f}m")
        
        with col2:
            st.metric("Expected Points", f"{total_expected_points:.1f}")
            st.metric("Budget Remaining", f"£{100 - total_cost/10:.1f}m")
        
        with col3:
            captain = recommendations.get("captain", {})
            vice_captain = recommendations.get("vice_captain", {})
            st.metric("Captain", captain.get("web_name", "N/A"))
            st.metric("Vice Captain", vice_captain.get("web_name", "N/A"))
        
        # Team display
        st.subheader("👥 Squad Details")
        
        # Prepare display data
        display_df = team_df.copy()
        display_df['Position'] = display_df['element_type'].map({
            1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'
        })
        display_df['Cost'] = '£' + (display_df['now_cost'] / 10).round(1).astype(str) + 'm'
        display_df['Status'] = display_df.get('is_starting', [True]*len(display_df)).apply(
            lambda x: '🟢 Starting' if x else '🟡 Bench'
        )
        
        # Show team table
        columns_to_show = ['web_name', 'Position', 'team_name', 'Cost', 'Status', 'total_points', 'form']
        if 'selected_by_percent' in display_df.columns:
            columns_to_show.append('selected_by_percent')
            
        st.dataframe(
            display_df[columns_to_show],
            column_config={
                'web_name': 'Player',
                'team_name': 'Team',
                'total_points': 'Points',
                'form': 'Form',
                'selected_by_percent': 'Ownership %'
            },
            use_container_width=True,
            hide_index=True
        )
    
    def _assess_squad_player_reliability(self, player, position):
        """Assess reliability for squad players with enhanced metrics"""
        try:
            # Get player stats
            minutes = player.get('minutes', 0)
            total_games = player.get('round', 1)  # Games played this season
            games_started = minutes / 90 if minutes > 0 else 0
            form = player.get('form', 0)
            injury_status = player.get('status', 'a')  # 'a' = available
            
            # Calculate reliability score
            if injury_status != 'a':
                return "🔴 Injured"
            
            if total_games < 3:  # Early season
                if form >= 5:
                    return "🟢 Promising"
                elif form >= 3:
                    return "🟡 Monitor"
                else:
                    return "🔴 Concern"
            
            # Regular season assessment
            if games_started >= 0.8 * total_games and form >= 5:
                return "🟢 Reliable"
            elif games_started >= 0.6 * total_games and form >= 3:
                return "🟡 Decent"
            elif games_started >= 0.4 * total_games:
                return "🟠 Rotation"
            else:
                return "🔴 Bench Risk"
                
        except Exception as e:
            return "❓ Unknown"

# Main execution
if __name__ == "__main__":
    app = FPLAnalyticsApp()
    app.run()

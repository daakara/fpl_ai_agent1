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
        """Process fixtures data into a structured DataFrame"""
        fixtures = self.load_fixtures()
        teams = self.load_teams()
        
        if not fixtures or not teams:
            return pd.DataFrame()
        
        team_lookup = {team['id']: team for team in teams}
        fixture_data = []
        
        for team in teams:
            team_id = team['id']
            team_name = team['name']
            team_short_name = team['short_name']
            
            next_fixtures = self.get_next_5_fixtures(team_id, fixtures)
            
            # If no fixtures found, create placeholder data to ensure all teams appear
            if not next_fixtures:
                for i in range(1, 6):
                    fixture_data.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'team_short_name': team_short_name,
                        'fixture_number': i,
                        'opponent_id': None,
                        'opponent_name': 'TBD',
                        'opponent_short_name': 'TBD',
                        'is_home': True,
                        'venue': 'H',
                        'kickoff_time': None,
                        'gameweek': None,
                        'fixture_id': None,
                        'difficulty': 3,
                        'opponent_strength': 3,
                        'opponent_strength_overall_home': 3,
                        'opponent_strength_overall_away': 3,
                        'opponent_strength_attack_home': 3,
                        'opponent_strength_attack_away': 3,
                        'opponent_strength_defence_home': 3,
                        'opponent_strength_defence_away': 3
                    })
            else:
                for i, fixture in enumerate(next_fixtures, 1):
                    is_home = fixture['team_h'] == team_id
                    opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                    opponent = team_lookup.get(opponent_id, {})
                    
                    # Use FPL's difficulty rating if available, otherwise use opponent strength
                    if is_home:
                        difficulty = fixture.get('team_h_difficulty', 3)
                    else:
                        difficulty = fixture.get('team_a_difficulty', 3)
                    
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
        
        return pd.DataFrame(fixture_data)

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
            
            if defence_strength <= 2:
                return 5
            elif defence_strength <= 2.5:
                return 4
            elif defence_strength <= 3.5:
                return 3
            elif defence_strength <= 4:
                return 2
            else:
                return 1
        
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
            
            if attack_strength >= 4.5:
                return 5
            elif attack_strength >= 4:
                return 4
            elif attack_strength >= 3:
                return 3
            elif attack_strength >= 2.5:
                return 2
            else:
                return 1
        
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
            "📈 Team Odds": "team_odds"
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
        """Enhanced Fixture Difficulty Ratings tab"""
        st.header("🎯 Fixture Difficulty Ratings (FDR)")
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
                                st.success("✅ Fixture data loaded!")
                                st.rerun()
                            else:
                                st.error("❌ No fixture data available")
                                # Show debug info
                                raw_fixtures = fixture_loader.load_fixtures()
                                st.write(f"Debug: Found {len(raw_fixtures)} raw fixtures")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            return
        
        fixtures_df = st.session_state.fixtures_df
        
        # Settings panel
        with st.expander("⚙️ FDR Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gameweeks_ahead = st.slider("Gameweeks to analyze:", 1, 10, 5)
                show_colors = st.checkbox("Show color coding", value=True)
            
            with col2:
                fdr_threshold = st.slider("Good fixture threshold:", 1.0, 4.0, 2.5, 0.1)
                show_opponents = st.checkbox("Show opponent names", value=True)
            
            with col3:
                sort_by = st.selectbox("Sort teams by:", ["Combined FDR", "Attack FDR", "Defense FDR", "Alphabetical"])
                ascending_sort = st.checkbox("Ascending order", value=True)
        
        # Create enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "⚔️ Attack Analysis", 
            "🛡️ Defense Analysis", 
            "🎯 Transfer Targets",
            "📈 Fixture Swings"
        ])
        
        with tab1:
            self._render_fdr_overview(fixtures_df, fdr_visualizer, gameweeks_ahead, sort_by, ascending_sort)
        
        with tab2:
            self._render_attack_analysis(fixtures_df, fdr_visualizer, fdr_threshold, show_opponents)
        
        with tab3:
            self._render_defense_analysis(fixtures_df, fdr_visualizer, fdr_threshold, show_opponents)
        
        with tab4:
            self._render_transfer_targets(fixtures_df, fdr_threshold)
        
        with tab5:
            self._render_fixture_swings(fixtures_df)
    
    def _render_fdr_overview(self, fixtures_df, fdr_visualizer, gameweeks_ahead, sort_by, ascending_sort):
        """Enhanced FDR overview with better metrics and insights"""
        st.subheader("📊 FDR Overview - Next 5 Fixtures")
        
        if fixtures_df.empty:
            st.warning("No fixture data available")
            return
        
        # Enhanced key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_fixtures = len(fixtures_df)
            teams_count = fixtures_df['team_short_name'].nunique()
            st.metric("📊 Fixtures Analyzed", f"{total_fixtures} ({teams_count} teams)")
        
        with col2:
            if 'attack_fdr' in fixtures_df.columns:
                avg_attack_fdr = fixtures_df['attack_fdr'].mean()
                best_attack_team = fixtures_df.groupby('team_short_name')['attack_fdr'].mean().idxmin()
                st.metric("⚔️ Best Attack Fixtures", f"{best_attack_team}", f"Avg: {avg_attack_fdr:.2f}")
            else:
                st.metric("⚔️ Attack FDR", "N/A")
        
        with col3:
            if 'defense_fdr' in fixtures_df.columns:
                avg_defense_fdr = fixtures_df['defense_fdr'].mean()
                best_defense_team = fixtures_df.groupby('team_short_name')['defense_fdr'].mean().idxmin()
                st.metric("🛡️ Best Defense Fixtures", f"{best_defense_team}", f"Avg: {avg_defense_fdr:.2f}")
            else:
                st.metric("🛡️ Defense FDR", "N/A")
        
        with col4:
            if 'combined_fdr' in fixtures_df.columns:
                avg_combined_fdr = fixtures_df['combined_fdr'].mean()
                best_overall_team = fixtures_df.groupby('team_short_name')['combined_fdr'].mean().idxmin()
                st.metric("🎯 Best Overall Fixtures", f"{best_overall_team}", f"Avg: {avg_combined_fdr:.2f}")
            else:
                st.metric("🎯 Combined FDR", "N/A")
        
        st.divider()
        
        # Enhanced FDR Heatmap with better styling
        if 'combined_fdr' in fixtures_df.columns:
            st.subheader("🌡️ FDR Heatmap - Next 5 Fixtures")
            
            # Add fixture opponent info to heatmap
            if 'opponent_short_name' in fixtures_df.columns:
                # Create enhanced heatmap with opponent names
                fig_heatmap = self._create_enhanced_fdr_heatmap(fixtures_df, 'combined')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                fig_heatmap = fdr_visualizer.create_fdr_heatmap(fixtures_df, 'combined')
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
        
        if fixtures_df.empty:
            st.warning("No data for team summary")
            return
        
        # Create comprehensive team summary
        team_summary = self._create_enhanced_team_summary(fixtures_df, sort_by, ascending_sort)
        
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
    
    def _render_attack_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents):
        """Enhanced attack FDR analysis"""
        st.subheader("⚔️ Attack FDR Analysis")
        st.info("🎯 Lower Attack FDR = Easier to score goals. Target these teams' forwards and attacking midfielders!")
        
        if fixtures_df.empty or 'attack_fdr' not in fixtures_df.columns:
            st.warning("Attack FDR data not available")
            return
        
        # Attack FDR insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Best attacking fixtures with detailed breakdown
            st.subheader("🟢 Best Attacking Fixtures")
            attack_summary = fixtures_df.groupby('team_short_name').agg({
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
            fig_attack = fdr_visualizer.create_fdr_heatmap(fixtures_df, 'attack')
            st.plotly_chart(fig_attack, use_container_width=True)
        
        # Attack fixture opportunities
        st.subheader("🎯 Attack Fixture Opportunities")
        opportunities = self._identify_attack_opportunities(fixtures_df, fdr_threshold)
        
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
    
    def _render_defense_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents):
        """Enhanced defense FDR analysis"""
        st.subheader("🛡️ Defense FDR Analysis")
        st.info("🏠 Lower Defense FDR = Easier to keep clean sheets. Target these teams' defenders and goalkeepers!")
        
        if fixtures_df.empty or 'defense_fdr' not in fixtures_df.columns:
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
            elif selected_page == "team_odds":
                self.render_team_odds()
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
        """Render player analysis page"""
        st.header("👥 Player Analysis")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first from the Dashboard.")
            return
        
        df = st.session_state.players_df
        
        if df.empty:
            st.warning("No player data available")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'position_name' in df.columns:
                positions = st.multiselect(
                    "Position:",
                    df['position_name'].unique(),
                    default=df['position_name'].unique()
                )
            else:
                positions = []
                st.warning("Position filter not available")
        
        with col2:
            if 'team_name' in df.columns:
                teams = st.multiselect(
                    "Team:",
                    sorted(df['team_name'].unique()),
                    default=sorted(df['team_name'].unique())
                )
            else:
                teams = []
                st.warning("Team filter not available")
        
        with col3:
            if 'cost_millions' in df.columns:
                min_price, max_price = st.slider(
                    "Price Range (£m):",
                    float(df['cost_millions'].min()),
                    float(df['cost_millions'].max()),
                    (float(df['cost_millions'].min()), float(df['cost_millions'].max())),
                    step=0.1
                )
            else:
                min_price, max_price = 0, 15
                st.warning("Price filter not available")
        
        # Apply filters
        filtered_df = df.copy()
        
        if 'position_name' in df.columns and positions:
            filtered_df = filtered_df[filtered_df['position_name'].isin(positions)]
        
        if 'team_name' in df.columns and teams:
            filtered_df = filtered_df[filtered_df['team_name'].isin(teams)]
        
        if 'cost_millions' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['cost_millions'] >= min_price) &
                (filtered_df['cost_millions'] <= max_price)
            ]
        
        st.write(f"📊 Showing {len(filtered_df)} players")
        
        # Player table
        base_cols = ['web_name', 'total_points']
        optional_cols = ['team_short_name', 'position_name', 'cost_millions', 'form', 'points_per_million', 'selected_by_percent']
        
        display_cols = base_cols + [col for col in optional_cols if col in filtered_df.columns]
        
        if 'total_points' in filtered_df.columns:
            st.dataframe(
                filtered_df[display_cols].sort_values('total_points', ascending=False),
                use_container_width=True
            )
        else:
            st.dataframe(filtered_df[display_cols], use_container_width=True)

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
        """Enhanced My FPL Team analysis and import"""
        st.header("👤 My FPL Team")
        
        # Team import section
        if 'my_team_loaded' not in st.session_state or not st.session_state.my_team_loaded:
            st.subheader("📥 Import Your FPL Team")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                team_id = st.text_input(
                    "Enter your FPL Team ID",
                    placeholder="e.g., 123456",
                    help="You can find your team ID in the URL when viewing your team on the FPL website"
                )
            
            with col2:
                if st.button("🔄 Load My Team", type="primary"):
                    if team_id:
                        with st.spinner("Loading your team..."):
                            team_data = self._load_fpl_team(team_id)
                            
                            if team_data:
                                st.session_state.my_team_data = team_data
                                st.session_state.my_team_id = team_id
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "👥 Current Squad", 
            "📊 Performance", 
            "🔄 Transfer Suggestions", 
            "📈 Benchmarking"
        ])
        
        with tab1:
            self._display_current_squad(team_data)
        
        with tab2:
            self._display_performance_analysis(team_data)
        
        with tab3:
            self._display_transfer_suggestions(team_data)
        
        with tab4:
            self._display_benchmarking(team_data)
        
        # Reset team button
        if st.button("🔄 Load Different Team"):
            st.session_state.my_team_loaded = False
            st.rerun()
    
    def _load_fpl_team(self, team_id):
        """Load FPL team data from API"""
        try:
            # FPL API endpoints
            entry_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
            picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/1/picks/"  # Latest gameweek
            
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
            except:
                entry_data['picks'] = []
            
            return entry_data
            
        except Exception as e:
            st.error(f"Error loading team: {str(e)}")
            return None
    
    def _display_current_squad(self, team_data):
        """Display current squad with player details"""
        st.subheader("� Current Squad")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see detailed squad analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        
        # Match picks with player data
        squad_data = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                squad_data.append({
                    'Player': player['web_name'],
                    'Position': player.get('position_name', 'Unknown'),
                    'Team': player.get('team_short_name', 'N/A'),
                    'Price': f"£{player['cost_millions']:.1f}m",
                    'Points': player['total_points'],
                    'Form': player.get('form', 0),
                    'Status': '👑 Captain' if pick.get('is_captain') else '🅒 Vice' if pick.get('is_vice_captain') else '🟢 Playing' if pick['position'] <= 11 else '🟡 Bench'
                })
        
        if squad_data:
            squad_df = pd.DataFrame(squad_data)
            st.dataframe(squad_df, use_container_width=True, hide_index=True)
            
            # Squad statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_value = sum([float(row['Price'].replace('£', '').replace('m', '')) for row in squad_data])
                st.metric("Squad Value", f"£{total_value:.1f}m")
            
            with col2:
                total_points = sum([row['Points'] for row in squad_data])
                st.metric("Total Points", f"{total_points:,}")
            
            with col3:
                avg_form = np.mean([row['Form'] for row in squad_data if row['Form'] > 0])
                st.metric("Average Form", f"{avg_form:.1f}")
    
    def _display_performance_analysis(self, team_data):
        """Display performance analysis"""
        st.subheader("📊 Performance Analysis")
        
        # Basic performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Season Performance**")
            st.write(f"• Overall Points: {team_data.get('summary_overall_points', 'N/A'):,}")
            st.write(f"• Overall Rank: {team_data.get('summary_overall_rank', 'N/A'):,}")
            st.write(f"• Highest Score: {team_data.get('highest_score', 'N/A')}")
        
        with col2:
            st.write("**🎯 Recent Performance**")
            st.write(f"• Gameweek Rank: {team_data.get('summary_event_rank', 'N/A'):,}")
            st.write(f"• Gameweek Points: {team_data.get('summary_event_points', 'N/A')}")
            st.write(f"• Form: {team_data.get('current_event', 'N/A')}")
        
        # Performance insights
        st.write("**💡 Performance Insights**")
        
        overall_rank = team_data.get('summary_overall_rank', 0)
        if overall_rank:
            if overall_rank <= 100000:
                st.success("🏆 Excellent performance! You're in the top tier of managers.")
            elif overall_rank <= 500000:
                st.info("👍 Good performance! You're above average.")
            else:
                st.warning("📈 Room for improvement. Consider the transfer suggestions!")
    
    def _display_transfer_suggestions(self, team_data):
        """Display transfer suggestions based on current squad"""
        st.subheader("🔄 Transfer Suggestions")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see transfer suggestions")
            return
        
        # Analyze current squad for weaknesses
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available for analysis")
            return
        
        players_df = st.session_state.players_df
        current_players = [pick['element'] for pick in picks]
        
        # Find players in poor form
        poor_form_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                if player.get('form', 0) < 4.0 and player['total_points'] > 20:
                    poor_form_players.append({
                        'name': player['web_name'],
                        'form': player.get('form', 0),
                        'position': player.get('position_name', 'Unknown')
                    })
        
        # Suggest alternatives
        st.write("**⚠️ Players to Consider Transferring Out**")
        if poor_form_players:
            for player in poor_form_players[:3]:
                st.warning(f"• {player['name']} - Poor form ({player['form']:.1f})")
        else:
            st.success("✅ No obvious transfer candidates - your team is in good form!")
        
        # Suggest transfer targets
        st.write("**🎯 Suggested Transfer Targets**")
        
        # Find good value players not in current squad
        available_players = players_df[~players_df['id'].isin(current_players)]
        
        if not available_players.empty and 'points_per_million' in available_players.columns:
            targets = available_players[
                (available_players['total_points'] > 30) &
                (available_players.get('form', 0) >= 6.0) &
                (available_players['cost_millions'] <= 10.0)
            ].nlargest(5, 'points_per_million')
            
            for _, player in targets.iterrows():
                st.success(f"• {player['web_name']} ({player.get('position_name', 'Unknown')}) - "
                          f"£{player['cost_millions']:.1f}m - Form: {player.get('form', 0):.1f}")
    
    def _display_benchmarking(self, team_data):
        """Display benchmarking against top teams"""
        st.subheader("� Benchmarking")
        
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

    def render_team_odds(self):
        """Enhanced Team Odds Analysis and FPL Betting Insights"""
        st.header("📈 Team Odds & FPL Betting Insights")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first from the Dashboard.")
            return
        
        # Team odds analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 League Odds", 
            "⚽ Match Odds", 
            "🎯 Player Odds", 
            "📊 Value Bets", 
            "🔮 Predictions"
        ])
        
        with tab1:
            self._render_league_odds()
        
        with tab2:
            self._render_match_odds()
        
        with tab3:
            self._render_player_odds()
        
        with tab4:
            self._render_value_bets()
        
        with tab5:
            self._render_predictions()
    
    def _render_league_odds(self):
        """Render Premier League season odds analysis"""
        st.subheader("🏆 Premier League Season Odds")
        
        # Simulated odds data (in a real app, this would come from betting APIs)
        league_odds = self._get_simulated_league_odds()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Title Odds**")
            title_df = pd.DataFrame(league_odds['title_odds'])
            
            # Add FPL relevance
            for idx, row in title_df.iterrows():
                odds = row['Odds']
                team = row['Team']
                
                # Convert odds to probability
                prob = self._odds_to_probability(odds)
                
                # FPL advice
                if prob > 0.3:
                    advice = "🟢 Target premium players"
                elif prob > 0.15:
                    advice = "🟡 Consider key players"
                else:
                    advice = "🔴 Avoid expensive assets"
                
                st.write(f"**{team}** - {odds} ({prob:.1%}) {advice}")
        
        with col2:
            st.write("**🥅 Top 4 Odds**")
            top4_df = pd.DataFrame(league_odds['top4_odds'])
            
            for _, row in top4_df.iterrows():
                odds = row['Odds']
                team = row['Team']
                prob = self._odds_to_probability(odds)
                
                # FPL advice for top 4 contenders
                if prob > 0.7:
                    advice = "🟢 Strong defensive picks"
                elif prob > 0.4:
                    advice = "🟡 Rotation risk moderate"
                else:
                    advice = "🔴 High rotation risk"
                
                st.write(f"**{team}** - {odds} ({prob:.1%}) {advice}")
        
        # Season insights
        st.subheader("💡 FPL Season Strategy Insights")
        
        # Get team performance data
        teams_df = st.session_state.teams_df
        players_df = st.session_state.players_df
        
        if not teams_df.empty and not players_df.empty:
            # Calculate team values and recommendations
            team_insights = self._calculate_team_season_insights(teams_df, players_df, league_odds)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🎯 Best Value Teams**")
                for insight in team_insights['value_teams'][:3]:
                    st.success(f"• **{insight['team']}** - {insight['reason']}")
            
            with col2:
                st.write("**⚠️ Overpriced Teams**")
                for insight in team_insights['overpriced_teams'][:3]:
                    st.warning(f"• **{insight['team']}** - {insight['reason']}")
            
            with col3:
                st.write("**💎 Differential Teams**")
                for insight in team_insights['differential_teams'][:3]:
                    st.info(f"• **{insight['team']}** - {insight['reason']}")
    
    def _render_match_odds(self):
        """Render upcoming match odds and predictions"""
        st.subheader("⚽ Match Odds & FPL Impact")
        
        # Simulate upcoming fixtures with odds
        match_odds = self._get_simulated_match_odds()
        
        st.write("**📅 This Gameweek's Key Fixtures**")
        
        for match in match_odds:
            with st.expander(f"🏟️ {match['home_team']} vs {match['away_team']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**📊 Match Odds**")
                    st.write(f"Home Win: {match['home_odds']}")
                    st.write(f"Draw: {match['draw_odds']}")
                    st.write(f"Away Win: {match['away_odds']}")
                
                with col2:
                    st.write("**🥅 Goals Market**")
                    st.write(f"Over 2.5: {match['over_2_5']}")
                    st.write(f"Under 2.5: {match['under_2_5']}")
                    st.write(f"BTTS: {match['btts']}")
                
                with col3:
                    st.write("**🎯 FPL Recommendations**")
                    
                    # Calculate FPL advice based on odds
                    home_prob = self._odds_to_probability(match['home_odds'])
                    goals_prob = self._odds_to_probability(match['over_2_5'])
                    
                    if home_prob > 0.6:
                        st.success(f"✅ Target {match['home_team']} attackers")
                    elif home_prob < 0.3:
                        st.success(f"✅ Target {match['away_team']} attackers")
                    else:
                        st.info("🤔 Balanced fixture")
                    
                    if goals_prob > 0.6:
                        st.warning("⚠️ Both defenses vulnerable")
                    else:
                        st.success("🛡️ Good for clean sheets")
        
        # Captain recommendations based on odds
        st.subheader("👑 Captain Recommendations by Odds")
        
        captain_recs = self._generate_captain_recommendations(match_odds)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔥 High Ceiling Captains**")
            for rec in captain_recs['high_ceiling'][:3]:
                st.write(f"• **{rec['player']}** vs {rec['opponent']} - {rec['reason']}")
        
        with col2:
            st.write("**🛡️ Safe Captain Options**")
            for rec in captain_recs['safe_picks'][:3]:
                st.write(f"• **{rec['player']}** vs {rec['opponent']} - {rec['reason']}")
    
    def _render_player_odds(self):
        """Render player-specific betting odds and FPL implications"""
        st.subheader("🎯 Player Odds & Performance Predictions")
        
        if not st.session_state.data_loaded:
            return
        
        players_df = st.session_state.players_df
        
        # Player performance predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**⚽ Goal Scorer Odds Analysis**")
            
            # Get top forwards and attacking midfielders
            attackers = players_df[
                (players_df['element_type'].isin([3, 4])) &
                (players_df['total_points'] > 50)
            ].nlargest(10, 'total_points')
            
            # Simulate goal scorer odds
            for _, player in attackers.head(5).iterrows():
                # Calculate implied odds based on form and fixtures
                form_score = player.get('form', 5)
                points_score = player['total_points'] / 100
                
                # Simulate odds (would be real odds in production)
                implied_odds = max(1.5, 8 - (form_score * 0.5) - (points_score * 2))
                prob = self._odds_to_probability(f"{implied_odds:.1f}")
                
                team = player.get('team_short_name', 'N/A')
                
                if prob > 0.3:
                    color = "🟢"
                    advice = "Strong captain option"
                elif prob > 0.2:
                    color = "🟡"
                    advice = "Consider for captaincy"
                else:
                    color = "🔴"
                    advice = "Risky captain choice"
                
                st.write(f"{color} **{player['web_name']}** ({team}) - "
                        f"{implied_odds:.1f} ({prob:.1%}) - {advice}")
        
        with col2:
            st.write("**🥅 Clean Sheet Odds Analysis**")
            
            # Get top defenders and goalkeepers
            defenders = players_df[
                (players_df['element_type'].isin([1, 2])) &
                (players_df['total_points'] > 40)
            ].nlargest(10, 'total_points')
            
            for _, player in defenders.head(5).iterrows():
                # Calculate clean sheet probability
                points_score = player['total_points'] / 100
                form_score = player.get('form', 5)
                
                # Simulate clean sheet odds
                cs_prob = min(0.8, max(0.1, (points_score * 0.3) + (form_score * 0.05)))
                cs_odds = 1 / cs_prob
                
                team = player.get('team_short_name', 'N/A')
                
                if cs_prob > 0.5:
                    color = "🟢"
                    advice = "Excellent defensive pick"
                elif cs_prob > 0.3:
                    color = "🟡"
                    advice = "Decent defensive option"
                else:
                    color = "🔴"
                    advice = "Avoid this defense"
                
                st.write(f"{color} **{player['web_name']}** ({team}) - "
                        f"{cs_odds:.1f} ({cs_prob:.1%}) - {advice}")
        
        # Assist odds
        st.write("**🅰️ Assist Odds Analysis**")
        
        # Get creative players
        creative_players = players_df[
            (players_df['element_type'].isin([2, 3, 4])) &
            (players_df['total_points'] > 40)
        ].nlargest(8, 'total_points')
        
        assist_analysis = []
        for _, player in creative_players.iterrows():
            # Simulate assist probability
            position_mult = 1.2 if player['element_type'] == 3 else 0.8
            form_score = player.get('form', 5) * 0.1
            assist_prob = min(0.4, max(0.05, form_score * position_mult))
            
            assist_analysis.append({
                'Player': player['web_name'],
                'Team': player.get('team_short_name', 'N/A'),
                'Assist Probability': f"{assist_prob:.1%}",
                'FPL Value': "High" if assist_prob > 0.25 else "Medium" if assist_prob > 0.15 else "Low"
            })
        
        assist_df = pd.DataFrame(assist_analysis)
        st.dataframe(assist_df, use_container_width=True, hide_index=True)
    
    def _render_value_bets(self):
        """Identify value betting opportunities related to FPL"""
        st.subheader("📊 Value Betting & FPL Arbitrage")
        
        st.info("💡 **Value Betting Concept**: When bookmaker odds suggest a lower probability than your analysis indicates")
        
        # Value bet analysis tabs
        value_tab1, value_tab2, value_tab3 = st.tabs([
            "🎯 Player Value Bets", 
            "🏆 Team Value Bets", 
            "📈 Market Inefficiencies"
        ])
        
        with value_tab1:
            st.write("**⚽ Player Performance Value Bets**")
            
            if st.session_state.data_loaded:
                players_df = st.session_state.players_df
                
                # Identify undervalued players based on form vs ownership
                value_players = self._identify_value_players(players_df)
                
                for player in value_players[:5]:
                    with st.expander(f"💎 {player['name']} - Value Opportunity"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📊 Player Stats**")
                            st.write(f"• Form: {player['form']:.1f}")
                            st.write(f"• Points: {player['points']}")
                            st.write(f"• Price: £{player['price']:.1f}m")
                            st.write(f"• Ownership: {player['ownership']:.1f}%")
                        
                        with col2:
                            st.write("**🎯 Value Analysis**")
                            st.write(f"• Expected vs Actual: {player['value_score']:.2f}")
                            st.write(f"• Market Inefficiency: {player['inefficiency']}")
                            st.write(f"• FPL Strategy: {player['strategy']}")
                            
                        st.success(f"**Recommendation**: {player['recommendation']}")
        
        with value_tab2:
            st.write("**🏆 Team Performance Value Bets**")
            
            # Team-based value analysis
            team_values = self._analyze_team_values()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Undervalued Teams**")
                for team in team_values['undervalued'][:3]:
                    st.success(f"✅ **{team['team']}** - {team['reason']}")
                    st.caption(f"FPL Strategy: {team['fpl_strategy']}")
            
            with col2:
                st.write("**📉 Overvalued Teams**")
                for team in team_values['overvalued'][:3]:
                    st.warning(f"⚠️ **{team['team']}** - {team['reason']}")
                    st.caption(f"FPL Strategy: {team['fpl_strategy']}")
        
        with value_tab3:
            st.write("**📈 Market Inefficiencies & Opportunities**")
            
            market_inefficiencies = [
                {
                    'opportunity': 'Low Ownership Premium Players',
                    'description': 'High-performing players with surprisingly low ownership',
                    'fpl_action': 'Transfer in before price rises',
                    'risk': 'Medium',
                    'timeframe': '1-2 gameweeks'
                },
                {
                    'opportunity': 'Fixture Swing Arbitrage',
                    'description': 'Players with improving fixtures but stable prices',
                    'fpl_action': 'Early transfer to beat price changes',
                    'risk': 'Low',
                    'timeframe': '2-4 gameweeks'
                },
                {
                    'opportunity': 'Form vs Price Disconnect',
                    'description': 'In-form players not yet reflected in FPL pricing',
                    'fpl_action': 'Quick transfer before algorithm adjusts',
                    'risk': 'High',
                    'timeframe': '1 gameweek'
                }
            ]
            
            for inefficiency in market_inefficiencies:
                with st.expander(f"💡 {inefficiency['opportunity']}"):
                    st.write(f"**Description**: {inefficiency['description']}")
                    st.write(f"**FPL Action**: {inefficiency['fpl_action']}")
                    st.write(f"**Risk Level**: {inefficiency['risk']}")
                    st.write(f"**Timeframe**: {inefficiency['timeframe']}")
    
    def _render_predictions(self):
        """AI-powered predictions and recommendations"""
        st.subheader("🔮 AI Predictions & Recommendations")
        
        # Prediction categories
        pred_tab1, pred_tab2, pred_tab3 = st.tabs([
            "📈 Next Gameweek", 
            "🗓️ Next 5 GWs", 
            "🏆 Season Predictions"
        ])
        
        with pred_tab1:
            st.write("**⚡ Next Gameweek Predictions**")
            
            # Generate next GW predictions
            gw_predictions = self._generate_gameweek_predictions()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🎯 Captain Picks**")
                for pred in gw_predictions['captains'][:3]:
                    confidence = "🟢" if pred['confidence'] > 80 else "🟡" if pred['confidence'] > 60 else "🔴"
                    st.write(f"{confidence} **{pred['player']}** ({pred['confidence']}%)")
                    st.caption(f"Expected: {pred['expected_points']:.1f} pts")
            
            with col2:
                st.write("**💎 Differential Picks**")
                for pred in gw_predictions['differentials'][:3]:
                    st.write(f"💎 **{pred['player']}** ({pred['ownership']:.1f}%)")
                    st.caption(f"Potential: {pred['potential']:.1f} pts")
            
            with col3:
                st.write("**⚠️ Avoid These**")
                for pred in gw_predictions['avoid'][:3]:
                    st.write(f"🔴 **{pred['player']}**")
                    st.caption(f"Risk: {pred['risk_reason']}")
        
        with pred_tab2:
            st.write("**📅 Next 5 Gameweeks Strategy**")
            
            # 5-gameweek predictions
            medium_term = self._generate_medium_term_predictions()
            
            st.write("**🎯 Transfer Targets (Next 5 GWs)**")
            for target in medium_term['targets'][:5]:
                with st.expander(f"⭐ {target['player']} - {target['team']}"):
                    st.write(f"**Expected Points**: {target['projected_points']:.1f}")
                    st.write(f"**Fixture Difficulty**: {target['fixture_rating']}/5")
                    st.write(f"**Price Trend**: {target['price_trend']}")
                    st.write(f"**Strategy**: {target['strategy']}")
                    
                    if target['fixture_rating'] <= 2:
                        st.success("🟢 Excellent fixtures ahead")
                    elif target['fixture_rating'] <= 3:
                        st.info("🟡 Decent fixtures")
                    else:
                        st.warning("🔴 Tough fixtures coming")
        
        with pred_tab3:
            st.write("**🏆 Season-Long Predictions**")
            
            # Season predictions
            season_predictions = self._generate_season_predictions()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Predicted Final Table (Top 10)**")
                for i, team in enumerate(season_predictions['league_table'][:10], 1):
                    if i <= 4:
                        icon = "🟢"
                    elif i <= 6:
                        icon = "🟡"
                    else:
                        icon = "⚪"
                    
                    st.write(f"{icon} {i}. **{team['team']}** ({team['predicted_points']} pts)")
            
            with col2:
                st.write("**⭐ Season-Long FPL Assets**")
                
                for asset in season_predictions['season_assets'][:5]:
                    st.write(f"💎 **{asset['player']}** ({asset['team']})")
                    st.caption(f"Projected: {asset['projected_total']} season points")
    
    def _get_simulated_league_odds(self):
        """Generate simulated league odds (would be real API data in production)"""
        return {
            'title_odds': [
                {'Team': 'Manchester City', 'Odds': '1.80'},
                {'Team': 'Arsenal', 'Odds': '3.50'},
                {'Team': 'Liverpool', 'Odds': '4.00'},
                {'Team': 'Manchester United', 'Odds': '8.00'},
                {'Team': 'Chelsea', 'Odds': '12.00'},
                {'Team': 'Newcastle', 'Odds': '25.00'},
                {'Team': 'Tottenham', 'Odds': '30.00'}
            ],
            'top4_odds': [
                {'Team': 'Manchester City', 'Odds': '1.05'},
                {'Team': 'Arsenal', 'Odds': '1.25'},
                {'Team': 'Liverpool', 'Odds': '1.40'},
                {'Team': 'Manchester United', 'Odds': '2.20'},
                {'Team': 'Chelsea', 'Odds': '2.80'},
                {'Team': 'Newcastle', 'Odds': '3.50'},
                {'Team': 'Tottenham', 'Odds': '4.00'}
            ]
        }
    
    def _get_simulated_match_odds(self):
        """Generate simulated match odds"""
        return [
            {
                'home_team': 'Arsenal',
                'away_team': 'Brighton',
                'home_odds': '1.65',
                'draw_odds': '4.20',
                'away_odds': '5.50',
                'over_2_5': '1.75',
                'under_2_5': '2.10',
                'btts': '1.90'
            },
            {
                'home_team': 'Manchester City',
                'away_team': 'Sheffield United',
                'home_odds': '1.15',
                'draw_odds': '8.00',
                'away_odds': '15.00',
                'over_2_5': '1.40',
                'under_2_5': '2.90',
                'btts': '2.50'
            },
            {
                'home_team': 'West Ham',
                'away_team': 'Liverpool',
                'home_odds': '4.50',
                'draw_odds': '3.80',
                'away_odds': '1.75',
                'over_2_5': '1.55',
                'under_2_5': '2.40',
                'btts': '1.70'
            }
        ]
    
    def _odds_to_probability(self, odds_str):
        """Convert decimal odds to probability"""
        try:
            odds = float(odds_str)
            return 1 / odds
        except:
            return 0.5
    
    def _calculate_team_season_insights(self, teams_df, players_df, league_odds):
        """Calculate team insights based on odds vs performance"""
        insights = {
            'value_teams': [],
            'overpriced_teams': [],
            'differential_teams': []
        }
        
        # Calculate team performance scores
        for _, team in teams_df.head(6).iterrows():  # Top 6 for example
            team_players = players_df[players_df['team'] == team['id']]
            
            if not team_players.empty:
                avg_points = team_players['total_points'].mean()
                total_value = team_players['cost_millions'].sum()
                
                # Simple value analysis
                if avg_points > 60 and total_value < 85:
                    insights['value_teams'].append({
                        'team': team['short_name'],
                        'reason': f'High points/cost ratio ({avg_points:.1f} avg pts)'
                    })
                elif avg_points < 45 and total_value > 90:
                    insights['overpriced_teams'].append({
                        'team': team['short_name'],
                        'reason': f'Low points/cost ratio ({avg_points:.1f} avg pts)'
                    })
                else:
                    insights['differential_teams'].append({
                        'team': team['short_name'],
                        'reason': 'Balanced risk/reward profile'
                    })
        
        return insights
    
    def _generate_captain_recommendations(self, match_odds):
        """Generate captain recommendations based on match odds"""
        return {
            'high_ceiling': [
                {'player': 'Haaland', 'opponent': 'Sheffield United', 'reason': 'Huge favorite, goals expected'},
                {'player': 'Salah', 'opponent': 'West Ham', 'reason': 'Good form, favorable odds'},
                {'player': 'Saka', 'opponent': 'Brighton', 'reason': 'Home advantage, creative threat'}
            ],
            'safe_picks': [
                {'player': 'Son', 'opponent': 'Burnley', 'reason': 'Consistent performer, decent odds'},
                {'player': 'Palmer', 'opponent': 'Luton', 'reason': 'On penalties, favorable fixture'},
                {'player': 'Watkins', 'opponent': 'Everton', 'reason': 'Good form, reasonable odds'}
            ]
        }
    
    def _identify_value_players(self, players_df):
        """Identify value players based on performance vs ownership"""
        value_players = []
        
        # Filter for players with good stats but low ownership
        candidates = players_df[
            (players_df['total_points'] > 50) &
            (players_df.get('selected_by_percent', 0) < 15) &
            (players_df.get('form', 0) >= 6)
        ].head(5)
        
        for _, player in candidates.iterrows():
            form = player.get('form', 5)
            ownership = player.get('selected_by_percent', 10)
            
            value_score = (form * 10) / max(ownership, 1)
            
            value_players.append({
                'name': player['web_name'],
                'form': form,
                'points': player['total_points'],
                'price': player['cost_millions'],
                'ownership': ownership,
                'value_score': value_score,
                'inefficiency': 'High' if value_score > 5 else 'Medium',
                'strategy': 'Differential pick' if ownership < 5 else 'Template avoid',
                'recommendation': f'Strong differential option with {form:.1f} form and only {ownership:.1f}% ownership'
            })
        
        return sorted(value_players, key=lambda x: x['value_score'], reverse=True)
    
    def _analyze_team_values(self):
        """Analyze team values for betting opportunities"""
        return {
            'undervalued': [
                {'team': 'Brighton', 'reason': 'Consistent performance vs low expectations', 'fpl_strategy': 'Target budget options'},
                {'team': 'Newcastle', 'reason': 'Strong defense, improving attack', 'fpl_strategy': 'Defensive assets + Wilson'},
                {'team': 'West Ham', 'reason': 'European distraction priced in too heavily', 'fpl_strategy': 'Bowen + Kudus potential'}
            ],
            'overvalued': [
                {'team': 'Manchester United', 'reason': 'High expectations vs inconsistent performance', 'fpl_strategy': 'Avoid except Rashford'},
                {'team': 'Chelsea', 'reason': 'Rotation risk not properly priced', 'fpl_strategy': 'Palmer only, avoid others'},
                {'team': 'Tottenham', 'reason': 'Inconsistency pattern continuing', 'fpl_strategy': 'Son + avoid defense'}
            ]
        }
    
    def _generate_gameweek_predictions(self):
        """Generate next gameweek predictions"""
        return {
            'captains': [
                {'player': 'Haaland', 'confidence': 85, 'expected_points': 12.5},
                {'player': 'Salah', 'confidence': 75, 'expected_points': 10.2},
                {'player': 'Son', 'confidence': 70, 'expected_points': 9.8}
            ],
            'differentials': [
                {'player': 'Isak', 'ownership': 8.2, 'potential': 11.5},
                {'player': 'Bowen', 'ownership': 12.1, 'potential': 9.2},
                {'player': 'Watkins', 'ownership': 15.3, 'potential': 8.8}
            ],
            'avoid': [
                {'player': 'Fernandes', 'risk_reason': 'Rotation risk + tough fixture'},
                {'player': 'Kane', 'risk_reason': 'Poor away form continuing'},
                {'player': 'Sterling', 'risk_reason': 'Inconsistent minutes'}
            ]
        }
    
    def _generate_medium_term_predictions(self):
        """Generate 5-gameweek predictions"""
        return {
            'targets': [
                {
                    'player': 'Saka',
                    'team': 'Arsenal',
                    'projected_points': 45.2,
                    'fixture_rating': 2.2,
                    'price_trend': 'Stable',
                    'strategy': 'Set and forget for 5 GWs'
                },
                {
                    'player': 'Palmer',
                    'team': 'Chelsea',
                    'projected_points': 42.8,
                    'fixture_rating': 2.8,
                    'price_trend': 'Rising',
                    'strategy': 'Get before price increase'
                },
                {
                    'player': 'Isak',
                    'team': 'Newcastle',
                    'projected_points': 38.5,
                    'fixture_rating': 2.0,
                    'price_trend': 'Stable',
                    'strategy': 'Excellent fixtures run'
                }
            ]
        }
    
    def _generate_season_predictions(self):
        """Generate season-long predictions"""
        return {
            'league_table': [
                {'team': 'Manchester City', 'predicted_points': 88},
                {'team': 'Arsenal', 'predicted_points': 84},
                {'team': 'Liverpool', 'predicted_points': 79},
                {'team': 'Chelsea', 'predicted_points': 72},
                {'team': 'Newcastle', 'predicted_points': 69},
                {'team': 'Manchester United', 'predicted_points': 67},
                {'team': 'Tottenham', 'predicted_points': 65},
                {'team': 'Brighton', 'predicted_points': 62},
                {'team': 'West Ham', 'predicted_points': 58},
                {'team': 'Aston Villa', 'predicted_points': 55}
            ],
            'season_assets': [
                {'player': 'Haaland', 'team': 'Man City', 'projected_total': 285},
                {'player': 'Salah', 'team': 'Liverpool', 'projected_total': 245},
                {'player': 'Saka', 'team': 'Arsenal', 'projected_total': 220},
                {'player': 'Son', 'team': 'Tottenham', 'projected_total': 205},
                {'player': 'Palmer', 'team': 'Chelsea', 'projected_total': 195}
            ]
        }

# Main execution
if __name__ == "__main__":
    app = FPLAnalyticsApp()
    app.run()

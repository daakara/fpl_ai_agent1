"""
FPL Analytics Dashboard - Complete self-contained application
"""
import pandas as pd
import requests
from typing import List, Dict, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import json
import logging
import numpy as np
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
    good_attack = attack_teams[attack_teams <= threshold].sort_values().head(5)
    opportunities['best_attack'] = [(team, fdr) for team, fdr in good_attack.items()]
    
    # Best defensive fixtures
    defense_teams = fixtures_df.groupby(['team_id', 'team_name'])['defense_fdr'].mean()
    good_defense = defense_teams[defense_teams <= threshold].sort_values().head(5)
    opportunities['best_defense'] = [(team, fdr) for team, fdr in good_defense.items()]
    
    # Best combined fixtures
    combined_teams = fixtures_df.groupby(['team_id', 'team_name'])['combined_fdr'].mean()
    good_combined = combined_teams[combined_teams <= threshold].sort_values().head(5)
    opportunities['best_combined'] = [(team, fdr) for team, fdr in good_combined.items()]
    
    # Worst fixtures to avoid
    bad_fixtures = combined_teams[combined_teams >= 4.0].sort_values(ascending=False).head(5)
    opportunities['worst_fixtures'] = [(team, fdr) for team, fdr in bad_fixtures.items()]
    
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
        self.base_url = "https://fantasy.premierleague.com/api"
        self.logger = logging.getLogger(__name__)
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
            
            # Handle form and ownership properly
            players_df['form'] = pd.to_numeric(players_df['form'], errors='coerce').fillna(0.0) if 'form' in players_df.columns else 0.0
            players_df['selected_by_percent'] = pd.to_numeric(players_df['selected_by_percent'], errors='coerce').fillna(0.0) if 'selected_by_percent' in players_df.columns else 0.0
            
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
            if 'form' in players_df.columns:
                players_df['form'] = pd.to_numeric(players_df['form'], errors='coerce').fillna(0.0)
            else:
                players_df['form'] = 0.0
            
            if 'selected_by_percent' in players_df.columns:
                players_df['selected_by_percent'] = pd.to_numeric(players_df['selected_by_percent'], errors='coerce').fillna(0.0)
            else:
                players_df['selected_by_percent'] = 0.0

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
                st.sidebar.info(f"Players: {len(st.session_state.players_df)}")
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
                - **Defense FDR**: How easy it is for a team's defenders to keep clean sheets (lower = weaker attacking opponents).
                - **Combined FDR**: Overall fixture difficulty considering both attack and defense.
                
                🎯 **How to use**: Green = Good fixtures, Red = Difficult fixtures.
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
    
    def _render_fixture_swings(self, fixtures_df):
        """Render fixture swing analysis"""
        st.subheader("📈 Fixture Swings Analysis")
        
        # This would analyze dramatic changes in fixture difficulty
        st.info("🚧 Fixture swing analysis coming soon!")
        
        # Placeholder for future implementation
        if not fixtures_df.empty:
            st.write("**🔄 Upcoming Fixture Changes**")
            st.write("• Teams moving from hard to easy fixtures")
            st.write("• Teams moving from easy to hard fixtures")
            st.write("• Optimal transfer timing recommendations")
            
            # Could add actual swing analysis here
            sample_swings = [
                {"team": "Arsenal", "current": "Hard", "upcoming": "Easy", "action": "Buy"},
                {"team": "Chelsea", "current": "Easy", "upcoming": "Hard", "action": "Sell"},
                {"team": "Newcastle", "current": "Mixed", "upcoming": "Good", "action": "Consider"}
            ]
            
            for swing in sample_swings:
                if swing["action"] == "Buy":
                    st.success(f"🟢 **{swing['team']}**: {swing['current']} → {swing['upcoming']} - {swing['action']}")
                elif swing["action"] == "Sell":
                    st.error(f"🔴 **{swing['team']}**: {swing['current']} → {swing['upcoming']} - {swing['action']}")
                else:
                    st.info(f"🟡 **{swing['team']}**: {swing['current']} → {swing['upcoming']} - {swing['action']}")

    def render_my_team(self):
        """Enhanced My FPL Team analysis and import"""
        st.header("👤 My FPL Team")
        
        # Team import section
        if 'my_team_loaded' not in st.session_state or not st.session_state.my_team_loaded:
            st.subheader("📥 Import Your FPL Team")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                team_id = st.number_input(
                    "Enter your FPL Team ID:",
                    min_value=1,
                    max_value=10000000,
                    value=1,
                    help="Find your Team ID in the FPL website URL"
                )
            
            with col2:
                if st.button("📥 Load Team", type="primary"):
                    team_data = self._load_fpl_team(team_id)
                    if team_data:
                        st.session_state.my_team_id = team_id
                        st.session_state.my_team_data = team_data
                        st.session_state.my_team_loaded = True
                        st.success("✅ Team loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load team. Please check your Team ID.")
            
            # Instructions
            with st.expander("💡 How to find your Team ID", expanded=False):
                st.markdown("""
                **Steps to find your FPL Team ID:**
                1. Go to the [FPL website](https://fantasy.premierleague.com/)
                2. Navigate to "Points" or "My Team"
                3. Look at the URL - it will show: `fantasy.premierleague.com/entry/YOUR_TEAM_ID/event/X`
                4. Your Team ID is the number after `/entry/`
                
                **Example:** If URL is `fantasy.premierleague.com/entry/123456/event/10`, your Team ID is `123456`
                """)
            
            return
        
        # Display loaded team
        team_data = st.session_state.my_team_data
        
        # Team overview with safe formatting
        team_name = team_data.get('entry_name', 'Your Team')
        team_id = st.session_state.my_team_id
        st.subheader(f"🏆 {team_name} (ID: {team_id})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Safe formatting for rank - handle None values
            overall_rank = team_data.get('summary_overall_rank')
            if overall_rank is not None and overall_rank > 0:
                st.metric("Overall Rank", f"{overall_rank:,}")
            else:
                st.metric("Overall Rank", "N/A")
        
        with col2:
            # Safe formatting for points - handle None values
            total_points = team_data.get('summary_overall_points')
            if total_points is not None:
                st.metric("Total Points", f"{total_points:,}")
            else:
                st.metric("Total Points", "N/A")
        
        with col3:
            # Safe formatting for gameweek rank - handle None values
            gw_rank = team_data.get('summary_event_rank')
            if gw_rank is not None and gw_rank > 0:
                st.metric("Gameweek Rank", f"{gw_rank:,}")
            else:
                st.metric("Gameweek Rank", "N/A")
        
        with col4:
            # Safe formatting for team value - handle None values
            team_value = team_data.get('value', 1000)
            if team_value is not None:
                st.metric("Team Value", f"£{team_value/10:.1f}m")
            else:
                st.metric("Team Value", "£100.0m")
        
        # Enhanced team analysis tabs with improved structure
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "👥 Squad Analysis", 
            "🔄 Transfer Intelligence", 
            "🎯 Strategy & Planning",
            "💎 Chip Strategy",
            "📈 SWOT Analysis"
        ])
        
        with tab1:
            self._display_squad_analysis_enhanced(team_data)
        
        with tab2:
            self._display_performance_benchmarking_enhanced(team_data)
        
        with tab3:
            self._display_transfer_intelligence(team_data)
        
        with tab4:
            self._display_strategy_planning(team_data)
        
        with tab5:
            self._display_chip_strategy_enhanced(team_data)
            
            # Add SWOT analysis to the chip strategy tab for now
            st.divider()
            st.subheader("📈 SWOT Analysis")
            self._display_swot_analysis_enhanced(team_data)
        
        # Reset team button
        if st.button("🔄 Load Different Team"):
            st.session_state.my_team_loaded = False
            st.rerun()

    def _load_fpl_team(self, team_id):
        """Load a user's FPL team data from the FPL API."""
        try:
            url = f"{self.base_url}/entry/{team_id}/"
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            entry_data = response.json()

            # Fetch current gameweek picks
            current_gameweek = entry_data.get('current_event')
            if current_gameweek:
                picks_url = f"{self.base_url}/entry/{team_id}/event/{current_gameweek}/picks/"
                picks_response = requests.get(picks_url, timeout=10, verify=False)
                picks_response.raise_for_status()
                picks_data = picks_response.json()
                entry_data['picks'] = picks_data.get('picks', [])

            # Fetch chip usage
            chips_url = f"{self.base_url}/entry/{team_id}/history/"
            chips_response = requests.get(chips_url, timeout=10, verify=False)
            chips_response.raise_for_status()
            history_data = chips_response.json()
            entry_data['chips'] = history_data.get('chips', [])
            
            st.session_state.my_team_loaded = True
            return entry_data
        except requests.RequestException as e:
            st.error(f"Failed to load team data for ID {team_id}: {e}")
            return None

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
            st.metric("👥 Total Players", len(df))
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
                display_cols = ['web_name', 'total_points']
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
                # Enhanced Best Value: Filter for players with significant minutes
                minutes_column = 'minutes' if 'minutes' in df.columns else 'total_points'
                
                # Use a more robust filter for value players
                value_df = df[df[minutes_column] > 0].nlargest(15, 'points_per_million')

                st.dataframe(
                    value_df[['web_name', 'team_short_name', 'position_name', 'cost_millions', 'total_points', 'points_per_million', 'form', 'selected_by_percent']],
                    column_config={
                        "web_name": "Player",
                        "team_short_name": "Team",
                        "position_name": "Pos",
                        "cost_millions": "Cost (£m)",
                        "total_points": "P",
                        "points_per_million": "PPM",
                        "form": "Form",
                        "selected_by_percent": "Own %"
                    },
                    use_container_width=True
                )
            else:
                st.warning("Best value data not available.")
        
        with tab3:
            if len(df) > 0 and 'form' in df.columns:
                # Enhanced Form Players: Add ownership and PPM
                form_df = df.nlargest(15, 'form')
                st.dataframe(
                    form_df[['web_name', 'team_short_name', 'position_name', 'cost_millions', 'total_points', 'form', 'points_per_million', 'selected_by_percent']],
                    column_config={
                        "web_name": "Player",
                        "team_short_name": "Team",
                        "position_name": "Pos",
                        "cost_millions": "Cost (£m)",
                        "total_points": "P",
                        "form": "Form",
                        "points_per_million": "PPM",
                        "selected_by_percent": "Own %"
                    },
                    use_container_width=True
                )
            else:
                st.warning("Form data not available.")

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
            st.subheader("💡 Team Insights")
            
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
        st.info("🚧 Performance trends analysis coming soon! This will include:")
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
                                        st.write(f"🔥 **Form**: {player['form']:.1f}")
                                    
                                    with col2:
                                        st.write(f"👥 **Ownership**: {player['ownership']:.1f}%")
                                        st.write(f"💎 **Value**: {player['value']:.1f} pts/£m")
                                        st.write(f"🤖 **AI Score**: {player['ai_score']:.1f}")
                
        with tab2:
            st.subheader("📈 Enhanced Form Analysis")
            
            # Form-based recommendations with enhanced insights
            form_analysis = self._analyze_player_form(df)
            
            # Display form statistics overview
            if form_analysis.get('form_stats'):
                stats = form_analysis['form_stats']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Form", f"{stats['avg_form']:.2f}")
                with col2:
                    st.metric("Hot Form Players", stats['high_form_count'])
                with col3:
                    st.metric("Cold Form Players", stats['low_form_count'])
                with col4:
                    st.metric("Form Volatility", f"{stats['form_volatility']:.2f}")
                
                st.divider()
            
            # Enhanced form tabs
            form_tab1, form_tab2, form_tab3 = st.tabs([
                "🔥 Hot Form Players", 
                "❄️ Cold Form Players", 
                "📈 Form Trends"
            ])
            
            with form_tab1:
                st.write("**🔥 Players in Excellent Form (6.0+ Rating)**")
                
                if form_analysis.get('hot_form'):
                    for i, player in enumerate(form_analysis['hot_form'][:8], 1):
                        with st.expander(f"#{i} {player['name']} ({player['team']}) - Form: {player['form']:.1f}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"• Form: {player['form']:.1f}")
                                st.write(f"• Total Points: {player['total_points']}")
                                st.write(f"• Position: {player['position']}")
                                st.write(f"• Minutes: {player['minutes']}")
                            
                            with col2:
                                st.write("**💰 Value Analysis**")
                                st.write(f"• Price: £{player['price']:.1f}m")
                                st.write(f"• Value Score: {player['value_score']:.1f}")
                                st.write(f"• Ownership: {player['ownership']:.1f}%")
                                
                                # Ownership category
                                if player['ownership'] < 5:
                                    st.write("• 💎 **Differential**")
                                elif player['ownership'] < 15:
                                    st.write("• 🎯 **Rising Star**")
                                elif player['ownership'] < 30:
                                    st.write("• 👥 **Popular Pick**")
                                else:
                                    st.write("• 🔥 **Template Player**")
                            
                            with col3:
                                st.write("**💡 FPL Recommendation**")
                                st.info(player['recommendation'])
                                
                                # Transfer urgency
                                if player['form'] >= 8.0 and player['ownership'] < 15:
                                    st.success("🚨 **URGENT**: Transfer in before price rise!")
                                elif player['form'] >= 7.0:
                                    st.success("⏰ **HIGH PRIORITY**: Consider this week")
                                else:
                                    st.info("👀 **MONITOR**: Keep watching")
                else:
                    st.info("No players currently meet the hot form criteria")
            
            with form_tab2:
                st.write("**❄️ Players in Poor Form (4.0- Rating)**")
                
                if form_analysis.get('cold_form'):
                    for i, player in enumerate(form_analysis['cold_form'][:8], 1):
                        with st.expander(f"#{i} {player['name']} ({player['team']}) - Form: {player['form']:.1f}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**📉 Poor Performance**")
                                st.write(f"• Form: {player['form']:.1f}")
                                st.write(f"• Total Points: {player['total_points']}")
                                st.write(f"• Position: {player['position']}")
                                st.write(f"• Minutes: {player['minutes']}")
                            
                            with col2:
                                st.write("**💸 Value Concern**")
                                st.write(f"• Price: £{player['price']:.1f}m")
                                st.write(f"• Value Score: {player['value_score']:.1f}")
                                st.write(f"• Ownership: {player['ownership']:.1f}%")
                                
                                # Risk assessment
                                if player['price'] >= 10:
                                    st.write("• ⚠️ **Premium Risk**")
                                elif player['ownership'] >= 30:
                                    st.write("• 🚨 **Template Trap**")
                                else:
                                    st.write("• 📉 **Underperformer**")
                            
                            with col3:
                                st.write("**⚠️ Transfer Advice**")
                                st.warning(player['recommendation'])
                                
                                # Transfer urgency for selling
                                if player['form'] <= 2.0 and player['price'] >= 8:
                                    st.error("🚨 **URGENT**: Transfer out immediately!")
                                elif player['form'] <= 3.0:
                                    st.warning("⏰ **HIGH PRIORITY**: Plan transfer out")
                                else:
                                    st.info("👀 **MONITOR**: Watch for improvement")
                else:
                    st.info("No players currently meet the cold form criteria")
            
            with form_tab3:
                st.write("**📈 Significant Form Trends & Pattern Analysis**")
                
                if form_analysis.get('form_trends'):
                    for trend in form_analysis['form_trends'][:10]:
                        if trend['trend'] == "📈 Hot Streak":
                            st.success(f"**{trend['name']}** ({trend['team']}) - {trend['trend']}")
                        elif trend['trend'] == "📉 Poor Patch":
                            st.error(f"**{trend['name']}** ({trend['team']}) - {trend['trend']}")
                        else:
                            st.info(f"**{trend['name']}** ({trend['team']}) - {trend['trend']}")
                        
                        st.caption(f"Current: {trend['form']:.1f} | Season Avg: {trend['season_avg']:.1f} | £{trend['price']:.1f}m | {trend['ownership']:.1f}% owned")
                        st.divider()
                else:
                    st.info("No significant form trends detected")
            
            # Form-based actionable insights
            st.subheader("💡 This Week's Form-Based Actions")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                st.write("**🎯 Immediate Transfer Targets**")
                urgent_targets = [p for p in form_analysis.get('hot_form', [])[:5] 
                                if p['form'] >= 7.5 and p['ownership'] < 20]
                
                if urgent_targets:
                    for player in urgent_targets[:3]:
                        st.success(f"✅ **{player['name']}** - {player['form']:.1f} form, {player['ownership']:.1f}% owned")
                else:
                    st.info("No urgent targets identified")
            
            with action_col2:
                st.write("**⚠️ Consider Transferring Out**")
                urgent_sells = [p for p in form_analysis.get('cold_form', [])[:5] 
                              if p['form'] <= 3.0 and p['price'] >= 7]
                
                if urgent_sells:
                    for player in urgent_sells[:3]:
                        st.warning(f"❌ **{player['name']}** - {player['form']:.1f} form, £{player['price']:.1f}m")
                else:
                    st.success("No urgent transfers out needed")
        
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
        if st.button("⚡ Generate Optimized Team", type="primary"):
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
                        budget=int(budget),
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
        
        # Handle is_starting column safely - create default values if it doesn't exist
        if 'is_starting' not in display_df.columns:
            # Assume first 11 players are starting (based on typical FPL setup)
            display_df['is_starting'] = display_df.index < 11
        
        display_df['Status'] = display_df['is_starting'].apply(
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
                    st.warning(f"⚠️ **{insight['team']}** - {insight['reason']}")

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
                if value_players:
                    st.write("**💎 Top 5 Value Player Bets**")
                    for player in value_players[:5]:
                        with st.expander(f"💎 {player['name']} (£{player['price']:.1f}m) - Value: {player['value_score']:.1f}"):
                            st.write(f"**{player['recommendation']}**")
                            st.write(f"Form: {player['form']:.1f} | Ownership: {player['ownership']:.1f}% | Points: {player['points']}")
        
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
        return value_players
    
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

    def _render_fdr_overview(self, fixtures_df, fdr_visualizer, gameweeks_ahead, sort_by, ascending_sort, analysis_type="Combined"):
        """Enhanced FDR overview with better metrics and insights"""
        
        if fixtures_df.empty:
            st.warning("No fixture data available")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_fixtures = len(fixtures_df)
            teams_count = fixtures_df['team_short_name'].nunique()
        
        with col2:
            if 'attack_fdr' in fixtures_df.columns:
                avg_attack_fdr = fixtures_df['attack_fdr'].mean()
                st.metric("⚔️ Avg Attack FDR", f"{avg_attack_fdr:.2f}")
        
        with col3:
            if 'defense_fdr' in fixtures_df.columns:
                avg_defense_fdr = fixtures_df['defense_fdr'].mean()
                st.metric("🛡️ Avg Defense FDR", f"{avg_defense_fdr:.2f}")
        
        with col4:
            if 'combined_fdr' in fixtures_df.columns:
                avg_combined_fdr = fixtures_df['combined_fdr'].mean()
                st.metric("🎯 Avg Combined FDR", f"{avg_combined_fdr:.2f}")
        
        # Enhanced FDR Heatmap
        if 'combined_fdr' in fixtures_df.columns:
            st.subheader("🌡️ FDR Heatmap")
            fig_heatmap = fdr_visualizer.create_fdr_heatmap(fixtures_df, 'combined')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Team summary table
        st.subheader("📋 Team FDR Summary")
        if not fixtures_df.empty:
            # Create team summary
            team_summary = fixtures_df.groupby('team_short_name').agg({
                'combined_fdr': 'mean',
                'attack_fdr': 'mean',
                'defense_fdr': 'mean'
            }).round(2)
            
            # Sort based on user preference
            if sort_by == "Combined FDR":
                team_summary = team_summary.sort_values('combined_fdr', ascending=ascending_sort)
            elif sort_by == "Attack FDR":
                team_summary = team_summary.sort_values('attack_fdr', ascending=ascending_sort)
            elif sort_by == "Defense FDR":
                team_summary = team_summary.sort_values('defense_fdr', ascending=ascending_sort)
            else:  # Alphabetical
                team_summary = team_summary.sort_index(ascending=ascending_sort)
            
            st.dataframe(team_summary, use_container_width=True)

    def _render_attack_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents):
        """Enhanced attack FDR analysis"""
        st.subheader("⚔️ Attack FDR Analysis")
        st.info("🎯 Lower Attack FDR = Easier to score goals. Target these teams' forwards and attacking midfielders!")
        
        if fixtures_df.empty or 'attack_fdr' not in fixtures_df.columns:
            st.warning("Attack FDR data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Best Attacking Fixtures")
            attack_summary = fixtures_df.groupby('team_short_name').agg({
                'attack_fdr': ['mean', 'min', 'count']
            }).round(2)
            
            attack_summary.columns = ['Avg_FDR', 'Best_FDR', 'Fixtures']
            attack_summary = attack_summary.sort_values('Avg_FDR').head(10)
            
            for idx, (team, row) in enumerate(attack_summary.iterrows()):
                avg_fdr = row['Avg_FDR']
                color = "🟢" if avg_fdr <= 2 else "🟡" if avg_fdr <= 2.5 else "🔴"
                
                st.write(f"{color} **{team}**: {avg_fdr:.2f} FDR")
        
        with col2:
            # Attack FDR heatmap
            fig_attack = fdr_visualizer.create_fdr_heatmap(fixtures_df, 'attack')
            st.plotly_chart(fig_attack, use_container_width=True)

    def _render_defense_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents):
        """Enhanced defense FDR analysis"""
        st.subheader("🛡️ Defense FDR Analysis")
        st.info("🛡️ Lower Defense FDR = Easier to keep clean sheets. Target these teams' defenders and goalkeepers!")
        
        if fixtures_df.empty or 'defense_fdr' not in fixtures_df.columns:
            st.warning("Defense FDR data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Best Defensive Fixtures")
            defense_summary = fixtures_df.groupby('team_short_name').agg({
                'defense_fdr': ['mean', 'min', 'count']
            }).round(2)
            
            defense_summary.columns = ['Avg_FDR', 'Best_FDR', 'Fixtures']
            defense_summary = defense_summary.sort_values('Avg_FDR').head(10)
            
            for idx, (team, row) in enumerate(defense_summary.iterrows()):
                avg_fdr = row['Avg_FDR']
                color = "🟢" if avg_fdr <= 2 else "🟡" if avg_fdr <= 2.5 else "🔴"
                
                st.write(f"{color} **{team}**: {avg_fdr:.2f} FDR")
        
        with col2:
            # Defense FDR heatmap
            fig_defense = fdr_visualizer.create_fdr_heatmap(fixtures_df, 'defense')
            st.plotly_chart(fig_defense, use_container_width=True)

    def _render_transfer_targets(self, fixtures_df, fdr_threshold):
        """Enhanced transfer targets based on fixture analysis"""
        st.subheader("🎯 Transfer Targets")
        
        if fixtures_df.empty:
            st.warning("No fixture data available")
            return
        
        # Find teams with good fixtures
        good_fixtures = fixtures_df[fixtures_df['combined_fdr'] <= fdr_threshold]
        
        if not good_fixtures.empty:
            target_teams = good_fixtures.groupby('team_short_name')['combined_fdr'].mean().sort_values()
            
            st.write("**🎯 Teams with Best Fixtures:**")
            for team, avg_fdr in target_teams.head(8).items():
                st.write(f" - **{team}**: Average FDR of {avg_fdr:.2f}")
        else:
            st.info("No teams meet the 'good fixture' threshold.")

    def _display_squad_analysis_enhanced(self, team_data):
        """Enhanced squad analysis with detailed insights"""
        st.subheader("👥 Squad Analysis")
        
        # Player details
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
            
        squad_ids = [pick['element'] for pick in picks]
        squad_df = st.session_state.players_df[st.session_state.players_df['id'].isin(squad_ids)]
        
        if squad_df.empty:
            st.warning("Could not retrieve squad player details.")
            return
            
        # Squad stats
        squad_stats = {
            'total_cost': squad_df['cost_millions'].sum(),
            'avg_form': squad_df.get('form', pd.Series(0)).mean(),
            'form_players': squad_df[squad_df.get('form', 0) >= 6.0].shape[0],
            'avg_ownership': squad_df.get('selected_by_percent', pd.Series(0)).mean()
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost", f"£{squad_stats['total_cost']:.1f}m")
        with col2:
            st.metric("Avg Form", f"{squad_stats['avg_form']:.1f}")
        with col3:
            st.metric("In-Form Players", squad_stats['form_players'])
        with col4:
            st.metric("Avg Ownership", f"{squad_stats['avg_ownership']:.1f}%")
            
        # Squad insights
        st.write("**💡 Squad Insights**")
        insights = []
        if squad_stats['avg_form'] >= 5.5:
            insights.append("🔥 Squad in excellent form")
        elif squad_stats['form_players'] <= 4:
            insights.append("📉 Poor squad form - transfers needed")
        
        for insight in insights:
            st.write(f"• {insight}")

    def _display_performance_benchmarking_enhanced(self, team_data):
        """Enhanced performance benchmarking"""
        
        # Current performance metrics
        total_points = team_data.get('summary_overall_points', 0)
        overall_rank = team_data.get('summary_overall_rank', 0)
        gw_points = team_data.get('summary_event_points', 0)
        gw_rank = team_data.get('summary_event_rank', 0)
        
        # Calculate benchmarks
        total_managers = 8000000  # Approximate
        percentile = (1 - (overall_rank / total_managers)) * 100 if overall_rank else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🏆 Season Performance**")
            st.metric("Total Points", f"{total_points:,}")
            st.metric("Overall Rank", f"{overall_rank:,}" if overall_rank else "N/A")
            st.metric("Percentile", f"{percentile:.1f}%")
            
        with col2:
            st.write("**📈 Recent Form**")
            st.metric("GW Points", gw_points)
            st.metric("GW Rank", f"{gw_rank:,}" if gw_rank else "N/A")
            
            # Performance vs average
            league_avg = 50
            diff = gw_points - league_avg
            st.metric("vs Average", f"{diff:+d} points")
            
        with col3:
            st.write("**💰 Financial Status**")
            team_value = team_data.get('value', 1000) / 10
            bank = team_data.get('bank', 0) / 10
            
            st.metric("Team Value", f"£{team_value:.1f}m")
            st.metric("In Bank", f"£{bank:.1f}m")
            st.metric("Total Budget", f"£{team_value + bank:.1f}m")
        
        # Performance insights
        st.write("**💡 Performance Insights**")
        
        if percentile >= 90:
            st.success("🏆 Elite performance! You're in the top 10%")
        elif percentile >= 75:
            st.success("🥇 Excellent performance! Top 25%")
        elif percentile >= 50:
            st.info("👍 Above average performance")
        elif percentile >= 25:
            st.warning("📈 Room for improvement")
        else:
            st.error("🔧 Major improvements needed")
        
        # Benchmarking vs league averages
        
        benchmarks = {
            "Elite (Top 1%)": {"points": 1200, "rank": 80000},
            "Excellent (Top 10%)": {"points": 900, "rank": 800000},
            "Good (Top 25%)": {"points": 750, "rank": 2000000},
            "Average (Top 50%)": {"points": 600, "rank": 4000000}
        }
        
        for tier, targets in benchmarks.items():
            points_diff = total_points - targets["points"]
            rank_diff = overall_rank - targets["rank"] if overall_rank else 0
            
            if points_diff >= 0 and rank_diff <= 0:
                st.success(f"✅ **{tier}**: Target achieved!")
            else:
                points_needed = max(0, targets["points"] - total_points)
                st.info(f"🎯 **{tier}**: Need {points_needed} more points")

    def _generate_transfer_recommendations(self, squad_df, players_df, fixtures_df):
        """Generate transfer recommendations based on a combination of factors."""
        squad_ids = squad_df['id'].tolist()

        # 1. Identify players to transfer out
        sell_candidates = []
        for _, player in squad_df.iterrows():
            score = 0
            reasons = []
            
            # Penalize for bad form
            if player.get('form', 5) < 2.5:
                score += (5 - player.get('form', 5)) * 2
                reasons.append(f"Poor form ({player.get('form', 0):.1f})")

            # Penalize for difficult fixtures
            if not fixtures_df.empty:
                player_team_id = player['team']
                team_fixtures = fixtures_df[fixtures_df['team_id'] == player_team_id]
                if not team_fixtures.empty:
                    avg_fdr = team_fixtures['combined_fdr'].mean()
                    if avg_fdr > 3.5:
                        score += (avg_fdr - 3.5) * 5
                        reasons.append(f"Tough fixtures (FDR: {avg_fdr:.2f})")
            
            if score > 5:
                sell_candidates.append({
                    'player': player,
                    'score': score,
                    'reasons': ", ".join(reasons)
                })

        # 2. Identify players to transfer in
        buy_candidates = players_df[~players_df['id'].isin(squad_ids)].copy()
        
        target_scores = []
        for _, player in buy_candidates.iterrows():
            score = 0
            
            # Reward for good form
            score += player.get('form', 0) * 1.5
            
            # Reward for good fixtures
            if not fixtures_df.empty:
                player_team_id = player['team']
                team_fixtures = fixtures_df[fixtures_df['team_id'] == player_team_id]
                if not team_fixtures.empty:
                    avg_fdr = team_fixtures['combined_fdr'].mean()
                    if avg_fdr < 2.5:
                        score += (2.5 - avg_fdr) * 5
            
            # Reward for value (points per million)
            score += player.get('points_per_million', 0) * 2

            if score > 15:
                 target_scores.append({'player': player, 'score': score})

        # Sort candidates
        sorted_sells = sorted(sell_candidates, key=lambda x: x['score'], reverse=True)
        sorted_buys = sorted(target_scores, key=lambda x: x['score'], reverse=True)

        return sorted_sells, sorted_buys

    def _display_transfer_intelligence(self, team_data):
        """Display transfer intelligence and recommendations"""
        st.subheader("🔄 Transfer Intelligence")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see transfer recommendations")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        fixtures_df = st.session_state.get('fixtures_df', pd.DataFrame())
        
        # Get current squad - use safer column access
        try:
            squad_ids = [pick['element'] for pick in picks]
            
            # Check if 'id' column exists, if not try other common ID columns
            if 'id' in players_df.columns:
                id_column = 'id'
            elif 'element' in players_df.columns:
                id_column = 'element'
            else:
                st.error("Could not find player ID column in data")
                return
            
            squad_df = players_df[players_df[id_column].isin(squad_ids)]
        except KeyError as e:
            st.error(f"Error accessing player data: {e}")
            return
        
        # Generate transfer recommendations
        sorted_sells, sorted_buys = self._generate_transfer_recommendations(squad_df, players_df, fixtures_df)
        
        # Transfer analysis tabs
        transfer_tab1, transfer_tab2, transfer_tab3 = st.tabs([
            "🔥 Priority Transfers",
            "💎 Value Targets", 
            "📈 Form Players"
        ])
        
        with transfer_tab1:
            st.write("**⚠️ Players to Consider Selling**")
            
            if sorted_sells:
                for candidate in sorted_sells[:5]:
                    player = candidate['player']
                    st.warning(f"🔴 **{player['web_name']}** (£{player.get('now_cost', 0)/10:.1f}m) - {candidate['reasons']}")
            else:
                st.success("✅ No obvious players to sell!")
        
        with transfer_tab2:
            st.write("**💰 Best Value Targets**")
            
            if sorted_buys:
                for candidate in sorted_buys[:5]:
                    player = candidate['player']
                    st.success(f"💎 **{player['web_name']}** (£{player.get('now_cost', 0)/10:.1f}m) - Score: {candidate['score']:.1f}")
        
        with transfer_tab3:
            st.write("**🔥 In-Form Players**")
            
            # Find form players not in squad
            available_players = players_df[~players_df['id'].isin(squad_ids)]
            
            form_targets = available_players[
                (available_players.get('form', 0) >= 6) &
                (available_players['total_points'] > 30)
            ].nlargest(5, 'form')
            
            for _, player in form_targets.iterrows():
                st.success(f"🔥 **{player['web_name']}** (£{player.get('now_cost', 0)/10:.1f}m) - Form: {player.get('form', 0):.1f}")

    def _display_strategy_planning(self, team_data):
        """Display strategy and planning recommendations"""
        st.subheader("🎯 Strategy & Planning")
        
        # Strategy tabs
        strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
            "📅 Short-term (1-3 GWs)",
            "🗓️ Medium-term (4-8 GWs)", 
            "🏆 Long-term (Season)"
        ])
        
        with strategy_tab1:
            st.write("**⚡ Immediate Priorities**")
            
            priorities = [
                "🎯 **Captain Selection**: Choose between premium options",
                "🪑 **Bench Order**: Optimize bench for potential returns",
                "🏥 **Injury Watch**: Monitor player fitness updates",
                "📰 **Team News**: Stay updated on lineup changes",
                "🔄 **Transfer Timing**: Plan moves around price changes"
            ]
            
            for priority in priorities:
                st.write(f"• {priority}")
        
        with strategy_tab2:
            st.write("**📈 Medium-term Strategy**")
            st.info("🔍 Planning fixture swings and player trends...")
            
            medium_goals = [
                "🎲 **Fixture Analysis**: Target teams with improving fixtures",
                "💰 **Price Movements**: Monitor and predict price changes",
                "🔄 **Transfer Planning**: Plan 2-3 move sequences",
                "🎯 **Template Moves**: Balance following vs differentials"
            ]
            
            for goal in medium_goals:
                st.write(f"• {goal}")
        
        with strategy_tab3:
            st.write("🏆 Season-Long Strategy")
            st.info("🎖️ Building towards end-of-season success...")
            
            term_strategies = [
                "🎪 **Chip Strategy**: Plan optimal timing for all chips",
                "📈 **Rank Targets**: Set and track ranking goals",
                "⚖️ **Risk Management**: Balance safety vs differentials",
                "🏆 **End Game**: Prepare for final gameweeks push",
            ]
            
            for strategy in term_strategies:
                st.markdown(f"- {strategy}")
        

    def _display_chip_strategy_enhanced(self, team_data):
        """Enhanced chip strategy analysis"""
        used_chips = [chip['name'] for chip in team_data.get('chips', [])]
        available_chips = []
        if 'wildcard' not in [chip.lower() for chip in used_chips]:
            available_chips.append('Wildcard')
        if 'bench_boost' not in [chip.lower() for chip in used_chips]:
            available_chips.append('Bench Boost')
        if 'triple_captain' not in [chip.lower() for chip in used_chips]:
            available_chips.append('Triple Captain')
        if 'free_hit' not in [chip.lower() for chip in used_chips]:
            available_chips.append('Free Hit')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Chips Used**")
            if used_chips:
                for chip in used_chips:
                    st.write(f"• {chip}")
            else:
                st.write("• None used yet")
        
        with col2:
            st.write("**💎 Chips Available**")
            if available_chips:
                for chip in available_chips:
                    st.write(f"• {chip}")
            else:
                st.write("• All chips used")
        
        # Chip recommendations
        if available_chips:
            st.write("**🎯 Chip Usage Recommendations**")
            
            for chip in available_chips:
                with st.expander(f"💎 {chip} Strategy"):
                    if chip == 'Wildcard':
                        st.write("**Optimal Timing:**")
                        st.write("• During international breaks")
                        st.write("• When 4+ players need changing")
                        st.write("• Before fixture swings")
                        
                    elif chip == 'Bench Boost':
                        st.write("**Optimal Timing:**")
                        st.write("• Double gameweeks")
                        st.write("• When bench has good fixtures")
                        st.write("• Late in season for rank pushes")
                        
                    elif chip == 'Triple Captain':
                        st.write("**Optimal Timing:**")
                        st.write("• Double gameweeks")
                        st.write("• Haaland vs weak opposition")
                        st.write("• Salah at home vs poor defense")
                        
                    elif chip == 'Free Hit':
                        st.write("**Optimal Timing:**")
                        st.write("• Blank gameweeks")
                        st.write("• When many players don't play")
                        st.write("• Cup final gameweeks")

    def _generate_swot_analysis(self, team_data, players_df, fixtures_df):
        """Generate a dynamic SWOT analysis for the user's team"""
        squad_ids = [pick['element'] for pick in team_data.get('picks', [])]
        squad_df = players_df[players_df['id'].isin(squad_ids)]

        if squad_df.empty:
            return [], [], [], []

        # Strengths
        strengths = []
        in_form_players = squad_df[squad_df.get('form', 0) > 5].shape[0]
        if in_form_players >= 5:
            strengths.append(f"Strong form with {in_form_players} players performing well.")
        
        premium_players = squad_df[squad_df['cost_millions'] >= 10].shape[0]
        if premium_players >= 2:
            strengths.append(f"Solid premium core with {premium_players} high-value assets.")

        # Weaknesses
        weaknesses = []
        out_of_form_players = squad_df[squad_df.get('form', 10) < 3].shape[0]
        if out_of_form_players > 2:
            weaknesses.append(f"Significant weakness with {out_of_form_players} players out of form.")
        
        # Fixed: Remove the problematic bench value calculation that was causing the is_starting error
        # Since 'is_starting' doesn't exist in FPL API picks data, we'll skip this calculation
        # bench_value = squad_df[~squad_df['id'].isin([p['element'] for p in team_data.get('picks', []) if p.get('is_starting', False)])]['cost_millions'].sum()
        # if bench_value > 20:
        #     weaknesses.append(f"Potentially expensive bench (£{bench_value:.1f}m), tying up funds.")

        # Opportunities
        opportunities = []
        if not fixtures_df.empty:
            squad_fixtures = fixtures_df[fixtures_df['team_id'].isin(squad_df['team'].unique())]
            if not squad_fixtures.empty:
                avg_fdr = squad_fixtures.groupby('team_name')['combined_fdr'].mean().nsmallest(3)
                for team, fdr in avg_fdr.items():
                    opportunities.append(f"Excellent upcoming fixtures for {team} (Avg FDR: {fdr:.2f}).")

        # Threats
        threats = []
        if not fixtures_df.empty:
            squad_fixtures = fixtures_df[fixtures_df['team_id'].isin(squad_df['team'].unique())]
            if not squad_fixtures.empty:
                bad_fdr = squad_fixtures.groupby('team_name')['combined_fdr'].mean().nlargest(3)
                for team, fdr in bad_fdr.items():
                    if fdr > 3.5:
                        threats.append(f"Tough fixture run for {team} (Avg FDR: {fdr:.2f}).")
        
        return strengths, weaknesses, opportunities, threats

    def _display_swot_analysis_enhanced(self, team_data):
        """Enhanced SWOT analysis display"""
        st.subheader("📈 SWOT Analysis")

        if 'fixtures_df' not in st.session_state or st.session_state.fixtures_df.empty:
            st.warning("Load Fixture Data from the 'Fixture Difficulty' tab for a full SWOT analysis.")
        
        strengths, weaknesses, opportunities, threats = self._generate_swot_analysis(
            team_data, 
            st.session_state.players_df,
            st.session_state.get('fixtures_df', pd.DataFrame())
        )
        
        # SWOT quadrants
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**💪 Strengths**")
            if strengths:
                for item in strengths:
                    st.write(f"• {item}")
            else:
                st.write("• No significant strengths identified.")

            st.error("**⚠️ Weaknesses**")
            if weaknesses:
                for item in weaknesses:
                    st.write(f"• {item}")
            else:
                st.write("• No significant weaknesses identified.")
        
        with col2:
            st.info("**🎯 Opportunities**")
            if opportunities:
                for item in opportunities:
                    st.write(f"• {item}")
            else:
                st.write("• No clear opportunities from fixture analysis.")

            st.warning("**⚡ Threats**")
            if threats:
                for item in threats:
                    st.write(f"• {item}")
            else:
                st.write("• No immediate threats from fixture analysis.")

    def _generate_ai_targets(self, df, price_range, positions, min_form, max_ownership):
        """Generate AI-powered player targets"""
        targets = {}
        
        # Position mapping
        position_map = {
            "Goalkeeper": 1,
            "Defender": 2, 
            "Midfielder": 3,
            "Forward": 4
        }
        
        for position in positions:
            if position in position_map:
                position_id = position_map[position]
                
                # Filter players
                filtered = df[
                    (df['element_type'] == position_id) &
                    (df['cost_millions'] >= price_range[0]) &
                    (df['cost_millions'] <= price_range[1]) &
                    (df.get('form', 0) >= min_form) &
                    (df.get('selected_by_percent', 0) <= max_ownership) &
                    (df['total_points'] > 20)
                ]
                
                # Calculate AI score
                if not filtered.empty:
                    filtered = filtered.copy()
                    filtered['ai_score'] = (
                        filtered.get('form', 0) * 2 +
                        filtered['total_points'] / 10 +
                        (100 - filtered.get('selected_by_percent', 50)) / 20 +
                        filtered.get('points_per_million', 0) / 2
                    )
                    
                    # Top players for this position
                    top_players = filtered.nlargest(5, 'ai_score')
                    
                    targets[position] = []
                    for _, player in top_players.iterrows():
                        targets[position].append({
                            'name': player['web_name'],
                            'team': player.get('team_short_name', 'Unknown'),
                            'price': player['cost_millions'],
                            'points': player['total_points'],
                            'form': player.get('form', 0),
                            'ownership': player.get('selected_by_percent', 0),
                            'value': player.get('points_per_million', 0),
                            'ai_score': player['ai_score']
                        })
        
        return targets

    def _analyze_player_form(self, df):
        """Enhanced player form analysis with comprehensive insights"""
        if 'form' not in df.columns:
            return {'hot_form': [], 'cold_form': [], 'form_trends': [], 'form_stats': {}}
        
        # Filter for active players with meaningful data
        active_players = df[
            (df['total_points'] > 20) & 
            (df.get('minutes', 0) > 200)  # Ensure they're getting game time
        ].copy()
        
        if active_players.empty:
            return {'hot_form': [], 'cold_form': [], 'form_trends': [], 'form_stats': {}}
        
        # Enhanced hot form analysis (trending upward)
        hot_form = active_players[
            (active_players['form'] >= 6.0) & 
            (active_players['total_points'] > 30) &
            (active_players.get('selected_by_percent', 100) <= 40)  # Not too template
        ].nlargest(15, 'form')
        
        # Enhanced cold form analysis (trending downward)
        cold_form = active_players[
            (active_players['form'] <= 4.0) & 
            (active_players['total_points'] > 40) &  # Established players only
            (active_players['cost_millions'] >= 6.0)  # Expensive disappointments
        ].nsmallest(15, 'form')
        
        # Form trend analysis (form vs season average)
        form_trends = []
        for _, player in active_players.iterrows():
            season_avg = player['total_points'] / 38 if player['total_points'] > 0 else 0
            current_form = player.get('form', 0)
            
            if current_form > season_avg * 1.5:  # Significantly outperforming
                trend = "📈 Hot Streak"
                trend_score = (current_form / max(season_avg, 1)) * 10
            elif current_form < season_avg * 0.6:  # Significantly underperforming
                trend = "📉 Poor Patch"
                trend_score = -(season_avg / max(current_form, 0.1)) * 5
            else:
                trend = "➡️ Stable"
                trend_score = 0
            
            if abs(trend_score) > 3:  # Only significant trends
                form_trends.append({
                    'name': player['web_name'],
                    'team': player.get('team_short_name', 'N/A'),
                    'form': current_form,
                    'season_avg': season_avg,
                    'trend': trend,
                    'trend_score': trend_score,
                    'price': player['cost_millions'],
                    'ownership': player.get('selected_by_percent', 0),
                    'minutes': player.get('minutes', 0)
                })
        
        # Sort form trends by significance
        form_trends.sort(key=lambda x: abs(x['trend_score']), reverse=True)
        
        # Calculate form statistics
        form_stats = {
            'avg_form': active_players['form'].mean(),
            'high_form_count': len(hot_form),
            'low_form_count': len(cold_form),
            'form_volatility': active_players['form'].std(),
            'top_form_player': active_players.loc[active_players['form'].idxmax(), 'web_name'] if not active_players.empty else 'N/A',
            'worst_form_player': active_players.loc[active_players['form'].idxmin(), 'web_name'] if not active_players.empty else 'N/A'
        }
        
        return {
            'hot_form': [
                {
                    'name': row['web_name'],
                    'team': row.get('team_short_name', 'N/A'),
                    'form': row['form'],
                    'total_points': row['total_points'],
                    'price': row['cost_millions'],
                    'ownership': row.get('selected_by_percent', 0),
                    'value_score': row.get('points_per_million', 0),
                    'position': row.get('position_name', 'Unknown'),
                    'minutes': row.get('minutes', 0),
                    'recommendation': self._get_form_recommendation(row, 'hot')
                }
                for _, row in hot_form.iterrows()
            ],
            'cold_form': [
                {
                    'name': row['web_name'],
                    'team': row.get('team_short_name', 'N/A'),
                    'form': row['form'],
                    'total_points': row['total_points'],
                    'price': row['cost_millions'],
                    'ownership': row.get('selected_by_percent', 0),
                    'value_score': row.get('points_per_million', 0),
                    'position': row.get('position_name', 'Unknown'),
                    'minutes': row.get('minutes', 0),
                    'recommendation': self._get_form_recommendation(row, 'cold')
                }
                for _, row in cold_form.iterrows()
            ],
            'form_trends': form_trends[:10],  # Top 10 significant trends
            'form_stats': form_stats
        }
    
    def _get_form_recommendation(self, player, form_type):
        """Generate specific recommendations based on player form"""
        ownership = player.get('selected_by_percent', 0)
        price = player['cost_millions']
        
        if form_type == 'hot':
            if ownership < 10:
                return "🔥 **DIFFERENTIAL**: Excellent form and low ownership. High reward potential."
            elif ownership < 25:
                return "📈 **BANDWAGON**: Jumping on a popular, in-form player. Good, but not unique."
            else:
                return "✅ **TEMPLATE**: Essential player to own right now. Don't get left behind."
        else:  # cold form
            if price >= 10:
                return "🚨 **PREMIUM SELL**: Expensive player not delivering. Free up funds."
            elif ownership >= 30:
                return "📉 **TEMPLATE SELL**: High ownership but poor form. Good time to sell."
            else:
                return "👀 **MONITOR**: Keep watching, but avoid transferring in."

    def _generate_transfer_advice(self, df):
        """Generate transfer advice based on various factors"""
        # Filter active players
        active_players = df[df['total_points'] > 20].copy()
        
        if active_players.empty:
            return {'targets': [], 'sells': []}
        
        targets = []
        sells = []
        
        # Transfer targets (high form, good value)
        if 'form' in active_players.columns and 'points_per_million' in active_players.columns:
            good_targets = active_players[
                (active_players['form'] >= 7) &
                (active_players['points_per_million'] >= 8) &
                (active_players.get('selected_by_percent', 50) <= 30)
            ].head(5)
            
            for _, player in good_targets.iterrows():
                targets.append({
                    'player': player['web_name'],
                    'reason': f"High form ({player['form']:.1f}) and great value ({player['points_per_million']:.1f} pts/£m)"
                })
        
        # Players to sell (poor form, high ownership)
        if 'form' in active_players.columns:
            sell_candidates = active_players[
                (active_players['form'] <= 3) &
                (active_players['cost_millions'] >= 7) &
                (active_players.get('selected_by_percent', 0) >= 20)
            ].head(5)
            
            for _, player in sell_candidates.iterrows():
                sells.append({
                    'player': player['web_name'],
                    'reason': f"Poor form ({player['form']:.1f}) despite high price (£{player['cost_millions']:.1f}m)"
                })
        
        return {
            'targets': targets,
            'sells': sells
        }

# Main execution
if __name__ == "__main__":
    app = FPLAnalyticsApp()
    app.run()

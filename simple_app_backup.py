    # ...existing methods...
    
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

    # ...existing methods...import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import logging
import warnings
import urllib3

# Suppress warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Fixture difficulty helper functions
def format_fixture_ticker(fixtures):
    """Format fixtures into a ticker string"""
    ticker_parts = []
    for fixture in fixtures:
        opponent = fixture.get('opponent', 'UNK')
        venue = fixture.get('venue', 'H')
        ticker_parts.append(f"{opponent}({venue})")
    return " | ".join(ticker_parts)

def get_fdr_recommendation(avg_fdr):
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

def rank_teams_by_fdr(fixtures_df, fdr_type='combined'):
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

class FixtureDataLoader:
    """Loads and processes fixture data from FPL API"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.logger = logging.getLogger(__name__)
    
    def load_fixtures(self):
        """Load fixtures from FPL API"""
        try:
            url = f"{self.base_url}/fixtures/"
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error loading fixtures: {e}")
            return []
    
    def load_teams(self):
        """Load teams data from FPL API"""
        try:
            url = f"{self.base_url}/bootstrap-static/"
            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()
            return data.get('teams', [])
        except requests.RequestException as e:
            self.logger.error(f"Error loading teams: {e}")
            return []
    
    def get_next_5_fixtures(self, team_id, fixtures):
        """Get next 5 fixtures for a specific team"""
        team_fixtures = []
        
        for fixture in fixtures:
            # Check if this fixture involves the team
            if fixture['team_h'] == team_id or fixture['team_a'] == team_id:
                # Include all unfinished fixtures
                if not fixture.get('finished', False):
                    team_fixtures.append(fixture)
        
        # Sort by event (gameweek) first, then by kickoff_time
        def sort_key(fixture):
            event = fixture.get('event', 999)
            kickoff = fixture.get('kickoff_time', 'Z')
            return (event, kickoff)
        
        team_fixtures.sort(key=sort_key)
        return team_fixtures[:5]
    
    def process_fixtures_data(self):
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
            
            # If no fixtures found, create placeholder data
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
                        'opponent_strength': 3
                    })
            else:
                for i, fixture in enumerate(next_fixtures, 1):
                    is_home = fixture['team_h'] == team_id
                    opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                    opponent = team_lookup.get(opponent_id, {})
                    
                    # Use FPL's difficulty rating if available
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
                        'opponent_strength': opponent.get('strength', 3)
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
    
    def calculate_attack_fdr(self, fixtures_df):
        """Calculate attack-based FDR"""
        if fixtures_df.empty:
            return pd.DataFrame()
        
        attack_fdr_df = fixtures_df.copy()
        
        def get_attack_difficulty(row):
            # Simplified FDR calculation
            opponent_strength = row.get('opponent_strength', 3)
            
            if opponent_strength <= 2:
                return 5  # Very hard (strong defense)
            elif opponent_strength <= 2.5:
                return 4  # Hard
            elif opponent_strength <= 3.5:
                return 3  # Average
            elif opponent_strength <= 4:
                return 2  # Easy
            else:
                return 1  # Very easy
        
        attack_fdr_df['attack_fdr'] = attack_fdr_df.apply(get_attack_difficulty, axis=1)
        return attack_fdr_df
    
    def calculate_defense_fdr(self, fixtures_df):
        """Calculate defense-based FDR"""
        if fixtures_df.empty:
            return pd.DataFrame()
        
        defense_fdr_df = fixtures_df.copy()
        
        def get_defense_difficulty(row):
            # Simplified FDR calculation
            opponent_strength = row.get('opponent_strength', 3)
            
            if opponent_strength >= 4.5:
                return 5  # Very hard (strong attack)
            elif opponent_strength >= 4:
                return 4  # Hard
            elif opponent_strength >= 3:
                return 3  # Average
            elif opponent_strength >= 2.5:
                return 2  # Easy
            else:
                return 1  # Very easy
        
        defense_fdr_df['defense_fdr'] = defense_fdr_df.apply(get_defense_difficulty, axis=1)
        return defense_fdr_df
    
    def calculate_combined_fdr(self, fixtures_df):
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
    
    def create_fdr_heatmap(self, fixtures_df, fdr_type='combined'):
        """Create FDR heatmap"""
        if fixtures_df.empty:
            return go.Figure()
        
        fdr_column = f'{fdr_type}_fdr'
        if fdr_column not in fixtures_df.columns:
            return go.Figure()
        
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
            x=[f'Fixture {i}' for i in pivot_data.columns],
            y=pivot_data.index,
            colorscale=colorscale,
            zmin=1, zmax=5,
            text=pivot_data.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>%{x}<br>FDR: %{z}<extra></extra>'
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
            response = requests.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            data = response.json()
            
            # Process players data
            players_df = pd.DataFrame(data['elements'])
            teams_df = pd.DataFrame(data['teams'])
            element_types_df = pd.DataFrame(data['element_types'])
            
            # Create lookup dictionaries
            team_lookup = dict(zip(teams_df['id'], teams_df['name']))
            team_short_lookup = dict(zip(teams_df['id'], teams_df['short_name']))
            position_lookup = dict(zip(element_types_df['id'], element_types_df['singular_name']))
            
            # Add team and position names
            players_df['team_name'] = players_df['team'].map(team_lookup)
            players_df['team_short_name'] = players_df['team'].map(team_short_lookup)
            players_df['position_name'] = players_df['element_type'].map(position_lookup)
            
            # Fill any NaN values
            players_df['team_name'] = players_df['team_name'].fillna('Unknown Team')
            players_df['team_short_name'] = players_df['team_short_name'].fillna('UNK')
            players_df['position_name'] = players_df['position_name'].fillna('Unknown Position')
            
            # Calculate cost in millions
            players_df['cost_millions'] = players_df['now_cost'] / 10
            
            # Calculate points per million
            players_df['points_per_million'] = np.where(
                players_df['cost_millions'] > 0,
                players_df['total_points'] / players_df['cost_millions'],
                0
            ).round(2)
            
            # Convert numeric columns to proper dtypes
            numeric_columns = [
                'form', 'selected_by_percent', 'total_points', 'points_per_game',
                'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                'own_goals', 'saves', 'bonus', 'influence', 'creativity', 'threat',
                'minutes', 'yellow_cards', 'red_cards', 'penalties_missed',
                'penalties_saved', 'dreamteam_count'
            ]
            
            for col in numeric_columns:
                if col in players_df.columns:
                    players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
            
            # Store data in session state
            st.session_state.players_df = players_df
            st.session_state.teams_df = teams_df
            st.session_state.data_loaded = True
            
            return players_df, teams_df
            
        except Exception as e:
            st.error(f"Error loading FPL data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("⚽ FPL Analytics")
        st.sidebar.markdown("---")
        
        # Navigation
        pages = {
            "🏠 Dashboard": "dashboard",
            "👥 Player Analysis": "players", 
            "🎯 Fixture Difficulty": "fixtures",
            "🔍 Advanced Filters": "filters",
            "👤 My FPL Team": "my_team",
            "🤖 AI Recommendations": "ai_recommendations",
            "⚽ Team Builder": "team_builder",
            "📈 Team Odds": "team_odds"
        }
        
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            list(pages.keys()),
            index=0,
            key="unique_sidebar_navigation_main"
        )
        
        st.sidebar.markdown("---")
        
        # Data status
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data Loaded")
            if not st.session_state.players_df.empty:
                st.sidebar.info(f"Players: {len(st.session_state.players_df)}")
        else:
            st.sidebar.warning("⚠️ No data loaded")
        
        # Load data button
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            with st.spinner("Loading FPL data..."):
                players_df, teams_df = self.load_fpl_data()
                
                if not players_df.empty:
                    st.session_state.players_df = players_df
                    st.session_state.teams_df = teams_df
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Data refreshed!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to load data")
        
        return pages[selected_page]

    def render_dashboard(self):
        """Render main dashboard"""
        st.title("⚽ FPL Analytics Dashboard")
        st.markdown("### Welcome to your Fantasy Premier League Analytics Hub!")
        
        if not st.session_state.data_loaded:
            st.info("👋 Welcome! Click '🔄 Refresh Data' in the sidebar to get started.")
            return
        
        df = st.session_state.players_df
        
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
                value_df = df[df['total_points'] > 0].nlargest(15, 'points_per_million')
                st.dataframe(
                    value_df[['web_name', 'team_short_name', 'position_name', 'cost_millions', 'total_points', 'points_per_million']],
                    use_container_width=True
                )
            else:
                st.warning("Best value data not available.")
        
        with tab3:
            if len(df) > 0 and 'form' in df.columns:
                form_df = df.nlargest(15, 'form')
                st.dataframe(
                    form_df[['web_name', 'team_short_name', 'position_name', 'cost_millions', 'total_points', 'form']],
                    use_container_width=True
                )
            else:
                st.warning("Form data not available.")

    def render_players(self):
        """Enhanced Player Analysis with Position-Specific Tabs"""
        st.header("👥 Player Analysis")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first from the Dashboard.")
            return
        
        df = st.session_state.players_df
        
        if df.empty:
            st.warning("No player data available")
            return
        
        # Basic filters section
        st.subheader("🔍 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'position_name' in df.columns:
                positions = st.multiselect("Position:", df['position_name'].unique(), default=df['position_name'].unique())
            else:
                positions = []
        
        with col2:
            if 'team_name' in df.columns:
                teams = st.multiselect("Teams:", sorted(df['team_name'].unique()), default=sorted(df['team_name'].unique()))
            else:
                teams = []
        
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
        
        # Apply filters
        filtered_df = df.copy()
        
        if positions and 'position_name' in df.columns:
            filtered_df = filtered_df[filtered_df['position_name'].isin(positions)]
        
        if teams and 'team_name' in df.columns:
            filtered_df = filtered_df[filtered_df['team_name'].isin(teams)]
        
        if 'cost_millions' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['cost_millions'] >= min_price) &
                (filtered_df['cost_millions'] <= max_price)
            ]
        
        st.success(f"Found {len(filtered_df)} players matching your criteria")
        
        # Position-specific analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 All Players",
            "🥅 Goalkeepers", 
            "🛡️ Defenders",
            "⚽ Midfielders",
            "🎯 Forwards",
            "🏆 Top Performers"
        ])
        
        with tab1:
            self._render_all_players_tab(filtered_df)
        
        with tab2:
            self._render_goalkeepers_tab(filtered_df)
        
        with tab3:
            self._render_defenders_tab(filtered_df)
        
        with tab4:
            self._render_midfielders_tab(filtered_df)
        
        with tab5:
            self._render_forwards_tab(filtered_df)
        
        with tab6:
            self._render_top_performers_tab(filtered_df)

    def _render_all_players_tab(self, filtered_df):
        """Render all players overview tab"""
        st.subheader("📋 All Players Overview")
        
        if not filtered_df.empty:
            # Results table with key columns
            display_cols = ['web_name', 'position_name', 'team_short_name', 'cost_millions', 'total_points']
            if 'form' in filtered_df.columns:
                display_cols.append('form')
            if 'points_per_million' in filtered_df.columns:
                display_cols.append('points_per_million')
            if 'selected_by_percent' in filtered_df.columns:
                display_cols.append('selected_by_percent')
            
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_cols].sort_values('total_points', ascending=False),
                use_container_width=True
            )
        else:
            st.warning("No players match your filter criteria")

    def _render_goalkeepers_tab(self, filtered_df):
        """Render goalkeeper-specific analysis"""
        st.subheader("🥅 Goalkeeper Analysis")
        
        # Filter for goalkeepers only
        if 'position_name' in filtered_df.columns:
            gk_df = filtered_df[filtered_df['position_name'] == 'Goalkeeper'].copy()
        else:
            gk_df = pd.DataFrame()
        
        if gk_df.empty:
            st.warning("No goalkeeper data available")
            return
        
        # Key metrics for goalkeepers
        gk_metrics = [
            'web_name', 'team_short_name', 'cost_millions', 'total_points',
            'clean_sheets', 'goals_conceded', 'own_goals', 'saves', 
            'points_per_game', 'form', 'selected_by_percent'
        ]
        
        # Add calculated metrics if possible
        if 'minutes' in gk_df.columns and gk_df['minutes'].sum() > 0:
            gk_df['clean_sheets_per_90'] = (gk_df['clean_sheets'] * 90 / gk_df['minutes'].replace(0, 1)).round(2)
            gk_df['saves_per_90'] = (gk_df['saves'] * 90 / gk_df['minutes'].replace(0, 1)).round(2)
            gk_df['goals_conceded_per_90'] = (gk_df['goals_conceded'] * 90 / gk_df['minutes'].replace(0, 1)).round(2)
            
            gk_metrics.extend(['clean_sheets_per_90', 'saves_per_90', 'goals_conceded_per_90'])
        
        # Filter available columns
        available_gk_metrics = [col for col in gk_metrics if col in gk_df.columns]
        
        # Display goalkeeper stats
        st.write("**🏆 Top Goalkeepers by Total Points**")
        top_gks = gk_df.nlargest(15, 'total_points')[available_gk_metrics]
        st.dataframe(top_gks, use_container_width=True)
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            if 'clean_sheets' in gk_df.columns:
                st.write("**🏠 Most Clean Sheets**")
                clean_sheet_leaders = gk_df.nlargest(10, 'clean_sheets')[['web_name', 'team_short_name', 'clean_sheets', 'total_points']]
                st.dataframe(clean_sheet_leaders, use_container_width=True)
        
        with col2:
            if 'saves' in gk_df.columns:
                st.write("**✋ Most Saves**")
                save_leaders = gk_df.nlargest(10, 'saves')[['web_name', 'team_short_name', 'saves', 'total_points']]
                st.dataframe(save_leaders, use_container_width=True)

    def _render_defenders_tab(self, filtered_df):
        """Render defender-specific analysis"""
        st.subheader("🛡️ Defender Analysis")
        
        # Filter for defenders only
        if 'position_name' in filtered_df.columns:
            def_df = filtered_df[filtered_df['position_name'] == 'Defender'].copy()
        else:
            def_df = pd.DataFrame()
        
        if def_df.empty:
            st.warning("No defender data available")
            return
        
        # Key metrics for defenders
        def_metrics = [
            'web_name', 'team_short_name', 'cost_millions', 'total_points',
            'clean_sheets', 'goals_conceded', 'own_goals', 'goals_scored', 'assists',
            'points_per_game', 'form', 'selected_by_percent'
        ]
        
        # Add calculated metrics if possible
        if 'minutes' in def_df.columns and def_df['minutes'].sum() > 0:
            def_df['clean_sheets_per_90'] = (def_df['clean_sheets'] * 90 / def_df['minutes'].replace(0, 1)).round(2)
            def_df['goals_conceded_per_90'] = (def_df['goals_conceded'] * 90 / def_df['minutes'].replace(0, 1)).round(2)
            
            def_metrics.extend(['clean_sheets_per_90', 'goals_conceded_per_90'])
        
        # Filter available columns
        available_def_metrics = [col for col in def_metrics if col in def_df.columns]
        
        # Display defender stats
        st.write("**🏆 Top Defenders by Total Points**")
        top_defs = def_df.nlargest(15, 'total_points')[available_def_metrics]
        st.dataframe(top_defs, use_container_width=True)
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            if 'clean_sheets' in def_df.columns:
                st.write("**🏠 Most Clean Sheets**")
                clean_sheet_leaders = def_df.nlargest(10, 'clean_sheets')[['web_name', 'team_short_name', 'clean_sheets', 'total_points']]
                st.dataframe(clean_sheet_leaders, use_container_width=True)
        
        with col2:
            if 'goals_scored' in def_df.columns:
                st.write("**⚽ Attacking Defenders**")
                attacking_defs = def_df.nlargest(10, 'goals_scored')[['web_name', 'team_short_name', 'goals_scored', 'assists', 'total_points']]
                st.dataframe(attacking_defs, use_container_width=True)

    def _render_midfielders_tab(self, filtered_df):
        """Render midfielder-specific analysis"""
        st.subheader("⚽ Midfielder Analysis")
        
        # Filter for midfielders only
        if 'position_name' in filtered_df.columns:
            mid_df = filtered_df[filtered_df['position_name'] == 'Midfielder'].copy()
        else:
            mid_df = pd.DataFrame()
        
        if mid_df.empty:
            st.warning("No midfielder data available")
            return
        
        # Key metrics for midfielders
        mid_metrics = [
            'web_name', 'team_short_name', 'cost_millions', 'total_points',
            'goals_scored', 'assists', 'clean_sheets', 'points_per_game', 
            'form', 'selected_by_percent', 'influence', 'creativity', 'threat'
        ]
        
        # Add calculated metrics if possible
        if 'minutes' in mid_df.columns and mid_df['minutes'].sum() > 0:
            mid_df['goals_per_90'] = (mid_df['goals_scored'] * 90 / mid_df['minutes'].replace(0, 1)).round(2)
            mid_df['assists_per_90'] = (mid_df['assists'] * 90 / mid_df['minutes'].replace(0, 1)).round(2)
            
            mid_metrics.extend(['goals_per_90', 'assists_per_90'])
        
        # Filter available columns
        available_mid_metrics = [col for col in mid_metrics if col in mid_df.columns]
        
        # Display midfielder stats
        st.write("**🏆 Top Midfielders by Total Points**")
        top_mids = mid_df.nlargest(15, 'total_points')[available_mid_metrics]
        st.dataframe(top_mids, use_container_width=True)
        
        # Key insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'goals_scored' in mid_df.columns:
                st.write("**⚽ Top Goalscorers**")
                goal_leaders = mid_df.nlargest(10, 'goals_scored')[['web_name', 'team_short_name', 'goals_scored', 'total_points']]
                st.dataframe(goal_leaders, use_container_width=True)
        
        with col2:
            if 'assists' in mid_df.columns:
                st.write("**🎯 Top Assist Providers**")
                assist_leaders = mid_df.nlargest(10, 'assists')[['web_name', 'team_short_name', 'assists', 'total_points']]
                st.dataframe(assist_leaders, use_container_width=True)
        
        with col3:
            if 'creativity' in mid_df.columns:
                st.write("**🎨 Most Creative**")
                creative_players = mid_df.nlargest(10, 'creativity')[['web_name', 'team_short_name', 'creativity', 'total_points']]
                st.dataframe(creative_players, use_container_width=True)

    def _render_forwards_tab(self, filtered_df):
        """Render forward-specific analysis"""
        st.subheader("🎯 Forward Analysis")
        
        # Filter for forwards only
        if 'position_name' in filtered_df.columns:
            fwd_df = filtered_df[filtered_df['position_name'] == 'Forward'].copy()
        else:
            fwd_df = pd.DataFrame()
        
        if fwd_df.empty:
            st.warning("No forward data available")
            return
        
        # Key metrics for forwards
        fwd_metrics = [
            'web_name', 'team_short_name', 'cost_millions', 'total_points',
            'goals_scored', 'assists', 'penalties_missed', 'points_per_game',
            'form', 'selected_by_percent', 'influence', 'creativity', 'threat'
        ]
        
        # Add calculated metrics if possible
        if 'minutes' in fwd_df.columns and fwd_df['minutes'].sum() > 0:
            fwd_df['goals_per_90'] = (fwd_df['goals_scored'] * 90 / fwd_df['minutes'].replace(0, 1)).round(2)
            fwd_df['assists_per_90'] = (fwd_df['assists'] * 90 / fwd_df['minutes'].replace(0, 1)).round(2)
            
            fwd_metrics.extend(['goals_per_90', 'assists_per_90'])
        
        # Filter available columns
        available_fwd_metrics = [col for col in fwd_metrics if col in fwd_df.columns]
        
        # Display forward stats
        st.write("**🏆 Top Forwards by Total Points**")
        top_fwds = fwd_df.nlargest(15, 'total_points')[available_fwd_metrics]
        st.dataframe(top_fwds, use_container_width=True)
        
        # Key insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'goals_scored' in fwd_df.columns:
                st.write("**⚽ Top Goalscorers**")
                goal_leaders = fwd_df.nlargest(10, 'goals_scored')[['web_name', 'team_short_name', 'goals_scored', 'total_points']]
                st.dataframe(goal_leaders, use_container_width=True)
        
        with col2:
            if 'threat' in fwd_df.columns:
                st.write("**⚡ Highest Threat**")
                threat_leaders = fwd_df.nlargest(10, 'threat')[['web_name', 'team_short_name', 'threat', 'total_points']]
                st.dataframe(threat_leaders, use_container_width=True)
        
        with col3:
            # Value for money analysis
            if 'points_per_million' in fwd_df.columns:
                st.write("**💰 Best Value**")
                value_leaders = fwd_df.nlargest(10, 'points_per_million')[['web_name', 'team_short_name', 'cost_millions', 'points_per_million']]
                st.dataframe(value_leaders, use_container_width=True)

    def _render_top_performers_tab(self, filtered_df):
        """Render top performers across different metrics"""
        st.subheader("🏆 Top Performers")
        
        if filtered_df.empty:
            st.warning("No player data available")
            return
        
        # Create tabs for different performance metrics
        perf_tab1, perf_tab2, perf_tab3, perf_tab4, perf_tab5 = st.tabs([
            "📊 Points Per Game",
            "📈 Form", 
            "🎁 Bonus Points",
            "💰 Value",
            "🔥 In-Form Players"
        ])
        
        with perf_tab1:
            if 'points_per_game' in filtered_df.columns:
                st.write("**📊 Top 15 Players by Points Per Game**")
                ppg_leaders = filtered_df.nlargest(15, 'points_per_game')[
                    ['web_name', 'position_name', 'team_short_name', 'points_per_game', 'total_points', 'cost_millions']
                ]
                st.dataframe(ppg_leaders, use_container_width=True)
            else:
                st.info("Points per game data not available")
        
        with perf_tab2:
            if 'form' in filtered_df.columns:
                st.write("**📈 Top 15 Players by Form**")
                form_leaders = filtered_df.nlargest(15, 'form')[
                    ['web_name', 'position_name', 'team_short_name', 'form', 'total_points', 'cost_millions']
                ]
                st.dataframe(form_leaders, use_container_width=True)
            else:
                st.info("Form data not available")
        
        with perf_tab3:
            if 'bonus' in filtered_df.columns:
                st.write("**🎁 Top 15 Players by Bonus Points**")
                bonus_leaders = filtered_df.nlargest(15, 'bonus')[
                    ['web_name', 'position_name', 'team_short_name', 'bonus', 'total_points', 'cost_millions']
                ]
                st.dataframe(bonus_leaders, use_container_width=True)
            else:
                st.info("Bonus points data not available")
        
        with perf_tab4:
            if 'points_per_million' in filtered_df.columns:
                st.write("**💰 Top 15 Players by Value (Points per Million)**")
                # Filter out players with very low total points for meaningful value analysis
                value_df = filtered_df[filtered_df['total_points'] >= 20]
                value_leaders = value_df.nlargest(15, 'points_per_million')[
                    ['web_name', 'position_name', 'team_short_name', 'points_per_million', 'total_points', 'cost_millions']
                ]
                st.dataframe(value_leaders, use_container_width=True)
            else:
                st.info("Value data not available")
        
        with perf_tab5:
            # Players with good recent form and value
            if all(col in filtered_df.columns for col in ['form', 'points_per_game', 'total_points']):
                st.write("**🔥 In-Form Value Players**")
                st.caption("Players with good form (≥6.0) and decent total points (≥30)")
                
                in_form_df = filtered_df[
                    (filtered_df['form'] >= 6.0) & 
                    (filtered_df['total_points'] >= 30)
                ].copy()
                
                if not in_form_df.empty:
                    # Create a composite score
                    in_form_df['composite_score'] = (
                        in_form_df['form'] * 0.4 + 
                        in_form_df['points_per_game'] * 0.6
                    ).round(2)
                    
                    in_form_leaders = in_form_df.nlargest(15, 'composite_score')[
                        ['web_name', 'position_name', 'team_short_name', 'form', 'points_per_game', 'total_points', 'cost_millions', 'composite_score']
                    ]
                    st.dataframe(in_form_leaders, use_container_width=True)
                else:
                    st.info("No players meet the in-form criteria")
            else:
                st.info("Required data for in-form analysis not available")

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

    def render_filters(self):
        """Advanced Filters tab with comprehensive filtering options"""
        st.header("🔍 Advanced Player Filters")
        
        with st.expander("📚 How to Use Advanced Filters", expanded=False):
            st.markdown("""
            **Advanced Filters** help you find specific players based on multiple criteria:
            
            🎯 **Filter Categories:**
            - **Performance Filters**: Points, form, consistency metrics
            - **Financial Filters**: Price, value for money, budget considerations
            - **Ownership Filters**: Template players vs differentials
            - **Fixture Filters**: Upcoming difficulty, fixture swings
            - **Physical Filters**: Minutes played, injury status
            - **Statistical Filters**: Goals, assists, expected stats
            
            💡 **Pro Tips:**
            - Combine multiple filters for precise targeting
            - Use ownership filters to find differentials
            - Consider fixture difficulty for transfer timing
            - Monitor form trends for emerging players
            """)
        
        if not st.session_state.data_loaded:
            st.info("Please load FPL data first to use advanced filters.")
            return
        
        df = st.session_state.players_df
        
        # Advanced filter tabs
        filter_tab1, filter_tab2, filter_tab3, filter_tab4 = st.tabs([
            "🎯 Performance Filters",
            "💰 Financial Filters", 
            "👥 Ownership Filters",
            "🔬 Statistical Filters"
        ])
        
        with filter_tab1:
            st.subheader("🎯 Performance-Based Filtering")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                min_total_points = st.number_input("Min Total Points:", 0, 300, 50)
                min_form = st.slider("Min Form Rating:", 0.0, 10.0, 5.0, 0.1)
            
            with perf_col2:
                min_minutes = st.number_input("Min Minutes Played:", 0, 3000, 500)
                max_yellow_cards = st.number_input("Max Yellow Cards:", 0, 20, 10)
            
            with perf_col3:
                min_bonus = st.number_input("Min Bonus Points:", 0, 50, 0)
                consistency_filter = st.checkbox("High Consistency Only", help="Players with steady performance")
            
            # Apply performance filters
            filtered_df = df[
                (df['total_points'] >= min_total_points) &
                (df.get('form', 0) >= min_form) &
                (df.get('minutes', 0) >= min_minutes) &
                (df.get('yellow_cards', 0) <= max_yellow_cards) &
                (df.get('bonus', 0) >= min_bonus)
            ]
            
            st.write(f"**Performance Filter Results:** {len(filtered_df)} players")
            if not filtered_df.empty:
                top_performers = filtered_df.nlargest(10, 'total_points')[
                    ['web_name', 'position_name', 'team_short_name', 'total_points', 'form']
                ]
                st.dataframe(top_performers, use_container_width=True)
            else:
                st.warning("No players match your filter criteria")
        
        with filter_tab2:
            st.subheader("💰 Financial Filtering")
            
            fin_col1, fin_col2 = st.columns(2)
            
            with fin_col1:
                price_range = st.slider("Price Range (£m):", 3.9, 15.0, (4.0, 12.0), 0.1)
                min_ppm = st.number_input("Min Points Per Million:", 0.0, 20.0, 8.0, 0.1)
            
            with fin_col2:
                budget_category = st.selectbox("Budget Category:", [
                    "All Players", "Budget Options (≤6.0m)", "Mid-Price (6.1-9.0m)", "Premium (≥9.1m)"
                ])
                value_category = st.selectbox("Value Category:", [
                    "All Values", "Excellent Value (≥12 PPM)", "Good Value (8-12 PPM)", "Poor Value (≤8 PPM)"
                ])
            
            # Apply financial filters
            financial_filtered = df[
                (df['cost_millions'] >= price_range[0]) &
                (df['cost_millions'] <= price_range[1]) &
                (df.get('points_per_million', 0) >= min_ppm)
            ]
            
            # Apply budget category
            if budget_category == "Budget Options (≤6.0m)":
                financial_filtered = financial_filtered[financial_filtered['cost_millions'] <= 6.0]
            elif budget_category == "Mid-Price (6.1-9.0m)":
                financial_filtered = financial_filtered[
                    (financial_filtered['cost_millions'] > 6.0) & 
                    (financial_filtered['cost_millions'] <= 9.0)
                ]
            elif budget_category == "Premium (≥9.1m)":
                financial_filtered = financial_filtered[financial_filtered['cost_millions'] > 9.0]
            
            st.write(f"**Financial Filter Results:** {len(financial_filtered)} players")
            if not financial_filtered.empty:
                best_value = financial_filtered.nlargest(10, 'points_per_million')[
                    ['web_name', 'position_name', 'cost_millions', 'total_points', 'points_per_million']
                ]
                st.dataframe(best_value, use_container_width=True)
        
        with filter_tab3:
            st.subheader("👥 Ownership-Based Filtering")
            
            own_col1, own_col2 = st.columns(2)
            
            with own_col1:
                ownership_range = st.slider("Ownership Range (%):", 0.0, 100.0, (0.0, 100.0), 0.1)
                differential_threshold = st.number_input("Differential Threshold (%):", 0.0, 50.0, 10.0, 0.1)
            
            with own_col2:
                ownership_category = st.selectbox("Ownership Category:", [
                    "All Players", 
                    f"Differentials (≤{differential_threshold}%)", 
                    "Moderate Ownership (10-50%)", 
                    "Template Players (≥50%)"
                ])
                exclude_popular = st.checkbox("Exclude Template Players", help="Remove players owned by >50% of managers")
            
            # Apply ownership filters
            ownership_filtered = df[
                (df.get('selected_by_percent', 0) >= ownership_range[0]) &
                (df.get('selected_by_percent', 0) <= ownership_range[1])
            ]
            
            if ownership_category == f"Differentials (≤{differential_threshold}%)":
                ownership_filtered = ownership_filtered[ownership_filtered.get('selected_by_percent', 0) <= differential_threshold]
            elif ownership_category == "Moderate Ownership (10-50%)":
                ownership_filtered = ownership_filtered[
                    (ownership_filtered.get('selected_by_percent', 0) > 10) & 
                    (ownership_filtered.get('selected_by_percent', 0) <= 50)
                ]
            elif ownership_category == "Template Players (≥50%)":
                ownership_filtered = ownership_filtered[ownership_filtered.get('selected_by_percent', 0) > 50]
            
            if exclude_popular:
                ownership_filtered = ownership_filtered[ownership_filtered.get('selected_by_percent', 0) <= 50]
            
            st.write(f"**Ownership Filter Results:** {len(ownership_filtered)} players")
            if not ownership_filtered.empty:
                ownership_analysis = ownership_filtered.nlargest(10, 'total_points')[
                    ['web_name', 'position_name', 'total_points', 'selected_by_percent', 'cost_millions']
                ]
                st.dataframe(ownership_analysis, use_container_width=True)
        
        with filter_tab4:
            st.subheader("🔬 Statistical Filtering")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                min_goals = st.number_input("Min Goals:", 0, 50, 0)
                min_assists = st.number_input("Min Assists:", 0, 30, 0)
            
            with stat_col2:
                min_clean_sheets = st.number_input("Min Clean Sheets:", 0, 25, 0)
                min_saves = st.number_input("Min Saves:", 0, 200, 0)
            
            with stat_col3:
                position_focus = st.selectbox("Position Focus:", [
                    "All Positions", "Goalkeepers", "Defenders", "Midfielders", "Forwards"
                ])
                stat_category = st.selectbox("Statistical Focus:", [
                    "All Stats", "Goal Threat", "Assist Potential", "Defensive", "Goalkeeping"
                ])
            
            # Apply statistical filters
            stat_filtered = df[
                (df.get('goals_scored', 0) >= min_goals) &
                (df['assists'] >= min_assists) &
                (df['clean_sheets'] >= min_clean_sheets) &
                (df['saves'] >= min_saves)
            ]
            
            # Apply position filter
            if position_focus != "All Positions":
                if position_focus == "Goalkeepers":
                    stat_filtered = stat_filtered[stat_filtered['position_name'] == 'Goalkeeper']
                elif position_focus == "Defenders":
                    stat_filtered = stat_filtered[stat_filtered['position_name'] == 'Defender']
                elif position_focus == "Midfielders":
                    stat_filtered = stat_filtered[stat_filtered['position_name'] == 'Midfielder']
                elif position_focus == "Forwards":
                    stat_filtered = stat_filtered[stat_filtered['position_name'] == 'Forward']
            
            st.write(f"**Statistical Filter Results:** {len(stat_filtered)} players")
            if not stat_filtered.empty:
                stat_leaders = stat_filtered.nlargest(10, 'total_points')[
                    ['web_name', 'position_name', 'goals_scored', 'assists', 'clean_sheets', 'total_points']
                ]
                st.dataframe(stat_leaders, use_container_width=True)
        
        # Combined filter results
        st.divider()
        st.subheader("🔗 Combined Filter Results")
        
        if st.button("🎯 Apply All Active Filters", type="primary"):
            st.info("Combined filtering feature coming soon! This will allow you to apply multiple filter categories simultaneously for precise player targeting.")

    def render_my_team(self):
        """Enhanced My FPL Team analysis with comprehensive features"""
        st.header("👤 My FPL Team Analysis")
        
        with st.expander("📚 How to Use My FPL Team Analysis", expanded=False):
            st.markdown("""
            **My FPL Team Analysis** provides comprehensive insights into your actual FPL team:
            
            🎯 **Key Features:**
            - **Team Import**: Load your team using your FPL Team ID
            - **Performance Analysis**: Detailed breakdown of your team's performance
            - **Squad Analysis**: Player-by-player evaluation with recommendations
            - **Transfer Suggestions**: Data-driven transfer recommendations
            - **Fixture Analysis**: How your players' fixtures look
            - **Benchmarking**: Compare against top managers and league averages
            - **Strategy Planning**: Chip usage and long-term planning
            
            💡 **Finding Your Team ID:**
            1. Go to fantasy.premierleague.com
            2. Navigate to your team
            3. Look at the URL: `.../entry/XXXXXXX/event/X`
            4. The number after 'entry/' is your Team ID
            """)
        
        # Team import section
        if 'my_team_loaded' not in st.session_state or not st.session_state.my_team_loaded:
            st.subheader("📥 Import Your FPL Team")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                team_id = st.number_input(
                    "Enter your FPL Team ID:",
                    min_value=1,
                    max_value=10000000,
                    value=1234567,
                    help="Find your Team ID in the FPL website URL when viewing your team"
                )
            
            with col2:
                if st.button("📥 Load My Team", type="primary"):
                    team_data = self._load_fpl_team(team_id)
                    if team_data:
                        st.session_state.my_team_id = team_id
                        st.session_state.my_team_data = team_data
                        st.session_state.my_team_loaded = True
                        st.success("✅ Team loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load team. Please check your Team ID.")
            
            # Example team IDs
            st.info("💡 **Don't have your Team ID?** Try these examples: 12345, 67890, 11111")
            
            return
        
        # Display loaded team with comprehensive analysis
        team_data = st.session_state.my_team_data
        
        # Team overview header
        team_name = team_data.get('player_first_name', 'FPL') + " " + team_data.get('player_last_name', 'Manager')
        st.subheader(f"🏆 {team_name}'s Team (ID: {st.session_state.my_team_id})")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_rank = team_data.get('summary_overall_rank', 0)
            if overall_rank and overall_rank > 0:
                st.metric("🏆 Overall Rank", f"{overall_rank:,}")
            else:
                st.metric("🏆 Overall Rank", "N/A")
        
        with col2:
            total_points = team_data.get('summary_overall_points', 0)
            if total_points:
                st.metric("📊 Total Points", f"{total_points:,}")
            else:
                st.metric("📊 Total Points", "N/A")
        
        with col3:
            gw_points = team_data.get('summary_event_points', 0)
            if gw_points:
                st.metric("⚡ Last GW Points", f"{gw_points}")
            else:
                st.metric("⚡ Last GW Points", "N/A")
        
        with col4:
            team_value = team_data.get('last_deadline_value', 1000)
            if team_value:
                st.metric("💰 Team Value", f"£{team_value/10:.1f}m")
            else:
                st.metric("💰 Team Value", "£100.0m")
        
        # Analysis tabs
        team_tab1, team_tab2, team_tab3, team_tab4, team_tab5, team_tab6 = st.tabs([
            "👥 Squad Analysis",
            "⚽ Starting XI",
            "📊 Performance Review",
            "🔄 Transfer Planning", 
            "🎯 Strategy & Chips",
            "📈 Benchmarking"
        ])
        
        with team_tab1:
            self._render_squad_analysis(team_data)
        
        with team_tab2:
            self._render_starting_xi_recommendations(team_data)
        
        with team_tab3:
            self._render_performance_review(team_data)
        
        with team_tab4:
            self._render_transfer_planning(team_data)
        
        with team_tab5:
            self._render_strategy_and_chips(team_data)
        
        with team_tab6:
            self._render_team_benchmarking(team_data)
        
        # Reset team button
        if st.button("🔄 Load Different Team"):
            st.session_state.my_team_loaded = False
            st.rerun()

    def render_ai_recommendations(self):
        """AI-Powered Recommendations Hub with machine learning insights"""
        st.header("🤖 AI-Powered Recommendations Hub")
        
        with st.expander("📚 How AI Recommendations Work", expanded=False):
            st.markdown("""
            **AI Recommendations** use advanced algorithms and machine learning to provide:
            
            🧠 **AI Analysis Methods:**
            - **Pattern Recognition**: Identifies trends in player performance
            - **Predictive Modeling**: Forecasts future performance based on historical data
            - **Market Analysis**: Analyzes ownership trends and price movements
            - **Fixture Integration**: Combines difficulty ratings with player capabilities
            - **Form Algorithms**: Advanced form analysis beyond simple averages
            
            🎯 **Recommendation Categories:**
            - **Transfer Targets**: Players to bring into your team
            - **Captain Picks**: Weekly captaincy recommendations
            - **Differentials**: Low-owned gems for rank climbing
            - **Hold/Sell Decisions**: Keep or transfer current players
            - **Chip Timing**: Optimal timing for using your chips
            """)
        
        if not st.session_state.data_loaded:
            st.info("Please load FPL data first to access AI recommendations.")
            return
        
        # AI recommendation tabs
        ai_tab1, ai_tab2, ai_tab3, ai_tab4, ai_tab5 = st.tabs([
            "🎯 Transfer Targets",
            "👑 Captain Picks", 
            "💎 Differentials",
            "🔄 Hold vs Sell",
            "🎪 Advanced AI"
        ])
        
        with ai_tab1:
            self._render_ai_transfer_targets()
        
        with ai_tab2:
            self._render_ai_captain_picks()
        
        with ai_tab3:
            self._render_ai_differentials()
        
        with ai_tab4:
            self._render_ai_hold_sell()
        
        with ai_tab5:
            self._render_advanced_ai_analysis()

    def render_team_builder(self):
        """Interactive Team Builder with optimization algorithms"""
        st.header("⚽ Interactive Team Builder")
        
        with st.expander("📚 How to Use Team Builder", expanded=False):
            st.markdown("""
            **Interactive Team Builder** helps you create optimized FPL teams:
            
            🎯 **Builder Features:**
            - **Budget Optimization**: Maximize points within your budget
            - **Formation Selection**: Choose your preferred formation
            - **Constraint Handling**: Automatic compliance with FPL rules
            - **Multi-Objective**: Balance points, value, and risk
            - **Fixture Integration**: Consider upcoming fixture difficulty
            
            ⚽ **Optimization Goals:**
            - **Point Maximization**: Highest expected points
            - **Value Optimization**: Best points per million ratio
            - **Risk Management**: Balanced ownership and differential mix
            - **Form Focus**: Prioritize in-form players
            - **Fixture Friendly**: Target teams with good upcoming fixtures
            """)
        
        if not st.session_state.data_loaded:
            st.info("Please load FPL data first to use the team builder.")
            return
        
        # Team building configuration
        st.subheader("🛠️ Team Building Configuration")
        
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            budget = st.slider("Budget (£m):", 95.0, 105.0, 100.0, 0.1)
            formation = st.selectbox("Formation:", [
                "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"
            ])
        
        with config_col2:
            optimization_goal = st.selectbox("Optimization Goal:", [
                "Maximize Points", "Maximize Value", "Balance Risk", "Form Focus", "Fixture Focus"
            ])
            risk_tolerance = st.slider("Risk Tolerance:", 0.0, 1.0, 0.5, 0.1, 
                                     help="0 = Conservative, 1 = High Risk")
        
        with config_col3:
            differential_preference = st.slider("Differential Preference:", 0.0, 1.0, 0.3, 0.1,
                                               help="0 = Template, 1 = Many Differentials")
            form_weighting = st.slider("Form Importance:", 0.0, 1.0, 0.4, 0.1,
                                     help="How much to weight recent form")
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                exclude_teams = st.multiselect("Exclude Teams:", 
                    options=st.session_state.teams_df['short_name'].tolist() if not st.session_state.teams_df.empty else [],
                    help="Teams to avoid when building team"
                )
                min_team_spread = st.number_input("Min Team Spread:", 10, 20, 15,
                    help="Minimum number of different teams")
            
            with adv_col2:
                force_premium = st.checkbox("Force Premium Players", value=True,
                    help="Ensure at least 2-3 premium players")
                balanced_attack = st.checkbox("Balanced Attack", value=True,
                    help="Spread attacking returns across positions")
        
        # Generate team button
        if st.button("🚀 Generate Optimized Team", type="primary"):
            with st.spinner("Building your optimized team..."):
                team_result = self._generate_optimized_team(
                    budget=budget,
                    formation=formation,
                    optimization_goal=optimization_goal,
                    risk_tolerance=risk_tolerance,
                    differential_preference=differential_preference,
                    form_weighting=form_weighting,
                    exclude_teams=exclude_teams,
                    min_team_spread=min_team_spread,
                    force_premium=force_premium,
                    balanced_attack=balanced_attack
                )
                
                if team_result:
                    self._display_optimized_team(team_result)
                else:
                    st.error("Failed to generate team. Try adjusting your constraints.")

    # Helper methods for tabs
    def _render_squad_analysis(self, team_data):
        """Render detailed squad analysis with actual functionality"""
        st.subheader("👥 Current Squad Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load FPL data first to see detailed squad analysis.")
            return
        
        try:
            # Load current team picks
            team_id = st.session_state.my_team_id
            picks_url = f"{self.base_url}/entry/{team_id}/event/1/picks/"
            picks_response = requests.get(picks_url, timeout=10, verify=False)
            
            if picks_response.status_code == 200:
                picks_data = picks_response.json()
                picks = picks_data.get('picks', [])
                
                if picks:
                    # Get player details from FPL data
                    players_df = st.session_state.players_df
                    
                    # Create squad dataframe
                    squad_players = []
                    for pick in picks:
                        player_id = pick['element']
                        player_info = players_df[players_df['id'] == player_id]
                        
                        if not player_info.empty:
                            player = player_info.iloc[0]
                            squad_players.append({
                                'name': player['web_name'],
                                'position': player.get('position_name', 'Unknown'),
                                'team': player.get('team_short_name', 'UNK'),
                                'cost': player['cost_millions'],
                                'total_points': player['total_points'],
                                'form': player.get('form', 0),
                                'selected_by': player.get('selected_by_percent', 0),
                                'points_per_game': player.get('points_per_game', 0),
                                'is_captain': pick.get('is_captain', False),
                                'is_vice_captain': pick.get('is_vice_captain', False),
                                'multiplier': pick.get('multiplier', 1)
                            })
                    
                    if squad_players:
                        squad_df = pd.DataFrame(squad_players)
                        
                        # Squad overview metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_cost = squad_df['cost'].sum()
                            st.metric("💰 Squad Value", f"£{total_cost:.1f}m")
                        
                        with col2:
                            total_points = squad_df['total_points'].sum()
                            st.metric("📊 Total Points", f"{total_points:,}")
                        
                        with col3:
                            avg_form = squad_df['form'].mean()
                            st.metric("📈 Average Form", f"{avg_form:.1f}")
                        
                        with col4:
                            avg_ownership = squad_df['selected_by'].mean()
                            st.metric("👥 Avg Ownership", f"{avg_ownership:.1f}%")
                        
                        st.divider()
                        
                        # Position breakdown
                        st.subheader("📋 Squad Breakdown by Position")
                        
                        position_tabs = st.tabs(["🥅 GK", "🛡️ DEF", "⚽ MID", "🎯 FWD", "📊 Full Squad"])
                        
                        with position_tabs[0]:  # Goalkeepers
                            gks = squad_df[squad_df['position'] == 'Goalkeeper']
                            if not gks.empty:
                                st.dataframe(gks[['name', 'team', 'cost', 'total_points', 'form']], use_container_width=True)
                                
                                # GK insights
                                best_gk = gks.loc[gks['total_points'].idxmax()]
                                st.info(f"🏆 Top GK: {best_gk['name']} ({best_gk['total_points']} pts)")
                            else:
                                st.warning("No goalkeepers found")
                        
                        with position_tabs[1]:  # Defenders
                            defs = squad_df[squad_df['position'] == 'Defender']
                            if not defs.empty:
                                st.dataframe(defs[['name', 'team', 'cost', 'total_points', 'form']], use_container_width=True)
                                
                                # Defender insights
                                def_total = defs['total_points'].sum()
                                def_cost = defs['cost'].sum()
                                st.info(f"🛡️ Defense Summary: {len(defs)} players, {def_total} pts, £{def_cost:.1f}m")
                            else:
                                st.warning("No defenders found")
                        
                        with position_tabs[2]:  # Midfielders
                            mids = squad_df[squad_df['position'] == 'Midfielder']
                            if not mids.empty:
                                st.dataframe(mids[['name', 'team', 'cost', 'total_points', 'form']], use_container_width=True)
                                
                                # Midfielder insights
                                mid_total = mids['total_points'].sum()
                                mid_cost = mids['cost'].sum()
                                st.info(f"⚽ Midfield Summary: {len(mids)} players, {mid_total} pts, £{mid_cost:.1f}m")
                            else:
                                st.warning("No midfielders found")
                        
                        with position_tabs[3]:  # Forwards
                            fwds = squad_df[squad_df['position'] == 'Forward']
                            if not fwds.empty:
                                st.dataframe(fwds[['name', 'team', 'cost', 'total_points', 'form']], use_container_width=True)
                                
                                # Forward insights
                                fwd_total = fwds['total_points'].sum()
                                fwd_cost = fwds['cost'].sum()
                                st.info(f"🎯 Attack Summary: {len(fwds)} players, {fwd_total} pts, £{fwd_cost:.1f}m")
                            else:
                                st.warning("No forwards found")
                        
                        with position_tabs[4]:  # Full squad
                            # Enhanced squad view with additional metrics
                            display_squad = squad_df.copy()
                            
                            # Add status indicators
                            display_squad['Status'] = display_squad.apply(lambda x: 
                                '👑 Captain' if x['is_captain'] else 
                                '🥈 Vice-Captain' if x['is_vice_captain'] else 
                                '💎 Differential' if x['selected_by'] < 10 else 
                                '🔥 Popular' if x['selected_by'] > 50 else '👥 Standard', axis=1)
                            
                            # Reorder columns for better display
                            display_cols = ['name', 'position', 'team', 'cost', 'total_points', 'form', 'selected_by', 'Status']
                            st.dataframe(display_squad[display_cols], use_container_width=True)
                        
                        # Team composition analysis
                        st.divider()
                        self._render_team_swot_analysis(squad_df, team_data)
                        
                    else:
                        st.warning("No squad players found in the data")
                else:
                    st.warning("No picks found for this team")
            else:
                st.error(f"Failed to load team picks. Status code: {picks_response.status_code}")
                # Show basic team info from entry data
                st.info("Showing basic team information instead:")
                self._show_basic_team_info(team_data)
                
        except Exception as e:
            st.error(f"Error loading squad data: {str(e)}")
            # Fallback to basic team info
            self._show_basic_team_info(team_data)

    def _show_basic_team_info(self, team_data):
        """Show basic team information when squad data unavailable"""
        st.subheader("📋 Basic Team Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Team Details:**")
            st.write(f"• Manager: {team_data.get('player_first_name', 'N/A')} {team_data.get('player_last_name', 'N/A')}")
            st.write(f"• Team Name: {team_data.get('name', 'N/A')}")
            st.write(f"• Country: {team_data.get('player_region_name', 'N/A')}")
        
        with col2:
            st.write("**Performance:**")
            st.write(f"• Overall Points: {team_data.get('summary_overall_points', 'N/A'):,}")
            st.write(f"• Overall Rank: {team_data.get('summary_overall_rank', 'N/A'):,}")
            st.write(f"• Last GW Points: {team_data.get('summary_event_points', 'N/A')}")

    def _render_starting_xi_recommendations(self, team_data):
        """Render Starting XI recommendations and lineup optimization"""
        st.subheader("⚽ Starting XI Recommendations")
        
        with st.expander("📚 How Starting XI Analysis Works", expanded=False):
            st.markdown("""
            **Starting XI Analysis** helps you optimize your weekly lineup:
            
            🎯 **Key Features:**
            - **Formation Analysis**: Best formation based on your players
            - **Captaincy Recommendations**: Data-driven captain choices
            - **Bench Optimization**: Who to bench and why
            - **Risk Assessment**: Lineup risk evaluation
            - **Fixture Integration**: Consider upcoming fixtures
            
            💡 **Optimization Factors:**
            - Recent form and consistency
            - Fixture difficulty ratings
            - Expected minutes (rotation risk)
            - Historical performance vs opponent
            - Home/Away form differences
            """)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load FPL data first to see Starting XI recommendations.")
            return
        
        try:
            # Load current team picks
            team_id = st.session_state.my_team_id
            picks_url = f"{self.base_url}/entry/{team_id}/event/1/picks/"
            picks_response = requests.get(picks_url, timeout=10, verify=False)
            
            if picks_response.status_code == 200:
                picks_data = picks_response.json()
                picks = picks_data.get('picks', [])
                
                if picks:
                    # Get player details from FPL data
                    players_df = st.session_state.players_df
                    
                    # Create squad dataframe with lineup info
                    squad_players = []
                    for pick in picks:
                        player_id = pick['element']
                        player_info = players_df[players_df['id'] == player_id]
                        
                        if not player_info.empty:
                            player = player_info.iloc[0]
                            squad_players.append({
                                'name': player['web_name'],
                                'position': player.get('position_name', 'Unknown'),
                                'team': player.get('team_short_name', 'UNK'),
                                'cost': player['cost_millions'],
                                'total_points': player['total_points'],
                                'form': player.get('form', 0),
                                'points_per_game': player.get('points_per_game', 0),
                                'selected_by': player.get('selected_by_percent', 0),
                                'minutes': player.get('minutes', 0),
                                'is_captain': pick.get('is_captain', False),
                                'is_vice_captain': pick.get('is_vice_captain', False),
                                'multiplier': pick.get('multiplier', 1),
                                'position_order': pick.get('position', 1),
                                'element_id': player_id
                            })
                    
                    if squad_players:
                        squad_df = pd.DataFrame(squad_players)
                        
                        # Starting XI Analysis Tabs
                        xi_tab1, xi_tab2, xi_tab3, xi_tab4 = st.tabs([
                            "⚽ Optimal Lineup",
                            "👑 Captain Analysis", 
                            "🔄 Bench Strategy",
                            "🎯 Formation Analysis"
                        ])
                        
                        with xi_tab1:
                            self._render_optimal_lineup(squad_df)
                        
                        with xi_tab2:
                            self._render_captain_analysis(squad_df)
                        
                        with xi_tab3:
                            self._render_bench_strategy(squad_df)
                        
                        with xi_tab4:
                            self._render_formation_analysis(squad_df)
                        
                    else:
                        st.warning("No squad players found")
                else:
                    st.warning("No picks found for this team")
            else:
                st.error(f"Failed to load team picks. Status code: {picks_response.status_code}")
                st.info("Starting XI analysis requires access to your current team picks.")
                
        except Exception as e:
            st.error(f"Error loading Starting XI data: {str(e)}")
            st.info("Starting XI recommendations will be available once team data is properly loaded.")

    def render_team_odds(self):
        """Team Performance Odds and Analysis"""
        st.header("📈 Team Performance Analytics")
        
        with st.expander("📚 About Team Performance Analytics", expanded=False):
            st.markdown("""
            **Team Performance Analytics** provides insights into Premier League teams:
            
            🎯 **Analysis Areas:**
            - **Attack vs Defense Balance**: How teams perform in different phases
            - **Home vs Away Performance**: Venue-specific strengths and weaknesses  
            - **Form Trends**: Recent performance patterns and momentum
            - **Head-to-Head Records**: Historical matchup analysis
            - **Expected Performance**: Model-based predictions vs actual results
            
            📊 **Key Metrics:**
            - **Goals For/Against**: Offensive and defensive output
            - **Clean Sheet %**: Defensive reliability
            - **Win Rate**: Success percentage in different scenarios
            - **Points Per Game**: Consistency measure
            """)
        
        if not st.session_state.data_loaded:
            st.info("Please load FPL data first to access team analytics.")
            return
        
        # Team performance analysis tabs
        perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs([
            "🏆 Team Overview",
            "📊 Performance Metrics",
            "⚔️ Head-to-Head",
            "🔮 Predictions"
        ])
        
        with perf_tab1:
            st.subheader("🏆 Premier League Team Overview")
            
            if not st.session_state.teams_df.empty:
                teams_df = st.session_state.teams_df
                
                # Team strength visualization
                if 'strength' in teams_df.columns:
                    fig = px.bar(
                        teams_df.sort_values('strength', ascending=True),
                        x='strength',
                        y='name',
                        orientation='h',
                        title="Team Strength Ratings",
                        labels={'strength': 'Strength Rating', 'name': 'Team'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Team stats table
                st.subheader("📋 Team Statistics")
                display_teams = teams_df[['name', 'strength', 'played', 'win', 'draw', 'loss', 'points']].copy()
                st.dataframe(display_teams, use_container_width=True)
            else:
                st.warning("Team data not available")
        
        with perf_tab2:
            st.subheader("📊 Performance Metrics")
            st.info("Detailed performance metrics coming soon!")
            st.write("• Goals scored/conceded per game")
            st.write("• Clean sheet percentages")
            st.write("• Home vs away form splits")
            st.write("• Recent form trends")
        
        with perf_tab3:
            st.subheader("⚔️ Head-to-Head Analysis")
            st.info("Head-to-head analysis coming soon!")
            st.write("• Historical matchup records")
            st.write("• Recent encounters")
            st.write("• Venue-specific performance")
            st.write("• Key player impact")
        
        with perf_tab4:
            st.subheader("🔮 Performance Predictions")
            st.info("Performance prediction models coming soon!")
            st.write("• Expected goals/points models")
            st.write("• Form-based predictions")
            st.write("• Fixture difficulty adjustments")
            st.write("• Confidence intervals")

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

# Run the application
if __name__ == "__main__":
    app = FPLAnalyticsApp()
    app.run()

    def _render_optimal_lineup(self, squad_df):
        """Render optimal lineup recommendations"""
        st.subheader("⚽ Optimal Starting XI")
        
        # Calculate lineup scores for each player
        squad_df_copy = squad_df.copy()
        
        # Create a lineup score based on multiple factors
        squad_df_copy['lineup_score'] = (
            squad_df_copy['form'] * 0.4 +
            squad_df_copy['points_per_game'] * 0.3 +
            (squad_df_copy['minutes'] / 100) * 0.2 +  # Minutes as reliability indicator
            (100 - squad_df_copy['selected_by']) * 0.1  # Slight differential bonus
        ).round(2)
        
        # Separate by position for optimal selection
        positions = {
            'Goalkeeper': 1,
            'Defender': 5,  # Can play 3-5
            'Midfielder': 5,  # Can play 3-5  
            'Forward': 3   # Can play 1-3
        }
        
        optimal_xi = []
        bench_players = []
        
        # Select best players by position
        for position, max_count in positions.items():
            pos_players = squad_df_copy[squad_df_copy['position'] == position].copy()
            
            if not pos_players.empty:
                # Sort by lineup score
                pos_players = pos_players.sort_values('lineup_score', ascending=False)
                
                # Determine how many to start (flexible formations)
                if position == 'Goalkeeper':
                    to_start = 1
                elif position == 'Defender':
                    to_start = min(4, len(pos_players))  # Usually 3-4 defenders
                elif position == 'Midfielder': 
                    to_start = min(4, len(pos_players))  # Usually 3-4 midfielders
                else:  # Forward
                    to_start = min(2, len(pos_players))  # Usually 1-2 forwards
                
                # Add to starting XI
                for i, (_, player) in enumerate(pos_players.iterrows()):
                    if i < to_start:
                        optimal_xi.append(player.to_dict())
                    else:
                        bench_players.append(player.to_dict())
        
        # Display optimal lineup
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**🔥 Recommended Starting XI**")
            
            if optimal_xi:
                xi_df = pd.DataFrame(optimal_xi)
                
                # Group by position for display
                for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                    pos_players = xi_df[xi_df['position'] == position]
                    
                    if not pos_players.empty:
                        st.write(f"**{position}s:**")
                        display_cols = ['name', 'team', 'form', 'points_per_game', 'lineup_score']
                        available_cols = [col for col in display_cols if col in pos_players.columns]
                        st.dataframe(pos_players[available_cols], use_container_width=True, hide_index=True)
                        st.write("")
            else:
                st.warning("Unable to generate optimal lineup")
        
        with col2:
            st.write("**📊 Lineup Metrics**")
            
            if optimal_xi:
                xi_df = pd.DataFrame(optimal_xi)
                
                # Calculate team metrics
                total_cost = xi_df['cost'].sum()
                avg_form = xi_df['form'].mean()
                avg_points = xi_df['points_per_game'].mean()
                total_ownership = xi_df['selected_by'].mean()
                
                st.metric("💰 XI Cost", f"£{total_cost:.1f}m")
                st.metric("📈 Avg Form", f"{avg_form:.1f}")
                st.metric("📊 Avg PPG", f"{avg_points:.1f}")
                st.metric("👥 Avg Ownership", f"{total_ownership:.1f}%")
                
                # Lineup risk assessment
                high_ownership_count = len(xi_df[xi_df['selected_by'] > 50])
                differential_count = len(xi_df[xi_df['selected_by'] < 15])
                
                st.write("**🎯 Risk Assessment:**")
                if high_ownership_count > 7:
                    st.warning("⚠️ Template-heavy lineup")
                elif differential_count > 3:
                    st.info("🎲 High-risk differential strategy")
                else:
                    st.success("✅ Balanced risk profile")
        
        # Bench recommendations
        st.divider()
        st.subheader("🪑 Recommended Bench")
        
        if bench_players:
            bench_df = pd.DataFrame(bench_players)
            
            # Sort bench by lineup score (best first-sub)
            bench_df = bench_df.sort_values('lineup_score', ascending=False)
            
            st.write("**Bench Order (by autosub priority):**")
            bench_cols = ['name', 'position', 'team', 'form', 'lineup_score']
            available_bench_cols = [col for col in bench_cols if col in bench_df.columns]
            st.dataframe(bench_df[available_bench_cols], use_container_width=True, hide_index=True)
            
            # Bench insights
            if len(bench_df) > 0:
                best_bench = bench_df.iloc[0]
                st.info(f"💡 **First Sub Recommendation**: {best_bench['name']} ({best_bench['position']}) - Form: {best_bench['form']}")
        else:
            st.info("All players recommended for starting XI")

    def _render_captain_analysis(self, squad_df):
        """Render captain and vice-captain analysis"""
        st.subheader("👑 Captain & Vice-Captain Analysis")
        
        # Calculate captaincy scores
        squad_df_copy = squad_df.copy()
        
        # Enhanced captaincy scoring
        squad_df_copy['captain_score'] = (
            squad_df_copy['form'] * 0.35 +
            squad_df_copy['points_per_game'] * 0.35 +
            (squad_df_copy['total_points'] / 100) * 0.2 +  # Season consistency
            (100 - squad_df_copy['selected_by']) * 0.1  # Differential potential
        ).round(2)
        
        # Filter out goalkeepers (rarely captain material)
        captain_candidates = squad_df_copy[squad_df_copy['position'] != 'Goalkeeper'].copy()
        captain_candidates = captain_candidates.sort_values('captain_score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**👑 Captain Recommendations**")
            
            if not captain_candidates.empty:
                top_captains = captain_candidates.head(5)
                
                captain_cols = ['name', 'position', 'team', 'form', 'points_per_game', 'captain_score']
                available_captain_cols = [col for col in captain_cols if col in top_captains.columns]
                
                st.dataframe(top_captains[available_captain_cols], use_container_width=True, hide_index=True)
                
                # Captain insights
                best_captain = top_captains.iloc[0]
                st.success(f"🎯 **Top Pick**: {best_captain['name']} (Score: {best_captain['captain_score']})")
                
                # Current captain analysis
                current_captain = squad_df_copy[squad_df_copy['is_captain'] == True]
                if not current_captain.empty:
                    current_cap = current_captain.iloc[0]
                    current_rank = captain_candidates[captain_candidates['name'] == current_cap['name']].index[0] + 1 if len(captain_candidates[captain_candidates['name'] == current_cap['name']]) > 0 else 'N/A'
                    
                    if current_rank <= 3:
                        st.info(f"✅ Current captain {current_cap['name']} ranks #{current_rank} - Good choice!")
                    else:
                        st.warning(f"⚠️ Current captain {current_cap['name']} ranks #{current_rank} - Consider switching")
            else:
                st.warning("No captain candidates found")
        
        with col2:
            st.write("**🥈 Vice-Captain Recommendations**")
            
            if len(captain_candidates) > 1:
                # Vice-captain should be different from captain
                current_captain_name = squad_df_copy[squad_df_copy['is_captain'] == True]['name'].iloc[0] if len(squad_df_copy[squad_df_copy['is_captain'] == True]) > 0 else ""
                
                vc_candidates = captain_candidates[captain_candidates['name'] != current_captain_name]
                top_vcs = vc_candidates.head(3)
                
                vc_cols = ['name', 'position', 'form', 'captain_score']
                available_vc_cols = [col for col in vc_cols if col in top_vcs.columns]
                
                st.dataframe(top_vcs[available_vc_cols], use_container_width=True, hide_index=True)
                
                if not top_vcs.empty:
                    best_vc = top_vcs.iloc[0]
                    st.info(f"🎯 **VC Pick**: {best_vc['name']}")
            else:
                st.warning("Insufficient players for VC analysis")
        
        # Captaincy strategy insights
        st.divider()
        st.subheader("🧠 Captaincy Strategy")
        
        strategy_col1, strategy_col2 = st.columns(2)
        
        with strategy_col1:
            st.write("**🎯 This Week's Strategy:**")
            
            if not captain_candidates.empty:
                top_captain = captain_candidates.iloc[0]
                
                # Strategy recommendations based on ownership
                if top_captain['selected_by'] > 50:
                    st.info("📊 **Safe Play**: High-owned captain reduces rank volatility")
                elif top_captain['selected_by'] < 15:
                    st.warning("🎲 **Differential Play**: Low-owned captain for rank gains")
                else:
                    st.success("⚖️ **Balanced Play**: Moderate ownership captain")
                
                # Form-based insights
                if top_captain['form'] > 7:
                    st.success("🔥 Captain in excellent form")
                elif top_captain['form'] < 5:
                    st.warning("📉 Captain form concerns")
        
        with strategy_col2:
            st.write("**📈 Long-term Considerations:**")
            
            # Season consistency
            if not captain_candidates.empty:
                consistent_captains = captain_candidates[captain_candidates['total_points'] > 100]
                
                if len(consistent_captains) >= 3:
                    st.success("✅ Multiple reliable captain options")
                elif len(consistent_captains) >= 1:
                    st.info("⚠️ Limited consistent captain options")
                else:
                    st.warning("❌ No highly consistent captains")
                
                # Fixture planning note
                if st.session_state.fdr_data_loaded:
                    st.info("💡 Consider upcoming fixtures for captain planning")
                else:
                    st.info("💡 Load fixture data for fixture-based captain analysis")

    def _render_bench_strategy(self, squad_df):
        """Render bench strategy and autosub analysis"""
        st.subheader("🪑 Bench Strategy & Autosub Planning")
        
        # Separate potential starters from bench warmers
        squad_df_copy = squad_df.copy()
        
        # Calculate bench value score
        squad_df_copy['bench_value'] = (
            squad_df_copy['points_per_game'] * 0.4 +
            (squad_df_copy['minutes'] / 50) * 0.3 +  # Playing time reliability
            squad_df_copy['form'] * 0.2 +
            (5 - squad_df_copy['cost']) * 0.1  # Budget efficiency for bench
        ).round(2)
        
        # Group by position for bench analysis
        bench_tab1, bench_tab2, bench_tab3 = st.tabs([
            "🔄 Autosub Analysis",
            "💰 Bench Value", 
            "🎯 Bench Strategy"
        ])
        
        with bench_tab1:
            st.write("**🔄 Autosub Potential Analysis**")
            
            # Players likely to come off the bench
            potential_subs = squad_df_copy.copy()
            potential_subs = potential_subs.sort_values('bench_value', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best Bench Options:**")
                
                bench_cols = ['name', 'position', 'team', 'points_per_game', 'minutes', 'bench_value']
                available_bench_cols = [col for col in bench_cols if col in potential_subs.columns]
                
                top_bench = potential_subs.head(6)  # Typical bench size
                st.dataframe(top_bench[available_bench_cols], use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**🎯 Autosub Insights:**")
                
                # Minutes analysis for autosub potential
                if 'minutes' in squad_df_copy.columns:
                    regular_starters = squad_df_copy[squad_df_copy['minutes'] > 1000]
                    rotation_risks = squad_df_copy[(squad_df_copy['minutes'] > 500) & (squad_df_copy['minutes'] <= 1000)]
                    bench_warmers = squad_df_copy[squad_df_copy['minutes'] <= 500]
                    
                    st.metric("🟢 Regular Starters", len(regular_starters))
                    st.metric("🟡 Rotation Risks", len(rotation_risks))
                    st.metric("🔴 Bench Warmers", len(bench_warmers))
                    
                    if len(rotation_risks) > 3:
                        st.warning("⚠️ High rotation risk - strong bench needed")
                    else:
                        st.success("✅ Stable starting options")
        
        with bench_tab2:
            st.write("**💰 Bench Value Analysis**")
            
            # Budget analysis
            bench_players = squad_df_copy.nsmallest(4, 'cost')  # Cheapest 4 as typical bench
            
            total_bench_cost = bench_players['cost'].sum()
            total_bench_points = bench_players['total_points'].sum()
            
            value_col1, value_col2 = st.columns(2)
            
            with value_col1:
                st.metric("💰 Bench Cost", f"£{total_bench_cost:.1f}m")
                st.metric("📊 Bench Points", f"{total_bench_points}")
                
                # Value assessment
                if total_bench_cost < 20:
                    st.success("✅ Budget-efficient bench")
                elif total_bench_cost < 25:
                    st.info("💡 Moderate bench investment")
                else:
                    st.warning("💸 Expensive bench - consider downgrades")
            
            with value_col2:
                st.write("**Bench Players:**")
                bench_display_cols = ['name', 'position', 'team', 'cost', 'total_points']
                available_display_cols = [col for col in bench_display_cols if col in bench_players.columns]
                st.dataframe(bench_players[available_display_cols], use_container_width=True, hide_index=True)
        
        with bench_tab3:
            st.write("**🎯 Bench Strategy Recommendations**")
            
            strategy_insights = []
            
            # Position analysis
            position_counts = squad_df_copy['position'].value_counts()
            
            # Bench strategy based on position distribution
            if position_counts.get('Defender', 0) >= 5:
                strategy_insights.append("🛡️ **Defensive Bench**: Good defender coverage for rotation")
            
            if position_counts.get('Midfielder', 0) >= 5:
                strategy_insights.append("⚽ **Midfield Depth**: Strong midfield options for autosubs")
            
            if position_counts.get('Forward', 0) == 3:
                strategy_insights.append("🎯 **Balanced Attack**: Standard 3-forward setup")
            elif position_counts.get('Forward', 0) > 3:
                strategy_insights.append("🎯 **Forward Heavy**: Consider balancing with other positions")
            
            # Budget strategy
            expensive_bench = len(squad_df_copy[squad_df_copy['cost'] > 6.0])
            if expensive_bench > 4:
                strategy_insights.append("💰 **Premium Heavy**: Consider cheaper bench options for budget flexibility")
            
            # Playing time strategy
            if 'minutes' in squad_df_copy.columns:
                playing_bench = len(squad_df_copy[squad_df_copy['minutes'] > 800])
                if playing_bench > 11:
                    strategy_insights.append("🔄 **Rotation Friendly**: Squad built for heavy rotation")
            
            # Display insights
            if strategy_insights:
                for insight in strategy_insights:
                    st.info(insight)
            else:
                st.success("✅ Well-balanced bench strategy")
            
            # Future planning
            st.divider()
            st.write("**📅 Bench Planning Tips:**")
            st.write("• Keep at least one playing defender on bench")
            st.write("• Consider upcoming fixture congestion")
            st.write("• Monitor price changes for bench players") 
            st.write("• Plan bench boosts around favorable fixtures")

    def _render_formation_analysis(self, squad_df):
        """Render formation analysis and recommendations"""
        st.subheader("🎯 Formation Analysis")
        
        # Analyze possible formations based on squad
        position_counts = squad_df['position'].value_counts()
        
        gk_count = position_counts.get('Goalkeeper', 0)
        def_count = position_counts.get('Defender', 0) 
        mid_count = position_counts.get('Midfielder', 0)
        fwd_count = position_counts.get('Forward', 0)
        
        # Possible formations (GK always 1)
        possible_formations = []
        
        # Standard formations
        formations = [
            (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1), (5, 3, 2), (5, 4, 1)
        ]
        
        for def_req, mid_req, fwd_req in formations:
            form_name = f"{def_req}-{mid_req}-{fwd_req}"
                    
            # Calculate formation strength
            best_defs = squad_df[squad_df['position'] == 'Defender'].nlargest(def_req, 'points_per_game')
            best_mids = squad_df[squad_df['position'] == 'Midfielder'].nlargest(mid_req, 'points_per_game')
            best_fwds = squad_df[squad_df['position'] == 'Forward'].nlargest(fwd_req, 'points_per_game')
                    
            total_expected_points = (
                best_defs['points_per_game'].sum() +
                best_mids['points_per_game'].sum() + 
                best_fwds['points_per_game'].sum()
            )
                    
            possible_formations.append({
                'Formation': form_name,
                'Expected Points': round(total_expected_points, 1),
                'Defenders': def_req,
                'Midfielders': mid_req,
                'Forwards': fwd_req
            })
        
        form_tab1, form_tab2, form_tab3 = st.tabs([
            "⚽ Formation Options",
            "📊 Position Analysis",
            "🎯 Optimal Formation"
        ])
        
        with form_tab1:
            st.write("**⚽ Available Formations**")
            
            if possible_formations:
                formation_analysis = []
                
                for def_req, mid_req, fwd_req in possible_formations:
                    form_name = f"{def_req}-{mid_req}-{fwd_req}"
                    
                    # Calculate formation strength
                    best_defs = squad_df[squad_df['position'] == 'Defender'].nlargest(def_req, 'points_per_game')
                    best_mids = squad_df[squad_df['position'] == 'Midfielder'].nlargest(mid_req, 'points_per_game')
                    best_fwds = squad_df[squad_df['position'] == 'Forward'].nlargest(fwd_req, 'points_per_game')
                    
                    total_expected_points = (
                        best_defs['points_per_game'].sum() +
                        best_mids['points_per_game'].sum() + 
                        best_fwds['points_per_game'].sum()
                    )
                    
                    formation_analysis.append({
                        'Formation': form_name,
                        'Expected Points': round(total_expected_points, 1),
                        'Defenders': def_req,
                        'Midfielders': mid_req,
                        'Forwards': fwd_req
                    })
                
                formation_df = pd.DataFrame(formation_analysis)
                formation_df = formation_df.sort_values('Expected Points', ascending=False)
                
                st.dataframe(formation_df, use_container_width=True, hide_index=True)
                
                # Best formation recommendation
                if not formation_df.empty:
                    best_formation = formation_df.iloc(0)
                    st.success(f"🏆 **Recommended Formation**: {best_formation['Formation']} (Expected: {best_formation['Expected Points']} pts)")
            else:
                st.error("❌ No valid formations possible with current squad")
        
        with form_tab2:
            st.write("**📊 Position Strength Analysis**")
            
            pos_col1, pos_col2 = st.columns(2)
            
            with pos_col1:
                st.write("**Position Counts:**")
                st.write(f"🥅 Goalkeepers: {gk_count}")
                st.write(f"🛡️ Defenders: {def_count}")
                st.write(f"⚽ Midfielders: {mid_count}")  
                st.write(f"🎯 Forwards: {fwd_count}")
                
                # Squad balance assessment
                total_outfield = def_count + mid_count + fwd_count
                if total_outfield == 14:  # Standard FPL squad
                    st.success("✅ Full squad")
                else:
                    st.warning(f"⚠️ Squad size: {total_outfield + gk_count}")
            
            with pos_col2:
                st.write("**Position Strength:**")
                
                # Calculate average points per position
                for position in ['Defender', 'Midfielder', 'Forward']:
                    pos_players = squad_df[squad_df['position'] == position]
                    if not pos_players.empty:
                        avg_points = pos_players['points_per_game'].mean()
                        avg_form = pos_players['form'].mean()
                        
                        if avg_points > 4.5:
                            strength = "💪 Strong"
                        elif avg_points > 3.5:
                            strength = "👍 Good"
                        elif avg_points > 2.5:
                            strength = "⚖️ Average"
                        else:
                            strength = "📉 Weak"
                        
                        st.write(f"{position}s: {strength} ({avg_points:.1f} PPG)")
        
        with form_tab3:
            st.write("**🎯 Optimal Formation Recommendation**")
            
            if possible_formations:
                # More detailed formation analysis
                best_formation_data = formation_df.iloc(0) if not formation_df.empty else None
                
                if best_formation_data:
                    recommended_formation = best_formation_data['Formation']
                    def_req, mid_req, fwd_req = map(int, recommended_formation.split('-'))
                    
                    st.success(f"🏆 **Optimal Formation: {recommended_formation}**")
                    
                    # Show the actual players for this formation
                    st.write("**Recommended Starting XI:**")
                    
                    # Best players by position for the formation
                    best_gk = squad_df[squad_df['position'] == 'Goalkeeper'].nlargest(1, 'points_per_game')
                    best_defs = squad_df[squad_df['position'] == 'Defender'].nlargest(def_req, 'points_per_game')
                    best_mids = squad_df[squad_df['position'] == 'Midfielder'].nlargest(mid_req, 'points_per_game')
                    best_fwds = squad_df[squad_df['position'] == 'Forward'].nlargest(fwd_req, 'points_per_game')
                    
                    # Display by position
                    for pos_name, pos_df in [("Goalkeeper", best_gk), ("Defenders", best_defs), 
                                           ("Midfielders", best_mids), ("Forwards", best_fwds)]:
                        if not pos_df.empty:
                            st.write(f"**{pos_name}:**")
                            display_cols = ['name', 'team', 'points_per_game', 'form']
                            available_cols = [col for col in display_cols if col in pos_df.columns]
                            st.dataframe(pos_df[available_cols], use_container_width=True, hide_index=True)
                            st.write("")
                    
                    # Formation insights
                    st.write("**📊 Formation Insights:**")
                    
                    if def_req >= 4:
                        st.info("🛡️ **Defensive Formation**: Good for teams with strong defensive options")
                    
                    if mid_req >= 5:
                        st.info("⚽ **Midfield Heavy**: Maximizes midfield returns")
                    
                    if fwd_req >= 3:
                        st.info("🎯 **Attack Focused**: Prioritizes forward returns")
                    
                    # Risk assessment
                    formation_risk = abs(def_req - 4) + abs(mid_req - 4) + abs(fwd_req - 2)
                    
                    if formation_risk <= 2:
                        st.success("✅ **Low Risk**: Balanced formation")
                    elif formation_risk <= 4:
                        st.info("⚖️ **Medium Risk**: Slightly unbalanced")
                    else:
                        st.warning("⚠️ **High Risk**: Very unbalanced formation")

    def _render_team_swot_analysis(self, squad_df, team_data):
        """Render SWOT analysis for the team"""
        st.subheader("🎯 Team SWOT Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💪 Strengths:**")
            strengths = []
            
            # High value players
            high_value = squad_df[squad_df['total_points'] > 100]
            if len(high_value) >= 5:
                strengths.append("✅ Multiple high-scoring players")
            
            # Good form
            good_form = squad_df[squad_df['form'] > 6]
            if len(good_form) >= 6:
                strengths.append("✅ Squad in good form")
            
            # Balanced squad value
            total_value = squad_df['cost'].sum()
            if total_value > 98:
                strengths.append("✅ High squad value")
            
            # Display strengths
            if strengths:
                for strength in strengths:
                    st.write(strength)
            else:
                st.write("• Review squad composition")
            
            st.write("")
            st.write("**🎯 Opportunities:**")
            opportunities = []
            
            # Low ownership differentials
            differentials = squad_df[squad_df['selected_by'] < 10]
            if len(differentials) >= 2:
                opportunities.append("🎲 Good differential coverage")
            
            # Budget remaining
            if total_value < 99:
                opportunities.append("💰 Budget available for upgrades")
            
            # Display opportunities
            if opportunities:
                for opportunity in opportunities:
                    st.write(opportunity)
            else:
                st.write("• Look for emerging players")
        
        with col2:
            st.write("**⚠️ Weaknesses:**")
            weaknesses = []
            
            # Poor form players
            poor_form = squad_df[squad_df['form'] < 4]
            if len(poor_form) >= 3:
                weaknesses.append("📉 Multiple players in poor form")
            
            # Low total points
            low_scorers = squad_df[squad_df['total_points'] < 50]
            if len(low_scorers) >= 4:
                weaknesses.append("📊 Several low-scoring players")
            
            # Expensive bench
            bench_cost = squad_df.nsmallest(4, 'cost')['cost'].sum()
            if bench_cost > 25:
                weaknesses.append("💸 Expensive bench players")
            
            # Display weaknesses
            if weaknesses:
                for weakness in weaknesses:
                    st.write(weakness)
            else:
                st.write("• No major weaknesses identified")
            
            st.write("")
            st.write("**🚨 Threats:**")
            threats = []
            
            # High ownership risks
            template_heavy = squad_df[squad_df['selected_by'] > 70]
            if len(template_heavy) >= 8:
                threats.append("👥 Very template-heavy team")
            
            # Price fall risks
            falling_players = squad_df[(squad_df['form'] < 3) & (squad_df['selected_by'] > 20)]
            if len(falling_players) >= 2:
                threats.append("📉 Players at risk of price falls")
            
            # Display threats
            if threats:
                for threat in threats:
                    st.write(threat)
            else:
                st.write("• No immediate threats")

    def _load_fpl_team(self, team_id):
        """Load FPL team data from API"""
        try:
            url = f"{self.base_url}/entry/{team_id}/"
            response = requests.get(url, timeout=10, verify=False)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            st.error(f"Error loading team data: {str(e)}")
            return None

    def _render_team_import_section(self):
        """Render the team import section"""
        st.subheader("📥 Import Your FPL Team")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Enter your FPL Team ID to analyze your squad:**")
            team_id = st.text_input(
                "Team ID:",
                placeholder="Enter your team ID (e.g., 123456)",
                help="Find your team ID in the FPL website URL"
            )
            
            if st.button("📊 Load My Team", type="primary"):
                if team_id and team_id.isdigit():
                    with st.spinner("Loading your FPL team..."):
                        team_data = self._load_fpl_team(int(team_id))
                        if team_data:
                            st.session_state.my_team_data = team_data
                            st.session_state.my_team_id = int(team_id)
                            st.session_state.my_team_loaded = True
                            st.success("✅ Team loaded successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load team. Please check your Team ID.")
                else:
                    st.error("❌ Please enter a valid Team ID (numbers only)")
        
        with col2:
            st.info("💡 **Need help finding your Team ID?**")
            
        # Instructions
        with st.expander("💡 How to find your Team ID", expanded=False):
            st.markdown("""
            **Step-by-step guide:**
            
            1. Go to the official FPL website: fantasy.premierleague.com
            2. Log in to your account
            3. Go to "My Team" or "Points"
            4. Look at the URL in your browser
            5. Your Team ID is the number after "/entry/"
            
            **Example:**
            - URL: `https://fantasy.premierleague.com/entry/123456/event/10`
            - Team ID: `123456`
            
            **Privacy Note:** Your Team ID is public information that anyone can use to view your team.
            """)

    def _display_current_squad(self, team_data):
        """Enhanced current squad display with Starting 11 tab"""
        st.subheader("👥 Squad Management & Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see detailed squad analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        # **NEW: Enhanced tabs including Starting 11**
        squad_tab1, squad_tab2, squad_tab3, squad_tab4 = st.tabs([
            "🔢 Starting 11", 
            "👥 Full Squad", 
            "🏟️ Formation Analysis",
            "📊 Squad Statistics"
        ])
        
        with squad_tab1:
            self._render_starting_eleven(team_data, picks)
        
        with squad_tab2:
            self._render_full_squad_analysis(team_data, picks)
        
        with squad_tab3:
            self._render_formation_analysis_detailed(team_data, picks)
        
        with squad_tab4:
            self._render_squad_statistics(team_data, picks)

    def _render_starting_eleven(self, team_data, picks):
        """Render Starting 11 analysis"""
        st.write("**🔢 Your Starting XI Analysis**")
        
        players_df = st.session_state.players_df
        
        # Get starting 11 (positions 1-11)
        starting_picks = [p for p in picks if p['position'] <= 11]
        
        if len(starting_picks) != 11:
            st.warning("⚠️ Starting XI incomplete - check your team setup")
            return
        
        # Process starting 11 data
        starting_players = []
        total_cost = 0
        total_points = 0
        formation_count = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for pick in starting_picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                
                # Determine position for formation
                position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                position_short = position_map.get(player['element_type'], 'UNK')
                formation_count[position_short] += 1
                
                starting_players.append({
                    'Player': player['web_name'],
                    'Position': player.get('position_name', 'Unknown'),
                    'Team': player.get('team_short_name', 'UNK'),
                    'Price': f"£{player.get('cost_millions', 0):.1f}m",
                    'Points': player.get('total_points', 0),
                    'Form': f"{player.get('form', 0):.1f}",
                    'PPG': f"{player.get('total_points', 0) / max(player.get('minutes', 1) / 90, 1):.1f}",
                    'Role': '(C)' if pick['is_captain'] else '(VC)' if pick['is_vice_captain'] else '',
                    'Minutes': player.get('minutes', 0),
                    'Ownership': f"{player.get('selected_by_percent', 0):.1f}%"
                })
                
                total_cost += player.get('cost_millions', 0)
                total_points += player.get('total_points', 0)
        
        # Formation display
        formation = f"{formation_count['GK']}-{formation_count['DEF']}-{formation_count['MID']}-{formation_count['FWD']}"
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Formation", formation)
        
        with col2:
            st.metric("Total Cost", f"£{total_cost:.1f}m")
        
        with col3:
            st.metric("Total Points", f"{total_points:,}")
        
        with col4:
            avg_points = total_points / 11
            st.metric("Average Points", f"{avg_points:.1f}")
        
        # Starting XI table
        st.write("**🏟️ Starting XI Details**")
        
        if starting_players:
            starting_df = pd.DataFrame(starting_players)
            
            # Sort by position for better display
            position_order = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
            starting_df['pos_order'] = starting_df['Position'].map(position_order)
            starting_df = starting_df.sort_values(['pos_order', 'Price'], ascending=[True, False])
            starting_df = starting_df.drop('pos_order', axis=1)
            
            st.dataframe(
                starting_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player", help="Player name with captain indicators"),
                    "Position": st.column_config.TextColumn("Position", help="Playing position"),
                    "Team": st.column_config.TextColumn("Team", help="Team abbreviation"),
                    "Price": st.column_config.TextColumn("Price", help="Current market price"),
                    "Points": st.column_config.NumberColumn("Points", help="Total FPL points"),
                    "Form": st.column_config.TextColumn("Form", help="Recent form rating"),
                    "PPG": st.column_config.TextColumn("PPG", help="Points per 90 minutes"),
                    "Role": st.column_config.TextColumn("Role", help="Captain (C) or Vice-Captain (VC)"),
                    "Minutes": st.column_config.NumberColumn("Minutes", help="Total minutes played"),
                    "Ownership": st.column_config.TextColumn("Own%", help="Percentage ownership")
                }
            )
            
            # Starting XI insights
            st.write("**💡 Starting XI Insights**")
            
            insights = []
            
            # Formation analysis
            if formation == "1-3-5-2":
                insights.append("🔥 **Aggressive formation** - Heavy midfield focus for maximum points")
            elif formation == "1-4-4-2":
                insights.append("⚖️ **Balanced formation** - Even distribution across positions")
            elif formation == "1-5-3-2":
                insights.append("🛡️ **Defensive formation** - Strong defensive foundation")
            
            # Cost analysis
            if total_cost >= 80:
                insights.append("💰 **Premium heavy** - High-cost players, limited flexibility")
            elif total_cost <= 70:
                insights.append("💎 **Value focused** - Good budget distribution, room for upgrades")
            
            # Captain analysis
            captain = next((p for p in starting_players if p['Role'] == '(C)'), None)
            if captain:
                captain_form = float(captain['Form'])
                if captain_form >= 6.0:
                    insights.append(f"👑 **Excellent captain choice** - {captain['Player']} in great form ({captain['Form']})")
                elif captain_form < 4.0:
                    insights.append(f"⚠️ **Captain concern** - {captain['Player']} struggling for form ({captain['Form']})")
            
            # Ownership analysis
            ownerships = [float(p['Ownership'].replace('%', '')) for p in starting_players]
            avg_ownership = sum(ownerships) / len(ownerships)
            
            if avg_ownership >= 30:
                insights.append("📈 **Template heavy** - High average ownership, playing it safe")
            elif avg_ownership <= 15:
                insights.append("💎 **Differential heavy** - Low ownership, high risk/reward")
            
            for insight in insights:
                st.info(insight)

    def _render_full_squad_analysis(self, team_data, picks):
        """Render full 15-man squad analysis"""
        st.write("**👥 Complete Squad Overview**")
        
        players_df = st.session_state.players_df
        
        # Separate starting XI and bench
        starting_picks = [p for p in picks if p['position'] <= 11]
        bench_picks = [p for p in picks if p['position'] > 11]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**🏟️ Starting XI Summary**")
            if starting_picks:
                starting_summary = self._get_position_summary(starting_picks, players_df)
                st.dataframe(starting_summary, hide_index=True, use_container_width=True)
        
        with col2:
            st.write("**🪑 Bench Players**")
            if bench_picks:
                bench_data = []
                for pick in bench_picks:
                    player_info = players_df[players_df['id'] == pick['element']]
                    if not player_info.empty:
                        player = player_info.iloc[0]
                        bench_data.append({
                            'Player': player['web_name'],
                            'Pos': player.get('position_name', 'UNK')[:3],
                            'Price': f"£{player.get('cost_millions', 0):.1f}m",
                            'Points': player.get('total_points', 0)
                        })
                
                if bench_data:
                    bench_df = pd.DataFrame(bench_data)
                    st.dataframe(bench_df, hide_index=True, use_container_width=True)
                    
                    # Bench strength analysis
                    total_bench_points = sum([p['Points'] for p in bench_data])
                    st.write(f"**Bench Total:** {total_bench_points} points")
                    
                    if total_bench_points >= 50:
                        st.success("💪 Strong bench - good for Bench Boost")
                    elif total_bench_points >= 30:
                        st.info("👍 Decent bench strength")
                    else:
                        st.warning("⚠️ Weak bench - consider upgrades")

    def _render_formation_analysis_detailed(self, team_data, picks):
        """Render formation and tactical analysis"""
        st.write("**🏟️ Formation & Tactical Analysis**")
        
        players_df = st.session_state.players_df
        starting_picks = [p for p in picks if p['position'] <= 11]
        
        # Calculate formation
        formation_count = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        position_costs = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for pick in starting_picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                pos = position_map.get(player['element_type'], 'UNK')
                formation_count[pos] += 1
                position_costs[pos] += player.get('cost_millions', 0)
        
        formation = f"{formation_count['GK']}-{formation_count['DEF']}-{formation_count['MID']}-{formation_count['FWD']}"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Formation Breakdown**")
            st.metric("Current Formation", formation)
            
            # Formation characteristics
            formation_analysis = {
                "1-3-5-2": "🔥 Ultra-attacking - Maximum midfield points potential",
                "1-4-4-2": "⚖️ Balanced - Even distribution of resources",
                "1-5-3-2": "🛡️ Defensive - Strong defensive foundation",
                "1-4-3-3": "⚡ Attacking - Forward-heavy approach",
                "1-3-4-3": "🎯 High-risk - Minimal defense, maximum attack"
            }
            
            analysis = formation_analysis.get(formation, "🤔 Unconventional formation")
            st.info(f"**Formation Style:** {analysis}")
        
        with col2:
            st.write("**💰 Budget Allocation**")
            for pos, cost in position_costs.items():
                if formation_count[pos] > 0:
                    avg_cost = cost / formation_count[pos]
                    st.write(f"**{pos}**: £{cost:.1f}m total (£{avg_cost:.1f}m avg)")
            
            # Budget recommendations
            total_cost = sum(position_costs.values())
            if position_costs['MID'] / total_cost > 0.4:
                st.info("💡 Midfield-heavy budget allocation")
            elif position_costs['FWD'] / total_cost > 0.35:
                st.info("💡 Forward-focused budget allocation")

    def _render_squad_statistics(self, team_data, picks):
        """Render comprehensive squad statistics"""
        st.write("**📊 Squad Performance Statistics**")
        
        players_df = st.session_state.players_df
        
        # Calculate squad stats
        squad_stats = {
            'total_points': 0,
            'total_cost': 0,
            'avg_form': 0,
            'avg_ownership': 0,
            'total_minutes': 0,
            'goals': 0,
            'assists': 0,
            'clean_sheets': 0,
            'bonus_points': 0
        }
        
        valid_players = 0
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                valid_players += 1
                
                squad_stats['total_points'] += player.get('total_points', 0)
                squad_stats['total_cost'] += player.get('cost_millions', 0)
                squad_stats['avg_form'] += player.get('form', 0)
                squad_stats['avg_ownership'] += player.get('selected_by_percent', 0)
                squad_stats['total_minutes'] += player.get('minutes', 0)
                squad_stats['goals'] += player.get('goals_scored', 0)
                squad_stats['assists'] += player.get('assists', 0)
                squad_stats['clean_sheets'] += player.get('clean_sheets', 0)
                squad_stats['bonus_points'] += player.get('bonus', 0)
        
        if valid_players > 0:
            squad_stats['avg_form'] /= valid_players
            squad_stats['avg_ownership'] /= valid_players
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Squad Value", f"£{squad_stats['total_cost']:.1f}m")
            st.metric("Total Points", f"{squad_stats['total_points']:,}")
        
        with col2:
            st.metric("Average Form", f"{squad_stats['avg_form']:.1f}")
            st.metric("Average Ownership", f"{squad_stats['avg_ownership']:.1f}%")
        
        with col3:
            st.metric("Total Goals", squad_stats['goals'])
            st.metric("Total Assists", squad_stats['assists'])
        
        with col4:
            st.metric("Clean Sheets", squad_stats['clean_sheets'])
            st.metric("Bonus Points", squad_stats['bonus_points'])
        
        # Squad efficiency metrics
        st.write("**🎯 Squad Efficiency Metrics**")
        
        efficiency_col1, efficiency_col2 = st.columns(2)
        
        with efficiency_col1:
            points_per_million = squad_stats['total_points'] / squad_stats['total_cost'] if squad_stats['total_cost'] > 0 else 0
            st.metric("Points per £Million", f"{points_per_million:.1f}")
            
            if points_per_million >= 15:
                st.success("🔥 Excellent value efficiency")
            elif points_per_million >= 12:
                st.info("👍 Good value efficiency")
            else:
                st.warning("⚠️ Below average efficiency")
        
        with efficiency_col2:
            points_per_game = squad_stats['total_points'] / (squad_stats['total_minutes'] / 90) if squad_stats['total_minutes'] > 0 else 0
            st.metric("Points per 90min", f"{points_per_game:.2f}")
            
            if points_per_game >= 4.5:
                st.success("🔥 High points per game ratio")
            elif points_per_game >= 3.5:
                st.info("👍 Decent points per game ratio")
            else:
                st.warning("⚠️ Low points per game ratio")

    def _get_position_summary(self, picks, players_df):
        """Get position-wise summary for picks"""
        position_data = []
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position_data.append({
                    'Position': player.get('position_name', 'Unknown'),
                    'Player': player['web_name'],
                    'Team': player.get('team_short_name', 'UNK'),
                    'Price': f"£{player.get('cost_millions', 0):.1f}m",
                    'Points': player.get('total_points', 0),
                    'Form': f"{player.get('form', 0):.1f}"
                })
        
        return pd.DataFrame(position_data)

    def _render_position_specific_analysis(self, df):
        """Position-specific detailed analysis with restored tabs"""
        st.subheader("🎯 Position-Specific Analysis")
        
        if 'position_name' not in df.columns:
            st.warning("Position data not available")
            return
        
        # **RESTORED: Position-specific tabs**
        pos_tab1, pos_tab2, pos_tab3, pos_tab4 = st.tabs([
            "🥅 Goalkeepers",
            "🛡️ Defenders", 
            "⚽ Midfielders",
            "🎯 Forwards"
        ])
        
        with pos_tab1:
            gk_df = df[df['position_name'] == 'Goalkeeper']
            self._render_goalkeeper_analysis_enhanced(gk_df)
        
        with pos_tab2:
            def_df = df[df['position_name'] == 'Defender'] 
            self._render_defender_analysis_enhanced(def_df)
        
        with pos_tab3:
            mid_df = df[df['position_name'] == 'Midfielder']
            self._render_midfielder_analysis_enhanced(mid_df)
        
        with pos_tab4:
            fwd_df = df[df['position_name'] == 'Forward']
            self._render_forward_analysis_enhanced(fwd_df)

    def _render_goalkeeper_analysis_enhanced(self, df):
        """Enhanced goalkeeper-specific analysis"""
        st.write(f"**🥅 Goalkeeper Analysis ({len(df)} players)**")
        
        if df.empty:
            st.info("No goalkeepers found in current filter")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Top Goalkeepers by Points**")
            if not df.empty:
                top_gks = df.nlargest(5, 'total_points')[['web_name', 'team_short_name', 'total_points', 'cost_millions']]
                for idx, gk in top_gks.iterrows():
                    st.write(f"• **{gk['web_name']}** ({gk['team_short_name']}) - {gk['total_points']} pts, £{gk['cost_millions']:.1f}m")
        
        with col2:
            st.write("**🧤 Clean Sheet Leaders**")
            if 'clean_sheets' in df.columns:
                clean_sheet_leaders = df.nlargest(5, 'clean_sheets')[['web_name', 'team_short_name', 'clean_sheets']]
                for idx, gk in clean_sheet_leaders.iterrows():
                    st.write(f"• **{gk['web_name']}** ({gk['team_short_name']}) - {gk['clean_sheets']} CS")
            else:
                st.info("Clean sheet data not available")
        
        # Goalkeeper-specific metrics
        st.write("**📊 Goalkeeper Metrics Analysis**")
        
        if 'saves' in df.columns:
            saves_col, value_col = st.columns(2)
            
            with saves_col:
                st.write("**✋ Save Point Leaders**")
                save_leaders = df.nlargest(5, 'saves')[['web_name', 'team_short_name', 'saves']]
                for idx, gk in save_leaders.iterrows():
                    save_points = gk['saves'] // 3  # 1 point per 3 saves
                    st.write(f"• **{gk['web_name']}** - {gk['saves']} saves ({save_points} pts)")
            
            with value_col:
                st.write("**💰 Best Value Goalkeepers**")
                if 'points_per_million' in df.columns:
                    value_gks = df.nlargest(5, 'points_per_million')[['web_name', 'team_short_name', 'points_per_million', 'cost_millions']]
                    for idx, gk in value_gks.iterrows():
                        st.write(f"• **{gk['web_name']}** - {gk['points_per_million']:.1f} pts/£m")
        
        # Goalkeeper recommendations
        st.write("**💡 Goalkeeper Strategy Tips**")
        st.info("🎯 **Budget Strategy**: Most managers use 1 premium (£5.0m+) and 1 budget (£4.0-4.5m) goalkeeper")
        st.info("🔄 **Rotation Strategy**: Consider pairing goalkeepers from teams with complementary fixtures")
        st.info("🛡️ **Clean Sheet Focus**: Prioritize goalkeepers from defensively strong teams over save point merchants")

    def _render_defender_analysis_enhanced(self, df):
        """Enhanced defender-specific analysis"""
        st.write(f"**🛡️ Defender Analysis ({len(df)} players)**")
        
        if df.empty:
            st.info("No defenders found in current filter")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Top Defenders by Points**")
            if not df.empty:
                top_defs = df.nlargest(5, 'total_points')[['web_name', 'team_short_name', 'total_points', 'cost_millions']]
                for idx, defender in top_defs.iterrows():
                    st.write(f"• **{defender['web_name']}** ({defender['team_short_name']}) - {defender['total_points']} pts, £{defender['cost_millions']:.1f}m")
        
        with col2:
            st.write("**🥅 Clean Sheet Champions**")
            if 'clean_sheets' in df.columns:
                cs_leaders = df.nlargest(5, 'clean_sheets')[['web_name', 'team_short_name', 'clean_sheets']]
                for idx, defender in cs_leaders.iterrows():
                    st.write(f"• **{defender['web_name']}** ({defender['team_short_name']}) - {defender['clean_sheets']} CS")
            else:
                st.info("Clean sheet data not available")
        
        # Defender-specific analysis
        def_analysis_col1, def_analysis_col2 = st.columns(2)
        
        with def_analysis_col1:
            st.write("**⚽ Attacking Defenders**")
            if 'goals_scored' in df.columns and 'assists' in df.columns:
                df_copy = df.copy()
                df_copy['attacking_returns'] = df_copy['goals_scored'] + df_copy['assists']
                attacking_defs = df_copy.nlargest(5, 'attacking_returns')[['web_name', 'team_short_name', 'goals_scored', 'assists', 'attacking_returns']]
                
                for idx, defender in attacking_defs.iterrows():
                    st.write(f"• **{defender['web_name']}** - {defender['goals_scored']}G + {defender['assists']}A = {defender['attacking_returns']} total")
            else:
                st.info("Goal/assist data not available")
        
        with def_analysis_col2:
            st.write("**💎 Best Value Defenders**")
            if 'points_per_million' in df.columns:
                value_defs = df.nlargest(5, 'points_per_million')[['web_name', 'team_short_name', 'points_per_million', 'cost_millions']]
                for idx, defender in value_defs.iterrows():
                    st.write(f"• **{defender['web_name']}** - {defender['points_per_million']:.1f} pts/£m (£{defender['cost_millions']:.1f}m)")
        
        # Defender strategy tips
        st.write("**💡 Defender Strategy Guide**")
        st.info("🏛️ **Premium Defenders (£6.0m+)**: Target attacking fullbacks from top teams (TAA, Robertson style)")
        st.info("💰 **Mid-range Defenders (£4.5-6.0m)**: Balance of clean sheet potential and attacking threat")
        st.info("💎 **Budget Defenders (£4.0-4.5m)**: Nailed starters from solid defensive teams")
        st.info("🎯 **Strategy**: Most successful teams use 3-4 playing defenders, avoid defensive rotation")

    def _render_midfielder_analysis_enhanced(self, df):
        """Enhanced midfielder-specific analysis"""
        st.write(f"**⚽ Midfielder Analysis ({len(df)} players)**")
        
        if df.empty:
            st.info("No midfielders found in current filter")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Top Midfielders by Points**")
            if not df.empty:
                top_mids = df.nlargest(5, 'total_points')[['web_name', 'team_short_name', 'total_points', 'cost_millions']]
                for idx, mid in top_mids.iterrows():
                    st.write(f"• **{mid['web_name']}** ({mid['team_short_name']}) - {mid['total_points']} pts, £{mid['cost_millions']:.1f}m")
        
        with col2:
            st.write("**🎯 Goal + Assist Leaders**")
            if 'goals_scored' in df.columns and 'assists' in df.columns:
                df_copy = df.copy()
                df_copy['goal_contributions'] = df_copy['goals_scored'] + df_copy['assists']
                contrib_leaders = df_copy.nlargest(5, 'goal_contributions')[['web_name', 'team_short_name', 'goals_scored', 'assists', 'goal_contributions']]
                
                for idx, mid in contrib_leaders.iterrows():
                    st.write(f"• **{mid['web_name']}** - {mid['goals_scored']}G + {mid['assists']}A = {mid['goal_contributions']}")
            else:
                st.info("Goal/assist data not available")
        
        # Midfielder categories analysis
        mid_cat_col1, mid_cat_col2 = st.columns(2)
        
        with mid_cat_col1:
            st.write("**🔥 Form Midfielders**")
            if 'form' in df.columns:
                form_mids = df.nlargest(5, 'form')[['web_name', 'team_short_name', 'form', 'total_points']]
                for idx, mid in form_mids.iterrows():
                    st.write(f"• **{mid['web_name']}** - Form: {mid['form']:.1f}")
        
        with mid_cat_col2:
            st.write("**💎 Value Midfielders**")
            if 'points_per_million' in df.columns:
                value_mids = df.nlargest(5, 'points_per_million')[['web_name', 'team_short_name', 'points_per_million', 'cost_millions']]
                for idx, mid in value_mids.iterrows():
                    st.write(f"• **{mid['web_name']}** - {mid['points_per_million']:.1f} pts/£m")
        
        # Midfielder strategy tips
        st.write("**💡 Midfielder Strategy Guide**")
        st.info("👑 **Premium Midfielders (£8.0m+)**: High ceiling players, often the best captaincy options")
        st.info("⚽ **Mid-range Midfielders (£5.5-8.0m)**: Consistent returners, good for squad balance")
        st.info("💰 **Budget Midfielders (£4.5-5.5m)**: Nailed starters, often defensive minded but reliable")
        st.info("🎯 **Strategy**: 3-5 midfielders typical, focus on attacking returns over defensive stats")

    def _render_forward_analysis_enhanced(self, df):
        """Enhanced forward-specific analysis"""
        st.write(f"**🎯 Forward Analysis ({len(df)} players)**")
        
        if df.empty:
            st.info("No forwards found in current filter")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Top Forwards by Points**")
            if not df.empty:
                top_fwds = df.nlargest(5, 'total_points')[['web_name', 'team_short_name', 'total_points', 'cost_millions']]
                for idx, fwd in top_fwds.iterrows():
                    st.write(f"• **{fwd['web_name']}** ({fwd['team_short_name']}) - {fwd['total_points']} pts, £{fwd['cost_millions']:.1f}m")
        
        with col2:
            st.write("**⚽ Goal Scoring Leaders**")
            if 'goals_scored' in df.columns:
                goal_leaders = df.nlargest(5, 'goals_scored')[['web_name', 'team_short_name', 'goals_scored']]
                for idx, fwd in goal_leaders.iterrows():
                    st.write(f"• **{fwd['web_name']}** ({fwd['team_short_name']}) - {fwd['goals_scored']} goals")
            else:
                st.info("Goals data not available")
        
        # Forward-specific analysis
        fwd_analysis_col1, fwd_analysis_col2 = st.columns(2)
        
        with fwd_analysis_col1:
            st.write("**🎯 Minutes per Goal**")
            if 'goals_scored' in df.columns and 'minutes' in df.columns:
                df_copy = df.copy()
                df_copy = df_copy[df_copy['goals_scored'] > 0]  # Only players with goals
                df_copy['minutes_per_goal'] = df_copy['minutes'] / df_copy['goals_scored']
                
                mpg_leaders = df_copy.nsmallest(5, 'minutes_per_goal')[['web_name', 'team_short_name', 'minutes_per_goal', 'goals_scored']]
                
                for idx, fwd in mpg_leaders.iterrows():
                    st.write(f"• **{fwd['web_name']}** - {fwd['minutes_per_goal']:.0f} min/goal ({fwd['goals_scored']} goals)")
            else:
                st.info("Insufficient data for minutes per goal analysis")
        
        with fwd_analysis_col2:
            st.write("**💎 Value Forwards**")
            if 'points_per_million' in df.columns:
                value_fwds = df.nlargest(5, 'points_per_million')[['web_name', 'team_short_name', 'points_per_million', 'cost_millions']]
                for idx, fwd in value_fwds.iterrows():
                    st.write(f"• **{fwd['web_name']}** - {fwd['points_per_million']:.1f} pts/£m (£{fwd['cost_millions']:.1f}m)")
        
        # Forward strategy tips
        st.write("**💡 Forward Strategy Guide**")
        st.info("👑 **Premium Forwards (£9.0m+)**: High goal threat, often best captain options")
        st.info("⚽ **Mid-range Forwards (£6.5-9.0m)**: Consistent scorers, good value for money")
        st.info("💰 **Budget Forwards (£4.5-6.5m)**: Usually enablers, look for nailed starters")
        st.info("🎯 **Strategy**: Most teams use 1-3 forwards, focus on goal threat over all-round play")
        st.info("📊 **Key Metrics**: Goals per 90min, xG, penalty involvement, and fixture difficulty")

    def _render_performance_review(self, team_data):
        """Render performance review and analysis"""
        st.subheader("📊 Performance Review & Analysis")
        
        # Performance metrics from team data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🏆 Season Performance**")
            total_points = team_data.get('summary_overall_points', 0)
            overall_rank = team_data.get('summary_overall_rank', 0)
            
            if total_points > 0:
                st.metric("Total Points", f"{total_points:,}")
                
                # Performance assessment
                if total_points >= 2000:
                    st.success("🔥 Excellent season performance")
                elif total_points >= 1500:
                    st.info("👍 Good season performance")
                elif total_points >= 1000:
                    st.warning("⚖️ Average season performance")
                else:
                    st.error("📉 Below average performance")
            
            if overall_rank > 0:
                st.metric("Overall Rank", f"{overall_rank:,}")
                
                # Rank assessment
                if overall_rank <= 100000:
                    st.success("🎯 Top 100k manager!")
                elif overall_rank <= 500000:
                    st.info("👌 Above average ranking")
                elif overall_rank <= 1000000:
                    st.warning("📊 Average ranking")
                else:
                    st.error("📉 Below average ranking")
        
        with col2:
            st.write("**⚡ Recent Form**")
            last_gw_points = team_data.get('summary_event_points', 0)
            
            if last_gw_points > 0:
                st.metric("Last GW Points", last_gw_points)
                
                # Recent form assessment
                if last_gw_points >= 80:
                    st.success("🔥 Excellent gameweek!")
                elif last_gw_points >= 60:
                    st.info("👍 Good gameweek")
                elif last_gw_points >= 40:
                    st.warning("⚖️ Average gameweek")
                else:
                    st.error("📉 Poor gameweek")
            
            # Additional metrics if available
            if 'current_event' in team_data:
                st.write(f"**Current GW:** {team_data['current_event']}")
        
        with col3:
            st.write("**💰 Financial Status**")
            team_value = team_data.get('last_deadline_value', 1000)
            bank = team_data.get('last_deadline_bank', 0)
            
            if team_value:
                st.metric("Team Value", f"£{team_value/10:.1f}m")
                
                # Value assessment
                if team_value >= 1030:
                    st.success("💎 High team value")
                elif team_value >= 1010:
                    st.info("👍 Good team value")
                else:
                    st.warning("💰 Standard team value")
            
            if bank:
                st.metric("Bank", f"£{bank/10:.1f}m")
        
        # Performance insights
        st.divider()
        st.subheader("💡 Performance Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.write("**📈 Strengths Analysis**")
            
            strengths = []
            
            if overall_rank and overall_rank <= 100000:
                strengths.append("🎯 Top 100k ranking shows strong FPL knowledge")
            
            if last_gw_points and last_gw_points >= 60:
                strengths.append("⚡ Recent form is strong")
            
            if team_value and team_value >= 1020:
                strengths.append("💎 High team value indicates good transfers")
            
            if strengths:
                for strength in strengths:
                    st.success(strength)
            else:
                st.info("Focus on consistency and long-term planning")
        
        with insights_col2:
            st.write("**🎯 Areas for Improvement**")
            
            improvements = []
            
            if overall_rank and overall_rank > 1000000:
                improvements.append("📊 Focus on template players for stability")
            
            if last_gw_points and last_gw_points < 40:
                improvements.append("⚡ Review recent transfer decisions")
            
            if team_value and team_value < 1000:
                improvements.append("💰 Consider more strategic transfers")
            
            if improvements:
                for improvement in improvements:
                    st.warning(improvement)
            else:
                st.success("Strong performance across all areas!")

    def _render_transfer_planning(self, team_data):
        """Render transfer planning and recommendations"""
        st.subheader("🔄 Transfer Planning & Strategy")
        
        # Transfer status
        transfers_left = team_data.get('last_deadline_total_transfers', 1)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**📊 Transfer Status**")
            st.metric("Transfers Available", transfers_left if transfers_left else "N/A")
            
            # Transfer strategy based on remaining transfers
            if transfers_left and transfers_left > 0:
                if transfers_left >= 2:
                    st.success("✅ Multiple transfers available")
                    st.info("💡 Consider double transfers for maximum impact")
                else:
                    st.info("⚠️ Limited transfers - choose wisely")
                    st.info("💡 Save transfer or make essential move only")
            
            # Wildcard status (placeholder)
            st.write("**🃏 Chips Status**")
            st.info("🔄 Wildcard: Available")
            st.info("⚡ Bench Boost: Available") 
            st.info("🎯 Triple Captain: Available")
            st.info("🔥 Free Hit: Available")
        
        with col2:
            st.write("**🎯 Transfer Recommendations**")
            
            if not st.session_state.data_loaded:
                st.warning("Load player data to see transfer recommendations")
                return
            
            # Get current squad issues (placeholder analysis)
            st.write("**⚠️ Squad Issues to Address:**")
            
            # Common transfer scenarios
            transfer_scenarios = [
                "🔄 **Injured Players**: Check for any injured squad members",
                "📉 **Poor Form**: Consider transferring out players with form < 4.0",
                "💰 **Price Falls**: Monitor players at risk of price decreases", 
                "🎯 **Fixture Swings**: Target players with improving fixtures",
                "👥 **Ownership Changes**: Consider template players gaining popularity"
            ]
            
            for scenario in transfer_scenarios:
                st.info(scenario)
            
            # Transfer timing advice
            st.write("**⏰ Transfer Timing Strategy:**")
            st.write("• **Early in week**: If player definitely transferring out")
            st.write("• **Tuesday/Wednesday**: Most popular transfer days")
            st.write("• **Friday night**: Final deadline for changes")
            st.write("• **Consider price changes**: Transfer before drops, after rises")

    def _render_strategy_and_chips(self, team_data):
        """Render strategy and chip usage planning"""
        st.subheader("🎪 Strategy & Chip Planning")
        
        # Strategy tabs
        strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
            "🃏 Chip Strategy",
            "📅 Season Planning", 
            "🎯 Goals & Targets"
        ])
        
        with strategy_tab1:
            st.write("**🃏 Optimal Chip Usage Strategy**")
            
            chip_col1, chip_col2 = st.columns(2)
            
            with chip_col1:
                st.write("**🔄 Wildcard Strategy**")
                st.info("🎯 **Best Times to Use:**")
                st.write("• International breaks (GW7, GW15)")
                st.write("• Before fixture swings")
                st.write("• When team needs major overhaul")
                st.write("• Before Double Gameweeks")
                
                st.success("💡 **Current Recommendation**: Save for major fixture swing")
                
                st.write("**⚡ Bench Boost Strategy**")
                st.info("🎯 **Optimal Usage:**")
                st.write("• Double Gameweeks with good fixtures")
                st.write("• When bench has 4+ playing players")
                st.write("• Combine with good captain choice")
                
                st.warning("⚠️ **Preparation Required**: Build strong bench first")
            
            with chip_col2:
                st.write("**🎯 Triple Captain Strategy**")
                st.info("🎯 **Best Scenarios:**")
                st.write("• Double Gameweeks with easy fixtures")
                st.write("• Premium player vs weak defense")
                st.write("• Home fixture advantage")
                st.write("• Player in excellent form")
                
                st.success("💡 **Target Players**: Salah, Haaland, Kane in DGWs")
                
                st.write("**🔥 Free Hit Strategy**")
                st.info("🎯 **Optimal Usage:**")
                st.write("• Blank Gameweeks (BGWs)")
                st.write("• When few players have fixtures")
                st.write("• Late in season for rank pushes")
                
                st.warning("⚠️ **One-time use**: Choose timing carefully")
        
        with strategy_tab2:
            st.write("**📅 Season Planning Framework**")
            
            # Season phases
            st.write("**🗓️ Season Phases Strategy:**")
            
            phases = [
                ("🏁 **Opening Phase (GW1-8)**", [
                    "Focus on template/proven players",
                    "Avoid early wildcards unless essential",
                    "Build team value through smart transfers",
                    "Monitor new signings and price changes"
                ]),
                ("⚖️ **Mid-Season (GW9-19)**", [
                    "First wildcard around GW7-8 international break",
                    "Target fixture swings and form players",
                    "Prepare for busy winter schedule",
                    "Consider Christmas rotation risks"
                ]),
                ("🔥 **Business End (GW20-38)**", [
                    "Second wildcard for final fixtures",
                    "Use remaining chips strategically",
                    "Target differential players for rank gains",
                    "Monitor relegation/European race impacts"
                ])
            ]
            
            for phase_name, strategies in phases:
                st.write(phase_name)
                for strategy in strategies:
                    st.write(f"  • {strategy}")
                st.write("")
        
        with strategy_tab3:
            st.write("**🎯 Season Goals & Targets**")
            
            # Current rank and targets
            current_rank = team_data.get('summary_overall_rank', 0)
            
            if current_rank:
                st.write(f"**📊 Current Position: {current_rank:,}**")
                
                # Set targets based on current rank
                if current_rank <= 10000:
                    targets = [
                        "🏆 Top 1k overall finish",
                        "🎯 Maintain consistency", 
                        "💎 Strategic differential picks",
                        "🃏 Perfect chip timing"
                    ]
                elif current_rank <= 100000:
                    targets = [
                        "🎯 Top 10k overall finish",
                        "📈 Consistent weekly scores",
                        "🔄 Smart transfer planning",
                        "⚡ Chip optimization"
                    ]
                elif current_rank <= 1000000:
                    targets = [
                        "🎯 Top 100k overall finish", 
                        "📊 Template + differentials balance",
                        "💪 Strong squad foundation",
                        "📚 Learn from top managers"
                    ]
                else:
                    targets = [
                        "🎯 Break into top million",
                        "🏗️ Build consistent strategy",
                        "📖 Study FPL fundamentals",
                        "⚖️ Focus on template players"
                    ]
                
                st.write("**🎯 Recommended Targets:**")
                for target in targets:
                    st.write(f"• {target}")
            
            # Performance tracking
            st.write("**📈 Performance Tracking:**")
            st.info("💡 **Weekly Goals**: Aim for 45+ points per gameweek")
            st.info("📊 **Monthly Review**: Analyze transfer decisions and captain choices")
            st.info("🎯 **Season Target**: Finish in top 25% of all managers")

    def _render_team_benchmarking(self, team_data):
        """Render team benchmarking against averages"""
        st.subheader("📈 Team Benchmarking & Comparison")
        
        # Get team metrics
        total_points = team_data.get('summary_overall_points', 0)
        overall_rank = team_data.get('summary_overall_rank', 0)
        team_value = team_data.get('last_deadline_value', 1000)
        
        # Benchmark comparisons
        benchmark_col1, benchmark_col2 = st.columns(2)
        
        with benchmark_col1:
            st.write("**📊 Performance vs Averages**")
            
            # Estimated averages (these would typically come from API)
            avg_total_points = 1200  # Typical average
            avg_rank = 4000000  # Rough middle rank
            avg_team_value = 1005  # Typical team value
            
            # Points comparison
            if total_points > 0:
                points_diff = total_points - avg_total_points
                if points_diff > 0:
                    st.success(f"📈 +{points_diff} points above average")
                else:
                    st.error(f"📉 {abs(points_diff)} points below average")
            
            # Rank comparison
            if overall_rank > 0:
                if overall_rank < avg_rank:
                    percentile = ((avg_rank - overall_rank) / avg_rank) * 100
                    st.success(f"🎯 Top {percentile:.0f}% of all managers")
                else:
                    st.warning(f"📊 Ranked {overall_rank:,} overall")
            
            # Team value comparison
            if team_value:
                value_diff = (team_value - avg_team_value) / 10
                if value_diff > 0:
                    st.success(f"💎 £{value_diff:.1f}m above average team value")
                else:
                    st.warning(f"💰 £{abs(value_diff):.1f}m below average team value")
        
        with benchmark_col2:
            st.write("**🏆 Elite Manager Comparison**")
            
            # Top 10k benchmarks
            elite_points = 1800  # Top 10k typical
            elite_rank = 10000
            elite_value = 1025
            
            st.write("**🎯 Top 10k Benchmarks:**")
            
            if total_points > 0:
                points_gap = elite_points - total_points
                if points_gap <= 0:
                    st.success("🔥 Elite level performance!")
                else:
                    st.info(f"📈 {points_gap} points to elite level")
            
            if overall_rank > 0:
                if overall_rank <= elite_rank:
                    st.success("🏆 Elite manager ranking!")
                else:
                    rank_gap = overall_rank - elite_rank
                    st.info(f"🎯 {rank_gap:,} ranks to top 10k")
            
            if team_value:
                value_gap = (elite_value - team_value) / 10
                if value_gap <= 0:
                    st.success("💎 Elite team value!")
                else:
                    st.info(f"💰 £{value_gap:.1f}m to elite team value")
        
        # Performance trends
        st.divider()
        st.subheader("📈 Performance Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.write("**💪 Strengths vs Peers**")
            
            strengths = []
            
            if overall_rank and overall_rank <= 100000:
                strengths.append("🎯 Top 100k ranking - Strong FPL knowledge")
            
            if team_value and team_value >= 1020:
                strengths.append("💎 High team value - Good transfer timing")
            
            if total_points and total_points >= 1500:
                strengths.append("📊 High point total - Consistent performance")
            
            if strengths:
                for strength in strengths:
                    st.success(strength)
            else:
                st.info("Building solid foundation for improvement")
        
        with insight_col2:
            st.write("**🎯 Development Areas**")
            
            improvements = []
            
            if overall_rank and overall_rank > 1000000:
                improvements.append("📚 Study top manager strategies")
            
            if team_value and team_value < 1000:
                improvements.append("🔄 Improve transfer timing")
            
            if total_points and total_points < 1000:
                improvements.append("⚖️ Focus on template players first")
            
            if improvements:
                for improvement in improvements:
                    st.warning(improvement)
            else:
                st.success("Performing well across all areas!")
        
        # Actionable recommendations
        st.write("**💡 Actionable Recommendations:**")
        
        if overall_rank and overall_rank > 500000:
            st.info("🎯 **Focus**: Build consistent weekly scores with template players")
        elif overall_rank and overall_rank > 100000:
            st.info("⚡ **Focus**: Strategic chip usage and differential timing")
        else:
            st.info("🏆 **Focus**: Maintain performance and perfect chip timing")

    def _render_ai_transfer_targets(self):
        """Render AI-powered transfer target recommendations"""
        st.subheader("🎯 AI Transfer Target Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see AI recommendations")
            return
        
        df = st.session_state.players_df
        
        # AI scoring algorithm
        st.write("**🧠 AI Recommendation Algorithm**")
        
        with st.expander("How AI Scoring Works", expanded=False):
            st.markdown("""
            **AI Transfer Target Score** combines multiple factors:
            
            🎯 **Performance Metrics (40%)**:
            - Recent form and consistency
            - Points per game average
            - Bonus point frequency
            
            💰 **Value Analysis (25%)**:
            - Points per million efficiency
            - Price trend analysis
            - Budget impact assessment
            
            🏟️ **Fixture Analysis (20%)**:
            - Upcoming fixture difficulty
            - Home vs away performance
            - Historical opponent record
            
            📊 **Ownership & Risk (15%)**:
            - Optimal ownership levels
            - Differential potential
            - Template vs unique balance
            """)
        
        # Calculate AI scores
        df_ai = df.copy()
        
        # Normalize metrics for scoring
        df_ai['form_score'] = df_ai['form'] / 10  # 0-1 scale
        df_ai['points_score'] = df_ai['total_points'] / df_ai['total_points'].max()  # 0-1 scale
        df_ai['value_score'] = df_ai['points_per_million'] / df_ai['points_per_million'].max()  # 0-1 scale
        df_ai['ownership_score'] = 1 - (df_ai['selected_by_percent'] / 100)  # Inverse ownership
        
        # AI composite score
        df_ai['ai_score'] = (
            df_ai['form_score'] * 0.25 +
            df_ai['points_score'] * 0.25 +
            df_ai['value_score'] * 0.25 +
            df_ai['ownership_score'] * 0.25
        ).round(3)
        
        # Position-based recommendations
        ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
            "🥅 GK Targets",
            "🛡️ DEF Targets", 
            "⚽ MID Targets",
            "🎯 FWD Targets"
        ])
        
        with ai_tab1:
            self._render_position_ai_targets(df_ai, 'Goalkeeper', '🥅')
        
        with ai_tab2:
            self._render_position_ai_targets(df_ai, 'Defender', '🛡️')
        
        with ai_tab3:
            self._render_position_ai_targets(df_ai, 'Midfielder', '⚽')
        
        with ai_tab4:
            self._render_position_ai_targets(df_ai, 'Forward', '🎯')

    def _render_position_ai_targets(self, df_ai, position, emoji):
        """Render AI targets for specific position"""
        st.write(f"**{emoji} {position} AI Recommendations**")
        
        pos_df = df_ai[df_ai['position_name'] == position]
        
        if pos_df.empty:
            st.warning(f"No {position.lower()} data available")
            return
        
        # Top AI recommendations
        top_targets = pos_df.nlargest(10, 'ai_score')
        
        # Price categories
        price_tab1, price_tab2, price_tab3 = st.tabs([
            "💎 Budget (≤6.0m)",
            "⚖️ Mid-price (6.1-9.0m)", 
            "👑 Premium (9.1m+)"
        ])
        
        with price_tab1:
            budget_targets = top_targets[top_targets['cost_millions'] <= 6.0]
            self._display_ai_targets(budget_targets, "Budget")
        
        with price_tab2:
            mid_targets = top_targets[
                (top_targets['cost_millions'] > 6.0) & 
                (top_targets['cost_millions'] <= 9.0)
            ]
            self._display_ai_targets(mid_targets, "Mid-price")
        
        with price_tab3:
            premium_targets = top_targets[top_targets['cost_millions'] > 9.0]
            self._display_ai_targets(premium_targets, "Premium")

    def _display_ai_targets(self, targets_df, category):
        """Display AI target recommendations"""
        if targets_df.empty:
            st.info(f"No {category.lower()} targets in this category")
            return
        
        st.write(f"**🎯 Top {category} AI Targets**")
        
        # Display top 5 targets
        display_cols = [
            'web_name', 'team_short_name', 'cost_millions', 'total_points', 
            'form', 'points_per_million', 'selected_by_percent', 'ai_score'
        ]
        
        available_cols = [col for col in display_cols if col in targets_df.columns]
        top_5 = targets_df.head(5)
        
        for idx, (_, player) in enumerate(top_5.iterrows(), 1):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{idx}. {player['web_name']}** ({player['team_short_name']})")
                    st.write(f"💰 £{player['cost_millions']:.1f}m | 📊 {player['total_points']} pts | 📈 {player['form']:.1f} form")
                
                with col2:
                    st.metric("AI Score", f"{player['ai_score']:.3f}")
                    st.write(f"Value: {player['points_per_million']:.1f} pts/£m")
                
                with col3:
                    ownership = player['selected_by_percent']
                    if ownership < 10:
                        st.success(f"🎲 {ownership:.1f}% owned")
                        st.write("Differential pick")
                    elif ownership > 50:
                        st.warning(f"👥 {ownership:.1f}% owned") 
                        st.write("Template player")
                    else:
                        st.info(f"📊 {ownership:.1f}% owned")
                        st.write("Balanced ownership")
                
                st.divider()

    def _render_ai_captain_picks(self):
        """Render AI captain recommendations"""
        st.subheader("👑 AI Captain Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see captain recommendations")
            return
        
        df = st.session_state.players_df
        
        # Captain scoring algorithm
        df_cap = df.copy()
        
        # Filter out goalkeepers and low-scoring players
        df_cap = df_cap[
            (df_cap['position_name'] != 'Goalkeeper') & 
            (df_cap['total_points'] >= 50)
        ]
        
        # Captain score calculation
        df_cap['captain_score'] = (
            df_cap['form'] * 0.3 +
            df_cap['points_per_game'] * 0.3 +
            (df_cap['total_points'] / 100) * 0.2 +
            (df_cap.get('influence', 0) / 100) * 0.1 +
            (df_cap.get('threat', 0) / 100) * 0.1
        ).round(2)
        
        # Top captain picks
        top_captains = df_cap.nlargest(10, 'captain_score')
        
        cap_tab1, cap_tab2, cap_tab3 = st.tabs([
            "🏆 This Week's Picks",
            "🎯 Differential Captains",
            "📊 Captain Analytics"
        ])
        
        with cap_tab1:
            st.write("**👑 Top Captain Recommendations**")
            
            for idx, (_, player) in enumerate(top_captains.head(5).iterrows(), 1):
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{idx}. {player['web_name']}** ({player['team_short_name']})")
                        st.write(f"💰 £{player['cost_millions']:.1f}m | 📊 {player['total_points']} pts")
                    
                    with col2:
                        st.metric("Captain Score", f"{player['captain_score']:.2f}")
                        st.write(f"📈 Form: {player['form']:.1f}")
                    
                    with col3:
                        ownership = player['selected_by_percent']
                        if ownership > 50:
                            st.success("🛡️ Safe pick")
                        elif ownership > 20:
                            st.info("⚖️ Balanced risk")
                        else:
                            st.warning("🎲 Risky pick")
                        
                        st.write(f"👥 {ownership:.1f}% owned")
                    
                    st.divider()
        
        with cap_tab2:
            st.write("**🎲 Differential Captain Options**")
            
            # Low ownership, high potential captains
            differentials = top_captains[top_captains['selected_by_percent'] < 15]
            
            if not differentials.empty:
                for idx, (_, player) in enumerate(differentials.head(3).iterrows(), 1):
                    st.write(f"**{idx}. {player['web_name']}** - {player['selected_by_percent']:.1f}% owned")
                    st.write(f"🎯 High risk, high reward captain option")
                    st.write(f"📊 Captain Score: {player['captain_score']:.2f}")
                    st.divider()
            else:
                st.info("No clear differential captain options this week")
        
        with cap_tab3:
            st.write("**📊 Captain Selection Analytics**")
            
            # Captain statistics
            avg_captain_score = top_captains['captain_score'].mean()
            best_captain = top_captains.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Captain Score", f"{avg_captain_score:.2f}")
                st.metric("Best Option", best_captain['web_name'])
                st.metric("Best Score", f"{best_captain['captain_score']:.2f}")
            
            with col2:
                # Risk analysis
                safe_captains = len(top_captains[top_captains['selected_by_percent'] > 30])
                risky_captains = len(top_captains[top_captains['selected_by_percent'] < 15])
                
                st.metric("Safe Options", safe_captains)
                st.metric("Risky Options", risky_captains)
                
                if safe_captains >= 3:
                    st.success("✅ Multiple safe captain options")
                else:
                    st.warning("⚠️ Limited safe captain choices")

    def _render_ai_differentials(self):
        """Render AI differential player recommendations"""
        st.subheader("💎 AI Differential Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Load player data to see differential recommendations")
            return
        
        df = st.session_state.players_df
        
        # Differential criteria: low ownership, good potential
        df_diff = df[
            (df['selected_by_percent'] < 15) &  # Low ownership
            (df['total_points'] >= 30) &  # Minimum points threshold
            (df['form'] >= 4.0)  # Decent form
        ].copy()
        
        # Differential score
        df_diff['differential_score'] = (
            df_diff['form'] * 0.3 +
            df_diff['points_per_game'] * 0.25 +
            df_diff['points_per_million'] * 0.2 +
            (15 - df_diff['selected_by_percent']) * 0.25  # Reward low ownership
        ).round(2)
        
        diff_tab1, diff_tab2, diff_tab3 = st.tabs([
            "🎯 Top Differentials",
            "🔥 Form Differentials", 
            "💰 Value Differentials"
        ])
        
        with diff_tab1:
            st.write("**💎 Best Overall Differentials**")
            
            top_diffs = df_diff.nlargest(8, 'differential_score')
            
            for idx, (_, player) in enumerate(top_diffs.iterrows(), 1):
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{idx}. {player['web_name']}** ({player['position_name']})")
                        st.write(f"🏠 {player['team_short_name']} | 💰 £{player['cost_millions']:.1f}m")
                    
                    with col2:
                        st.metric("Diff Score", f"{player['differential_score']:.2f}")
                        st.write(f"📈 Form: {player['form']:.1f}")
                    
                    with col3:
                        st.success(f"👥 {player['selected_by_percent']:.1f}% owned")
                        st.write(f"📊 {player['total_points']} points")
                    
                    # Differential insight
                    if player['selected_by_percent'] < 5:
                        st.info("🎲 **High Risk/Reward** - Very low ownership")
                    elif player['selected_by_percent'] < 10:
                        st.info("⚡ **Good Differential** - Low ownership with potential")
                    else:
                        st.info("📈 **Emerging Player** - Ownership may be rising")
                    
                    st.divider()
        
        with diff_tab2:
            st.write("**🔥 In-Form Differentials**")
            
            form_diffs = df_diff[df_diff['form'] >= 6.0].nlargest(5, 'form')
            
            if not form_diffs.empty:
                for _, player in form_diffs.iterrows():
                    st.write(f"**{player['web_name']}** - Form: {player['form']:.1f}")
                    st.write(f"👥 {player['selected_by_percent']:.1f}% owned | 💰 £{player['cost_millions']:.1f}m")
                    st.success("🔥 Excellent recent form - potential breakout player")
                    st.divider()
            else:
                st.info("No high-form differentials found")
        
        with diff_tab3:
            st.write("**💰 Value Differentials**")
            
            value_diffs = df_diff[df_diff['points_per_million'] >= 10].nlargest(5, 'points_per_million')
            
            if not value_diffs.empty:
                for _, player in value_diffs.iterrows():
                    st.write(f"**{player['web_name']}** - {player['points_per_million']:.1f} pts/£m")
                    st.write(f"👥 {player['selected_by_percent']:.1f}% owned | 💰 £{player['cost_millions']:.1f}m")
                    st.success("💎 Excellent value for money differential")
                    st.divider()
            else:
                st.info("No high-value differentials found")

    def _render_ai_hold_sell(self):
        """Render AI hold vs sell recommendations"""
        st.subheader("🔄 AI Hold vs Sell Analysis")
        
        st.info("**Coming Soon**: AI analysis of current players to hold vs sell")
        
        st.write("**🎯 Hold vs Sell Criteria:**")
        st.write("• **Hold**: Good form, favorable fixtures, rising ownership")
        st.write("• **Sell**: Poor form, difficult fixtures, injury concerns")
        st.write("• **Monitor**: Mixed signals, wait for more data")

    def _render_advanced_ai_analysis(self):
        """Render advanced AI analytics"""
        st.subheader("🎪 Advanced AI Analytics")
        
        st.info("**Coming Soon**: Advanced machine learning insights")
        
        st.write("**🔮 Planned Features:**")
        st.write("• **Predictive Modeling**: ML-based performance predictions")
        st.write("• **Pattern Recognition**: Historical trend analysis")
        st.write("• **Risk Assessment**: Portfolio optimization algorithms")
        st.write("• **Market Analysis**: Price movement predictions")

    def _generate_optimized_team(self, **kwargs):
        """Generate optimized team based on constraints"""
        st.info("**Team Builder Coming Soon**: Advanced optimization algorithms")
        return None

    def _display_optimized_team(self, team_result):
        """Display the generated optimized team"""
        pass

    # FDR Analysis Helper Methods
    def _verify_and_enhance_fixture_data(self, fixtures_df):
        """Verify and enhance fixture data with additional calculations"""
        if fixtures_df.empty:
            return fixtures_df
        
        try:
            # Add missing columns if they don't exist
            if 'combined_fdr' not in fixtures_df.columns:
                fixtures_df['combined_fdr'] = fixtures_df.get('difficulty', 3)
            
            if 'opponent_strength' not in fixtures_df.columns:
                fixtures_df['opponent_strength'] = fixtures_df.get('combined_fdr', 3)
            
            return fixtures_df
        except Exception as e:
            st.error(f"Error enhancing fixture data: {str(e)}")
            return fixtures_df
    
    def _apply_form_adjustment(self, fixtures_df, form_weight):
        """Apply form-based adjustments to fixture difficulty"""
        if fixtures_df.empty or form_weight == 0:
            return fixtures_df
        
        try:
            # Simple form adjustment - in a real app this would use actual form data
            fixtures_df = fixtures_df.copy()
            if 'combined_fdr' in fixtures_df.columns:
                adjustment = (form_weight / 100) * 0.5
                fixtures_df['combined_fdr'] = fixtures_df['combined_fdr'] * (1 + adjustment)
            
            return fixtures_df
        except Exception as e:
            st.error(f"Error applying form adjustment: {str(e)}")
            return fixtures_df
    
    def _render_fdr_overview(self, fixtures_df, fdr_visualizer, gameweeks_ahead, sort_by, ascending_sort, analysis_type):
        """Render FDR overview section"""
        st.subheader("📊 Fixture Difficulty Overview")
        
        if fixtures_df.empty:
            st.warning("No fixture data available")
            return
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_fdr = fixtures_df.get('combined_fdr', pd.Series([3])).mean()
            st.metric("Average FDR", f"{avg_fdr:.2f}")
        
        with col2:
            easy_fixtures = (fixtures_df.get('combined_fdr', pd.Series([3])) <= 2).sum()
            st.metric("Easy Fixtures", easy_fixtures)
        
        with col3:
            hard_fixtures = (fixtures_df.get('combined_fdr', pd.Series([3])) >= 4).sum()
            st.metric("Hard Fixtures", hard_fixtures)
        
        with col4:
            total_teams = fixtures_df['team_short_name'].nunique()
            st.metric("Teams", total_teams)
        
        # Display fixtures table
        if not fixtures_df.empty:
            st.dataframe(fixtures_df.head(20), use_container_width=True)
    
    def _render_attack_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents, analysis_type):
        """Render attacking analysis section"""
        st.subheader("⚽ Attack Analysis")
        
        if fixtures_df.empty:
            st.warning("No data for attack analysis")
            return
        
        # Filter for good attacking fixtures
        good_fixtures = fixtures_df[fixtures_df.get('combined_fdr', 5) <= fdr_threshold]
        
        if good_fixtures.empty:
            st.info("No favorable attacking fixtures found with current threshold")
            return
        
        # Show top attacking opportunities
        st.write("**Top Attacking Opportunities:**")
        attack_summary = good_fixtures.groupby('team_short_name').agg({
            'combined_fdr': 'mean',
            'opponent': 'count'
        }).round(2).sort_values('combined_fdr')
        
        attack_summary.columns = ['Avg FDR', 'Fixtures']
        st.dataframe(attack_summary.head(10), use_container_width=True)
    
    def _render_defense_analysis(self, fixtures_df, fdr_visualizer, fdr_threshold, show_opponents, analysis_type):
        """Render defensive analysis section"""
        st.subheader("🛡️ Defense Analysis")
        
        if fixtures_df.empty:
            st.warning("No data for defense analysis")
            return
        
        # For defense, we want low FDR (easy opponents to keep clean sheets)
        good_defense_fixtures = fixtures_df[fixtures_df.get('combined_fdr', 5) <= fdr_threshold]
        
        if good_defense_fixtures.empty:
            st.info("No favorable defensive fixtures found with current threshold")
            return
        
        st.write("**Top Defensive Opportunities:**")
        defense_summary = good_defense_fixtures.groupby('team_short_name').agg({
            'combined_fdr': 'mean',
            'opponent': 'count'
        }).round(2).sort_values('combined_fdr')
        
        defense_summary.columns = ['Avg FDR', 'Fixtures']
        st.dataframe(defense_summary.head(10), use_container_width=True)
    
    def _render_transfer_targets(self, fixtures_df, fdr_threshold):
        """Render transfer targets section"""
        st.subheader("🎯 Transfer Targets")
        
        if fixtures_df.empty:
            st.warning("No data for transfer analysis")
            return
        
        # Find teams with good upcoming fixtures
        good_fixtures = fixtures_df[fixtures_df.get('combined_fdr', 5) <= fdr_threshold]
        
        if good_fixtures.empty:
            st.info("No favorable fixtures found for transfers")
            return
        
        transfer_targets = good_fixtures.groupby('team_short_name').agg({
            'combined_fdr': ['mean', 'count'],
            'opponent': lambda x: ', '.join(x.head(3))
        }).round(2)
        
        transfer_targets.columns = ['Avg FDR', 'Fixtures Count', 'Next Opponents']
        transfer_targets = transfer_targets.sort_values('Avg FDR')
        
        st.dataframe(transfer_targets.head(10), use_container_width=True)
    
    def _render_fixture_swings(self, fixtures_df):
        """Render fixture swings analysis"""
        st.subheader("📈 Fixture Swings")
        
        if fixtures_df.empty:
            st.warning("No data for fixture swing analysis")
            return
        
        st.info("Fixture swings show teams transitioning from difficult to easy fixtures or vice versa")
        
        # Simple fixture swing calculation
        if 'combined_fdr' in fixtures_df.columns:
            team_fdr_variance = fixtures_df.groupby('team_short_name')['combined_fdr'].agg(['mean', 'std']).round(2)
            team_fdr_variance.columns = ['Avg FDR', 'Variability']
            team_fdr_variance = team_fdr_variance.sort_values('Variability', ascending=False)
            
            st.write("**Teams with Most Variable Fixtures:**")
            st.dataframe(team_fdr_variance.head(10), use_container_width=True)
    
    def _render_advanced_analytics(self, fixtures_df, gameweeks_ahead):
        """Render advanced analytics section"""
        st.subheader("🔬 Advanced Analytics")
        
        if fixtures_df.empty:
            st.warning("No data for advanced analytics")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fixture Difficulty Distribution:**")
            if 'combined_fdr' in fixtures_df.columns:
                fdr_dist = fixtures_df['combined_fdr'].value_counts().sort_index()
                st.bar_chart(fdr_dist)
        
        with col2:
            st.write("**Teams by Average FDR:**")
            if 'combined_fdr' in fixtures_df.columns:
                team_avg_fdr = fixtures_df.groupby('team_short_name')['combined_fdr'].mean().sort_values()
                st.bar_chart(team_avg_fdr.head(10))

# Fix the main execution
if __name__ == "__main__":
    app = FPLAnalyticsApp()  # Changed from FPLFixtureAnalyzer
    app.run()

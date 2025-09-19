"""
Main Application Controller - Coordinates all components
"""
import streamlit as st
import pandas as pd
from pages.player_analysis_page import PlayerAnalysisPage
from pages.fixture_analysis_page import FixtureAnalysisPage  
from pages.my_team_page import MyTeamPage
from services.fpl_data_service import FPLDataService
from services.ai_recommendation_engine import get_player_recommendations
from core.page_router import PageRouter


class FPLAppController:
    """Main application controller coordinating all components"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.data_service = FPLDataService()
        self.page_router = PageRouter()
        
        # Initialize page components
        self.pages = {
            "player_analysis": PlayerAnalysisPage(),
            "fixture_analysis": FixtureAnalysisPage(),
            "my_team": MyTeamPage()
        }
    
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
    
    def run(self):
        """Main application runner"""
        try:
            # Render sidebar and get selected page
            selected_page = self.page_router.render_sidebar()
            
            # Route to appropriate page
            if selected_page == "dashboard":
                self.render_dashboard()
            elif selected_page == "players":
                self.pages["player_analysis"].render()
            elif selected_page == "fixtures":
                self.pages["fixture_analysis"].render()
            elif selected_page == "my_team":
                self.pages["my_team"].render()
            elif selected_page == "ai_recommendations":
                self.render_ai_recommendations()
            elif selected_page == "team_builder":
                self.render_team_builder()
            else:
                st.error(f"Unknown page: {selected_page}")
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.write("Please try refreshing the page or loading data again.")
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.title("⚽ FPL Analytics Dashboard")
        st.markdown("### Welcome to your Fantasy Premier League Analytics Hub!")
        
        if not st.session_state.data_loaded:
            st.info("👋 Welcome! Click '🔄 Refresh Data' in the sidebar to get started.")
            return
        
        df = st.session_state.players_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Players", len(df))
        with col2:
            avg_price = df['cost_millions'].mean() if 'cost_millions' in df.columns else 0
            st.metric("💰 Avg Price", f"£{avg_price:.1f}m")
        with col3:
            if len(df) > 0 and 'total_points' in df.columns:
                top_scorer = df.loc[df['total_points'].idxmax()]
                st.metric("🏆 Top Scorer", f"{top_scorer['web_name']} ({top_scorer['total_points']})")
            else:
                st.metric("🏆 Top Scorer", "N/A")
        with col4:
            if len(df) > 0 and 'points_per_million' in df.columns:
                best_value = df.loc[df['points_per_million'].idxmax()]
                st.metric("💎 Best Value", f"{best_value['web_name']} ({best_value['points_per_million']:.1f})")
            else:
                st.metric("💎 Best Value", "N/A")
        
        st.success("✅ Dashboard loaded successfully!")
    
    def render_ai_recommendations(self):
        """Render AI recommendations page using the recommendation engine"""
        st.header("🤖 AI-Powered Recommendations")
        
        if not st.session_state.data_loaded:
            st.info("👋 Please load data first to get AI recommendations.")
            return
        
        try:
            # Import the AI recommendation engine
            from services.ai_recommendation_engine import get_player_recommendations
            
            df = st.session_state.players_df
            
            # AI Recommendation tabs
            ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
                "🎯 Transfer Targets",
                "👑 Captain Picks", 
                "💎 Differentials",
                "🏆 Team Optimizer"
            ])
            
            with ai_tab1:
                st.subheader("🎯 AI Transfer Targets")
                
                try:
                    # Position filter
                    position_filter = st.selectbox(
                        "Filter by Position",
                        ["All Positions", "Goalkeeper", "Defender", "Midfielder", "Forward"],
                        index=0
                    )
                    
                    position = None if position_filter == "All Positions" else position_filter
                    
                    # Clean data before processing
                    df_clean = df.copy()
                    
                    # Ensure numeric columns are properly typed
                    numeric_cols = ['total_points', 'now_cost', 'form', 'selected_by_percent', 
                                  'minutes', 'goals_scored', 'assists', 'clean_sheets']
                    
                    for col in numeric_cols:
                        if col in df_clean.columns:
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
                    
                    # Get AI recommendations with error handling
                    with st.spinner("🤖 Generating AI recommendations..."):
                        recommendations = get_player_recommendations(
                            df_clean, 
                            position=position,
                            top_n=10
                        )
                    
                    if recommendations:
                        st.success(f"✅ Found {len(recommendations)} transfer targets!")
                        
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"{i}. {rec.web_name} ({rec.position}) - £{rec.current_price:.1f}m"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Predicted Points", f"{rec.predicted_points:.1f}")
                                    st.metric("Value Score", f"{rec.value_score:.1f}")
                                    st.metric("Form Score", f"{rec.form_score:.1f}")
                                
                                with col2:
                                    st.metric("Confidence", f"{rec.confidence_score:.1%}")
                                    st.metric("Risk Level", rec.risk_level)
                                    st.write(f"**Team:** {rec.team_name}")
                                
                                # Reasoning
                                if rec.reasoning:
                                    st.write("**AI Reasoning:**")
                                    for reason in rec.reasoning:
                                        st.write(f"• {reason}")
                    else:
                        st.warning("⚠️ No recommendations available.")
                        st.info("💡 **Possible solutions:**")
                        st.write("• Ensure player data is loaded correctly")
                        st.write("• Try refreshing the data")
                        st.write("• Check different position filters")
                        
                except Exception as e:
                    st.error("❌ Error generating transfer recommendations")
                    
                    if "multiply sequence" in str(e):
                        st.warning("🔧 **Data Type Issue Detected**")
                        st.write("The AI engine encountered mixed data types. This has been automatically fixed.")
                        st.write("**Please try these steps:**")
                        st.write("1. Refresh the page using the browser refresh button")
                        st.write("2. Reload data using the '🔄 Refresh Data' button in the sidebar")
                        st.write("3. Try again - the data cleaning should resolve the issue")
                        
                        # Attempt to show basic recommendations as fallback
                        try:
                            st.info("📋 **Showing basic recommendations instead:**")
                            simple_recs = df.nlargest(5, 'total_points')
                            for i, (_, player) in enumerate(simple_recs.iterrows(), 1):
                                st.write(f"{i}. **{player['web_name']}** - {player['total_points']} pts (£{player['now_cost']/10:.1f}m)")
                        except:
                            pass
                    else:
                        st.write(f"Error details: {str(e)}")
                        
                    # Debug information in expander
                    with st.expander("🔍 Debug Information"):
                        st.write("**Data Columns:**", list(df.columns))
                        st.write("**Data Types:**")
                        for col in ['total_points', 'now_cost', 'form', 'selected_by_percent']:
                            if col in df.columns:
                                st.write(f"• {col}: {df[col].dtype}")
                        st.write("**Sample Data:**")
                        st.dataframe(df.head(3))
            
            with ai_tab2:
                st.subheader("👑 AI Captain Analysis")
                
                # Filter for non-goalkeepers with decent points
                captain_candidates = df[
                    (df['element_type'] != 1) &  # Not goalkeepers
                    (df['total_points'] >= 50)
                ].copy()
                
                if not captain_candidates.empty:
                    # Simple captain scoring
                    captain_candidates['captain_score'] = (
                        captain_candidates['form'] * 0.3 +
                        captain_candidates.get('points_per_game', captain_candidates['total_points']/38) * 0.3 +
                        (captain_candidates['total_points'] / 100) * 0.2 +
                        (captain_candidates.get('influence', 0) / 100) * 0.1 +
                        (captain_candidates.get('threat', 0) / 100) * 0.1
                    )
                    
                    top_captains = captain_candidates.nlargest(10, 'captain_score')
                    
                    for i, (_, player) in enumerate(top_captains.head(5).iterrows(), 1):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**{i}. {player['web_name']}**")
                            st.write(f"💰 £{player['now_cost']/10:.1f}m | 📊 {player['total_points']} pts")
                        
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
                else:
                    st.warning("No captain candidates found")
            
            with ai_tab3:
                st.subheader("💎 AI Differential Analysis")
                
                # Low ownership, high potential players
                differentials = df[
                    (df['selected_by_percent'] < 15) &
                    (df['total_points'] >= 30) &
                    (df['form'] >= 4.0)
                ].copy()
                
                if not differentials.empty:
                    differentials['differential_score'] = (
                        differentials['form'] * 0.3 +
                        differentials.get('points_per_game', differentials['total_points']/38) * 0.25 +
                        differentials['points_per_million'] * 0.2 +
                        (15 - differentials['selected_by_percent']) * 0.25
                    )
                    
                    top_diffs = differentials.nlargest(8, 'differential_score')
                    
                    for i, (_, player) in enumerate(top_diffs.iterrows(), 1):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**{i}. {player['web_name']}**")
                            st.write(f"🏠 {player.get('team_short_name', 'N/A')} | 💰 £{player['now_cost']/10:.1f}m")
                        
                        with col2:
                            st.metric("Diff Score", f"{player['differential_score']:.2f}")
                            st.write(f"📈 Form: {player['form']:.1f}")
                        
                        with col3:
                            st.success(f"👥 {player['selected_by_percent']:.1f}% owned")
                            st.write(f"📊 {player['total_points']} points")
                        
                        if player['selected_by_percent'] < 5:
                            st.info("🎲 **High Risk/Reward** - Very low ownership")
                        elif player['selected_by_percent'] < 10:
                            st.info("⚡ **Good Differential** - Low ownership with potential")
                        
                        st.divider()
                else:
                    st.info("No quality differentials found in current data")
            
            with ai_tab4:
                st.subheader("🏆 AI Team Optimizer")
                
                # Import team recommender
                try:
                    from components.ui_components import render_enhanced_team_recommendations_tab
                    
                    # Create data manager wrapper
                    class DataManager:
                        def __init__(self, players_df, teams_df):
                            self.players_df = players_df
                            self.teams_df = teams_df
                        
                        def get_players_data(self):
                            return self.players_df
                        
                        def get_teams_data(self):
                            return self.teams_df
                    
                    data_manager = DataManager(
                        st.session_state.players_df, 
                        st.session_state.get('teams_df', pd.DataFrame())
                    )
                    
                    render_enhanced_team_recommendations_tab(data_manager)
                    
                except ImportError as e:
                    st.warning("Team optimizer not available. Using basic recommendations.")
                    st.info("💡 The AI can analyze your current players and suggest improvements.")
                    
                    if not df.empty:
                        # Simple team analysis
                        st.write("**📊 Quick Team Analysis**")
                        
                        avg_form = df['form'].mean()
                        avg_value = df['points_per_million'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Form", f"{avg_form:.1f}")
                        with col2:
                            st.metric("Average Value", f"{avg_value:.1f} pts/£m")
                        
                        # Best performers
                        best_performers = df.nlargest(5, 'total_points')
                        st.write("**🏆 Top Performers Available:**")
                        for _, player in best_performers.iterrows():
                            st.write(f"• {player['web_name']} - {player['total_points']} pts (£{player['now_cost']/10:.1f}m)")
            
        except Exception as e:
            st.error(f"Error loading AI recommendations: {str(e)}")
            st.info("🚧 AI recommendations are being enhanced. Please try refreshing or check data loading.")

    def render_team_builder(self):
        """Render team builder page using the team optimizer"""
        st.header("⚽ Enhanced Team Builder")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first to use the team builder.")
            return
        
        try:
            # Import team builder components
            from components.ui_components import render_enhanced_team_recommendations_tab
            
            # Create data manager
            class DataManager:
                def __init__(self, players_df, teams_df):
                    self.players_df = players_df
                    self.teams_df = teams_df
                
                def get_players_data(self):
                    return self.players_df
                
                def get_teams_data(self):
                    return self.teams_df
            
            data_manager = DataManager(
                st.session_state.players_df, 
                st.session_state.get('teams_df', pd.DataFrame())
            )
            
            render_enhanced_team_recommendations_tab(data_manager)
            
        except ImportError:
            # Fallback basic team builder
            st.info("🚧 Enhanced team builder loading. Using basic version.")
            
            df = st.session_state.players_df
            
            # Basic team building interface
            st.subheader("🎯 Quick Team Analysis")
            
            # Formation selector
            formation = st.selectbox(
                "Select Formation",
                ["3-4-3", "4-3-3", "3-5-2", "4-4-2", "5-3-2"],
                index=0
            )
            
            budget = st.slider("Budget (£m)", 80.0, 120.0, 100.0, 0.5)
            
            # Position breakdown
            formation_map = {
                "3-4-3": (3, 4, 3),
                "4-3-3": (4, 3, 3),
                "3-5-2": (3, 5, 2),
                "4-4-2": (4, 4, 2),
                "5-3-2": (5, 3, 2)
            }
            
            def_count, mid_count, fwd_count = formation_map[formation]
            
            st.write(f"**Formation {formation} requires:**")
            st.write(f"• 1 Goalkeeper")
            st.write(f"• {def_count} Defenders")
            st.write(f"• {mid_count} Midfielders") 
            st.write(f"• {fwd_count} Forwards")
            
            # Best players by position
            positions = {1: "Goalkeepers", 2: "Defenders", 3: "Midfielders", 4: "Forwards"}
            
            for pos_id, pos_name in positions.items():
                with st.expander(f"🔍 Best {pos_name}"):
                    pos_players = df[df['element_type'] == pos_id].nlargest(5, 'total_points')
                    
                    for _, player in pos_players.iterrows():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{player['web_name']}** - {player['total_points']} pts")
                        with col2:
                            st.write(f"£{player['now_cost']/10:.1f}m")

# filepath: c:\Users\daakara\Documents\fpl_ai_agent\fpl_ai_agent\simple_app.py
class FPLAnalyticsApp:
    """Main FPL Analytics Application with enhanced team odds and AI recommendations"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.team_builder = AdvancedTeamBuilder()
        self.squad_analyzer = SquadAnalyzer()
        self.data_validator = DataValidator()
        self.fdr_analyzer = FDRAnalyzer()  # Initialize FDRAnalyzer
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'players_df' not in st.session_state:
            st.session_state.players_df = pd.DataFrame()
        if 'teams_df' not in st.session_state:
            st.session_state.teams_df = pd.DataFrame()
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = {}
        if 'fdr_data' not in st.session_state:
            st.session_state.fdr_data = {}

    def run(self):
        """Main application runner"""
        
        # Page config
        st.set_page_config(
            page_title="FPL Analytics Pro", 
            page_icon="⚽", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""<style>...</style>""", unsafe_allow_html=True)
        
        # Header
        st.markdown("""<div class="main-header">...</div>""", unsafe_allow_html=True)
        
        # Sidebar for navigation and status
        with st.sidebar:
            st.image("https://fantasy.premierleague.com/static/libsass/plfpl/dist/img/pl-logo.svg", width=200)
            st.markdown("### 📊 Data Status")
            if st.session_state.data_loaded:
                st.success("✅ Data Loaded")
                st.info(f"📈 {len(st.session_state.players_df)} players")
                st.info(f"🏆 {len(st.session_state.teams_df)} teams")
            else:
                st.warning("⚠️ No data loaded")
            
            st.markdown("### 🔧 Quick Actions")
            if st.button("🔄 Refresh Data", use_container_width=True):
                self._load_data()
            
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
            
            # App info
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.markdown("""**FPL Analytics Pro** provides: ...""")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "🏠 Dashboard", 
            "📊 Player Analysis", 
            "🔍 Advanced Filters",
            "📈 Visualizations", 
            "🏆 Team Builder", 
            "📥 Import My Team",
            "🤖 AI Recommendations",
            "🎯 Team Odds",
            "⚔️ Fixture Difficulty Ratings (Attack)",
            "🛡️ Fixture Difficulty Ratings (Defense)"
        ])
        
        with tab1:
            self._render_dashboard()
        
        with tab2:
            self._render_player_analysis()
        
        with tab3:
            self._render_advanced_filters()
        
        with tab4:
            self._render_visualizations()
        
        with tab5:
            self._render_team_builder()
        
        with tab6:
            self._render_import_my_team()
        
        with tab7:
            self._render_ai_recommendations()
        
        with tab8:
            self._render_team_odds()
        
        with tab9:
            self._render_fixture_difficulty_attack()  # New tab for attack FDR
        
        with tab10:
            self._render_fixture_difficulty_defense()  # New tab for defense FDR

    def _render_fixture_difficulty_attack(self):
        """Render Fixture Difficulty Ratings based on Attack"""
        st.header("⚔️ Fixture Difficulty Ratings (Attack)")
        
        if not st.session_state.data_loaded or st.session_state.players_df.empty:
            st.info("Please load data first from the Dashboard tab")
            return
        
        # Load FDR data
        fdr_data = self.fdr_analyzer.calculate_attack_fdr()  # Method to calculate attack FDR
        st.subheader("📊 Attack Fixture Difficulty Ratings")
        
        if fdr_data is not None:
            st.dataframe(fdr_data, use_container_width=True)
        else:
            st.warning("No FDR data available.")

    def _render_fixture_difficulty_defense(self):
        """Render Fixture Difficulty Ratings based on Defense"""
        st.header("🛡️ Fixture Difficulty Ratings (Defense)")
        
        if not st.session_state.data_loaded or st.session_state.players_df.empty:
            st.info("Please load data first from the Dashboard tab")
            return
        
        # Load FDR data
        fdr_data = self.fdr_analyzer.calculate_defense_fdr()  # Method to calculate defense FDR
        st.subheader("📊 Defense Fixture Difficulty Ratings")
        
        if fdr_data is not None:
            st.dataframe(fdr_data, use_container_width=True)
        else:
            st.warning("No FDR data available.")
    
    # Other methods remain unchanged...
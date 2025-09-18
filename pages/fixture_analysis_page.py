"""
Fixture Analysis Page - Handles fixture difficulty ratings and analysis
"""
import streamlit as st
import pandas as pd
from fixtures.fixture_data_loader import FixtureDataLoader
from fixtures.fdr_analyzer import FDRAnalyzer
from fixtures.fdr_visualizer import FDRVisualizer


class FixtureAnalysisPage:
    """Handles fixture analysis functionality"""
    
    def __init__(self):
        self.fixture_loader = FixtureDataLoader()
        self.fdr_analyzer = FDRAnalyzer()
        self.fdr_visualizer = FDRVisualizer()
    
    def render(self):
        """Main render method for fixture analysis page"""
        st.header("🎯 Fixture Difficulty Ratings (FDR)")
        
        # Comprehensive explanation
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
            """)
        
        st.markdown("### Analyze team fixtures to identify transfer targets and avoid traps")
        
        # Load fixtures data
        if not st.session_state.get('fdr_data_loaded', False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("Click the button to load fixture data and begin analysis")
            
            with col2:
                if st.button("📊 Load Fixture Data", type="primary"):
                    self._load_fixture_data()
            return
        
        fixtures_df = st.session_state.fixtures_df
        
        # Settings panel
        with st.expander("⚙️ FDR Settings", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                gameweeks_ahead = st.slider("Gameweeks to analyze:", 1, 15, 5)
                show_colors = st.checkbox("Show color coding", value=True)
            
            with col2:
                fdr_threshold = st.slider("Good fixture threshold:", 1.0, 4.0, 2.5, 0.1)
                show_opponents = st.checkbox("Show opponent names", value=True)
            
            with col3:
                sort_by = st.selectbox("Sort by:", ["Combined FDR", "Attack FDR", "Defense FDR", "Team Name"])
                ascending_sort = st.checkbox("Ascending order", value=True)
            
            with col4:
                use_form_adjustment = st.checkbox("Apply form adjustment", value=False)
                if use_form_adjustment:
                    form_weight = st.slider("Form weight:", 0.1, 0.5, 0.3, 0.1)
                else:
                    form_weight = 0.0
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "🎯 Analysis Focus:",
            ["All Fixtures", "Home Only", "Away Only", "Next 3 Fixtures"],
            help="Choose what type of fixtures to analyze"
        )
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "⚔️ Attack Analysis", 
            "🛡️ Defense Analysis", 
            "🎯 Transfer Targets"
        ])
        
        # Apply form adjustment if enabled
        if use_form_adjustment and st.session_state.get('data_loaded', False):
            fixtures_df = self.fdr_analyzer.apply_form_adjustment(fixtures_df, form_weight)
        
        with tab1:
            self._render_fdr_overview(fixtures_df, gameweeks_ahead, sort_by, ascending_sort, analysis_type)
        
        with tab2:
            self._render_attack_analysis(fixtures_df, fdr_threshold, show_opponents, analysis_type)
        
        with tab3:
            self._render_defense_analysis(fixtures_df, fdr_threshold, show_opponents, analysis_type)
        
        with tab4:
            self._render_transfer_targets(fixtures_df, fdr_threshold)
    
    def _load_fixture_data(self):
        """Load and process fixture data"""
        with st.spinner("Loading fixture data..."):
            try:
                # Load and process fixtures
                fixtures_df = self.fixture_loader.process_fixtures_data()
                
                if not fixtures_df.empty:
                    # Calculate FDR ratings
                    fixtures_df = self.fdr_analyzer.calculate_combined_fdr(fixtures_df)
                    
                    # Store in session state
                    st.session_state.fixtures_df = fixtures_df
                    st.session_state.fdr_data_loaded = True
                    st.success("✅ Fixture data loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load fixture data")
            except Exception as e:
                st.error(f"❌ Error loading fixture data: {str(e)}")
    
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
            else:
                return fixtures_df
        except Exception as e:
            st.warning(f"Fixture filtering failed: {str(e)}")
            return fixtures_df
    
    def _render_fdr_overview(self, fixtures_df, gameweeks_ahead, sort_by, ascending_sort, analysis_type):
        """Render FDR overview with heatmaps and summaries"""
        st.subheader(f"📊 FDR Overview - {analysis_type}")
        
        # Filter fixtures based on analysis type
        filtered_fixtures = self._filter_fixtures_by_type(fixtures_df, analysis_type)
        
        if filtered_fixtures.empty:
            st.warning("No fixture data available")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_fixtures = len(filtered_fixtures)
            teams_count = filtered_fixtures['team_short_name'].nunique()
            st.metric("📊 Fixtures Analyzed", f"{total_fixtures} ({teams_count} teams)")
        
        with col2:
            if 'attack_fdr' in filtered_fixtures.columns:
                avg_attack_fdr = filtered_fixtures['attack_fdr'].mean()
                st.metric("⚔️ Avg Attack FDR", f"{avg_attack_fdr:.2f}")
            else:
                st.metric("⚔️ Avg Attack FDR", "N/A")
        
        with col3:
            if 'defense_fdr' in filtered_fixtures.columns:
                avg_defense_fdr = filtered_fixtures['defense_fdr'].mean()
                st.metric("🛡️ Avg Defense FDR", f"{avg_defense_fdr:.2f}")
            else:
                st.metric("🛡️ Avg Defense FDR", "N/A")
        
        with col4:
            if 'combined_fdr' in filtered_fixtures.columns:
                avg_combined_fdr = filtered_fixtures['combined_fdr'].mean()
                st.metric("🎯 Avg Combined FDR", f"{avg_combined_fdr:.2f}")
            else:
                st.metric("🎯 Avg Combined FDR", "N/A")
        
        st.divider()
        
        # FDR Heatmap
        if 'combined_fdr' in filtered_fixtures.columns:
            st.subheader(f"🌡️ FDR Heatmap - {analysis_type}")
            
            fig_heatmap = self.fdr_visualizer.create_fdr_heatmap(filtered_fixtures, 'combined')
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # FDR Legend
            st.markdown("""
            **🎯 FDR Guide:**
            - 🟢 **1-2**: Excellent fixtures - Target these teams' players
            - 🟡 **3**: Average fixtures - Neutral stance
            - 🟠 **4**: Difficult fixtures - Consider avoiding
            - 🔴 **5**: Very difficult - Strong avoid
            """)
        
        # Team summary table
        st.subheader("📋 Team FDR Summary")
        
        if not filtered_fixtures.empty and 'combined_fdr' in filtered_fixtures.columns:
            team_summary = filtered_fixtures.groupby('team_short_name').agg({
                'combined_fdr': ['mean', 'min', 'max'],
                'fixture_number': 'count'
            }).round(2)
            
            team_summary.columns = ['Avg_FDR', 'Best_FDR', 'Worst_FDR', 'Fixtures']
            team_summary = team_summary.reset_index()
            
            # Sort based on user selection
            if sort_by == "Combined FDR":
                team_summary = team_summary.sort_values('Avg_FDR', ascending=ascending_sort)
            elif sort_by == "Team Name":
                team_summary = team_summary.sort_values('team_short_name', ascending=ascending_sort)
            
            st.dataframe(team_summary, use_container_width=True, hide_index=True)
        else:
            st.warning("No FDR data available for team summary")
    
    def _render_attack_analysis(self, fixtures_df, fdr_threshold, show_opponents, analysis_type):
        """Render attack FDR analysis"""
        st.subheader(f"⚔️ Attack FDR Analysis - {analysis_type}")
        st.info("🎯 Lower Attack FDR = Easier to score goals. Target these teams' forwards and attacking midfielders!")
        
        # Filter fixtures
        filtered_fixtures = self._filter_fixtures_by_type(fixtures_df, analysis_type)
        
        if filtered_fixtures.empty or 'attack_fdr' not in filtered_fixtures.columns:
            st.warning("Attack FDR data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Best attacking fixtures
            st.subheader("🟢 Best Attacking Fixtures")
            attack_summary = filtered_fixtures.groupby('team_short_name')['attack_fdr'].mean().sort_values().head(10)
            
            for team, fdr in attack_summary.items():
                color = "🟢" if fdr <= 2 else "🟡" if fdr <= 3 else "🔴"
                st.write(f"{color} **{team}**: {fdr:.2f} Attack FDR")
        
        with col2:
            # Attack FDR visualization
            fig_attack = self.fdr_visualizer.create_fdr_bar_chart(filtered_fixtures, 'attack')
            st.plotly_chart(fig_attack, use_container_width=True)
    
    def _render_defense_analysis(self, fixtures_df, fdr_threshold, show_opponents, analysis_type):
        """Render defense FDR analysis"""
        st.subheader(f"🛡️ Defense FDR Analysis - {analysis_type}")
        st.info("🏠 Lower Defense FDR = Easier to keep clean sheets. Target these teams' defenders and goalkeepers!")
        
        # Filter fixtures
        filtered_fixtures = self._filter_fixtures_by_type(fixtures_df, analysis_type)
        
        if filtered_fixtures.empty or 'defense_fdr' not in filtered_fixtures.columns:
            st.warning("Defense FDR data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Best defensive fixtures
            st.subheader("🟢 Best Defensive Fixtures")
            defense_summary = filtered_fixtures.groupby('team_short_name')['defense_fdr'].mean().sort_values().head(10)
            
            for team, fdr in defense_summary.items():
                color = "🟢" if fdr <= 2 else "🟡" if fdr <= 3 else "🔴"
                st.write(f"{color} **{team}**: {fdr:.2f} Defense FDR")
        
        with col2:
            # Defense FDR visualization
            fig_defense = self.fdr_visualizer.create_fdr_bar_chart(filtered_fixtures, 'defense')
            st.plotly_chart(fig_defense, use_container_width=True)
    
    def _render_transfer_targets(self, fixtures_df, fdr_threshold):
        """Render transfer targets based on fixtures"""
        st.subheader("🎯 Transfer Targets & Recommendations")
        st.info("💡 Based on fixture difficulty analysis - players to target or avoid")
        
        if fixtures_df.empty or 'combined_fdr' not in fixtures_df.columns:
            st.warning("No fixture data for transfer analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Teams to Target")
            good_fixtures = fixtures_df.groupby('team_short_name')['combined_fdr'].mean()
            target_teams = good_fixtures[good_fixtures <= fdr_threshold].sort_values().head(8)
            
            if not target_teams.empty:
                st.write("**Teams with excellent upcoming fixtures:**")
                for team, fdr in target_teams.items():
                    st.write(f"• **{team}**: {fdr:.2f} FDR")
                    
                st.info("💡 Consider players from these teams for transfers IN")
            else:
                st.warning("No teams meet the good fixture criteria")
        
        with col2:
            st.subheader("⚠️ Teams to Avoid")
            bad_fixtures = fixtures_df.groupby('team_short_name')['combined_fdr'].mean()
            avoid_teams = bad_fixtures[bad_fixtures >= 4.0].sort_values(ascending=False).head(8)
            
            if not avoid_teams.empty:
                st.write("**Teams with difficult upcoming fixtures:**")
                for team, fdr in avoid_teams.items():
                    st.write(f"• **{team}**: {fdr:.2f} FDR")
                    
                st.warning("⚠️ Consider transferring OUT players from these teams")
            else:
                st.info("No teams have particularly difficult fixtures")


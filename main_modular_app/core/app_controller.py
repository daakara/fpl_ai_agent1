"""
Enhanced Application Controller with Modern Features
"""
import streamlit as st
import pandas as pd
from pages.player_analysis_page import PlayerAnalysisPage
from pages.fixture_analysis_page import FixtureAnalysisPage  
from pages.my_team_page import MyTeamPage
from services.fpl_data_service import FPLDataService
from services.ai_recommendation_engine import get_player_recommendations
from core.page_router import PageRouter
from utils.modern_ui_components import ModernUIComponents, NavigationManager, DataVisualization, render_loading_spinner, create_success_animation
from utils.enhanced_cache import display_cache_metrics, cached_load_fpl_data
from utils.error_handling import handle_errors, logger, perf_monitor
from config.app_config import config
import time


class EnhancedFPLAppController:
    """Enhanced application controller with modern features and performance monitoring"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.data_service = FPLDataService()
        self.page_router = PageRouter()
        self.nav_manager = NavigationManager()
        self.ui_components = ModernUIComponents()
        
        # Initialize page components
        self.pages = {
            "player_analysis": PlayerAnalysisPage(),
            "fixture_analysis": FixtureAnalysisPage(),
            "my_team": MyTeamPage()
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'page_loads': 0,
            'data_refreshes': 0,
            'errors': 0,
            'session_start': time.time()
        }
    
    def setup_page_config(self):
        """Setup enhanced Streamlit page configuration"""
        st.set_page_config(
            page_title=config.ui.page_title,
            page_icon=config.ui.page_icon,
            layout=config.ui.layout,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/fpl-analytics',
                'Report a bug': "https://github.com/your-repo/fpl-analytics/issues",
                'About': "# FPL Analytics Dashboard\nAdvanced Fantasy Premier League analytics powered by AI"
            }
        )
        
        # Custom CSS for modern styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .feature-highlight {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .stButton > button {
            border-radius: 8px;
            border: none;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize enhanced session state variables"""
        defaults = {
            'data_loaded': False,
            'players_df': pd.DataFrame(),
            'teams_df': pd.DataFrame(),
            'fdr_data_loaded': False,
            'fixtures_df': pd.DataFrame(),
            'current_gameweek': 4,  # Set default current gameweek to 4
            'user_preferences': {},
            'performance_mode': 'standard',
            'theme': 'light',
            'debug_mode': config.ui.show_debug_info,
            'feature_flags': {
                'advanced_analytics': config.features.advanced_analytics,
                'real_time_updates': config.features.real_time_updates,
                'ai_recommendations': True,
                'export_features': True
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @handle_errors("Application error occurred", show_details=True)
    def run(self):
        """Enhanced application runner with error handling and performance monitoring"""
        start_time = time.time()
        selected_page = "dashboard"  # Initialize with default value
        
        try:
            # Ensure session state is properly initialized
            self.initialize_session_state()
            
            # Render header
            self.render_enhanced_header()
            
            # Handle quick actions
            quick_action = self.nav_manager.render_quick_actions()
            if quick_action:
                self.handle_quick_action(quick_action)
            
            # Render navigation and get selected page
            selected_page = self.page_router.render_sidebar()
            
            # Add to navigation history
            self.nav_manager.add_to_history(selected_page)
            
            # Render breadcrumbs
            self.nav_manager.render_breadcrumbs(selected_page)
            
            # Check for recent page navigation
            recent_page = self.nav_manager.render_recent_pages()
            if recent_page:
                selected_page = recent_page
                st.rerun()
            
            # Display performance metrics if debug mode
            if st.session_state.debug_mode:
                self.display_debug_info()
            
            # Route to appropriate page with performance monitoring
            perf_monitor.start_timer(f"page_render_{selected_page}")
            
            if selected_page == "dashboard":
                self.render_enhanced_dashboard()
            elif selected_page == "players":
                self.pages["player_analysis"].render()
            elif selected_page == "fixtures":
                self.pages["fixture_analysis"].render()
            elif selected_page == "my_team":
                self.pages["my_team"].render()
            elif selected_page == "ai_recommendations":
                self.render_ai_recommendations()
            elif selected_page == "automated_iteration":
                self.render_automated_iteration_page()
            elif selected_page == "team_builder":
                self.render_team_builder()
            elif selected_page == "settings":
                self.render_settings_page()
            else:
                st.error(f"Unknown page: {selected_page}")
            
            perf_monitor.end_timer(f"page_render_{selected_page}")
            
            # Update performance metrics
            self.performance_metrics['page_loads'] += 1
            
            # Display cache metrics in sidebar
            display_cache_metrics()
            
            # Render footer
            self.render_footer()
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            logger.log_error(e, "app_controller_run")
            raise
        
        finally:
            # Log performance - selected_page is now always defined
            execution_time = time.time() - start_time
            logger.log_performance("app_run", execution_time, {"page": selected_page})
    
    def render_enhanced_header(self):
        """Render modern application header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0;">⚽ FPL Analytics Dashboard</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Advanced Fantasy Premier League Analytics powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            data_status = "🟢 Loaded" if st.session_state.data_loaded else "🔴 Not Loaded"
            self.ui_components.create_metric_card(
                "Data Status", data_status, icon="📊"
            )
        
        with col2:
            player_count = len(st.session_state.players_df) if not st.session_state.players_df.empty else 0
            self.ui_components.create_metric_card(
                "Players", str(player_count), icon="👥"
            )
        
        with col3:
            session_duration = int(time.time() - self.performance_metrics['session_start'])
            self.ui_components.create_metric_card(
                "Session", f"{session_duration//60}m {session_duration%60}s", icon="⏱️"
            )
        
        with col4:
            page_loads = self.performance_metrics['page_loads']
            self.ui_components.create_metric_card(
                "Page Loads", str(page_loads), icon="📄"
            )
    
    def render_enhanced_dashboard(self):
        """Render enhanced dashboard with modern components"""
        st.markdown("### 🎯 Dashboard Overview")
        
        if not st.session_state.data_loaded:
            # Enhanced onboarding
            st.markdown("""
            <div class="feature-highlight">
                <h3>👋 Welcome to FPL Analytics!</h3>
                <p>Get started by loading the latest FPL data. This will enable all advanced features including:</p>
                <ul>
                    <li>🤖 AI-powered player recommendations</li>
                    <li>📊 Advanced performance analytics</li>
                    <li>🎯 Transfer planning assistance</li>
                    <li>⚽ Comprehensive fixture analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Get Started - Load FPL Data", type="primary"):
                with st.spinner("Loading FPL data..."):
                    render_loading_spinner("Fetching latest player data...")
                    players_df, teams_df = cached_load_fpl_data()
                    
                    if not players_df.empty:
                        st.session_state.players_df = players_df
                        st.session_state.teams_df = teams_df
                        st.session_state.data_loaded = True
                        create_success_animation("Data loaded successfully!")
                        st.rerun()
            return
        
        df = st.session_state.players_df
        
        # Enhanced key metrics with modern cards
        st.markdown("### 📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui_components.create_metric_card(
                "Total Players", f"{len(df):,}", 
                delta=f"+{len(df)-500} from last season", icon="👥"
            )
        
        with col2:
            if 'cost_millions' in df.columns:
                avg_price = df['cost_millions'].mean()
                self.ui_components.create_metric_card(
                    "Average Price", f"£{avg_price:.1f}m",
                    delta="Market stable", icon="💰"
                )
        
        with col3:
            if 'total_points' in df.columns and len(df) > 0:
                top_scorer = df.loc[df['total_points'].idxmax()]
                self.ui_components.create_metric_card(
                    "Top Scorer", f"{top_scorer['web_name']}", 
                    delta=f"{top_scorer['total_points']} points", icon="🏆"
                )
        
        with col4:
            if 'points_per_million' in df.columns and len(df) > 0:
                best_value = df.loc[df['points_per_million'].idxmax()]
                self.ui_components.create_metric_card(
                    "Best Value", f"{best_value['web_name']}", 
                    delta=f"{best_value['points_per_million']:.1f} pts/£m", icon="💎"
                )
        
        # Interactive visualizations
        st.markdown("### 📈 Performance Insights")
        
        if len(df) > 0:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if 'total_points' in df.columns and 'cost_millions' in df.columns:
                    DataVisualization.create_performance_chart(
                        df, 'cost_millions', 'total_points', 
                        "Price vs Performance"
                    )
            
            with viz_col2:
                if 'element_type' in df.columns:
                    position_counts = df['element_type'].value_counts()
                    position_names = {1: 'Goalkeepers', 2: 'Defenders', 3: 'Midfielders', 4: 'Forwards'}
                    composition = {position_names.get(k, f'Position {k}'): v for k, v in position_counts.items()}
                    DataVisualization.create_team_balance_chart(composition)
        
        # Feature highlights
        st.markdown("### ✨ Available Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            ai_enabled = self.ui_components.create_feature_card(
                "AI Recommendations", 
                "Get personalized player suggestions powered by machine learning",
                "🤖", 
                enabled=st.session_state.feature_flags.get('ai_recommendations', True)
            )
            st.session_state.feature_flags['ai_recommendations'] = ai_enabled
            
            analytics_enabled = self.ui_components.create_feature_card(
                "Advanced Analytics",
                "Deep performance insights and statistical analysis", 
                "📊",
                enabled=st.session_state.feature_flags.get('advanced_analytics', True)
            )
            st.session_state.feature_flags['advanced_analytics'] = analytics_enabled
        
        with feature_col2:
            realtime_enabled = self.ui_components.create_feature_card(
                "Real-time Updates",
                "Live data updates and price change monitoring",
                "⚡", 
                enabled=st.session_state.feature_flags.get('real_time_updates', False),
                beta=True
            )
            st.session_state.feature_flags['real_time_updates'] = realtime_enabled
            
            export_enabled = self.ui_components.create_feature_card(
                "Data Export",
                "Export analysis results and custom reports",
                "💾",
                enabled=st.session_state.feature_flags.get('export_features', True)
            )
            st.session_state.feature_flags['export_features'] = export_enabled
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        with action_col1:
            if st.button("🎯 Get Transfer Suggestions", key="quick_transfers"):
                st.switch_page("ai_recommendations")
        
        with action_col2:
            if st.button("📊 Analyze My Team", key="quick_team"):
                st.switch_page("my_team")
        
        with action_col3:
            if st.button("📅 Check Fixtures", key="quick_fixtures"):
                st.switch_page("fixtures")
        
        with action_col4:
            if st.button("🔄 Refresh Data", key="quick_refresh"):
                self.handle_quick_action("refresh_data")
    
    def handle_quick_action(self, action: str):
        """Handle quick action buttons"""
        if action == "refresh_data":
            with st.spinner("Refreshing data..."):
                try:
                    # Clear cache and reload
                    from utils.enhanced_cache import cache_manager
                    cache_manager.clear_cache()
                    
                    players_df, teams_df = cached_load_fpl_data()
                    
                    if not players_df.empty:
                        st.session_state.players_df = players_df
                        st.session_state.teams_df = teams_df
                        st.session_state.data_loaded = True
                        self.performance_metrics['data_refreshes'] += 1
                        
                        create_success_animation("Data refreshed successfully!")
                        logger.log_user_action("data_refresh", {"success": True})
                    else:
                        st.error("Failed to refresh data")
                        logger.log_user_action("data_refresh", {"success": False})
                        
                except Exception as e:
                    st.error(f"Error refreshing data: {str(e)}")
                    logger.log_error(e, "data_refresh")
        
        elif action == "export_data":
            self.export_current_data()
        
        elif action == "settings":
            st.switch_page("settings")
    
    def display_debug_info(self):
        """Display debug information"""
        with st.sidebar.expander("🔧 Debug Info"):
            st.write("**Performance Metrics:**")
            st.json(self.performance_metrics)
            
            st.write("**Session State Keys:**")
            st.write(list(st.session_state.keys()))
            
            st.write("**Feature Flags:**")
            st.json(st.session_state.feature_flags)
    
    def render_footer(self):
        """Render application footer"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>FPL Analytics Dashboard v2.0 | Built with ❤️ for FPL managers</p>
            <p>Data provided by the Official Fantasy Premier League API</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_ai_recommendations(self):
        """Render AI recommendations page"""
        st.markdown("### 🤖 AI-Powered Player Recommendations")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load FPL data first to get AI recommendations.")
            if st.button("Load Data Now"):
                st.switch_page("dashboard")
            return
        
        df = st.session_state.players_df
        
        # Add filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            position_filter = st.selectbox(
                "Filter by Position",
                ["All"] + list(df['position_name'].unique()) if 'position_name' in df.columns else ["All"],
                index=0
            )
        
        with col2:
            budget_max = st.number_input(
                "Max Budget (£m)",
                min_value=4.0,
                max_value=15.0,
                value=12.0,
                step=0.5
            )
        
        with col3:
            top_n = st.selectbox(
                "Number of Recommendations",
                [5, 10, 15, 20],
                index=1
            )
        
        try:
            # Get AI recommendations
            position = None if position_filter == "All" else position_filter
            recommendations = get_player_recommendations(
                df, 
                position=position, 
                budget=budget_max, 
                top_n=top_n
            )
            
            if recommendations:
                st.success(f"🎯 Generated {len(recommendations)} AI recommendations!")
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"#{i} Recommendation: {rec.web_name} ({rec.team_name})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Price", f"£{rec.current_price:.1f}m")
                            st.metric("Total Points", int(rec.predicted_points))
                            st.metric("Position", rec.position)
                        
                        with col2:
                            st.metric("Value Score", f"{rec.value_score:.2f}")
                            st.metric("Form Score", f"{rec.form_score:.2f}")
                            st.metric("Risk Level", rec.risk_level)
                        
                        # Show confidence and ownership
                        conf_col1, conf_col2 = st.columns(2)
                        with conf_col1:
                            st.metric("Confidence", f"{rec.confidence_score:.1%}")
                        with conf_col2:
                            st.metric("Expected ROI", f"{rec.expected_roi:.1f}%")
                        
                        # Display reasoning
                        if rec.reasoning:
                            st.markdown("**🧠 AI Reasoning:**")
                            for reason in rec.reasoning:
                                st.write(f"• {reason}")
                        else:
                            st.write("**Recommendation:** Strong statistical performance indicates good value.")
                
                # Summary statistics
                st.markdown("### 📊 Recommendation Summary")
                
                avg_price = sum(rec.current_price for rec in recommendations) / len(recommendations)
                avg_value_score = sum(rec.value_score for rec in recommendations) / len(recommendations)
                risk_distribution = {}
                for rec in recommendations:
                    risk_distribution[rec.risk_level] = risk_distribution.get(rec.risk_level, 0) + 1
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Average Price", f"£{avg_price:.1f}m")
                
                with summary_col2:
                    st.metric("Average Value Score", f"{avg_value_score:.1f}")
                
                with summary_col3:
                    most_common_risk = max(risk_distribution.items(), key=lambda x: x[1])[0]
                    st.metric("Most Common Risk", most_common_risk)
                
                # Risk distribution chart
                if len(risk_distribution) > 1:
                    st.markdown("#### Risk Level Distribution")
                    st.bar_chart(risk_distribution)
            
            else:
                st.info("No recommendations available with the current filters.")
        
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            st.info("💡 Try loading fresh data or adjusting your filters.")
            logger.log_error(e, "ai_recommendations")
    
    def render_team_builder(self):
        """Render team builder page"""
        st.markdown("### 🏗️ Team Builder")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load FPL data first to use the team builder.")
            if st.button("Load Data Now"):
                st.switch_page("dashboard")
            return
        
        st.info("🚧 Team Builder feature is coming soon! This will allow you to build and optimize your FPL team.")
        
        # Placeholder for team builder functionality
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Available Features (Coming Soon)")
            st.write("- 🎯 Optimal team selection")
            st.write("- 💰 Budget optimization")
            st.write("- 🔄 Transfer planning")
            st.write("- 📊 Team analysis")
        
        with col2:
            st.markdown("#### Current Status")
            st.write("- ✅ Data loading")
            st.write("- ✅ Player analysis")
            st.write("- 🚧 Team optimization (in development)")
            st.write("- 🚧 Budget management (in development)")
    
    def render_settings_page(self):
        """Render settings and preferences page"""
        st.markdown("### ⚙️ Settings & Preferences")
        
        # Theme settings
        st.markdown("#### 🎨 Theme Settings")
        theme = st.selectbox(
            "Choose theme",
            ["light", "dark", "auto"],
            index=0 if st.session_state.theme == "light" else 1
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.success(f"Theme updated to {theme}")
        
        # Performance settings
        st.markdown("#### ⚡ Performance Settings")
        performance_mode = st.selectbox(
            "Performance Mode",
            ["standard", "fast", "detailed"],
            index=["standard", "fast", "detailed"].index(st.session_state.performance_mode)
        )
        if performance_mode != st.session_state.performance_mode:
            st.session_state.performance_mode = performance_mode
            st.success(f"Performance mode updated to {performance_mode}")
        
        # Debug settings
        st.markdown("#### 🔧 Debug Settings")
        debug_mode = st.checkbox(
            "Enable debug mode",
            value=st.session_state.debug_mode
        )
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.success(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
        
        # Feature flags
        st.markdown("#### 🚀 Feature Flags")
        for feature, enabled in st.session_state.feature_flags.items():
            new_value = st.checkbox(
                f"Enable {feature.replace('_', ' ').title()}",
                value=enabled,
                key=f"feature_{feature}"
            )
            if new_value != enabled:
                st.session_state.feature_flags[feature] = new_value
                st.success(f"{feature.replace('_', ' ').title()} {'enabled' if new_value else 'disabled'}")
        
        # Data management
        st.markdown("#### 📊 Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh All Data"):
                self.handle_quick_action("refresh_data")
        
        with col2:
            if st.button("💾 Export Data"):
                self.handle_quick_action("export_data")
        
        # Cache management
        st.markdown("#### 🗄️ Cache Management")
        cache_info = {
            "Cache Status": "Enabled" if config.cache.enabled else "Disabled",
            "TTL": f"{config.cache.ttl_seconds} seconds",
            "Max Size": f"{config.cache.max_size_mb} MB"
        }
        st.json(cache_info)
        
        if st.button("🗑️ Clear Cache"):
            try:
                from utils.enhanced_cache import cache_manager
                cache_manager.clear_cache()
                st.success("Cache cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing cache: {str(e)}")
    
    def export_current_data(self):
        """Export current data to various formats"""
        st.markdown("### 💾 Export Data")
        
        if not st.session_state.data_loaded:
            st.warning("No data to export. Please load data first.")
            return
        
        df = st.session_state.players_df
        
        export_format = st.selectbox(
            "Choose export format",
            ["CSV", "Excel", "JSON"]
        )
        
        if st.button(f"Export as {export_format}"):
            try:
                import io
                
                if export_format == "CSV":
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name="fpl_players_data.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "Excel":
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name="fpl_players_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "JSON":
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="fpl_players_data.json",
                        mime="application/json"
                    )
                
                st.success(f"Data exported successfully as {export_format}!")
                logger.log_user_action("data_export", {"format": export_format, "rows": len(df)})
                
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
                logger.log_error(e, "data_export")

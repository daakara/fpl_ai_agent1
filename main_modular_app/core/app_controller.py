"""
Enhanced Application Controller with Modern Features
"""
import streamlit as st
import pandas as pd
import time
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
            'current_gameweek': 4,
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
        selected_page = "dashboard"
        
        try:
            self.initialize_session_state()
            self.render_enhanced_header()
            
            quick_action = self.nav_manager.render_quick_actions()
            if quick_action:
                self.handle_quick_action(quick_action)
            
            selected_page = self.page_router.render_sidebar()
            self.nav_manager.add_to_history(selected_page)
            self.nav_manager.render_breadcrumbs(selected_page)
            
            recent_page = self.nav_manager.render_recent_pages()
            if recent_page:
                selected_page = recent_page
                st.rerun()
            
            if st.session_state.debug_mode:
                self.display_debug_info()
            
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
            self.performance_metrics['page_loads'] += 1
            display_cache_metrics()
            self.render_footer()
            
        except Exception as e:
            self.performance_metrics['errors'] += 1
            logger.log_error(e, "app_controller_run")
            raise
        
        finally:
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
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            data_status = "🟢 Loaded" if st.session_state.data_loaded else "🔴 Not Loaded"
            self.ui_components.create_metric_card("Data Status", data_status, icon="📊")
        
        with col2:
            player_count = len(st.session_state.players_df) if not st.session_state.players_df.empty else 0
            self.ui_components.create_metric_card("Players", str(player_count), icon="👥")
        
        with col3:
            session_duration = int(time.time() - self.performance_metrics['session_start'])
            self.ui_components.create_metric_card("Session", f"{session_duration//60}m {session_duration%60}s", icon="⏱️")
        
        with col4:
            page_loads = self.performance_metrics['page_loads']
            self.ui_components.create_metric_card("Page Loads", str(page_loads), icon="📄")
    
    def render_enhanced_dashboard(self):
        """Render enhanced dashboard with modern components"""
        st.markdown("### 🎯 Dashboard Overview")
        
        if not st.session_state.data_loaded:
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
            position = None if position_filter == "All" else position_filter
            recommendations = get_player_recommendations(
                df, 
                position=position, 
                budget=budget_max, 
                top_n=top_n
            )
            
            if recommendations:
                st.success(f"🎯 Generated {len(recommendations)} AI recommendations!")
                
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
                        
                        if rec.reasoning:
                            st.markdown("**🧠 AI Reasoning:**")
                            for reason in rec.reasoning:
                                st.write(f"• {reason}")
                        else:
                            st.write("**Recommendation:** Strong statistical performance indicates good value.")
            
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
    
    def render_settings_page(self):
        """Render settings and preferences page"""
        st.markdown("### ⚙️ Settings & Preferences")
        
        st.markdown("#### 🎨 Theme Settings")
        theme = st.selectbox(
            "Choose theme",
            ["light", "dark", "auto"],
            index=0 if st.session_state.theme == "light" else 1
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.success(f"Theme updated to {theme}")
        
        st.markdown("#### 🔧 Debug Settings")
        debug_mode = st.checkbox(
            "Enable debug mode",
            value=st.session_state.debug_mode
        )
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.success(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
    
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
                
                st.success(f"Data exported successfully as {export_format}!")
                logger.log_user_action("data_export", {"format": export_format, "rows": len(df)})
                
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
                logger.log_error(e, "data_export")
    
    def render_automated_iteration_page(self):
        """Render the automated iteration recommendations page"""
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load FPL data first to use automated recommendations.")
            if st.button("Load Data Now"):
                st.switch_page("dashboard")
            return
        
        # Import the UI components for automated iteration
        from components.automated_iteration_ui import (
            render_automated_recommendations_tab,
            render_feedback_summary,
            display_automated_iteration_help
        )
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["🤖 AI Recommendations", "📊 Feedback Analysis", "❓ Help"])
        
        with tab1:
            render_automated_recommendations_tab(st.session_state.players_df)
        
        with tab2:
            render_feedback_summary()
        
        with tab3:
            display_automated_iteration_help()

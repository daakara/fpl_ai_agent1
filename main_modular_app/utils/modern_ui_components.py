"""
Modern UI Components and Enhanced Navigation
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class ModernUIComponents:
    """Modern UI components for better user experience"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, 
                          delta_color: str = "normal", icon: str = "📊") -> None:
        """Create a modern metric card with styling"""
        delta_html = ""
        if delta:
            color = {"normal": "#28a745", "inverse": "#dc3545", "off": "#6c757d"}[delta_color]
            delta_html = f'<div style="color: {color}; font-size: 14px; margin-top: 5px;">{delta}</div>'
        
        card_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 10px 0;
            color: white;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
                <h3 style="margin: 0; font-size: 16px; opacity: 0.9;">{title}</h3>
            </div>
            <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">{value}</div>
            {delta_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_progress_bar(label: str, value: float, max_value: float = 100, 
                           color: str = "#667eea") -> None:
        """Create a styled progress bar"""
        percentage = (value / max_value) * 100
        
        progress_html = f"""
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 500;">{label}</span>
                <span style="color: #666;">{value:.1f}/{max_value}</span>
            </div>
            <div style="
                background: #e9ecef;
                border-radius: 10px;
                height: 10px;
                overflow: hidden;
            ">
                <div style="
                    background: {color};
                    height: 100%;
                    width: {percentage}%;
                    border-radius: 10px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_alert_box(message: str, alert_type: str = "info", 
                        dismissible: bool = True) -> None:
        """Create styled alert boxes"""
        colors = {
            "success": {"bg": "#d4edda", "border": "#c3e6cb", "text": "#155724"},
            "info": {"bg": "#d1ecf1", "border": "#bee5eb", "text": "#0c5460"},
            "warning": {"bg": "#fff3cd", "border": "#ffeaa7", "text": "#856404"},
            "error": {"bg": "#f8d7da", "border": "#f5c6cb", "text": "#721c24"}
        }
        
        style = colors.get(alert_type, colors["info"])
        dismiss_btn = "×" if dismissible else ""
        
        alert_html = f"""
        <div style="
            background-color: {style['bg']};
            border: 1px solid {style['border']};
            color: {style['text']};
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            position: relative;
        ">
            {message}
            {f'<span style="position: absolute; top: 10px; right: 15px; cursor: pointer; font-size: 20px;">{dismiss_btn}</span>' if dismissible else ''}
        </div>
        """
        st.markdown(alert_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_feature_card(title: str, description: str, icon: str, 
                           enabled: bool = True, beta: bool = False) -> bool:
        """Create feature cards with enable/disable toggle"""
        status_badge = ""
        if beta:
            status_badge = '<span style="background: #ff6b6b; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; margin-left: 10px;">BETA</span>'
        
        opacity = "1" if enabled else "0.6"
        
        card_html = f"""
        <div style="
            border: 2px solid {'#28a745' if enabled else '#dc3545'};
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            opacity: {opacity};
            transition: all 0.3s ease;
            background: {'#f8fff8' if enabled else '#fff8f8'};
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 28px; margin-right: 15px;">{icon}</span>
                <div>
                    <h4 style="margin: 0; color: #333;">{title}{status_badge}</h4>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        return st.checkbox(f"Enable {title}", value=enabled, key=f"feature_{title.lower().replace(' ', '_')}")
    
    @staticmethod
    def create_comparison_table(data: List[Dict], highlight_best: bool = True) -> None:
        """Create a comparison table with highlighting"""
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Style function for highlighting
        def highlight_max(s):
            if highlight_best and s.dtype in ['int64', 'float64']:
                is_max = s == s.max()
                return ['background-color: #28a745; color: white' if v else '' for v in is_max]
            return [''] * len(s)
        
        styled_df = df.style.apply(highlight_max, axis=0)
        st.dataframe(styled_df, use_container_width=True)

class NavigationManager:
    """Enhanced navigation with breadcrumbs and quick actions"""
    
    def __init__(self):
        self.navigation_history = st.session_state.get('nav_history', [])
        self.quick_actions = {
            "🔄 Refresh Data": "refresh_data",
            "📊 Quick Analysis": "quick_analysis", 
            "💾 Export Data": "export_data",
            "⚙️ Settings": "settings"
        }
    
    def render_breadcrumbs(self, current_page: str) -> None:
        """Render navigation breadcrumbs"""
        breadcrumbs = ["🏠 Home"]
        
        if current_page != "dashboard":
            page_names = {
                "players": "👥 Players",
                "fixtures": "📅 Fixtures", 
                "my_team": "⚽ My Team",
                "ai_recommendations": "🤖 AI Recommendations",
                "team_builder": "🏗️ Team Builder"
            }
            breadcrumbs.append(page_names.get(current_page, current_page))
        
        breadcrumb_html = " > ".join(breadcrumbs)
        st.markdown(f"**Navigation:** {breadcrumb_html}")
    
    def render_quick_actions(self) -> Optional[str]:
        """Render quick action buttons"""
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        action_triggered = None
        
        for action_name, action_id in self.quick_actions.items():
            if st.sidebar.button(action_name, key=f"qa_{action_id}"):
                action_triggered = action_id
        
        return action_triggered
    
    def add_to_history(self, page: str) -> None:
        """Add page to navigation history"""
        if not self.navigation_history or self.navigation_history[-1] != page:
            self.navigation_history.append(page)
            # Keep only last 10 pages
            self.navigation_history = self.navigation_history[-10:]
            st.session_state.nav_history = self.navigation_history
    
    def render_recent_pages(self) -> Optional[str]:
        """Render recently visited pages"""
        if len(self.navigation_history) > 1:
            st.sidebar.markdown("### 📚 Recent Pages")
            
            page_names = {
                "dashboard": "🏠 Dashboard",
                "players": "👥 Player Analysis",
                "fixtures": "📅 Fixture Analysis",
                "my_team": "⚽ My Team", 
                "ai_recommendations": "🤖 AI Recommendations",
                "team_builder": "🏗️ Team Builder"
            }
            
            for page in reversed(self.navigation_history[-5:]):
                if st.sidebar.button(page_names.get(page, page), key=f"recent_{page}"):
                    return page
        
        return None

class DataVisualization:
    """Enhanced data visualization components"""
    
    @staticmethod
    def create_performance_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                               title: str = "Performance Chart") -> None:
        """Create interactive performance charts"""
        fig = px.scatter(
            df.head(20), 
            x=x_col, 
            y=y_col,
            hover_data=['web_name', 'team_short_name'],
            title=title,
            template="plotly_white"
        )
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_form_trend_chart(player_data: Dict) -> None:
        """Create form trend visualization"""
        # Simulate form data (in real app, this would come from API)
        gameweeks = list(range(1, 11))
        form_points = [player_data.get(f'gw_{i}_points', 0) for i in gameweeks]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gameweeks,
            y=form_points,
            mode='lines+markers',
            name='Points',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Form Trend (Last 10 GWs)",
            xaxis_title="Gameweek",
            yaxis_title="Points",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_team_balance_chart(team_composition: Dict) -> None:
        """Create team balance visualization"""
        positions = list(team_composition.keys())
        counts = list(team_composition.values())
        
        fig = px.pie(
            values=counts,
            names=positions,
            title="Squad Composition",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_loading_spinner(message: str = "Loading...") -> None:
    """Render a modern loading spinner"""
    loading_html = f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 40px;">
        <div style="
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin-right: 15px;
        "></div>
        <span style="font-size: 16px; color: #667eea;">{message}</span>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    st.markdown(loading_html, unsafe_allow_html=True)

def create_success_animation(message: str = "Success!") -> None:
    """Create success animation"""
    success_html = f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
    ">
        <span style="font-size: 24px; margin-right: 10px;">✅</span>
        <span style="font-size: 18px; font-weight: 500;">{message}</span>
    </div>
    <style>
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """
    st.markdown(success_html, unsafe_allow_html=True)
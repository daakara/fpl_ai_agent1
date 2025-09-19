"""
Enhanced UI Components for FPL Analytics App
Provides reusable, accessible, and responsive UI components with modern design
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModernUIComponents:
    """Modern UI components with enhanced UX"""
    
    @staticmethod
    def render_hero_section(title: str, subtitle: str, metrics: Dict[str, Any] = None):
        """Render hero section with key metrics"""
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem; color: white;'>
            <h1 style='margin: 0; font-size: 2.5rem; font-weight: 700;'>{title}</h1>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if metrics:
            cols = st.columns(len(metrics))
            for i, (label, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(label, value)
    
    @staticmethod
    def render_insights_panel(insights: List[Dict[str, str]], title: str = "💡 Key Insights"):
        """Render insights panel with categorized insights"""
        st.subheader(title)
        
        # Categorize insights by type
        categories = {
            'positive': {'icon': '✅', 'color': '#28a745', 'items': []},
            'warning': {'icon': '⚠️', 'color': '#ffc107', 'items': []},
            'negative': {'icon': '❌', 'color': '#dc3545', 'items': []},
            'info': {'icon': 'ℹ️', 'color': '#17a2b8', 'items': []}
        }
        
        for insight in insights:
            insight_type = insight.get('type', 'info')
            if insight_type in categories:
                categories[insight_type]['items'].append(insight)
        
        for category, data in categories.items():
            if data['items']:
                for item in data['items']:
                    st.markdown(f"""
                    <div style='padding: 0.75rem; margin: 0.5rem 0; border-left: 4px solid {data['color']}; background-color: {data['color']}15; border-radius: 0 8px 8px 0;'>
                        <strong>{data['icon']} {item.get('title', 'Insight')}</strong><br>
                        {item.get('description', '')}
                    </div>
                    """, unsafe_allow_html=True)
    
    @staticmethod
    def render_performance_dashboard(data: Dict[str, Any]):
        """Render comprehensive performance dashboard"""
        st.subheader("📊 Performance Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        metrics = data.get('metrics', {})
        
        with col1:
            st.metric(
                "Overall Score",
                f"{metrics.get('overall_score', 0)}/100",
                delta=f"{metrics.get('score_change', 0):+.1f}"
            )
        
        with col2:
            st.metric(
                "Form Rating",
                f"{metrics.get('form_rating', 0):.1f}/10",
                delta=f"{metrics.get('form_change', 0):+.1f}"
            )
        
        with col3:
            st.metric(
                "Value Efficiency",
                f"{metrics.get('value_efficiency', 0):.1f}",
                delta=f"{metrics.get('efficiency_change', 0):+.1f}"
            )
        
        with col4:
            st.metric(
                "Risk Level",
                metrics.get('risk_level', 'Medium'),
                help="Lower risk = more predictable performance"
            )
        
        # Performance breakdown
        if 'breakdown' in data:
            breakdown = data['breakdown']
            
            # Create radar chart for multi-dimensional performance
            categories = list(breakdown.keys())
            values = list(breakdown.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Performance',
                line_color='rgb(102, 126, 234)',
                fillcolor='rgba(102, 126, 234, 0.25)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Performance Breakdown",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_smart_filters(filter_config: Dict[str, Dict], data: pd.DataFrame, key_prefix: str = "filter") -> Dict[str, Any]:
        """Render smart filter panel with search and presets"""
        st.subheader("🎛️ Smart Filters")
        
        # Filter presets
        presets = {
            "🏆 Premium Players": {"min_cost": 9.0, "min_points": 100},
            "💎 Budget Gems": {"max_cost": 6.0, "min_form": 6.0},
            "🔥 In Form": {"min_form": 7.0, "min_minutes": 800},
            "💰 Best Value": {"min_ppm": 8.0, "max_ownership": 15.0}
        }
        
        selected_preset = st.selectbox("Quick Filters:", ["Custom"] + list(presets.keys()))
        
        filter_values = {}
        
        # Apply preset if selected
        if selected_preset != "Custom":
            preset_values = presets[selected_preset]
            st.info(f"Applied preset: {selected_preset}")
        else:
            preset_values = {}
        
        # Render individual filters
        cols = st.columns(min(len(filter_config), 4))
        
        for i, (filter_name, config) in enumerate(filter_config.items()):
            col_idx = i % 4
            
            with cols[col_idx]:
                filter_type = config.get('type', 'text')
                label = config.get('label', filter_name)
                default_value = preset_values.get(filter_name, config.get('default'))
                
                if filter_type == 'slider' and config.get('column') in data.columns:
                    col_data = data[config['column']]
                    min_val = config.get('min_value', col_data.min())
                    max_val = config.get('max_value', col_data.max())
                    
                    if isinstance(default_value, (list, tuple)):
                        filter_values[filter_name] = st.slider(
                            label, min_val, max_val, default_value,
                            key=f"{key_prefix}_{filter_name}"
                        )
                    else:
                        filter_values[filter_name] = st.slider(
                            label, min_val, max_val, (min_val, max_val),
                            key=f"{key_prefix}_{filter_name}"
                        )
                
                elif filter_type == 'multiselect':
                    options = config.get('options', [])
                    if not options and config.get('column') in data.columns:
                        options = sorted(data[config['column']].unique())
                    
                    filter_values[filter_name] = st.multiselect(
                        label, options, default=default_value or [],
                        key=f"{key_prefix}_{filter_name}"
                    )
        
        return filter_values
    
    @staticmethod
    def render_comparison_matrix(players: List[Dict], metrics: List[str]):
        """Render advanced player comparison matrix"""
        if not players or not metrics:
            st.warning("No players or metrics provided for comparison")
            return
        
        st.subheader("⚖️ Advanced Player Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for player in players:
            row = {'Player': player.get('name', 'Unknown')}
            for metric in metrics:
                row[metric] = player.get(metric, 0)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Normalize metrics for visualization
        normalized_df = df.copy()
        for metric in metrics:
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    normalized_df[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_df[f'{metric}_norm'] = 0.5
        
        # Create heatmap
        heatmap_data = normalized_df[[f'{m}_norm' for m in metrics if f'{m}_norm' in normalized_df.columns]]
        
        if not heatmap_data.empty:
            fig = px.imshow(
                heatmap_data.T,
                labels=dict(x="Players", y="Metrics", color="Performance"),
                x=df['Player'],
                y=metrics,
                color_continuous_scale="RdYlGn",
                title="Player Performance Heatmap"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def render_trend_analysis(data: pd.DataFrame, x_col: str, y_col: str, group_col: str = None):
        """Render trend analysis with forecasting"""
        st.subheader("📈 Trend Analysis")
        
        if data.empty or x_col not in data.columns or y_col not in data.columns:
            st.warning("Invalid data for trend analysis")
            return
        
        fig = go.Figure()
        
        if group_col and group_col in data.columns:
            # Group by category
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group]
                fig.add_trace(go.Scatter(
                    x=group_data[x_col],
                    y=group_data[y_col],
                    mode='lines+markers',
                    name=str(group),
                    line=dict(width=3)
                ))
        else:
            # Single line
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name='Trend',
                line=dict(width=3, color='rgb(102, 126, 234)')
            ))
        
        # Add trend line
        if len(data) > 1:
            z = np.polyfit(range(len(data)), data[y_col], 1)
            trend_line = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=trend_line(range(len(data))),
                mode='lines',
                name='Trend Line',
                line=dict(dash='dash', color='red')
            ))
        
        fig.update_layout(
            title=f"{y_col.replace('_', ' ').title()} over {x_col.replace('_', ' ').title()}",
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_action_center(actions: List[Dict[str, Any]]):
        """Render action center with recommended actions"""
        st.subheader("🎯 Recommended Actions")
        
        priority_order = {'high': 1, 'medium': 2, 'low': 3}
        sorted_actions = sorted(actions, key=lambda x: priority_order.get(x.get('priority', 'medium'), 2))
        
        for action in sorted_actions:
            priority = action.get('priority', 'medium')
            priority_colors = {
                'high': '#dc3545',
                'medium': '#ffc107', 
                'low': '#28a745'
            }
            priority_icons = {
                'high': '🔥',
                'medium': '⚠️',
                'low': '💡'
            }
            
            color = priority_colors.get(priority, '#17a2b8')
            icon = priority_icons.get(priority, '📋')
            
            with st.container():
                st.markdown(f"""
                <div style='padding: 1rem; margin: 0.5rem 0; border-left: 4px solid {color}; background-color: {color}15; border-radius: 0 8px 8px 0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong>{icon} {action.get('title', 'Action Required')}</strong>
                            <p style='margin: 0.5rem 0 0 0; opacity: 0.8;'>{action.get('description', '')}</p>
                        </div>
                        <div style='text-align: right;'>
                            <small>Priority: {priority.title()}</small><br>
                            <small>Impact: {action.get('impact', 'Unknown')}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if action.get('show_button', False):
                    if st.button(f"Execute: {action.get('button_text', 'Take Action')}", key=f"action_{hash(action.get('title', ''))}"):
                        st.success(f"Action executed: {action.get('title')}")

class InteractiveCharts:
    """Interactive chart components with enhanced functionality"""
    
    @staticmethod
    def create_performance_radar(player_data: Dict[str, float], categories: List[str], 
                               title: str = "Player Performance Radar") -> go.Figure:
        """Create interactive radar chart for player performance"""
        fig = go.Figure()
        
        values = [player_data.get(cat, 0) for cat in categories]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=title,
            line_color='rgb(102, 126, 234)',
            fillcolor='rgba(102, 126, 234, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title=title,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_fixture_difficulty_matrix(fixture_data: pd.DataFrame) -> go.Figure:
        """Create interactive fixture difficulty matrix"""
        if fixture_data.empty:
            return go.Figure()
        
        # Pivot data for heatmap
        pivot_data = fixture_data.pivot_table(
            index='team_name',
            columns='gameweek',
            values='difficulty',
            fill_value=3
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=[
                [0, '#00FF87'],    # Very Easy
                [0.25, '#01FF70'], # Easy  
                [0.5, '#FFDC00'],  # Average
                [0.75, '#FF851B'], # Hard
                [1, '#FF4136']     # Very Hard
            ],
            zmin=1,
            zmax=5,
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>GW%{x}<br>Difficulty: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Fixture Difficulty Matrix",
            xaxis_title="Gameweek",
            yaxis_title="Team",
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_points_projection(historical_data: pd.DataFrame, 
                               projection_weeks: int = 5) -> go.Figure:
        """Create points projection chart with confidence intervals"""
        fig = go.Figure()
        
        if not historical_data.empty and 'gameweek' in historical_data.columns and 'points' in historical_data.columns:
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_data['gameweek'],
                y=historical_data['points'],
                mode='lines+markers',
                name='Historical Points',
                line=dict(color='blue', width=3)
            ))
            
            # Simple projection (in real implementation, use advanced forecasting)
            last_gw = historical_data['gameweek'].max()
            avg_points = historical_data['points'].tail(5).mean()
            
            projection_gws = list(range(last_gw + 1, last_gw + projection_weeks + 1))
            projection_points = [avg_points] * projection_weeks
            
            # Add uncertainty bands
            upper_bound = [p * 1.3 for p in projection_points]
            lower_bound = [p * 0.7 for p in projection_points]
            
            # Projection line
            fig.add_trace(go.Scatter(
                x=projection_gws,
                y=projection_points,
                mode='lines+markers',
                name='Projected Points',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence bands
            fig.add_trace(go.Scatter(
                x=projection_gws + projection_gws[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Points Projection with Confidence Intervals",
            xaxis_title="Gameweek",
            yaxis_title="Points",
            height=400
        )
        
        return fig

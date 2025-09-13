"""
Enhanced UI Components with Modern Design and Better UX
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional

class ModernUIComponents:
    """Modern UI components with enhanced UX"""
    
    @staticmethod
    def render_metric_cards(metrics: List[Dict[str, Any]], cols: int = 4):
        """Render enhanced metric cards with better styling"""
        columns = st.columns(cols)
        
        for i, metric in enumerate(metrics):
            with columns[i % cols]:
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem;
                        border-radius: 15px;
                        color: white;
                        text-align: center;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                        margin: 0.5rem 0;
                    ">
                        <h3 style="margin: 0; font-size: 2.5rem; font-weight: bold;">
                            {metric.get('value', 'N/A')}
                        </h3>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem;">
                            {metric.get('label', '')}
                        </p>
                        {f"<small style='opacity: 0.8;'>{metric.get('subtitle', '')}</small>" if metric.get('subtitle') else ""}
                    </div>
                    """, unsafe_allow_html=True)

    @staticmethod
    def render_search_and_filters(df: pd.DataFrame, filter_config: Dict) -> pd.DataFrame:
        """Advanced search and filtering interface"""
        st.markdown("### 🔍 Smart Search & Filters")
        
        # Search bar
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input(
                "🔍 Search players, teams, or positions",
                placeholder="e.g. Salah, Liverpool, Forward",
                help="Search across player names, teams, and positions"
            )
        
        with col2:
            show_filters = st.toggle("Show Advanced Filters", value=False)
        
        filtered_df = df.copy()
        
        # Apply search
        if search_term:
            search_cols = ['web_name', 'team_name', 'position_name']
            mask = pd.concat([
                filtered_df[col].astype(str).str.contains(search_term, case=False, na=False) 
                for col in search_cols if col in filtered_df.columns
            ], axis=1).any(axis=1)
            filtered_df = filtered_df[mask]
        
        # Advanced filters
        if show_filters:
            with st.expander("🎛️ Advanced Filters", expanded=True):
                filter_cols = st.columns(3)
                
                with filter_cols[0]:
                    price_range = st.slider(
                        "Price Range (£m)",
                        min_value=float(df['cost_millions'].min()),
                        max_value=float(df['cost_millions'].max()),
                        value=(float(df['cost_millions'].min()), float(df['cost_millions'].max()))
                    )
                
                with filter_cols[1]:
                    if 'position_name' in df.columns:
                        positions = st.multiselect(
                            "Positions",
                            options=df['position_name'].unique(),
                            default=df['position_name'].unique()
                        )
                        if positions:
                            filtered_df = filtered_df[filtered_df['position_name'].isin(positions)]
                
                with filter_cols[2]:
                    min_points = st.number_input(
                        "Minimum Points",
                        min_value=0,
                        max_value=int(df['total_points'].max()),
                        value=0
                    )
                
                # Apply filters
                filtered_df = filtered_df[
                    (filtered_df['cost_millions'] >= price_range[0]) &
                    (filtered_df['cost_millions'] <= price_range[1]) &
                    (filtered_df['total_points'] >= min_points)
                ]
        
        # Results summary
        if len(filtered_df) != len(df):
            st.info(f"📊 Showing {len(filtered_df)} of {len(df)} players")
        
        return filtered_df

    @staticmethod
    def render_interactive_table(df: pd.DataFrame, title: str = "", key: str = "table"):
        """Enhanced interactive table with sorting and pagination"""
        if df.empty:
            st.warning("No data to display")
            return
        
        st.markdown(f"### {title}" if title else "")
        
        # Pagination
        items_per_page = st.selectbox(
            "Items per page",
            [10, 25, 50, 100],
            index=1,
            key=f"{key}_pagination"
        )
        
        total_pages = (len(df) - 1) // items_per_page + 1
        
        if total_pages > 1:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"{key}_page"
            )
            
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            display_df = df.iloc[start_idx:end_idx]
            
            st.caption(f"Page {page} of {total_pages} ({len(df)} total items)")
        else:
            display_df = df
        
        # Enhanced table display
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "cost_millions": st.column_config.NumberColumn(
                    "Price",
                    format="£%.1f"
                ),
                "total_points": st.column_config.NumberColumn(
                    "Points",
                    format="%d"
                ),
                "selected_by_percent": st.column_config.NumberColumn(
                    "Ownership",
                    format="%.1f%%"
                )
            }
        )

    @staticmethod
    def render_comparison_chart(data: Dict[str, List], title: str = "Comparison"):
        """Enhanced comparison visualization"""
        if not data:
            st.warning("No data available for comparison")
            return
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Add traces
        for i, (key, values) in enumerate(data.items()):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines+markers',
                    name=key,
                    line=dict(width=3),
                    marker=dict(size=8)
                )
            )
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_status_indicators(status_data: Dict[str, str]):
        """Render status indicators with color coding"""
        cols = st.columns(len(status_data))
        
        status_colors = {
            'excellent': '#00ff87',
            'good': '#01ff70', 
            'average': '#ffdc00',
            'poor': '#ff851b',
            'critical': '#ff4136'
        }
        
        for i, (label, status) in enumerate(status_data.items()):
            with cols[i]:
                color = status_colors.get(status.lower(), '#cccccc')
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    padding: 1rem;
                    border-radius: 10px;
                    text-align: center;
                    color: black;
                    font-weight: bold;
                    margin: 0.5rem 0;
                ">
                    {label}<br>
                    <small>{status.title()}</small>
                </div>
                """, unsafe_allow_html=True)

class NavigationManager:
    """Enhanced navigation with breadcrumbs and quick actions"""
    
    @staticmethod
    def render_breadcrumbs(path: List[str]):
        """Render breadcrumb navigation"""
        if len(path) > 1:
            breadcrumb = " > ".join(path)
            st.markdown(f"**Navigation:** {breadcrumb}")
            st.markdown("---")
    
    @staticmethod
    def render_quick_actions():
        """Render quick action buttons"""
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", key="quick_refresh"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("📊 Export", key="quick_export"):
                st.info("Export functionality coming soon!")
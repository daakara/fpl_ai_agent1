"""
Enhanced UI Components for FPL Analytics App
Provides reusable, accessible, and responsive UI components
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import pandas as pd

class UIComponents:
    """Collection of reusable UI components"""
    
    @staticmethod
    def render_loading_state(message: str = "Loading...", progress: Optional[float] = None):
        """Render loading state with optional progress"""
        try:
            import streamlit as st
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if progress is not None:
                    st.progress(progress)
                
                with st.spinner(message):
                    st.empty()
        
        except ImportError:
            print(f"Loading: {message}")
    
    @staticmethod
    def render_metric_card(title: str, value: str, delta: Optional[str] = None, 
                          delta_color: str = "normal", help_text: Optional[str] = None):
        """Render enhanced metric card"""
        try:
            import streamlit as st
            
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )
        
        except ImportError:
            print(f"{title}: {value} {delta or ''}")
    
    @staticmethod
    def render_data_table(data: pd.DataFrame, 
                         title: str = "",
                         searchable: bool = True,
                         sortable: bool = True,
                         column_config: Optional[Dict] = None,
                         selection_mode: str = "single"):
        """Render enhanced data table"""
        try:
            import streamlit as st
            
            if title:
                st.subheader(title)
            
            # Search functionality
            if searchable and not data.empty:
                search_term = st.text_input("🔍 Search", key=f"search_{title}")
                
                if search_term:
                    # Simple text search across all columns
                    mask = data.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    data = data[mask]
            
            # Display table
            if not data.empty:
                st.dataframe(
                    data,
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No data to display")
        
        except ImportError:
            print(f"Table: {title}")
            print(data.head() if not data.empty else "No data")
    
    @staticmethod
    def render_filter_panel(filters: Dict[str, Dict], 
                           data: pd.DataFrame,
                           key_prefix: str = "filter") -> Dict[str, Any]:
        """Render dynamic filter panel"""
        try:
            import streamlit as st
            
            st.subheader("🎛️ Filters")
            
            filter_values = {}
            
            for filter_name, filter_config in filters.items():
                filter_type = filter_config.get('type', 'text')
                label = filter_config.get('label', filter_name)
                
                if filter_type == 'selectbox':
                    options = filter_config.get('options', [])
                    if not options and filter_config.get('column') in data.columns:
                        options = sorted(data[filter_config['column']].unique())
                    
                    filter_values[filter_name] = st.selectbox(
                        label,
                        options,
                        key=f"{key_prefix}_{filter_name}"
                    )
                
                elif filter_type == 'multiselect':
                    options = filter_config.get('options', [])
                    if not options and filter_config.get('column') in data.columns:
                        options = sorted(data[filter_config['column']].unique())
                    
                    filter_values[filter_name] = st.multiselect(
                        label,
                        options,
                        default=filter_config.get('default', []),
                        key=f"{key_prefix}_{filter_name}"
                    )
                
                elif filter_type == 'slider':
                    col_data = data[filter_config['column']] if filter_config.get('column') in data.columns else []
                    min_val = filter_config.get('min_value', col_data.min() if len(col_data) > 0 else 0)
                    max_val = filter_config.get('max_value', col_data.max() if len(col_data) > 0 else 100)
                    
                    filter_values[filter_name] = st.slider(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=filter_config.get('default', (min_val, max_val)),
                        key=f"{key_prefix}_{filter_name}"
                    )
                
                elif filter_type == 'number_input':
                    filter_values[filter_name] = st.number_input(
                        label,
                        min_value=filter_config.get('min_value', 0),
                        max_value=filter_config.get('max_value', 1000),
                        value=filter_config.get('default', 0),
                        key=f"{key_prefix}_{filter_name}"
                    )
                
                elif filter_type == 'text':
                    filter_values[filter_name] = st.text_input(
                        label,
                        value=filter_config.get('default', ''),
                        key=f"{key_prefix}_{filter_name}"
                    )
            
            return filter_values
        
        except ImportError:
            print("Filter panel rendered")
            return {}
    
    @staticmethod
    def render_comparison_cards(items: List[Dict[str, Any]], 
                               title: str = "Comparison",
                               max_columns: int = 3):
        """Render comparison cards for multiple items"""
        try:
            import streamlit as st
            
            st.subheader(title)
            
            # Create columns based on number of items
            num_items = len(items)
            num_cols = min(num_items, max_columns)
            
            if num_cols > 0:
                cols = st.columns(num_cols)
                
                for i, item in enumerate(items):
                    col_idx = i % num_cols
                    
                    with cols[col_idx]:
                        # Card container
                        with st.container():
                            st.markdown(f"**{item.get('name', f'Item {i+1}')}**")
                            
                            # Render metrics
                            for key, value in item.items():
                                if key != 'name':
                                    formatted_key = key.replace('_', ' ').title()
                                    st.metric(formatted_key, value)
        
        except ImportError:
            print(f"Comparison: {title}")
            for item in items:
                print(f"  - {item}")
    
    @staticmethod
    def render_alert(message: str, alert_type: str = "info", 
                    dismissible: bool = False, icon: Optional[str] = None):
        """Render styled alert messages"""
        try:
            import streamlit as st
            
            alert_icons = {
                'success': '✅',
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌'
            }
            
            display_icon = icon or alert_icons.get(alert_type, 'ℹ️')
            formatted_message = f"{display_icon} {message}"
            
            if alert_type == 'success':
                st.success(formatted_message)
            elif alert_type == 'warning':
                st.warning(formatted_message)
            elif alert_type == 'error':
                st.error(formatted_message)
            else:
                st.info(formatted_message)
        
        except ImportError:
            print(f"Alert ({alert_type}): {message}")
    
    @staticmethod
    def render_progress_tracker(steps: List[str], current_step: int):
        """Render progress tracker for multi-step processes"""
        try:
            import streamlit as st
            
            st.markdown("### Progress")
            
            progress_container = st.container()
            
            with progress_container:
                cols = st.columns(len(steps))
                
                for i, step in enumerate(steps):
                    with cols[i]:
                        if i < current_step:
                            st.markdown(f"✅ **{step}**")
                        elif i == current_step:
                            st.markdown(f"🔄 **{step}**")
                        else:
                            st.markdown(f"⏳ {step}")
                
                # Progress bar
                progress = (current_step + 1) / len(steps)
                st.progress(progress)
        
        except ImportError:
            print(f"Progress: Step {current_step + 1} of {len(steps)}")
    
    @staticmethod
    def render_expandable_section(title: str, content_func: Callable, 
                                 expanded: bool = False, key: Optional[str] = None):
        """Render expandable section with dynamic content"""
        try:
            import streamlit as st
            
            with st.expander(title, expanded=expanded):
                content_func()
        
        except ImportError:
            print(f"Section: {title}")
            content_func()
    
    @staticmethod
    def render_sidebar_navigation(pages: Dict[str, str], 
                                 current_page: str = None) -> str:
        """Render sidebar navigation menu"""
        try:
            import streamlit as st
            
            st.sidebar.title("🧭 Navigation")
            
            # Create navigation buttons
            selected_page = None
            
            for page_name, page_key in pages.items():
                if st.sidebar.button(
                    page_name, 
                    key=f"nav_{page_key}",
                    use_container_width=True,
                    type="primary" if page_key == current_page else "secondary"
                ):
                    selected_page = page_key
            
            return selected_page or current_page
        
        except ImportError:
            print("Navigation menu")
            return current_page
    
    @staticmethod
    def render_status_badge(status: str, status_config: Optional[Dict] = None):
        """Render status badge with color coding"""
        try:
            import streamlit as st
            
            default_config = {
                'active': {'color': 'green', 'icon': '🟢'},
                'inactive': {'color': 'red', 'icon': '🔴'},
                'pending': {'color': 'orange', 'icon': '🟡'},
                'warning': {'color': 'yellow', 'icon': '⚠️'},
                'info': {'color': 'blue', 'icon': 'ℹ️'}
            }
            
            config = status_config or default_config
            status_info = config.get(status.lower(), {'color': 'gray', 'icon': '⚪'})
            
            icon = status_info.get('icon', '⚪')
            st.markdown(f"{icon} **{status.title()}**")
        
        except ImportError:
            print(f"Status: {status}")

class ChartComponents:
    """Specialized chart components"""
    
    @staticmethod
    def render_performance_chart(data: pd.DataFrame, 
                               x_column: str, 
                               y_column: str,
                               title: str = "Performance Chart"):
        """Render performance chart with trend line"""
        try:
            import streamlit as st
            import plotly.express as px
            
            if not data.empty and x_column in data.columns and y_column in data.columns:
                fig = px.line(
                    data, 
                    x=x_column, 
                    y=y_column,
                    title=title,
                    markers=True
                )
                
                fig.update_layout(
                    xaxis_title=x_column.replace('_', ' ').title(),
                    yaxis_title=y_column.replace('_', ' ').title(),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for chart")
        
        except ImportError:
            print(f"Chart: {title}")
    
    @staticmethod
    def render_comparison_chart(data: Dict[str, List], 
                              chart_type: str = "bar",
                              title: str = "Comparison Chart"):
        """Render comparison chart"""
        try:
            import streamlit as st
            import plotly.express as px
            import pandas as pd
            
            # Convert dict to DataFrame
            df = pd.DataFrame(data)
            
            if not df.empty:
                if chart_type == "bar":
                    fig = px.bar(df, title=title)
                elif chart_type == "line":
                    fig = px.line(df, title=title)
                elif chart_type == "scatter":
                    fig = px.scatter(df, title=title)
                else:
                    fig = px.bar(df, title=title)  # Default to bar
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for chart")
        
        except ImportError:
            print(f"Chart: {title}")

# Global UI components instance
ui = UIComponents()
charts = ChartComponents()

def create_responsive_layout(mobile_columns: int = 1, 
                           tablet_columns: int = 2, 
                           desktop_columns: int = 3):
    """Create responsive layout based on screen size"""
    try:
        import streamlit as st
        
        # For now, use desktop layout (could be enhanced with JS for responsive detection)
        return st.columns(desktop_columns)
    
    except ImportError:
        return [None] * desktop_columns

def render_footer():
    """Render application footer"""
    try:
        import streamlit as st
        
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 14px;'>
                <p>FPL Analytics Dashboard | Built with ❤️ for FPL managers</p>
                <p>Data provided by Fantasy Premier League API</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    except ImportError:
        print("Footer: FPL Analytics Dashboard")

def apply_custom_css():
    """Apply custom CSS styling"""
    try:
        import streamlit as st
        
        st.markdown("""
            <style>
            .stMetric {
                background-color: #f0f2f6;
                border: 1px solid #e0e0e0;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            
            .stAlert {
                border-radius: 0.5rem;
                margin: 1rem 0;
            }
            
            .stExpander {
                border: 1px solid #e0e0e0;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            
            .stDataFrame {
                border: 1px solid #e0e0e0;
                border-radius: 0.5rem;
            }
            
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)
    
    except ImportError:
        pass

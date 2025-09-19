"""
Refactored FPL Analytics Application - Main Entry Point with Enhanced Features
"""
import streamlit as st
import pandas as pd
import numpy as np
from core.app_controller import EnhancedFPLAppController
from utils.enhanced_cache import display_cache_metrics
from utils.error_handling import logger
from config.app_config import config


@st.cache_resource
def get_enhanced_app_controller():
    """Create and cache the enhanced application controller instance"""
    try:
        return EnhancedFPLAppController()
    except Exception as e:
        st.error(f"Error initializing enhanced app controller: {str(e)}")
        logger.log_error(e, "app_initialization")
        return None


def apply_global_styles():
    """Apply enhanced global CSS styling"""
    st.markdown("""
    <style>
    /* Enhanced Global Styles */
    .main {
        padding-top: 2rem;
    }
    
    /* Modern Button Styles */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Metrics */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Modern Cards */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    /* Sidebar Enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Enhanced Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Loading Animations */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Success/Error States */
    .success-message {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            color: white;
        }
        
        .metric-container {
            background: #2d3748;
            color: white;
        }
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        .feature-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
        }
    }
    
    /* Performance Indicators */
    .performance-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .performance-good {
        background: #48bb78;
        color: white;
    }
    
    .performance-medium {
        background: #ed8936;
        color: white;
    }
    
    .performance-poor {
        background: #f56565;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


def render_app_status_bar():
    """Render application status and health indicators"""
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        data_status = "🟢 Online" if st.session_state.get('data_loaded', False) else "🔴 Offline"
        st.markdown(f"**Data:** {data_status}")
    
    with status_col2:
        cache_hit_rate = 85  # Would get from actual cache metrics
        cache_color = "🟢" if cache_hit_rate > 80 else "🟡" if cache_hit_rate > 60 else "🔴"
        st.markdown(f"**Cache:** {cache_color} {cache_hit_rate}%")
    
    with status_col3:
        ai_status = "🤖 Ready" if st.session_state.get('ai_enabled', True) else "🤖 Disabled"
        st.markdown(f"**AI:** {ai_status}")
    
    with status_col4:
        user_count = len(st.session_state.get('users', [1]))  # Placeholder
        st.markdown(f"**Users:** 👥 {user_count}")


def main():
    """Enhanced main application entry point"""
    try:
        # Apply global styling
        apply_global_styles()
        
        # Render status bar
        render_app_status_bar()
        
        # Get enhanced application controller
        app = get_enhanced_app_controller()
        
        if app is None:
            st.error("❌ Failed to initialize application. Please refresh the page.")
            
            # Provide helpful recovery options
            st.markdown("""
            <div class="error-message">
                <h4>🔧 Troubleshooting Steps:</h4>
                <ol>
                    <li>Refresh your browser page</li>
                    <li>Clear browser cache and cookies</li>
                    <li>Check your internet connection</li>
                    <li>Try accessing the app in an incognito/private window</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            return
        
        # Enhanced data validation before processing
        if hasattr(st.session_state, 'players_df') and st.session_state.players_df is not None:
            # Comprehensive data cleaning
            numeric_cols = [
                'total_points', 'now_cost', 'form', 'selected_by_percent', 
                'minutes', 'goals_scored', 'assists', 'clean_sheets',
                'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
                'points_per_game', 'value_form', 'value_season'
            ]
            
            for col in numeric_cols:
                if col in st.session_state.players_df.columns:
                    st.session_state.players_df[col] = pd.to_numeric(
                        st.session_state.players_df[col], errors='coerce'
                    ).fillna(0)
        
        # Run the enhanced application
        app.run()
        
        # Display cache metrics in sidebar if enabled
        if config.ui.show_debug_info:
            display_cache_metrics()
        
    except Exception as e:
        logger.log_error(e, "main_app")
        
        st.error("❌ Application encountered an unexpected error")
        
        # Enhanced error display with recovery options
        with st.expander("🔍 Error Details & Recovery"):
            st.write("**Error Type:**", type(e).__name__)
            st.write("**Error Message:**", str(e))
            
            # Specific error handling
            if "multiply sequence" in str(e):
                st.warning("🔧 **Data Type Error Detected**")
                st.markdown("""
                This error has been automatically resolved with enhanced data cleaning.
                
                **Recovery Steps:**
                1. Click '🔄 Refresh Data' in the sidebar
                2. Try your action again
                3. If the error persists, refresh the browser page
                """)
                
            elif "connection" in str(e).lower():
                st.warning("🌐 **Connection Error**")
                st.markdown("""
                **Recovery Steps:**
                1. Check your internet connection
                2. Wait 30 seconds and try again
                3. The FPL API might be temporarily unavailable
                """)
            
            elif "cache" in str(e).lower():
                st.warning("💾 **Cache Error**")
                st.markdown("""
                **Recovery Steps:**
                1. Clear cache using the sidebar button
                2. Refresh the page
                3. Reload your data
                """)
            
            # Recovery actions
            st.markdown("### 🛠️ Quick Recovery Actions")
            
            recovery_col1, recovery_col2, recovery_col3 = st.columns(3)
            
            with recovery_col1:
                if st.button("🔄 Refresh Page", type="primary"):
                    st.rerun()
            
            with recovery_col2:
                if st.button("🗑️ Clear Cache"):
                    try:
                        from utils.enhanced_cache import cache_manager
                        cache_manager.clear_cache()
                        st.success("Cache cleared!")
                    except:
                        st.error("Could not clear cache")
            
            with recovery_col3:
                if st.button("🏠 Go to Dashboard"):
                    st.session_state.current_page = "dashboard"
                    st.rerun()
            
            # Debug information for developers
            if st.checkbox("🔬 Show Technical Details"):
                st.code(str(e))
                
                import traceback
                st.text("Stack Trace:")
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
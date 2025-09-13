"""
Enhanced App Performance Configuration
Optimizes Streamlit performance and caching
"""
import streamlit as st
from functools import wraps
import pandas as pd
import time
from typing import Dict, Any, Optional

class PerformanceConfig:
    """Configuration for app performance optimizations"""
    
    # Cache settings
    CACHE_TTL = 300  # 5 minutes
    MAX_CACHE_ENTRIES = 100
    
    # Data loading settings
    API_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    BATCH_SIZE = 1000
    
    # UI settings
    MAX_ROWS_DISPLAY = 50
    PAGINATION_SIZE = 25
    LAZY_LOADING = True

@st.cache_data(ttl=PerformanceConfig.CACHE_TTL, max_entries=PerformanceConfig.MAX_CACHE_ENTRIES)
def cached_api_call(url: str, params: Optional[Dict] = None):
    """Cached API call wrapper"""
    import requests
    try:
        response = requests.get(url, params=params, timeout=PerformanceConfig.API_TIMEOUT, verify=False)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

@st.cache_data(ttl=PerformanceConfig.CACHE_TTL)
def cached_data_processing(data: Dict) -> pd.DataFrame:
    """Cached data processing"""
    # Process FPL data with caching
    return pd.DataFrame(data)

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if execution_time > 2.0:  # Log slow functions
            st.warning(f"Slow operation detected: {func.__name__} took {execution_time:.2f}s")
        
        return result
    return wrapper

class DataValidator:
    """Validates data integrity and completeness"""
    
    @staticmethod
    def validate_players_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate player data quality"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        required_columns = ['id', 'web_name', 'team', 'element_type', 'now_cost', 'total_points']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check for reasonable data ranges
        if not df.empty:
            if df['now_cost'].max() > 200 or df['now_cost'].min() < 30:
                validation_results['warnings'].append("Unusual price ranges detected")
            
            if df['total_points'].max() > 400:
                validation_results['warnings'].append("Unusually high points detected")
        
        return validation_results
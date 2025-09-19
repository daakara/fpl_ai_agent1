"""
Enhanced Cache Manager with Performance Optimization
"""
import streamlit as st
import pandas as pd
import hashlib
import pickle
import time
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import os

class EnhancedCacheManager:
    """Advanced caching with TTL, size limits, and performance monitoring"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.cache_dir = "cache"
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'cache_size_mb': 0
        }
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate unique cache key from arguments"""
        key_string = f"{args}_{kwargs}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            
            if not os.path.exists(cache_file):
                self.metrics['misses'] += 1
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check TTL
            if datetime.now() > cache_data['expires']:
                os.remove(cache_file)
                self.metrics['misses'] += 1
                return None
            
            self.metrics['hits'] += 1
            return cache_data['data']
            
        except Exception:
            self.metrics['misses'] += 1
            return None
        finally:
            self.metrics['total_requests'] += 1
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache with TTL"""
        try:
            ttl = ttl or self.default_ttl
            expires = datetime.now() + timedelta(seconds=ttl)
            
            cache_data = {
                'data': data,
                'expires': expires,
                'created': datetime.now()
            }
            
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self._cleanup_if_needed()
            return True
            
        except Exception:
            return False
    
    def _cleanup_if_needed(self):
        """Clean up cache if size exceeds limit"""
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in os.listdir(self.cache_dir)
            if f.endswith('.cache')
        )
        
        self.metrics['cache_size_mb'] = total_size / (1024 * 1024)
        
        if self.metrics['cache_size_mb'] > self.max_size_mb:
            # Remove oldest files
            cache_files = [
                os.path.join(self.cache_dir, f)
                for f in os.listdir(self.cache_dir)
                if f.endswith('.cache')
            ]
            
            cache_files.sort(key=os.path.getctime)
            
            # Remove oldest 25% of files
            files_to_remove = len(cache_files) // 4
            for file_path in cache_files[:files_to_remove]:
                try:
                    os.remove(file_path)
                except:
                    pass
    
    def get_metrics(self) -> dict:
        """Get cache performance metrics"""
        hit_rate = (self.metrics['hits'] / max(self.metrics['total_requests'], 1)) * 100
        return {
            **self.metrics,
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    def clear_cache(self):
        """Clear all cache files"""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.cache'):
                    os.remove(os.path.join(self.cache_dir, file))
            
            self.metrics = {
                'hits': 0,
                'misses': 0,
                'total_requests': 0,
                'cache_size_mb': 0
            }
            return True
        except:
            return False

# Global cache manager
cache_manager = EnhancedCacheManager()

def cached_function(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}_{func.__name__}_{cache_manager.get_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl)
            
            # Log performance if enabled
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.sidebar.info(f"🔄 {func.__name__}: {execution_time:.2f}s (cached)")
            
            return result
        
        return wrapper
    return decorator

@cached_function(ttl=3600, key_prefix="fpl_data")
def cached_load_fpl_data():
    """Cached FPL data loading"""
    from services.fpl_data_service import FPLDataService
    service = FPLDataService()
    return service.load_fpl_data()

@cached_function(ttl=1800, key_prefix="team_data")
def cached_load_team_data(team_id: str, gameweek: int):
    """Cached team data loading"""
    from services.fpl_data_service import FPLDataService
    service = FPLDataService()
    return service.load_team_data(team_id, gameweek)

def display_cache_metrics():
    """Display cache performance metrics in sidebar"""
    metrics = cache_manager.get_metrics()
    
    with st.sidebar.expander("📊 Cache Performance"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Hit Rate", f"{metrics['hit_rate_percent']:.1f}%")
            st.metric("Cache Size", f"{metrics['cache_size_mb']:.1f}MB")
        
        with col2:
            st.metric("Hits", metrics['hits'])
            st.metric("Misses", metrics['misses'])
        
        if st.button("Clear Cache"):
            cache_manager.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
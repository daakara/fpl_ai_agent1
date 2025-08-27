"""
Advanced Caching and Performance System for FPL Analytics App
Provides intelligent caching, data persistence, and performance optimization
"""
import sqlite3
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Callable
from pathlib import Path
import threading
from functools import wraps

class FPLCache:
    """Advanced caching system for FPL data"""
    
    def __init__(self, cache_dir: str = "fpl_cache", ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialize SQLite database for caching"""
        with sqlite3.connect(self.db_path) as conn:
            # Create table with basic structure first
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)
            
            # Check existing columns and add missing ones
            cursor = conn.execute("PRAGMA table_info(cache)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add missing columns for enhanced caching
            if 'created_at' not in existing_columns:
                conn.execute("ALTER TABLE cache ADD COLUMN created_at TIMESTAMP")
            
            if 'expires_at' not in existing_columns:
                conn.execute("ALTER TABLE cache ADD COLUMN expires_at TIMESTAMP")
                
            if 'hit_count' not in existing_columns:
                conn.execute("ALTER TABLE cache ADD COLUMN hit_count INTEGER DEFAULT 0")
                
            if 'data_type' not in existing_columns:
                conn.execute("ALTER TABLE cache ADD COLUMN data_type TEXT")
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON cache(created_at)
            """)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value, expires_at, data_type, hit_count
                    FROM cache 
                    WHERE key = ? AND expires_at > ?
                """, (key, datetime.now()))
                
                row = cursor.fetchone()
                
                if row:
                    # Update hit count
                    conn.execute("""
                        UPDATE cache 
                        SET hit_count = hit_count + 1 
                        WHERE key = ?
                    """, (key,))
                    
                    value_blob, expires_at, data_type, hit_count = row
                    
                    try:
                        if data_type == 'json':
                            return json.loads(value_blob)
                        else:
                            return pickle.loads(value_blob)
                    except (json.JSONDecodeError, pickle.PickleError):
                        # Remove corrupted cache entry
                        self.delete(key)
                        return None
                
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl_seconds or self.ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        # Determine data type and serialize
        try:
            # Try JSON first (more efficient)
            value_blob = json.dumps(value)
            data_type = 'json'
        except (TypeError, ValueError):
            # Fall back to pickle
            value_blob = pickle.dumps(value)
            data_type = 'pickle'
        
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache 
                        (key, value, created_at, expires_at, data_type, hit_count)
                        VALUES (?, ?, ?, ?, ?, 0)
                    """, (key, value_blob, datetime.now(), expires_at, data_type))
                
                return True
            except Exception:
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return cursor.rowcount > 0
    
    def clear_expired(self) -> int:
        """Clear expired cache entries"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache 
                    WHERE expires_at <= ?
                """, (datetime.now(),))
                return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            total_count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            
            # Expired entries
            expired_count = conn.execute("""
                SELECT COUNT(*) FROM cache 
                WHERE expires_at <= ?
            """, (datetime.now(),)).fetchone()[0]
            
            # Cache size
            cache_size = conn.execute("""
                SELECT SUM(LENGTH(value)) FROM cache
            """).fetchone()[0] or 0
            
            # Hit statistics
            hit_stats = conn.execute("""
                SELECT AVG(hit_count), MAX(hit_count), MIN(hit_count)
                FROM cache 
                WHERE expires_at > ?
            """, (datetime.now(),)).fetchone()
            
            return {
                'total_entries': total_count,
                'active_entries': total_count - expired_count,
                'expired_entries': expired_count,
                'cache_size_bytes': cache_size,
                'cache_size_mb': cache_size / (1024 * 1024),
                'avg_hits': hit_stats[0] or 0,
                'max_hits': hit_stats[1] or 0,
                'min_hits': hit_stats[2] or 0
            }

# Global cache instance
cache = FPLCache()

def cached(ttl_seconds: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}:{cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator

class DataManager:
    """Enhanced data management with caching and validation"""
    
    def __init__(self, cache_instance: FPLCache = None):
        self.cache = cache_instance or cache
        self.api_call_count = 0
        self.cache_hit_count = 0
    
    @cached(ttl_seconds=1800, key_prefix="fpl_api:")  # 30 minutes
    def fetch_bootstrap_data(self) -> Optional[Dict]:
        """Fetch and cache FPL bootstrap data"""
        try:
            import requests
            
            self.api_call_count += 1
            
            response = requests.get(
                "https://fantasy.premierleague.com/api/bootstrap-static/",
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            print(f"Error fetching bootstrap data: {e}")
            return None
    
    @cached(ttl_seconds=3600, key_prefix="fpl_fixtures:")  # 1 hour
    def fetch_fixtures_data(self) -> Optional[List[Dict]]:
        """Fetch and cache fixtures data"""
        try:
            import requests
            
            self.api_call_count += 1
            
            response = requests.get(
                "https://fantasy.premierleague.com/api/fixtures/",
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            print(f"Error fetching fixtures data: {e}")
            return None
    
    @cached(ttl_seconds=1800, key_prefix="fpl_team:")  # 30 minutes
    def fetch_team_data(self, team_id: str) -> Optional[Dict]:
        """Fetch and cache team data"""
        try:
            import requests
            
            self.api_call_count += 1
            
            response = requests.get(
                f"https://fantasy.premierleague.com/api/entry/{team_id}/",
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            print(f"Error fetching team data: {e}")
            return None
    
    def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        total_requests = self.api_call_count + self.cache_hit_count
        cache_hit_rate = (self.cache_hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'api_calls': self.api_call_count,
            'cache_hits': self.cache_hit_count,
            'total_requests': total_requests,
            'cache_hit_rate': cache_hit_rate,
            **self.cache.get_stats()
        }

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def lazy_load_data(data_loader_func: Callable, session_key: str):
        """Lazy load data with session state caching"""
        try:
            import streamlit as st
            
            if session_key not in st.session_state:
                with st.spinner("Loading data..."):
                    st.session_state[session_key] = data_loader_func()
            
            return st.session_state[session_key]
        
        except ImportError:
            # Fallback if streamlit not available
            return data_loader_func()
    
    @staticmethod
    def paginate_data(data: List[Any], page_size: int = 50, page: int = 1) -> Dict[str, Any]:
        """Paginate large datasets"""
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        return {
            'data': data[start_idx:end_idx],
            'pagination': {
                'current_page': page,
                'total_pages': total_pages,
                'total_items': total_items,
                'page_size': page_size,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        }
    
    @staticmethod
    def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
        """Process items in batches for better performance"""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# Global data manager instance
data_manager = DataManager()

def cleanup_cache():
    """Clean up expired cache entries"""
    removed_count = cache.clear_expired()
    print(f"Removed {removed_count} expired cache entries")
    return removed_count

def cache_maintenance():
    """Perform cache maintenance operations"""
    stats = cache.get_stats()
    
    # Clear expired entries
    cleanup_cache()
    
    # If cache is too large, remove least accessed entries
    if stats['cache_size_mb'] > 100:  # 100MB limit
        with sqlite3.connect(cache.db_path) as conn:
            # Remove 20% of least accessed entries
            conn.execute("""
                DELETE FROM cache 
                WHERE key IN (
                    SELECT key FROM cache 
                    ORDER BY hit_count ASC, created_at ASC 
                    LIMIT (SELECT COUNT(*) * 0.2 FROM cache)
                )
            """)
    
    print("Cache maintenance completed")

# Utility functions for common caching patterns
def cache_player_data(player_id: int, ttl_seconds: int = 3600):
    """Cache player-specific data"""
    return cached(ttl_seconds=ttl_seconds, key_prefix=f"player_{player_id}:")

def cache_gameweek_data(gameweek: int, ttl_seconds: int = 1800):
    """Cache gameweek-specific data"""
    return cached(ttl_seconds=ttl_seconds, key_prefix=f"gw_{gameweek}:")

def invalidate_cache_pattern(pattern: str):
    """Invalidate cache entries matching a pattern"""
    with sqlite3.connect(cache.db_path) as conn:
        cursor = conn.execute("SELECT key FROM cache WHERE key LIKE ?", (f"%{pattern}%",))
        keys_to_delete = [row[0] for row in cursor.fetchall()]
        
        for key in keys_to_delete:
            cache.delete(key)
        
        return len(keys_to_delete)

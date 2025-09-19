"""
Enhanced Performance Cache Manager
Multi-tier caching with memory and disk persistence for optimal performance
"""

import sqlite3
import pickle
import json
import time
import hashlib
from typing import Any, Optional, Dict, List
from pathlib import Path
from threading import Lock
from functools import wraps
import logging
from config.enhanced_app_config import get_config

logger = logging.getLogger(__name__)

class MemoryCache:
    """In-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                
                # Check expiration
                entry = self.cache[key]
                if time.time() < entry['expires_at']:
                    return entry['data']
                else:
                    del self.cache[key]
                    self.access_order.remove(key)
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        with self.lock:
            # Remove if exists
            if key in self.cache:
                self.access_order.remove(key)
            
            # Add new entry
            self.cache[key] = {
                'data': value,
                'expires_at': time.time() + ttl_seconds,
                'created_at': time.time()
            }
            self.access_order.append(key)
            
            # Evict if necessary
            while len(self.cache) > self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
    
    def delete(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def size(self) -> int:
        return len(self.cache)

class DiskCache:
    """Persistent disk cache using SQLite"""
    
    def __init__(self, cache_dir: str = "fpl_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.lock = Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize the cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    expires_at REAL,
                    created_at REAL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)')
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT data, expires_at FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        data_blob, expires_at = row
                        if time.time() < expires_at:
                            # Update access count
                            conn.execute(
                                'UPDATE cache_entries SET access_count = access_count + 1 WHERE key = ?',
                                (key,)
                            )
                            return pickle.loads(data_blob)
                        else:
                            # Expired, delete
                            conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                
                return None
            except Exception as e:
                logger.error(f"Error reading from disk cache: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        with self.lock:
            try:
                data_blob = pickle.dumps(value)
                size_bytes = len(data_blob)
                expires_at = time.time() + ttl_seconds
                created_at = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (key, data, expires_at, created_at, access_count, size_bytes)
                        VALUES (?, ?, ?, ?, 0, ?)
                    ''', (key, data_blob, expires_at, created_at, size_bytes))
                
            except Exception as e:
                logger.error(f"Error writing to disk cache: {e}")
    
    def delete(self, key: str):
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            except Exception as e:
                logger.error(f"Error deleting from disk cache: {e}")
    
    def clear_expired(self):
        """Remove expired entries"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM cache_entries WHERE expires_at < ?', (time.time(),))
            except Exception as e:
                logger.error(f"Error clearing expired entries: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN expires_at > ? THEN 1 END) as valid_entries
                    FROM cache_entries
                ''', (time.time(),))
                
                row = cursor.fetchone()
                return {
                    'total_entries': row[0] or 0,
                    'total_size_mb': (row[1] or 0) / (1024 * 1024),
                    'avg_access_count': row[2] or 0,
                    'valid_entries': row[3] or 0
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

class EnhancedCacheManager:
    """Multi-tier cache manager with memory and disk caching"""
    
    def __init__(self):
        config = get_config()
        self.memory_cache = MemoryCache(config.cache.max_size) if config.cache.use_memory_cache else None
        self.disk_cache = DiskCache(config.cache.cache_dir) if config.cache.use_disk_cache else None
        self.enabled = config.cache.enabled
        self.default_ttl = config.cache.ttl_seconds
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key"""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str, cache_type: str = "default") -> Optional[Any]:
        """Get value from cache (memory first, then disk)"""
        if not self.enabled:
            return None
        
        # Try memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Try disk cache
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Warm memory cache
                if self.memory_cache:
                    config = get_config()
                    ttl = config.get_cache_ttl(cache_type)
                    self.memory_cache.set(key, value, ttl)
                return value
        
        return None
    
    def set(self, key: str, value: Any, cache_type: str = "default"):
        """Set value in both memory and disk cache"""
        if not self.enabled:
            return
        
        config = get_config()
        ttl = config.get_cache_ttl(cache_type)
        
        # Set in memory cache
        if self.memory_cache:
            self.memory_cache.set(key, value, ttl)
        
        # Set in disk cache
        if self.disk_cache:
            self.disk_cache.set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete from both caches"""
        if self.memory_cache:
            self.memory_cache.delete(key)
        if self.disk_cache:
            self.disk_cache.delete(key)
    
    def clear(self, cache_type: Optional[str] = None):
        """Clear cache(s)"""
        if cache_type is None:
            # Clear all
            if self.memory_cache:
                self.memory_cache.clear()
            if self.disk_cache:
                self.disk_cache.clear_expired()
    
    def get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Public method to generate cache keys"""
        return self._generate_key(prefix, *args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'enabled': self.enabled,
            'memory_cache': {},
            'disk_cache': {}
        }
        
        if self.memory_cache:
            stats['memory_cache'] = {
                'size': self.memory_cache.size(),
                'max_size': self.memory_cache.max_size
            }
        
        if self.disk_cache:
            stats['disk_cache'] = self.disk_cache.get_stats()
        
        return stats

# Global cache instance
cache_manager = EnhancedCacheManager()

def cached(cache_type: str = "default", ttl_override: Optional[int] = None):
    """Decorator for automatic caching of function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_manager.get_cache_key(f"{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            result = cache_manager.get(key, cache_type)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(key, result, cache_type)
            
            return result
        return wrapper
    return decorator

# Convenience functions
def get_cache() -> EnhancedCacheManager:
    """Get the global cache manager instance"""
    return cache_manager

def invalidate_cache(pattern: Optional[str] = None):
    """Invalidate cache entries matching pattern"""
    cache_manager.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return cache_manager.get_stats()
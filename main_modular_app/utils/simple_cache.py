"""
Simple fallback caching module for when main caching fails
"""
import pickle
import time
from pathlib import Path
from typing import Any, Optional

class SimpleFPLCache:
    """Fallback cache implementation using simple file storage"""
    
    def __init__(self, cache_dir: str = "simple_cache", ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if expired
            if time.time() - data['timestamp'] > self.ttl_seconds:
                cache_file.unlink()  # Delete expired cache
                return None
            
            return data['value']
        except Exception:
            return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            data = {
                'value': value,
                'timestamp': time.time()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Fail silently
    
    def clear(self):
        """Clear all cache"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception:
            pass

# Simple cached decorator for fallback
def simple_cached(ttl_seconds: int = 3600):
    """Simple caching decorator"""
    cache = SimpleFPLCache(ttl_seconds=ttl_seconds)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate simple key
            key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator

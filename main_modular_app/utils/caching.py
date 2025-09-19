"""
Advanced Caching System for FPL Analytics App
"""
import pickle
import hashlib
import json
import time
import asyncio
from pathlib import Path
from typing import Any, Optional, Callable, Dict, Union
from datetime import datetime, timedelta
import streamlit as st
import threading
from concurrent.futures import ThreadPoolExecutor


class CacheBackend:
    """Base class for cache backends"""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError
    
    def delete(self, key: str) -> None:
        raise NotImplementedError
    
    def clear(self) -> None:
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """In-memory cache backend"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if entry['expires'] > time.time():
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created'])
            del self.cache[oldest_key]
        
        expires = time.time() + (ttl or 3600)
        self.cache[key] = {
            'value': value,
            'created': time.time(),
            'expires': expires
        }
    
    def delete(self, key: str) -> None:
        self.cache.pop(key, None)
    
    def clear(self) -> None:
        self.cache.clear()


class FileCache(CacheBackend):
    """File-based cache backend"""
    
    def __init__(self, cache_dir: str = "fpl_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        file_path = self._get_file_path(key)
        
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                if entry['expires'] > time.time():
                    return entry['value']
                else:
                    file_path.unlink()
            except Exception:
                # Corrupted cache file
                file_path.unlink(missing_ok=True)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        file_path = self._get_file_path(key)
        expires = time.time() + (ttl or 3600)
        
        entry = {
            'value': value,
            'created': time.time(),
            'expires': expires
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logging.warning(f"Failed to cache to file: {e}")
    
    def delete(self, key: str) -> None:
        file_path = self._get_file_path(key)
        file_path.unlink(missing_ok=True)
    
    def clear(self) -> None:
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink()


class CacheManager:
    """Main cache manager"""
    
    def __init__(self, backend: str = "memory", **kwargs):
        if backend == "memory":
            self.backend = MemoryCache(**kwargs)
        elif backend == "file":
            self.backend = FileCache(**kwargs)
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
    
    def get_or_set(self, key: str, func: Callable, ttl: Optional[int] = None) -> Any:
        """Get from cache or execute function and cache result"""
        cached_value = self.backend.get(key)
        if cached_value is not None:
            return cached_value
        
        # Execute function and cache result
        value = func()
        self.backend.set(key, value, ttl)
        return value
    
    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all keys matching pattern (simple implementation)"""
        # This is a simplified implementation
        # In production, you might want a more sophisticated pattern matching
        pass


class AsyncCacheManager(CacheManager):
    """Async-capable cache manager with background refresh"""
    
    def __init__(self, backend: str = "memory", **kwargs):
        super().__init__(backend, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._background_tasks = {}
    
    def get_or_set_async(self, key: str, func: Callable, ttl: Optional[int] = None, 
                        background_refresh: bool = False) -> Any:
        """Get from cache or execute function with optional background refresh"""
        cached_value = self.backend.get(key)
        
        if cached_value is not None:
            # If background refresh is enabled and cache is near expiry, refresh in background
            if background_refresh and key not in self._background_tasks:
                self._schedule_background_refresh(key, func, ttl)
            return cached_value
        
        # Execute function and cache result
        value = func()
        self.backend.set(key, value, ttl)
        return value
    
    def _schedule_background_refresh(self, key: str, func: Callable, ttl: Optional[int]):
        """Schedule background cache refresh"""
        def refresh_cache():
            try:
                new_value = func()
                self.backend.set(key, new_value, ttl)
            except Exception as e:
                st.warning(f"Background cache refresh failed: {e}")
            finally:
                self._background_tasks.pop(key, None)
        
        future = self.executor.submit(refresh_cache)
        self._background_tasks[key] = future


# Enhanced global cache instance
async_cache_manager = AsyncCacheManager()


def cached(ttl_seconds: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            
            # Add arguments to key
            if args:
                key_parts.extend([str(arg) for arg in args])
            if kwargs:
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            
            cache_key = ":".join(key_parts)
            
            return cache_manager.get_or_set(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl_seconds
            )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


def cached_with_refresh(ttl_seconds: int = 3600, key_prefix: str = "", background_refresh: bool = True):
    """Enhanced caching decorator with background refresh capability"""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            
            # Add arguments to key
            if args:
                key_parts.extend([str(arg) for arg in args])
            if kwargs:
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            
            cache_key = ":".join(key_parts)
            
            return async_cache_manager.get_or_set_async(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl_seconds,
                background_refresh
            )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# Streamlit-specific caching helpers
def clear_app_cache():
    """Clear application cache"""
    cache_manager.backend.clear()
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    if isinstance(cache_manager.backend, MemoryCache):
        return {
            "backend": "memory",
            "entries": len(cache_manager.backend.cache),
            "max_size": cache_manager.backend.max_size
        }
    elif isinstance(cache_manager.backend, FileCache):
        cache_files = list(cache_manager.backend.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "backend": "file", 
            "entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024)
        }
    
    return {"backend": "unknown"}

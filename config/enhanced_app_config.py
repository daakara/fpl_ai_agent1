"""
Enhanced App Configuration for FPL Analytics
Advanced configuration management with performance optimization and feature flags
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes
    max_size: int = 1000
    cache_dir: str = "fpl_cache"
    use_memory_cache: bool = True
    use_disk_cache: bool = True

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 100
    enable_data_compression: bool = True
    lazy_loading: bool = True
    enable_profiling: bool = False

@dataclass
class UIConfig:
    """UI/UX configuration"""
    theme: str = "light"
    enable_animations: bool = True
    items_per_page: int = 20
    enable_real_time_updates: bool = True
    chart_animation_duration: int = 500
    enable_tooltips: bool = True

@dataclass
class DataConfig:
    """Data processing configuration"""
    auto_refresh_interval: int = 3600  # 1 hour
    max_retry_attempts: int = 3
    timeout_seconds: int = 30
    enable_data_validation: bool = True
    backup_data: bool = True
    data_compression_level: int = 6

@dataclass
class AIConfig:
    """AI/ML configuration"""
    enable_predictions: bool = True
    model_confidence_threshold: float = 0.7
    enable_transfer_recommendations: bool = True
    enable_captain_suggestions: bool = True
    prediction_horizon_gws: int = 5
    enable_sentiment_analysis: bool = False

@dataclass
class AdvancedFeatures:
    """Advanced feature flags"""
    enable_fixture_analysis: bool = True
    enable_xg_xa_analysis: bool = True
    enable_ownership_tracking: bool = True
    enable_price_change_predictions: bool = True
    enable_form_analysis: bool = True
    enable_team_comparison: bool = True
    enable_chip_strategy: bool = True

class EnhancedAppConfig:
    """Enhanced application configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/app_config.json"
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize configurations
        self.cache = CacheConfig()
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.data = DataConfig()
        self.ai = AIConfig()
        self.features = AdvancedFeatures()
        
        # Load from file if exists
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configurations
                if 'cache' in config_data:
                    self._update_dataclass(self.cache, config_data['cache'])
                if 'performance' in config_data:
                    self._update_dataclass(self.performance, config_data['performance'])
                if 'ui' in config_data:
                    self._update_dataclass(self.ui, config_data['ui'])
                if 'data' in config_data:
                    self._update_dataclass(self.data, config_data['data'])
                if 'ai' in config_data:
                    self._update_dataclass(self.ai, config_data['ai'])
                if 'features' in config_data:
                    self._update_dataclass(self.features, config_data['features'])
                    
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            'cache': self._dataclass_to_dict(self.cache),
            'performance': self._dataclass_to_dict(self.performance),
            'ui': self._dataclass_to_dict(self.ui),
            'data': self._dataclass_to_dict(self.data),
            'ai': self._dataclass_to_dict(self.ai),
            'features': self._dataclass_to_dict(self.features)
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def _update_dataclass(self, obj, data):
        """Update dataclass object with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _dataclass_to_dict(self, obj):
        """Convert dataclass to dictionary"""
        return {
            field.name: getattr(obj, field.name)
            for field in obj.__dataclass_fields__.values()
        }
    
    def get_cache_ttl(self, cache_type: str = "default") -> int:
        """Get cache TTL for specific cache type"""
        cache_ttls = {
            "players": self.cache.ttl_seconds,
            "fixtures": self.cache.ttl_seconds * 2,
            "teams": self.cache.ttl_seconds * 3,
            "my_team": 60,  # 1 minute for team data
            "default": self.cache.ttl_seconds
        }
        return cache_ttls.get(cache_type, self.cache.ttl_seconds)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a specific feature is enabled"""
        return getattr(self.features, f"enable_{feature_name}", False)
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance optimization settings"""
        return {
            'parallel_processing': self.performance.enable_parallel_processing,
            'max_workers': self.performance.max_workers,
            'chunk_size': self.performance.chunk_size,
            'compression': self.performance.enable_data_compression,
            'lazy_loading': self.performance.lazy_loading
        }
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI configuration settings"""
        return {
            'theme': self.ui.theme,
            'animations': self.ui.enable_animations,
            'pagination': self.ui.items_per_page,
            'real_time': self.ui.enable_real_time_updates,
            'tooltips': self.ui.enable_tooltips
        }
    
    def update_feature_flag(self, feature_name: str, enabled: bool):
        """Update a feature flag"""
        if hasattr(self.features, f"enable_{feature_name}"):
            setattr(self.features, f"enable_{feature_name}", enabled)
            self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.cache = CacheConfig()
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.data = DataConfig()
        self.ai = AIConfig()
        self.features = AdvancedFeatures()
        self.save_config()

# Global configuration instance
app_config = EnhancedAppConfig()

# Convenience functions
def get_config() -> EnhancedAppConfig:
    """Get the global configuration instance"""
    return app_config

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    return app_config.is_feature_enabled(feature_name)

def get_cache_ttl(cache_type: str = "default") -> int:
    """Get cache TTL for specific type"""
    return app_config.get_cache_ttl(cache_type)

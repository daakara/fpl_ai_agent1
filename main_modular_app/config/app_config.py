"""
Enhanced Configuration Management for FPL Analytics App
Centralized settings and feature flags
"""
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import streamlit as st

@dataclass
class APIConfig:
    """API configuration settings"""
    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    verify_ssl: bool = False

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour
    max_size_mb: int = 100
    cache_dir: str = "fpl_cache"

@dataclass
class UIConfig:
    """UI configuration settings"""
    page_title: str = "FPL Analytics Dashboard"
    page_icon: str = "⚽"
    layout: str = "wide"
    theme: str = "light"
    show_debug_info: bool = False

@dataclass
class FeatureFlags:
    """Feature flags for experimental features"""
    advanced_analytics: bool = True
    machine_learning: bool = False
    real_time_updates: bool = False
    social_features: bool = False
    betting_integration: bool = True
    mobile_optimization: bool = False

@dataclass
class DataConfig:
    """Data processing configuration"""
    default_gameweeks_ahead: int = 5
    min_player_points_threshold: int = 10
    default_budget: float = 100.0
    max_players_per_team: int = 3
    form_weight: float = 0.3
    fixture_weight: float = 0.2

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self.api = APIConfig()
        self.cache = CacheConfig()
        self.ui = UIConfig()
        self.features = FeatureFlags()
        self.data = DataConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from Streamlit secrets if available
        self._load_from_secrets()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API Configuration
        if os.getenv('FPL_API_TIMEOUT'):
            self.api.request_timeout = int(os.getenv('FPL_API_TIMEOUT'))
        
        if os.getenv('FPL_SSL_VERIFY'):
            self.api.verify_ssl = os.getenv('FPL_SSL_VERIFY').lower() == 'true'
        
        # Cache Configuration
        if os.getenv('CACHE_TTL'):
            self.cache.ttl_seconds = int(os.getenv('CACHE_TTL'))
        
        if os.getenv('CACHE_ENABLED'):
            self.cache.enabled = os.getenv('CACHE_ENABLED').lower() == 'true'
        
        # Feature Flags
        if os.getenv('ENABLE_ML_FEATURES'):
            self.features.machine_learning = os.getenv('ENABLE_ML_FEATURES').lower() == 'true'
        
        if os.getenv('ENABLE_REALTIME'):
            self.features.real_time_updates = os.getenv('ENABLE_REALTIME').lower() == 'true'
        
        # Debug mode
        if os.getenv('DEBUG_MODE'):
            self.ui.show_debug_info = os.getenv('DEBUG_MODE').lower() == 'true'
    
    def _load_from_secrets(self):
        """Load configuration from Streamlit secrets"""
        try:
            if hasattr(st, 'secrets'):
                # API settings
                if 'api' in st.secrets:
                    api_secrets = st.secrets['api']
                    if 'timeout' in api_secrets:
                        self.api.request_timeout = api_secrets['timeout']
                    if 'max_retries' in api_secrets:
                        self.api.max_retries = api_secrets['max_retries']
                
                # Feature flags
                if 'features' in st.secrets:
                    feature_secrets = st.secrets['features']
                    for feature, value in feature_secrets.items():
                        if hasattr(self.features, feature):
                            setattr(self.features, feature, value)
                
                # UI settings
                if 'ui' in st.secrets:
                    ui_secrets = st.secrets['ui']
                    for setting, value in ui_secrets.items():
                        if hasattr(self.ui, setting):
                            setattr(self.ui, setting, value)
        
        except Exception as e:
            # Gracefully handle missing secrets
            pass
    
    def get_api_headers(self) -> Dict[str, str]:
        """Get standard API headers"""
        return {
            'User-Agent': 'FPL-Analytics-App/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return getattr(self.features, feature_name, False)
    
    def get_cache_key(self, key_parts: list) -> str:
        """Generate standardized cache key"""
        return ":".join(str(part) for part in key_parts)
    
    def update_config(self, section: str, key: str, value: Any):
        """Update configuration at runtime"""
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

# Global configuration instance
config = ConfigManager()

# Configuration presets for different environments
DEVELOPMENT_CONFIG = {
    'ui': {'show_debug_info': True},
    'cache': {'ttl_seconds': 600},  # 10 minutes
    'features': {'machine_learning': True, 'real_time_updates': True}
}

PRODUCTION_CONFIG = {
    'ui': {'show_debug_info': False},
    'cache': {'ttl_seconds': 3600},  # 1 hour
    'features': {'machine_learning': False, 'real_time_updates': False}
}

def apply_environment_config(env: str = 'development'):
    """Apply environment-specific configuration"""
    global config
    
    env_config = DEVELOPMENT_CONFIG if env == 'development' else PRODUCTION_CONFIG
    
    for section, settings in env_config.items():
        for key, value in settings.items():
            config.update_config(section, key, value)

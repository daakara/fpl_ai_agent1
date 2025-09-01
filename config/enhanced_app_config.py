"""
Enhanced Configuration Management for FPL Analytics App
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class APIConfig:
    """FPL API configuration"""
    base_url: str = "https://fantasy.premierleague.com/api"
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_ttl: int = 3600  # 1 hour
    rate_limit_delay: float = 0.5


@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    backend: str = "memory"  # memory, redis, file
    max_size: int = 1000
    ttl_seconds: int = 3600
    cache_dir: str = "fpl_cache"


@dataclass
class UIConfig:
    """UI/UX configuration"""
    page_title: str = "FPL Analytics Dashboard"
    page_icon: str = "soccer"  # Use text instead of emoji for Windows compatibility
    layout: str = "wide"
    sidebar_state: str = "expanded"
    theme: str = "light"


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    model_cache_dir: str = "models"
    prediction_horizon: int = 5  # gameweeks
    min_training_samples: int = 100
    feature_importance_threshold: float = 0.05


@dataclass
class AppConfig:
    """Main application configuration"""
    api: APIConfig = APIConfig()
    cache: CacheConfig = CacheConfig()
    ui: UIConfig = UIConfig()
    ml: MLConfig = MLConfig()
    
    # Environment settings
    debug: bool = False
    log_level: str = "INFO"
    enable_profiling: bool = False
    
    # Feature flags
    enable_ai_recommendations: bool = True
    enable_premium_features: bool = False
    enable_social_features: bool = False


# Global configuration instance
config = AppConfig()

# Environment-based overrides
if os.getenv("FPL_DEBUG", "").lower() == "true":
    config.debug = True
    config.log_level = "DEBUG"

if os.getenv("FPL_CACHE_BACKEND"):
    config.cache.backend = os.getenv("FPL_CACHE_BACKEND")

if os.getenv("FPL_API_TIMEOUT"):
    config.api.request_timeout = int(os.getenv("FPL_API_TIMEOUT"))


def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config


def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration with new values"""
    global config
    
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Handle nested config updates
            parts = key.split('.')
            if len(parts) == 2:
                section, setting = parts
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, setting):
                        setattr(section_config, setting, value)

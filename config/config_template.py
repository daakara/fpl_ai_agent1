# config/config_template.py
"""Configuration template for FPL AI Agent"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # API Keys (set via environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    # App Settings
    APP_TITLE = "FPL AI Agent"
    PAGE_ICON = "⚽"
    LAYOUT = "wide"
    
    # Data Settings
    CACHE_TTL = 3600  # 1 hour
    MAX_PLAYERS_DISPLAY = 100
    DEFAULT_BUDGET = 100.0
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CACHE_DIR = PROJECT_ROOT / "fpl_cache"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        missing_keys = []
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not cls.COHERE_API_KEY:
            missing_keys.append("COHERE_API_KEY")
        
        return missing_keys

# Create .env template
ENV_TEMPLATE = """
# FPL AI Agent Environment Variables
# Copy this file to .env and fill in your API keys

# AI Services
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Optional: Team ID for personal data
FPL_TEAM_ID=your_fpl_team_id_here
"""
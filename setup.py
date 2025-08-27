#!/usr/bin/env python3
"""
Quick Setup Script for FPL Analytics App
Helps integrate new infrastructure with existing application
"""

import os
import sys
import shutil
from pathlib import Path

def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "config",
        "utils", 
        "core",
        "tests",
        "docs",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory in ["config", "utils", "core"]:
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
    
    print("✅ Directory structure created")

def update_main_app():
    """Update simple_app.py to use new infrastructure"""
    
    # Read the current simple_app.py
    simple_app_path = Path("simple_app.py")
    if not simple_app_path.exists():
        print("❌ simple_app.py not found")
        return
    
    # Backup original file
    backup_path = Path("simple_app_backup.py")
    shutil.copy2(simple_app_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Read current content
    content = simple_app_path.read_text()
    
    # Add new imports at the top
    imports_to_add = """
# Enhanced Infrastructure Imports
try:
    from config.app_config import config
    from utils.error_handling import handle_errors, FPLLogger
    from utils.caching import cached, DataManager
    from utils.ui_enhancements import ui
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Infrastructure modules not available: {e}")
    CONFIG_AVAILABLE = False

# Initialize logger if available
if CONFIG_AVAILABLE:
    logger = FPLLogger(__name__)
    data_manager = DataManager()
else:
    logger = None
    data_manager = None
"""
    
    # Insert imports after existing imports
    lines = content.split('\n')
    import_end = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
    
    lines.insert(import_end, imports_to_add)
    
    # Update the content
    updated_content = '\n'.join(lines)
    simple_app_path.write_text(updated_content)
    
    print("✅ simple_app.py updated with new infrastructure imports")

def create_requirements():
    """Create comprehensive requirements.txt"""
    requirements = """
# Core Dependencies
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
plotly>=5.15.0

# Data Processing
scikit-learn>=1.3.0
scipy>=1.11.0

# HTTP & Async
httpx>=0.24.0
aiohttp>=3.8.0

# Database
sqlite3

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
typing-extensions>=4.7.0

# Development (Optional)
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0

# Visualization
seaborn>=0.12.0
matplotlib>=3.7.0

# Web Scraping (if needed)
beautifulsoup4>=4.12.0
selenium>=4.11.0
"""
    
    Path("requirements.txt").write_text(requirements.strip())
    print("✅ requirements.txt created")

def create_env_template():
    """Create .env template file"""
    env_template = """
# FPL Analytics App Configuration

# API Configuration
FPL_API_BASE_URL=https://fantasy.premierleague.com/api
API_REQUEST_TIMEOUT=30
API_RATE_LIMIT=60

# Cache Configuration
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=1000

# UI Configuration
APP_TITLE="FPL Analytics Dashboard"
APP_ICON="⚽"
PAGE_LAYOUT="wide"
THEME="light"

# Feature Flags
ENABLE_AI_RECOMMENDATIONS=true
ENABLE_BETTING_ODDS=true
ENABLE_MY_TEAM=true
ENABLE_ADVANCED_FILTERS=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/fpl_app.log

# Optional API Keys (for enhanced features)
# OPENAI_API_KEY=your_openai_key_here
# ODDS_API_KEY=your_odds_api_key_here
"""
    
    Path(".env.template").write_text(env_template.strip())
    print("✅ .env.template created")

def create_startup_script():
    """Create startup script"""
    startup_script = """#!/bin/bash

# FPL Analytics App Startup Script

echo "🚀 Starting FPL Analytics App..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env file with your configuration"
fi

# Create logs directory
mkdir -p logs

# Start the application
echo "🎯 Starting Streamlit app..."
streamlit run simple_app.py --server.port 8501 --server.address 0.0.0.0
"""
    
    script_path = Path("start.sh")
    script_path.write_text(startup_script.strip())
    script_path.chmod(0o755)
    print("✅ start.sh script created")

def main():
    """Main setup function"""
    print("🔧 FPL Analytics App - Quick Setup")
    print("=" * 40)
    
    try:
        create_directory_structure()
        update_main_app()
        create_requirements()
        create_env_template()
        create_startup_script()
        
        print("\n🎉 Setup Complete!")
        print("\nNext steps:")
        print("1. Review and edit .env file with your configuration")
        print("2. Run: ./start.sh")
        print("3. Or manually: pip install -r requirements.txt && streamlit run simple_app.py")
        print("\n📚 See implementation_roadmap.md for detailed next steps")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

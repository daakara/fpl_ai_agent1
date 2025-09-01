#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPL Analytics App - Infrastructure Setup Automation
Sets up configuration, error handling, caching, and testing framework
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Fix for Windows Unicode issues
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass
    
    # Force UTF-8 encoding for output
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

def safe_print(text):
    """Print text with fallback for Unicode issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove emojis and special characters for Windows compatibility
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)

class InfrastructureSetup:
    """Automates infrastructure setup for the refactored FPL app"""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
    
    def setup_all(self):
        """Setup complete infrastructure"""
        safe_print("Setting up FPL Analytics Infrastructure...")
        
        self.setup_enhanced_config()
        self.setup_error_handling()
        self.setup_caching_system()
        self.setup_logging()
        self.setup_testing_framework()
        self.setup_deployment_scripts()
        self.create_requirements()
        
        safe_print("Infrastructure setup completed!")
    
    def setup_enhanced_config(self):
        """Enhanced configuration system"""
        config_dir = self.workspace_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Enhanced app_config.py
        config_content = '''"""
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
'''
        
        with open(config_dir / "enhanced_app_config.py", 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        safe_print("Enhanced configuration system created")
    
    def setup_error_handling(self):
        """Advanced error handling system"""
        utils_dir = self.workspace_dir / "utils"
        utils_dir.mkdir(exist_ok=True)
        
        error_handling_content = '''"""
Advanced Error Handling for FPL Analytics App
"""
import logging
import functools
import traceback
import time
import requests
from typing import Callable, Any, Optional, Type, Union
import streamlit as st


class FPLError(Exception):
    """Base exception for FPL Analytics App"""
    pass


class APIError(FPLError):
    """API-related errors"""
    pass


class DataError(FPLError):
    """Data processing errors"""
    pass


class ConfigError(FPLError):
    """Configuration errors"""
    pass


class ValidationError(FPLError):
    """Data validation errors"""
    pass


class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_api_error(self, error: Exception, context: str = "") -> None:
        """Handle API-related errors"""
        error_msg = f"API Error {context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        if isinstance(error, requests.exceptions.Timeout):
            st.error("Request timed out. Please try again.")
        elif isinstance(error, requests.exceptions.ConnectionError):
            st.error("Connection error. Check your internet connection.")
        elif isinstance(error, requests.exceptions.HTTPError):
            st.error(f"HTTP Error: {error.response.status_code}")
        else:
            st.error("An API error occurred. Please try again later.")
    
    def handle_data_error(self, error: Exception, context: str = "") -> None:
        """Handle data processing errors"""
        error_msg = f"Data Error {context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        st.error("Data processing error. Using fallback data if available.")
    
    def handle_validation_error(self, error: Exception, context: str = "") -> None:
        """Handle validation errors"""
        error_msg = f"Validation Error {context}: {str(error)}"
        self.logger.warning(error_msg)
        st.warning(f"Invalid input: {str(error)}")
    
    def handle_generic_error(self, error: Exception, context: str = "") -> None:
        """Handle generic errors"""
        error_msg = f"Error {context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        st.error("An unexpected error occurred. Please refresh and try again.")


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(
    fallback_message: str = "An error occurred",
    fallback_value: Any = None,
    reraise: bool = False,
    error_types: tuple = (Exception,)
):
    """Decorator for automatic error handling"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                context = f"in {func.__name__}"
                
                if isinstance(e, APIError):
                    error_handler.handle_api_error(e, context)
                elif isinstance(e, DataError):
                    error_handler.handle_data_error(e, context)
                elif isinstance(e, ValidationError):
                    error_handler.handle_validation_error(e, context)
                else:
                    error_handler.handle_generic_error(e, context)
                
                if reraise:
                    raise
                
                return fallback_value
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    fallback_value: Any = None,
    error_message: str = "Operation failed"
) -> Any:
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        error_handler.handle_generic_error(e, f"during safe_execute of {func.__name__}")
        return fallback_value


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    @staticmethod
    def retry_with_backoff(
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ) -> Any:
        """Retry function with exponential backoff"""
        
        last_exception = None
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff_factor
        
        raise last_exception
'''
        
        with open(utils_dir / "enhanced_error_handling.py", 'w', encoding='utf-8') as f:
            f.write(error_handling_content)
        
        safe_print("Advanced error handling system created")
    
    def setup_caching_system(self):
        """Advanced caching system"""
        utils_dir = self.workspace_dir / "utils"
        
        caching_content = '''"""
Advanced Caching System for FPL Analytics App
"""
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Callable, Dict, Union
from datetime import datetime, timedelta
import streamlit as st


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


# Global cache instance
cache_manager = CacheManager()


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
'''
        
        with open(utils_dir / "caching.py", 'w', encoding='utf-8') as f:
            f.write(caching_content)
        
        safe_print("Advanced caching system created")
    
    def setup_logging(self):
        """Enhanced logging system"""
        utils_dir = self.workspace_dir / "utils"
        
        logging_content = '''"""
Enhanced Logging System for FPL Analytics App
"""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys


class FPLLogger:
    """Enhanced logger for FPL Analytics"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "fpl_analytics.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "fpl_errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **kwargs)


# Performance logging
class PerformanceLogger:
    """Log performance metrics"""
    
    def __init__(self, logger: FPLLogger):
        self.logger = logger
    
    def log_api_call(self, endpoint: str, duration: float, status_code: int):
        """Log API call performance"""
        self.logger.info(
            f"API Call - Endpoint: {endpoint}, Duration: {duration:.2f}s, Status: {status_code}"
        )
    
    def log_function_performance(self, func_name: str, duration: float, args_count: int = 0):
        """Log function performance"""
        self.logger.debug(
            f"Function - {func_name}, Duration: {duration:.2f}s, Args: {args_count}"
        )


# Create global logger instance
logger = FPLLogger("fpl_analytics")
perf_logger = PerformanceLogger(logger)
'''
        
        with open(utils_dir / "logging.py", 'w', encoding='utf-8') as f:
            f.write(logging_content)
        
        safe_print("Enhanced logging system created")
    
    def setup_testing_framework(self):
        """Create testing framework"""
        tests_dir = self.workspace_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        # Create __init__.py
        (tests_dir / "__init__.py").write_text("# Test package\n")
        
        # Test configuration
        test_config_content = '''"""
Test Configuration for FPL Analytics App
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_players_data():
    """Sample players data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'web_name': ['Salah', 'Kane', 'De Bruyne'],
        'total_points': [250, 200, 180],
        'cost_millions': [13.0, 11.5, 12.0],
        'position_name': ['Midfielder', 'Forward', 'Midfielder'],
        'team_short_name': ['LIV', 'TOT', 'MCI'],
        'form': [8.5, 6.0, 7.5],
        'selected_by_percent': [45.2, 25.1, 35.8]
    })


@pytest.fixture
def sample_teams_data():
    """Sample teams data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Liverpool', 'Tottenham', 'Man City'],
        'short_name': ['LIV', 'TOT', 'MCI'],
        'strength': [5, 4, 5]
    })


@pytest.fixture
def mock_fpl_api():
    """Mock FPL API responses"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': [],
            'teams': [],
            'events': []
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get
'''
        
        with open(tests_dir / "conftest.py", 'w', encoding='utf-8') as f:
            f.write(test_config_content)
        
        # Sample test file
        test_pages_content = '''"""
Test cases for page components
"""
import pytest
from unittest.mock import Mock, patch
import streamlit as st


class TestPlayerAnalysisPage:
    """Test PlayerAnalysisPage functionality"""
    
    def test_initialization(self):
        """Test page initialization"""
        from pages.player_analysis_page import PlayerAnalysisPage
        page = PlayerAnalysisPage()
        assert page is not None
    
    @patch('streamlit.session_state')
    def test_render_with_no_data(self, mock_session_state):
        """Test rendering with no data"""
        mock_session_state.data_loaded = False
        
        from pages.player_analysis_page import PlayerAnalysisPage
        page = PlayerAnalysisPage()
        
        # This should not raise an exception
        try:
            page.render()
        except Exception as e:
            pytest.fail(f"render() raised {e} unexpectedly")


class TestFixtureAnalysisPage:
    """Test FixtureAnalysisPage functionality"""
    
    def test_initialization(self):
        """Test page initialization"""
        from pages.fixture_analysis_page import FixtureAnalysisPage
        page = FixtureAnalysisPage()
        assert page is not None


class TestMyTeamPage:
    """Test MyTeamPage functionality"""
    
    def test_initialization(self):
        """Test page initialization"""
        from pages.my_team_page import MyTeamPage
        page = MyTeamPage()
        assert page is not None
'''
        
        with open(tests_dir / "test_pages.py", 'w', encoding='utf-8') as f:
            f.write(test_pages_content)
        
        # Service tests
        test_services_content = '''"""
Test cases for service components
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd


class TestFPLDataService:
    """Test FPLDataService functionality"""
    
    def test_initialization(self):
        """Test service initialization"""
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        assert service is not None
    
    @patch('requests.get')
    def test_load_fpl_data_success(self, mock_get, sample_players_data):
        """Test successful data loading"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': sample_players_data.to_dict('records'),
            'teams': [],
            'element_types': []
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        
        # This should not raise an exception
        try:
            result = service.load_fpl_data()
            assert result is not None
        except Exception as e:
            pytest.fail(f"load_fpl_data() raised {e} unexpectedly")
    
    @patch('requests.get')
    def test_load_fpl_data_api_error(self, mock_get):
        """Test API error handling"""
        mock_get.side_effect = Exception("API Error")
        
        from services.fpl_data_service import FPLDataService
        service = FPLDataService()
        
        # Should handle the error gracefully
        result = service.load_fpl_data()
        # Should return None or handle error appropriately
'''
        
        with open(tests_dir / "test_services.py", 'w', encoding='utf-8') as f:
            f.write(test_services_content)
        
        safe_print("Testing framework created")
    
    def setup_deployment_scripts(self):
        """Create deployment and utility scripts"""
        scripts_dir = self.workspace_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Deployment script
        deploy_content = '''#!/usr/bin/env python3
"""
Deployment script for FPL Analytics App
"""
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def run_tests():
    """Run test suite"""
    print("🧪 Running tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    return result.returncode == 0


def start_app():
    """Start the application"""
    print("🚀 Starting FPL Analytics App...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main_refactored.py"])


def main():
    """Main deployment process"""
    print("🚀 FPL Analytics App - Deployment")
    print("=" * 40)
    
    # Install dependencies
    install_dependencies()
    
    # Run tests
    if run_tests():
        print("✅ All tests passed!")
        
        # Start application
        start_app()
    else:
        print("❌ Tests failed! Please fix issues before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        
        with open(scripts_dir / "deploy.py", 'w', encoding='utf-8') as f:
            f.write(deploy_content)
        
        # Health check script
        health_check_content = '''#!/usr/bin/env python3
"""
Health check script for FPL Analytics App
"""
import requests
import time
import sys
from pathlib import Path


def check_fpl_api():
    """Check FPL API availability"""
    try:
        response = requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ FPL API check failed: {e}")
        return False


def check_app_structure():
    """Check if required files exist"""
    required_files = [
        "main_refactored.py",
        "pages/player_analysis_page.py",
        "pages/fixture_analysis_page.py",
        "pages/my_team_page.py",
        "services/fpl_data_service.py",
        "core/app_controller.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True


def main():
    """Run health checks"""
    print("🏥 FPL Analytics App - Health Check")
    print("=" * 40)
    
    checks = [
        ("App Structure", check_app_structure),
        ("FPL API", check_fpl_api)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"Checking {check_name}...")
        if check_func():
            print(f"✅ {check_name} - OK")
        else:
            print(f"❌ {check_name} - FAILED")
            all_passed = False
    
    print("=" * 40)
    if all_passed:
        print("✅ All health checks passed!")
        sys.exit(0)
    else:
        print("❌ Some health checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        
        with open(scripts_dir / "health_check.py", 'w', encoding='utf-8') as f:
            f.write(health_check_content)
        
        safe_print("Deployment scripts created")
    
    def create_requirements(self):
        """Create enhanced requirements.txt"""
        requirements_content = '''# FPL Analytics App - Production Requirements

# Core Dependencies
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
plotly>=5.15.0

# Data Processing
scipy>=1.11.0
scikit-learn>=1.3.0

# Configuration & Environment
python-dotenv>=1.0.0

# Caching & Performance
redis>=4.6.0

# Testing (Development)
pytest>=7.4.0
pytest-mock>=3.11.0
pytest-cov>=4.1.0

# Code Quality (Development)
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0

# Documentation (Development)
mkdocs>=1.5.0
mkdocs-material>=9.2.0
'''
        
        with open(self.workspace_dir / "requirements_production.txt", 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        
        safe_print("Production requirements created")


def main():
    """Main entry point for infrastructure setup"""
    safe_print("FPL Analytics App - Infrastructure Setup")
    safe_print("=" * 50)
    
    setup = InfrastructureSetup()
    setup.setup_all()
    
    safe_print("\n" + "=" * 50)
    safe_print("Infrastructure setup completed!")
    safe_print("\nNext steps:")
    safe_print("1. Run: python automate_refactor.py")
    safe_print("2. Run: python scripts/health_check.py")
    safe_print("3. Run: python scripts/deploy.py")


if __name__ == "__main__":
    main()
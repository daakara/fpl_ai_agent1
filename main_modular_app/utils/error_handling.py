"""
Enhanced Logging and Error Handling for FPL Analytics App
Provides structured logging and graceful error handling
"""
import logging
import traceback
import functools
from typing import Any, Callable, Optional
from datetime import datetime
import streamlit as st

class FPLLogger:
    """Enhanced logging for FPL Analytics App"""
    
    def __init__(self, name: str = "fpl_analytics"):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with appropriate handlers and formatters"""
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler('fpl_analytics.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.DEBUG)
    
    def log_api_call(self, endpoint: str, status_code: int, response_time: float):
        """Log API call details"""
        self.logger.info(
            f"API Call - Endpoint: {endpoint}, Status: {status_code}, "
            f"Response Time: {response_time:.2f}s"
        )
    
    def log_user_action(self, action: str, details: dict = None):
        """Log user actions for analytics"""
        details_str = f" - Details: {details}" if details else ""
        self.logger.info(f"User Action: {action}{details_str}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context"""
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        self.logger.error(f"Error in {context}: {error_details}")
    
    def log_performance(self, operation: str, duration: float, details: dict = None):
        """Log performance metrics"""
        details_str = f" - {details}" if details else ""
        self.logger.info(f"Performance - {operation}: {duration:.2f}s{details_str}")

# Global logger instance
logger = FPLLogger()

class FPLError(Exception):
    """Base exception for FPL Analytics App"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()

class APIError(FPLError):
    """API-related errors"""
    pass

class DataValidationError(FPLError):
    """Data validation errors"""
    pass

class ConfigurationError(FPLError):
    """Configuration-related errors"""
    pass

def handle_errors(error_message: str = "An error occurred", 
                 show_details: bool = False):
    """Decorator for handling errors in Streamlit functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except FPLError as e:
                logger.log_error(e, func.__name__)
                
                if show_details:
                    st.error(f"{error_message}: {str(e)}")
                    with st.expander("Error Details"):
                        st.write(f"Error Code: {e.error_code}")
                        st.write(f"Timestamp: {e.timestamp}")
                        if e.details:
                            st.json(e.details)
                else:
                    st.error(error_message)
                
                return None
            
            except Exception as e:
                logger.log_error(e, func.__name__)
                
                if show_details:
                    st.error(f"{error_message}: {str(e)}")
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc())
                else:
                    st.error(error_message)
                
                return None
        
        return wrapper
    return decorator

def validate_team_id(team_id: str) -> bool:
    """Validate FPL team ID format"""
    if not team_id:
        raise DataValidationError("Team ID cannot be empty", "EMPTY_TEAM_ID")
    
    if not team_id.isdigit():
        raise DataValidationError("Team ID must be numeric", "INVALID_TEAM_ID_FORMAT")
    
    if len(team_id) < 5 or len(team_id) > 8:
        raise DataValidationError("Team ID must be 5-8 digits", "INVALID_TEAM_ID_LENGTH")
    
    return True

def validate_budget(budget: float) -> bool:
    """Validate FPL budget"""
    if budget < 80 or budget > 120:
        raise DataValidationError("Budget must be between £80m and £120m", "INVALID_BUDGET")
    
    return True

def validate_formation(formation: tuple) -> bool:
    """Validate formation tuple"""
    if len(formation) != 3:
        raise DataValidationError("Formation must have 3 positions", "INVALID_FORMATION_LENGTH")
    
    if sum(formation) != 10:
        raise DataValidationError("Formation must sum to 10 outfield players", "INVALID_FORMATION_SUM")
    
    # Valid formations check
    valid_formations = [(3,4,3), (4,3,3), (3,5,2), (4,4,2), (5,3,2), (5,4,1)]
    if formation not in valid_formations:
        raise DataValidationError(f"Invalid formation: {formation}", "INVALID_FORMATION")
    
    return True

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {'start_time': datetime.now()}
    
    def end_timer(self, operation: str, details: dict = None):
        """End timing and log the operation"""
        if operation in self.metrics:
            start_time = self.metrics[operation]['start_time']
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.log_performance(operation, duration, details)
            
            # Clean up
            del self.metrics[operation]
            
            return duration
        
        return 0

def performance_monitor(operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                logger.log_performance(op_name, duration)
                
                return result
            
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.log_performance(f"{op_name}_FAILED", duration)
                raise
        
        return wrapper
    return decorator

# Global performance monitor
perf_monitor = PerformanceMonitor()

def safe_api_call(func: Callable, *args, **kwargs) -> Optional[Any]:
    """Safely execute API calls with proper error handling"""
    try:
        logger.log_user_action("API_CALL_START", {"function": func.__name__})
        
        start_time = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.log_api_call(func.__name__, 200, duration)
        logger.log_user_action("API_CALL_SUCCESS", {"function": func.__name__})
        
        return result
    
    except requests.exceptions.Timeout:
        logger.log_error(APIError("API request timed out"), func.__name__)
        st.warning("⏱️ Request timed out. Please try again.")
        return None
    
    except requests.exceptions.ConnectionError:
        logger.log_error(APIError("Connection error"), func.__name__)
        st.error("🌐 Connection error. Please check your internet connection.")
        return None
    
    except requests.exceptions.HTTPError as e:
        logger.log_error(APIError(f"HTTP error: {e.response.status_code}"), func.__name__)
        st.error(f"❌ API error: {e.response.status_code}")
        return None
    
    except Exception as e:
        logger.log_error(e, func.__name__)
        st.error("❌ An unexpected error occurred. Please try again.")
        return None

def display_error_summary():
    """Display error summary in sidebar"""
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    
    if st.session_state.error_count > 0:
        st.sidebar.warning(f"⚠️ {st.session_state.error_count} errors occurred")
        
        if st.sidebar.button("Clear Errors"):
            st.session_state.error_count = 0
            st.rerun()

# Import requests here to avoid circular imports
try:
    import requests
except ImportError:
    pass

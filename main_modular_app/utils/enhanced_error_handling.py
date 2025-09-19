"""
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

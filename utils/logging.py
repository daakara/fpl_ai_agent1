"""
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

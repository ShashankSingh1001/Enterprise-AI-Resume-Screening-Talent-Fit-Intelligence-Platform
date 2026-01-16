import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for other formatters
        record.levelname = levelname
        
        return result


def setup_logging(
    log_level: str = None,
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_colors: bool = True
):
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Convert string to logging level
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # ============ Console Handler ============
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        
        # Simple format for console
        console_format = '%(levelname)s - %(name)s - %(message)s'
        
        if enable_colors:
            console_formatter = ColoredFormatter(console_format)
        else:
            console_formatter = logging.Formatter(console_format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # ============ File Handlers ============
    if enable_file:
        # Detailed format for file logs
        file_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'
        )
        file_formatter = logging.Formatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 1. Main application log (all messages)
        app_log_file = log_path / 'app.log'
        app_handler = RotatingFileHandler(
            app_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(file_formatter)
        root_logger.addHandler(app_handler)
        
        # 2. Error log (only ERROR and CRITICAL)
        error_log_file = log_path / 'error.log'
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # 3. API log (for API-specific logging)
        api_log_file = log_path / 'api.log'
        api_handler = RotatingFileHandler(
            api_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        api_handler.setLevel(logging.INFO)
        api_handler.setFormatter(file_formatter)
        
        # Add filter to only log from api module
        api_handler.addFilter(lambda record: record.name.startswith('api'))
        root_logger.addHandler(api_handler)
    
    # Log the initialization
    root_logger.info(f"Logging system initialized - Level: {log_level}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exception: Exception, message: str = ""):
    if message:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"{type(exception).__name__}: {str(exception)}", exc_info=True)


def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}", exc_info=True)
            raise
    return wrapper

setup_logging()
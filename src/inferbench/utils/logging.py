"""
Logging configuration for InferBench Framework.

Uses loguru for structured logging with rotation, filtering,
and multiple output formats.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    json_format: bool = False,
) -> None:
    """
    Configure logging for the framework.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        rotation: Log rotation size/time
        retention: How long to keep old logs
        json_format: Whether to use JSON format for file logs
    """
    # Remove default handler
    logger.remove()
    
    # Console format - colorful and readable
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if json_format:
            logger.add(
                str(log_file),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                level=level,
                rotation=rotation,
                retention=retention,
                serialize=True,  # JSON format
                backtrace=True,
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            )
            logger.add(
                str(log_file),
                format=file_format,
                level=level,
                rotation=rotation,
                retention=retention,
                backtrace=True,
            )


def get_logger(name: str) -> "logger":
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name for the logger (usually __name__)
        
    Returns:
        Logger instance bound to the given name
    """
    return logger.bind(name=name)


# Create convenience functions
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
exception = logger.exception


__all__ = [
    "setup_logging",
    "get_logger",
    "logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
]

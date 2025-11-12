"""Logging utilities for the tracker."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "ndt",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name (default: "ndt")
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        format_string: Optional custom format string

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("ndt", level=logging.DEBUG, log_file="tracking.log")
        >>> logger.info("Starting tracking...")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

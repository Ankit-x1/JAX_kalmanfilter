"""
Professional logging configuration for JAX Kalman Filter
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, component: str = "root"
) -> logging.Logger:
    """Setup structured logging with file and console output"""

    logger = logging.getLogger(f"kalman_filter.{component}")
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path("logs") / f"{component}_{datetime.now().strftime('%Y%m%d')}.log"
        log_path.parent.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ComponentLogger:
    """Context-aware component logger"""

    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(f"kalman_filter.{component}")

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)

    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs)

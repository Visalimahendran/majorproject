"""
Logging Configuration
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name, level=logging.INFO, log_file=None):
    """Setup and configure logger"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

class NeuroMotorLogger:
    """Custom logger for NeuroMotor system"""
    
    def __init__(self, component_name):
        self.logger = setup_logger(f"NeuroMotor.{component_name}")
        self.component_name = component_name
    
    def info(self, message):
        self.logger.info(f"[{self.component_name}] {message}")
    
    def warning(self, message):
        self.logger.warning(f"[{self.component_name}] {message}")
    
    def error(self, message):
        self.logger.error(f"[{self.component_name}] {message}")
    
    def debug(self, message):
        self.logger.debug(f"[{self.component_name}] {message}")
    
    def start(self, operation):
        self.info(f"Starting {operation}")
        return datetime.now()
    
    def end(self, operation, start_time):
        duration = (datetime.now() - start_time).total_seconds()
        self.info(f"Completed {operation} in {duration:.2f}s")
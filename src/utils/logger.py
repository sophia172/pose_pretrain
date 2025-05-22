"""
Logging and monitoring utilities for Human Pose Estimation models.

This module provides logging and monitoring utilities for tracking
training progress and performance metrics.
"""
import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from colorama import Fore, Style, init

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

# Define custom logging level
DEBUGML = 15  # Between DEBUG (10) and INFO (20)
logging.addLevelName(DEBUGML, "DEBUGML")
# Add DEBUGML as an attribute to the logging module so it can be used as logging.DEBUGML
setattr(logging, "DEBUGML", DEBUGML)

def get_logger(name=None):
    # You can add more configuration here if needed
    return logging.getLogger(name)

def setup_logging(output_dir: str, debug: bool = False) -> None:
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug == 1 else logging.DEBUGML if debug == 2 else logging.INFO
    
    # Create log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file handler
    log_file = os.path.join(log_dir, "train.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class Logger(logging.Logger):
    """
    Logger class for tracking training progress and performance metrics.
    
    Supports console logging, file logging, TensorBoard, and local metric tracking.
    """
    
    def debugml(self, msg, *args, **kwargs):
        """Log 'msg % args' with severity 'DEBUGML'."""
        if self.isEnabledFor(DEBUGML):
            self.log(logging.DEBUGML, msg, *args, **kwargs)
    
    def gradient(self, msg, *args, **kwargs):
        # You can customize formatting or handling here
        self.log(logging.DEBUGML, f"[GRADIENT] {msg}", *args, **kwargs)

    def loss(self, msg, *args, **kwargs):
        self.log(logging.DEBUGML, f"[LOSS] {msg}", *args, **kwargs)

    def header(self, text, width=80, *args, **kwargs):
        """Print a centered header with decoration, without log prefix."""
        padding = max(0, (width - len(text) - 4) // 2)
        
        # Also write to any log files (without colors or timestamp prefix)
        lines = [
            f"\n{'=' * width}",
            f"={' ' * padding}{text}{' ' * padding}=",
            f"{'=' * width}\n"
        ]
        self._bypass_formatter(lines)

    def _bypass_formatter(self, lines):
        """Bypass the formatter"""
        # Find any FileHandlers in this logger or parent loggers
        for handler in logging.getLogger().handlers + self.handlers:
            try:
                for line in lines:
                    handler.stream.write(line + "\n")
                handler.stream.flush()
            except Exception as e:
                # Fall back to regular logging if direct write fails
                self.warning(f"Could not write header directly to log file: {e}")

logging.setLoggerClass(Logger)

class ProgressBar:
    """
    Progress bar for tracking training or evaluation progress.
    
    Wrapper around tqdm with logging integration.
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        unit: str = "it",
        log_interval: int = 10
    ):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of iterations
            desc: Description of the progress bar
            unit: Unit of iterations
            log_interval: Interval for logging progress
        """
        self.pbar = tqdm(total=total, desc=desc, unit=unit)
        self.log_interval = log_interval
        self.last_log_step = 0
        self.metrics = {}
        
    def update(self, n: int = 1, **kwargs) -> None:
        """
        Update progress bar.
        
        Args:
            n: Number of iterations to increment
            **kwargs: Additional metrics to display
        """
        # Update metrics
        self.metrics.update(kwargs)
        
        # Format metrics for display
        postfix = {}
        for k, v in self.metrics.items():
            if isinstance(v, float):
                postfix[k] = f"{v:.4f}"
            else:
                postfix[k] = str(v)
                
        # Update progress bar
        self.pbar.set_postfix(postfix)
        self.pbar.update(n)
        
        # Log progress
        if self.pbar.n % self.log_interval == 0 and self.pbar.n != self.last_log_step:
            self.last_log_step = self.pbar.n
    
    def close(self) -> None:
        """Close progress bar."""
        self.pbar.close()
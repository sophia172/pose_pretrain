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

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Logger class for tracking training progress and performance metrics.
    
    Supports console logging, file logging, TensorBoard, and local metric tracking.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        log_freq: int = 100,
        save_freq: int = 1000,
        log_level: int = logging.INFO
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            log_freq: Frequency of logging metrics
            save_freq: Frequency of saving metrics
            log_level: Logging level
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.log_freq = log_freq
        self.save_freq = save_freq
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up console and file logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'experiment.log'))
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Set up TensorBoard
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        else:
            self.writer = None
            
        # Metrics tracking
        self.metrics = {}
        self.metric_history = {}
        self.current_step = 0
        
        self.logger.info(f"Logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Logs will be saved to: {self.log_dir}")
        if self.use_tensorboard:
            self.logger.info(f"TensorBoard logs available at: {os.path.join(self.log_dir, 'tensorboard')}")
        
    def log_metrics(
        self, 
        metrics: Dict[str, Union[float, int, np.ndarray, torch.Tensor]], 
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to console, file, and TensorBoard.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step (default: self.current_step)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.current_step
            
        # Add prefix to metric names if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Update current metrics
        self.metrics.update(metrics)
        
        # Update metric history
        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = []
                
            # Convert to Python scalar if needed
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.item() if hasattr(value, 'item') else float(value)
                
            self.metric_history[name].append((step, value))
        
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) or (isinstance(value, (torch.Tensor, np.ndarray)) and value.size == 1):
                    # Convert to Python scalar if needed
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        value = value.item() if hasattr(value, 'item') else float(value)
                    self.writer.add_scalar(name, value, step)
        
        # Log to console every log_freq steps
        if step % self.log_freq == 0:
            log_str = f"Step {step}: "
            log_str += ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
            self.logger.info(log_str)
            
        # Save metrics every save_freq steps
        if step % self.save_freq == 0:
            self.save_metrics()
            
        # Update current step
        self.current_step = step + 1
    
    def log_histogram(
        self, 
        name: str, 
        values: Union[torch.Tensor, np.ndarray], 
        step: Optional[int] = None,
        bins: str = 'auto'
    ) -> None:
        """
        Log histogram to TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Values to plot in the histogram
            step: Current step (default: self.current_step)
            bins: Number of bins or 'auto'
        """
        if step is None:
            step = self.current_step
            
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_histogram(name, values, step, bins=bins)
            
        # Update current step
        self.current_step = step + 1
    
    def log_image(
        self, 
        name: str, 
        image: Union[torch.Tensor, np.ndarray], 
        step: Optional[int] = None
    ) -> None:
        """
        Log image to TensorBoard.
        
        Args:
            name: Name of the image
            image: Image to log (HxWxC or CxHxW)
            step: Current step (default: self.current_step)
        """
        if step is None:
            step = self.current_step
            
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_image(name, image, step)
            
        # Update current step
        self.current_step = step + 1
    
    def log_figure(
        self, 
        name: str, 
        figure: Figure, 
        step: Optional[int] = None,
        close_figure: bool = True
    ) -> None:
        """
        Log Matplotlib figure to TensorBoard.
        
        Args:
            name: Name of the figure
            figure: Matplotlib figure to log
            step: Current step (default: self.current_step)
            close_figure: Whether to close the figure after logging
        """
        if step is None:
            step = self.current_step
            
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_figure(name, figure, step)
            
        # Save figure to disk
        figure_dir = os.path.join(self.log_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, f"{name}_{step}.png")
        figure.savefig(figure_path)
        
        # Close figure if requested
        if close_figure:
            plt.close(figure)
            
        # Update current step
        self.current_step = step + 1
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_graph(model, input_tensor)
    
    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text to TensorBoard.
        
        Args:
            name: Name of the text
            text: Text to log
            step: Current step (default: self.current_step)
        """
        if step is None:
            step = self.current_step
            
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_text(name, text, step)
            
        # Update current step
        self.current_step = step + 1
    
    def save_metrics(self) -> None:
        """Save metrics to disk."""
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        
        # Convert metric history to serializable format
        serializable_history = {}
        for name, values in self.metric_history.items():
            serializable_history[name] = [[int(step), float(value)] for step, value in values]
            
        with open(metrics_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
            
        self.logger.debug(f"Metrics saved to: {metrics_file}")
    
    def plot_metrics(
        self, 
        metrics: Optional[List[str]] = None, 
        start_step: int = 0, 
        end_step: Optional[int] = None,
        figsize: tuple = (10, 6),
        save: bool = True
    ) -> Figure:
        """
        Plot metrics from history.
        
        Args:
            metrics: List of metrics to plot (default: all metrics)
            start_step: Start step for plotting
            end_step: End step for plotting (default: latest step)
            figsize: Figure size
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = list(self.metric_history.keys())
            
        if not metrics:
            self.logger.warning("No metrics available for plotting")
            return
            
        if end_step is None:
            end_step = self.current_step
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each metric
        for name in metrics:
            if name not in self.metric_history:
                self.logger.warning(f"Metric '{name}' not found in history")
                continue
                
            steps, values = zip(*self.metric_history[name])
            steps = np.array(steps)
            values = np.array(values)
            
            # Filter by step range
            mask = (steps >= start_step) & (steps <= end_step)
            steps = steps[mask]
            values = values[mask]
            
            ax.plot(steps, values, label=name)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Training Metrics')
        ax.grid(True)
        ax.legend()
        
        # Save figure if requested
        if save:
            figures_dir = os.path.join(self.log_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            
            timestamp = int(time.time())
            figure_path = os.path.join(figures_dir, f"metrics_{timestamp}.png")
            fig.savefig(figure_path)
            self.logger.info(f"Metrics plot saved to: {figure_path}")
        
        return fig
    
    def close(self) -> None:
        """Close logger."""
        # Save final metrics
        self.save_metrics()
        
        # Close TensorBoard writer
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
            
        self.logger.info("Logger closed")


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


def get_logger(
    log_dir: str,
    experiment_name: str,
    use_tensorboard: bool = True,
    log_freq: int = 100,
    save_freq: int = 1000,
    log_level: int = logging.INFO
) -> Logger:
    """
    Create logger with specified configuration.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard
        log_freq: Frequency of logging metrics
        save_freq: Frequency of saving metrics
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        log_freq=log_freq,
        save_freq=save_freq,
        log_level=log_level
    ) 
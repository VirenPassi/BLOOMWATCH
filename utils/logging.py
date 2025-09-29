"""
Logging utilities for BloomWatch project.

This module provides comprehensive logging setup with different handlers,
formatters, and integration with experiment tracking.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    
    Adds color codes to log messages based on their level
    for better readability in terminal output.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON for easier parsing
    and integration with log analysis tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def setup_logging(
    name: str = 'bloomwatch',
    level: str = 'INFO',
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging for BloomWatch.
    
    Args:
        name: Logger name
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_dir: Directory for log files (optional)
        console_output: Whether to output to console
        file_output: Whether to output to file
        json_format: Whether to use JSON format for file output
        max_file_size: Maximum size for log files before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Use colored formatter for console
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        log_file = log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Choose formatter
        if json_format:
            file_formatter = JsonFormatter()
        else:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            file_formatter = logging.Formatter(file_format)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log
        error_log_file = log_dir / f"{name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    logger.info(f"Logging setup completed for {name}")
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (if None, uses calling module name)
        
    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'bloomwatch')
    
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Logger specifically for experiment tracking.
    
    Provides structured logging for experiments with
    metrics tracking and experiment lifecycle events.
    """
    
    def __init__(self, experiment_name: str, log_dir: str):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment-specific logger
        self.logger = setup_logging(
            name=f"experiment_{experiment_name}",
            log_dir=str(self.log_dir),
            json_format=True
        )
        
        # Metrics log
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        
        # Experiment start
        self.log_experiment_start()
    
    def log_experiment_start(self):
        """Log experiment start."""
        self.logger.info(
            "Experiment started",
            extra={
                'event_type': 'experiment_start',
                'experiment_name': self.experiment_name
            }
        )
    
    def log_experiment_end(self, status: str = 'completed'):
        """
        Log experiment end.
        
        Args:
            status: Experiment status ('completed', 'failed', 'cancelled')
        """
        self.logger.info(
            f"Experiment {status}",
            extra={
                'event_type': 'experiment_end',
                'experiment_name': self.experiment_name,
                'status': status
            }
        )
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.logger.info(
            f"Epoch {epoch}/{total_epochs} started",
            extra={
                'event_type': 'epoch_start',
                'epoch': epoch,
                'total_epochs': total_epochs
            }
        )
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """
        Log epoch end with metrics.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        self.logger.info(
            f"Epoch {epoch} completed",
            extra={
                'event_type': 'epoch_end',
                'epoch': epoch,
                **metrics
            }
        )
        
        # Also log to metrics file
        self.log_metrics(epoch, metrics)
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics to separate file.
        
        Args:
            step: Training step/epoch
            metrics: Dictionary of metrics
        """
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
    
    def log_model_checkpoint(self, epoch: int, checkpoint_path: str, metrics: Dict[str, float]):
        """
        Log model checkpoint.
        
        Args:
            epoch: Epoch number
            checkpoint_path: Path to saved checkpoint
            metrics: Current metrics
        """
        self.logger.info(
            f"Model checkpoint saved at epoch {epoch}",
            extra={
                'event_type': 'model_checkpoint',
                'epoch': epoch,
                'checkpoint_path': checkpoint_path,
                **metrics
            }
        )
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.logger.info(
            "Hyperparameters logged",
            extra={
                'event_type': 'hyperparameters',
                **hyperparams
            }
        )


class PerformanceLogger:
    """
    Logger for performance monitoring.
    
    Tracks system metrics, timing information,
    and resource usage during training.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """
        Log operation timing.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional context
        """
        self.logger.info(
            f"{operation} completed in {duration:.3f}s",
            extra={
                'event_type': 'timing',
                'operation': operation,
                'duration_seconds': duration,
                **kwargs
            }
        )
    
    def log_memory_usage(self, context: str = 'general'):
        """
        Log current memory usage.
        
        Args:
            context: Context for memory measurement
        """
        try:
            import psutil
            import torch
            
            # System memory
            memory = psutil.virtual_memory()
            
            # GPU memory if available
            gpu_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory[f'gpu_{i}_allocated'] = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    gpu_memory[f'gpu_{i}_cached'] = torch.cuda.memory_reserved(i) / 1024**3  # GB
            
            self.logger.info(
                f"Memory usage - {context}",
                extra={
                    'event_type': 'memory_usage',
                    'context': context,
                    'ram_used_gb': memory.used / 1024**3,
                    'ram_available_gb': memory.available / 1024**3,
                    'ram_percent': memory.percent,
                    **gpu_memory
                }
            )
        
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
    
    def log_model_size(self, model, model_name: str = 'model'):
        """
        Log model size information.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
        """
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate model size in MB
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024**2
            
            self.logger.info(
                f"Model size - {model_name}",
                extra={
                    'event_type': 'model_size',
                    'model_name': model_name,
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': model_size_mb
                }
            )
        
        except Exception as e:
            self.logger.error(f"Error logging model size: {e}")


# Context manager for timing operations
class LoggedTimer:
    """
    Context manager for timing operations with automatic logging.
    """
    
    def __init__(self, logger: logging.Logger, operation_name: str, **kwargs):
        """
        Initialize timer.
        
        Args:
            logger: Logger instance
            operation_name: Name of the operation being timed
            **kwargs: Additional context to log
        """
        self.logger = logger
        self.operation_name = operation_name
        self.context = kwargs
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.info(
                    f"{self.operation_name} completed in {duration:.3f}s",
                    extra={
                        'event_type': 'timing',
                        'operation': self.operation_name,
                        'duration_seconds': duration,
                        'status': 'success',
                        **self.context
                    }
                )
            else:
                self.logger.error(
                    f"{self.operation_name} failed after {duration:.3f}s",
                    extra={
                        'event_type': 'timing',
                        'operation': self.operation_name,
                        'duration_seconds': duration,
                        'status': 'error',
                        'error_type': exc_type.__name__,
                        **self.context
                    }
                )


# Convenience function for timing with logging
def log_time(logger: logging.Logger, operation_name: str, **kwargs):
    """
    Decorator/context manager factory for timing operations.
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation
        **kwargs: Additional context
        
    Returns:
        LoggedTimer: Context manager for timing
    """
    return LoggedTimer(logger, operation_name, **kwargs)
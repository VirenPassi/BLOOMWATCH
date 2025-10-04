"""
Helper utilities for BloomWatch project.

This module provides various utility functions for common tasks
like directory management, device detection, random seed setting, etc.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, List, Any
import subprocess
import platform
import psutil
import json
from datetime import datetime

def ensure_dir(directory: Union[str, Path]) -> Path:
 """
 Ensure directory exists, create if it doesn't.
 
 Args:
 directory: Directory path
 
 Returns:
 Path object for the directory
 """
 dir_path = Path(directory)
 dir_path.mkdir(parents=True, exist_ok=True)
 return dir_path

def get_device(
 preferred_device: Optional[str] = None,
 verbose: bool = True
) -> torch.device:
 """
 Get the best available device for PyTorch.
 
 Args:
 preferred_device: Preferred device ('cpu', 'cuda', 'mps', 'auto')
 verbose: Whether to print device information
 
 Returns:
 torch.device: Selected device
 """
 if preferred_device and preferred_device != 'auto':
 device = torch.device(preferred_device)
 if verbose:
 print(f"Using preferred device: {device}")
 return device
 
 # Auto-detect best device
 if torch.cuda.is_available():
 device = torch.device('cuda')
 if verbose:
 gpu_count = torch.cuda.device_count()
 gpu_name = torch.cuda.get_device_name()
 print(f"Using CUDA device: {gpu_name} ({gpu_count} GPU{'s' if gpu_count > 1 else ''})")
 elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
 device = torch.device('mps')
 if verbose:
 print("Using Apple Metal Performance Shaders (MPS)")
 else:
 device = torch.device('cpu')
 if verbose:
 print("Using CPU")
 
 return device

def set_seed(seed: int = 42, deterministic: bool = False):
 """
 Set random seeds for reproducibility.
 
 Args:
 seed: Random seed value
 deterministic: Whether to use deterministic algorithms (slower but reproducible)
 """
 random.seed(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 
 if torch.cuda.is_available():
 torch.cuda.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 
 if deterministic:
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False
 os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_system_info() -> dict:
 """
 Get comprehensive system information.
 
 Returns:
 Dictionary with system information
 """
 info = {
 'platform': platform.platform(),
 'python_version': platform.python_version(),
 'cpu_count': psutil.cpu_count(),
 'memory_gb': psutil.virtual_memory().total / (1024**3),
 'pytorch_version': torch.__version__
 }
 
 # GPU information
 if torch.cuda.is_available():
 info['cuda_available'] = True
 info['cuda_version'] = torch.version.cuda
 info['gpu_count'] = torch.cuda.device_count()
 info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
 info['gpu_memory_gb'] = [
 torch.cuda.get_device_properties(i).total_memory / (1024**3)
 for i in range(torch.cuda.device_count())
 ]
 else:
 info['cuda_available'] = False
 
 # MPS information (Apple Silicon)
 if hasattr(torch.backends, 'mps'):
 info['mps_available'] = torch.backends.mps.is_available()
 
 return info

def format_time(seconds: float) -> str:
 """
 Format time duration in human-readable format.
 
 Args:
 seconds: Time duration in seconds
 
 Returns:
 Formatted time string
 """
 if seconds < 60:
 return f"{seconds:.1f}s"
 elif seconds < 3600:
 minutes = seconds / 60
 return f"{minutes:.1f}m"
 else:
 hours = seconds / 3600
 return f"{hours:.1f}h"

def format_bytes(bytes_count: int) -> str:
 """
 Format byte count in human-readable format.
 
 Args:
 bytes_count: Number of bytes
 
 Returns:
 Formatted size string
 """
 for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
 if bytes_count < 1024.0:
 return f"{bytes_count:.1f}{unit}"
 bytes_count /= 1024.0
 return f"{bytes_count:.1f}PB"

def count_files(directory: Union[str, Path], extensions: Optional[List[str]] = None) -> dict:
 """
 Count files in directory by extension.
 
 Args:
 directory: Directory to scan
 extensions: List of extensions to count (e.g., ['.jpg', '.png'])
 
 Returns:
 Dictionary with file counts by extension
 """
 directory = Path(directory)
 
 if not directory.exists():
 return {}
 
 file_counts = {}
 total_count = 0
 
 for file_path in directory.rglob('*'):
 if file_path.is_file():
 ext = file_path.suffix.lower()
 
 if extensions is None or ext in extensions:
 file_counts[ext] = file_counts.get(ext, 0) + 1
 total_count += 1
 
 file_counts['total'] = total_count
 return file_counts

def get_git_info() -> dict:
 """
 Get git repository information if available.
 
 Returns:
 Dictionary with git information
 """
 git_info = {}
 
 try:
 # Get current commit hash
 result = subprocess.run(
 ['git', 'rev-parse', 'HEAD'],
 capture_output=True,
 text=True,
 check=True
 )
 git_info['commit_hash'] = result.stdout.strip()
 
 # Get branch name
 result = subprocess.run(
 ['git', 'branch', '--show-current'],
 capture_output=True,
 text=True,
 check=True
 )
 git_info['branch'] = result.stdout.strip()
 
 # Get commit message
 result = subprocess.run(
 ['git', 'log', '-1', '--pretty=%B'],
 capture_output=True,
 text=True,
 check=True
 )
 git_info['commit_message'] = result.stdout.strip()
 
 # Check for uncommitted changes
 result = subprocess.run(
 ['git', 'status', '--porcelain'],
 capture_output=True,
 text=True,
 check=True
 )
 git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
 
 except (subprocess.CalledProcessError, FileNotFoundError):
 git_info['error'] = 'Git not available or not a git repository'
 
 return git_info

def save_experiment_metadata(
 save_path: Union[str, Path],
 config: dict,
 model_info: Optional[dict] = None,
 additional_metadata: Optional[dict] = None
):
 """
 Save comprehensive experiment metadata.
 
 Args:
 save_path: Path to save metadata JSON file
 config: Experiment configuration
 model_info: Model information (parameters, size, etc.)
 additional_metadata: Additional metadata to include
 """
 metadata = {
 'timestamp': datetime.now().isoformat(),
 'system_info': get_system_info(),
 'git_info': get_git_info(),
 'config': config
 }
 
 if model_info:
 metadata['model_info'] = model_info
 
 if additional_metadata:
 metadata.update(additional_metadata)
 
 save_path = Path(save_path)
 save_path.parent.mkdir(parents=True, exist_ok=True)
 
 with open(save_path, 'w') as f:
 json.dump(metadata, f, indent=2, default=str)

def load_experiment_metadata(load_path: Union[str, Path]) -> dict:
 """
 Load experiment metadata from JSON file.
 
 Args:
 load_path: Path to metadata file
 
 Returns:
 Dictionary with experiment metadata
 """
 with open(load_path, 'r') as f:
 return json.load(f)

class ProgressTracker:
 """
 Simple progress tracker for long-running operations.
 """
 
 def __init__(self, total_steps: int, description: str = "Progress"):
 """
 Initialize progress tracker.
 
 Args:
 total_steps: Total number of steps
 description: Description of the operation
 """
 self.total_steps = total_steps
 self.current_step = 0
 self.description = description
 self.start_time = datetime.now()
 
 def update(self, steps: int = 1):
 """Update progress by specified number of steps."""
 self.current_step += steps
 self._print_progress()
 
 def set_progress(self, step: int):
 """Set current progress to specific step."""
 self.current_step = step
 self._print_progress()
 
 def _print_progress(self):
 """Print progress bar."""
 if self.total_steps == 0:
 return
 
 percentage = (self.current_step / self.total_steps) * 100
 elapsed_time = (datetime.now() - self.start_time).total_seconds()
 
 # Estimate remaining time
 if self.current_step > 0:
 estimated_total_time = elapsed_time * (self.total_steps / self.current_step)
 remaining_time = estimated_total_time - elapsed_time
 remaining_str = format_time(remaining_time)
 else:
 remaining_str = "??"
 
 # Create progress bar
 bar_length = 30
 filled_length = int(bar_length * self.current_step / self.total_steps)
 bar = '' * filled_length + '-' * (bar_length - filled_length)
 
 print(f'\r{self.description}: |{bar}| {percentage:.1f}% ({self.current_step}/{self.total_steps}) ETA: {remaining_str}', end='')
 
 if self.current_step >= self.total_steps:
 print() # New line when complete

def check_dependencies() -> dict:
 """
 Check if all required dependencies are installed.
 
 Returns:
 Dictionary with dependency status
 """
 dependencies = {
 'torch': None,
 'torchvision': None,
 'numpy': None,
 'pandas': None,
 'pillow': None,
 'matplotlib': None,
 'plotly': None,
 'fastapi': None,
 'boto3': None,
 'omegaconf': None,
 'pytest': None
 }
 
 for dep in dependencies:
 try:
 __import__(dep)
 dependencies[dep] = 'installed'
 except ImportError:
 dependencies[dep] = 'missing'
 
 return dependencies

def validate_image_file(file_path: Union[str, Path]) -> dict:
 """
 Validate image file format and properties.
 
 Args:
 file_path: Path to image file
 
 Returns:
 Dictionary with validation results
 """
 from PIL import Image
 
 file_path = Path(file_path)
 result = {
 'valid': False,
 'error': None,
 'format': None,
 'size': None,
 'mode': None
 }
 
 try:
 if not file_path.exists():
 result['error'] = 'File does not exist'
 return result
 
 with Image.open(file_path) as img:
 result['valid'] = True
 result['format'] = img.format
 result['size'] = img.size
 result['mode'] = img.mode
 
 except Exception as e:
 result['error'] = str(e)
 
 return result

def create_directory_structure(base_dir: Union[str, Path], structure: dict):
 """
 Create directory structure from nested dictionary.
 
 Args:
 base_dir: Base directory path
 structure: Nested dictionary representing directory structure
 """
 base_dir = Path(base_dir)
 
 def _create_recursive(current_path: Path, struct: dict):
 for name, content in struct.items():
 new_path = current_path / name
 
 if isinstance(content, dict):
 new_path.mkdir(parents=True, exist_ok=True)
 _create_recursive(new_path, content)
 else:
 # Create file with content
 new_path.parent.mkdir(parents=True, exist_ok=True)
 if content is not None:
 with open(new_path, 'w') as f:
 f.write(str(content))
 
 _create_recursive(base_dir, structure)

def cleanup_experiment_files(
 experiment_dir: Union[str, Path],
 keep_checkpoints: bool = True,
 keep_logs: bool = True,
 keep_plots: bool = True
):
 """
 Clean up experiment files based on specified criteria.
 
 Args:
 experiment_dir: Experiment directory
 keep_checkpoints: Whether to keep model checkpoints
 keep_logs: Whether to keep log files
 keep_plots: Whether to keep plot files
 """
 experiment_dir = Path(experiment_dir)
 
 if not experiment_dir.exists():
 return
 
 for file_path in experiment_dir.rglob('*'):
 if file_path.is_file():
 should_delete = False
 
 # Check file types to delete
 if not keep_checkpoints and file_path.suffix in ['.pth', '.pt', '.ckpt']:
 should_delete = True
 elif not keep_logs and file_path.suffix in ['.log', '.txt']:
 should_delete = True
 elif not keep_plots and file_path.suffix in ['.png', '.jpg', '.pdf', '.svg']:
 should_delete = True
 
 if should_delete:
 try:
 file_path.unlink()
 print(f"Deleted: {file_path}")
 except Exception as e:
 print(f"Failed to delete {file_path}: {e}")

def estimate_training_time(
 dataset_size: int,
 batch_size: int,
 epochs: int,
 seconds_per_batch: float = 1.0
) -> dict:
 """
 Estimate training time based on dataset and hardware characteristics.
 
 Args:
 dataset_size: Number of training samples
 batch_size: Batch size
 epochs: Number of training epochs
 seconds_per_batch: Estimated time per batch
 
 Returns:
 Dictionary with time estimates
 """
 batches_per_epoch = dataset_size / batch_size
 total_batches = batches_per_epoch * epochs
 total_seconds = total_batches * seconds_per_batch
 
 return {
 'batches_per_epoch': batches_per_epoch,
 'total_batches': total_batches,
 'estimated_time_seconds': total_seconds,
 'estimated_time_formatted': format_time(total_seconds),
 'time_per_epoch_seconds': batches_per_epoch * seconds_per_batch,
 'time_per_epoch_formatted': format_time(batches_per_epoch * seconds_per_batch)
 }

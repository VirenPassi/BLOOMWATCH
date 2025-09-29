"""
Configuration management for the FastAPI application.

This module handles loading and managing configuration settings
for the BloomWatch API application.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path


class AppConfig:
    """
    Application configuration class.
    
    Manages configuration settings loaded from environment variables
    and configuration files.
    """
    
    def __init__(self):
        """Initialize configuration with default values and load from environment."""
        # Model settings
        self.default_model_path: Optional[str] = os.getenv('BLOOMWATCH_DEFAULT_MODEL_PATH')
        self.models_dir: str = os.getenv('BLOOMWATCH_MODELS_DIR', './models')
        self.max_models: int = int(os.getenv('BLOOMWATCH_MAX_MODELS', '5'))
        
        # API settings
        self.host: str = os.getenv('BLOOMWATCH_HOST', '0.0.0.0')
        self.port: int = int(os.getenv('BLOOMWATCH_PORT', '8000'))
        self.reload: bool = os.getenv('BLOOMWATCH_RELOAD', 'false').lower() == 'true'
        self.workers: int = int(os.getenv('BLOOMWATCH_WORKERS', '1'))
        
        # Image processing settings
        self.max_batch_size: int = int(os.getenv('BLOOMWATCH_MAX_BATCH_SIZE', '50'))
        self.max_image_size: int = int(os.getenv('BLOOMWATCH_MAX_IMAGE_SIZE', str(10 * 1024 * 1024)))  # 10MB
        self.allowed_image_types: List[str] = [
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/bmp', 'image/tiff', 'image/webp'
        ]
        self.default_image_size: tuple = (224, 224)
        
        # Performance settings
        self.model_timeout: int = int(os.getenv('BLOOMWATCH_MODEL_TIMEOUT', '30'))  # seconds
        self.enable_caching: bool = os.getenv('BLOOMWATCH_ENABLE_CACHING', 'true').lower() == 'true'
        self.cache_ttl: int = int(os.getenv('BLOOMWATCH_CACHE_TTL', '3600'))  # 1 hour
        self.max_concurrent_requests: int = int(os.getenv('BLOOMWATCH_MAX_CONCURRENT', '100'))
        
        # Security settings
        self.api_key_required: bool = os.getenv('BLOOMWATCH_API_KEY_REQUIRED', 'false').lower() == 'true'
        self.api_keys: List[str] = self._load_api_keys()
        self.rate_limit_enabled: bool = os.getenv('BLOOMWATCH_RATE_LIMIT', 'false').lower() == 'true'
        self.rate_limit_requests: int = int(os.getenv('BLOOMWATCH_RATE_LIMIT_REQUESTS', '100'))
        self.rate_limit_window: int = int(os.getenv('BLOOMWATCH_RATE_LIMIT_WINDOW', '3600'))  # 1 hour
        
        # Logging settings
        self.log_level: str = os.getenv('BLOOMWATCH_LOG_LEVEL', 'INFO').upper()
        self.log_dir: Optional[str] = os.getenv('BLOOMWATCH_LOG_DIR')
        self.enable_access_logs: bool = os.getenv('BLOOMWATCH_ACCESS_LOGS', 'true').lower() == 'true'
        self.enable_error_logs: bool = os.getenv('BLOOMWATCH_ERROR_LOGS', 'true').lower() == 'true'
        
        # Monitoring settings
        self.enable_metrics: bool = os.getenv('BLOOMWATCH_ENABLE_METRICS', 'true').lower() == 'true'
        self.metrics_endpoint: str = os.getenv('BLOOMWATCH_METRICS_ENDPOINT', '/metrics')
        self.health_check_interval: int = int(os.getenv('BLOOMWATCH_HEALTH_CHECK_INTERVAL', '60'))
        
        # CORS settings
        self.cors_origins: List[str] = self._load_cors_origins()
        self.cors_methods: List[str] = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.cors_headers: List[str] = ['*']
        
        # Development settings
        self.debug: bool = os.getenv('BLOOMWATCH_DEBUG', 'false').lower() == 'true'
        self.development_mode: bool = os.getenv('BLOOMWATCH_DEVELOPMENT', 'false').lower() == 'true'
        
        # GPU/Device settings
        self.force_device: Optional[str] = os.getenv('BLOOMWATCH_DEVICE')  # 'cpu', 'cuda', 'mps'
        self.gpu_memory_fraction: float = float(os.getenv('BLOOMWATCH_GPU_MEMORY_FRACTION', '0.8'))
        
        # Validate configuration
        self._validate_config()
    
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variable."""
        api_keys_str = os.getenv('BLOOMWATCH_API_KEYS', '')
        if api_keys_str:
            return [key.strip() for key in api_keys_str.split(',') if key.strip()]
        return []
    
    def _load_cors_origins(self) -> List[str]:
        """Load CORS origins from environment variable."""
        origins_str = os.getenv('BLOOMWATCH_CORS_ORIGINS', '*')
        if origins_str == '*':
            return ['*']
        return [origin.strip() for origin in origins_str.split(',') if origin.strip()]
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate port range
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Invalid port number: {self.port}")
        
        # Validate batch size
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be positive: {self.max_batch_size}")
        
        # Validate image size
        if self.max_image_size < 1024:  # Minimum 1KB
            raise ValueError(f"max_image_size too small: {self.max_image_size}")
        
        # Validate timeout
        if self.model_timeout < 1:
            raise ValueError(f"model_timeout must be positive: {self.model_timeout}")
        
        # Validate GPU memory fraction
        if not (0.0 < self.gpu_memory_fraction <= 1.0):
            raise ValueError(f"gpu_memory_fraction must be between 0 and 1: {self.gpu_memory_fraction}")
        
        # Create directories if they don't exist
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get the full path for a model file.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Full path to the model file
        """
        return str(Path(self.models_dir) / f"{model_name}.pth")
    
    def is_valid_api_key(self, api_key: str) -> bool:
        """
        Check if an API key is valid.
        
        Args:
            api_key: API key to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not self.api_key_required:
            return True
        
        return api_key in self.api_keys
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict with configuration values
        """
        return {
            'default_model_path': self.default_model_path,
            'models_dir': self.models_dir,
            'max_models': self.max_models,
            'host': self.host,
            'port': self.port,
            'reload': self.reload,
            'workers': self.workers,
            'max_batch_size': self.max_batch_size,
            'max_image_size': self.max_image_size,
            'allowed_image_types': self.allowed_image_types,
            'default_image_size': self.default_image_size,
            'model_timeout': self.model_timeout,
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl,
            'max_concurrent_requests': self.max_concurrent_requests,
            'api_key_required': self.api_key_required,
            'rate_limit_enabled': self.rate_limit_enabled,
            'rate_limit_requests': self.rate_limit_requests,
            'rate_limit_window': self.rate_limit_window,
            'log_level': self.log_level,
            'log_dir': self.log_dir,
            'enable_access_logs': self.enable_access_logs,
            'enable_error_logs': self.enable_error_logs,
            'enable_metrics': self.enable_metrics,
            'metrics_endpoint': self.metrics_endpoint,
            'health_check_interval': self.health_check_interval,
            'cors_origins': self.cors_origins,
            'cors_methods': self.cors_methods,
            'cors_headers': self.cors_headers,
            'debug': self.debug,
            'development_mode': self.development_mode,
            'force_device': self.force_device,
            'gpu_memory_fraction': self.gpu_memory_fraction
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        # Hide sensitive information
        if 'api_keys' in config_dict:
            config_dict['api_keys'] = ['***' for _ in self.api_keys]
        
        return f"AppConfig({config_dict})"


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """
    Get the global configuration instance.
    
    Returns:
        AppConfig: Global configuration object
    """
    return config


def reload_config() -> AppConfig:
    """
    Reload configuration from environment variables.
    
    Returns:
        AppConfig: New configuration object
    """
    global config
    config = AppConfig()
    return config
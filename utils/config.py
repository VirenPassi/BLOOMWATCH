"""
Configuration management for BloomWatch project.

This module handles loading and managing configuration files
using Hydra/OmegaConf for flexible experiment management.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from omegaconf import OmegaConf, DictConfig
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manage configuration files and settings for BloomWatch experiments.
    
    Supports loading from YAML files, environment variables,
    and command-line arguments using OmegaConf.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to main config file
        """
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> DictConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            DictConfig: Loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load base configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.config = OmegaConf.create(config_dict)
        
        # Resolve interpolations and merge with defaults
        self.config = OmegaConf.to_container(self.config, resolve=True)
        self.config = OmegaConf.create(self.config)
        
        logger.info(f"Configuration loaded from: {config_path}")
        return self.config
    
    def merge_config(self, override_config: Union[str, Dict, DictConfig]) -> DictConfig:
        """
        Merge configuration with overrides.
        
        Args:
            override_config: Override config (file path, dict, or DictConfig)
            
        Returns:
            DictConfig: Merged configuration
        """
        if isinstance(override_config, str):
            # Load from file
            with open(override_config, 'r') as f:
                override_dict = yaml.safe_load(f)
            override_config = OmegaConf.create(override_dict)
        elif isinstance(override_config, dict):
            override_config = OmegaConf.create(override_config)
        
        if self.config is None:
            self.config = override_config
        else:
            self.config = OmegaConf.merge(self.config, override_config)
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key: Dot-separated key path (e.g., 'model.num_classes')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self.config is None:
            return default
        
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key path.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        if self.config is None:
            self.config = OmegaConf.create({})
        
        OmegaConf.set(self.config, key, value)
    
    def save_config(self, save_path: str):
        """
        Save current configuration to file.
        
        Args:
            save_path: Path to save config file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            OmegaConf.save(config=self.config, f=f)
        
        logger.info(f"Configuration saved to: {save_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if self.config is None:
            return {}
        return OmegaConf.to_container(self.config, resolve=True)
    
    def pretty_print(self) -> str:
        """Get pretty-printed configuration string."""
        if self.config is None:
            return "No configuration loaded"
        return OmegaConf.to_yaml(self.config)
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration and return validation report.
        
        Returns:
            Dict with validation results
        """
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if self.config is None:
            validation_report['valid'] = False
            validation_report['errors'].append("No configuration loaded")
            return validation_report
        
        # Required fields validation
        required_fields = [
            'project.name',
            'data.batch_size',
            'model.num_classes',
            'training.epochs',
            'training.learning_rate'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                validation_report['valid'] = False
                validation_report['errors'].append(f"Missing required field: {field}")
        
        # Value validation
        if self.get('data.batch_size', 0) <= 0:
            validation_report['valid'] = False
            validation_report['errors'].append("data.batch_size must be positive")
        
        if self.get('model.num_classes', 0) <= 0:
            validation_report['valid'] = False
            validation_report['errors'].append("model.num_classes must be positive")
        
        if self.get('training.epochs', 0) <= 0:
            validation_report['valid'] = False
            validation_report['errors'].append("training.epochs must be positive")
        
        if self.get('training.learning_rate', 0) <= 0:
            validation_report['valid'] = False
            validation_report['errors'].append("training.learning_rate must be positive")
        
        # Warnings
        if self.get('data.batch_size', 32) > 128:
            validation_report['warnings'].append("Large batch size may cause memory issues")
        
        if self.get('training.learning_rate', 0.001) > 0.1:
            validation_report['warnings'].append("High learning rate may cause training instability")
        
        return validation_report


def load_config(config_path: str, override_config: Optional[Union[str, Dict]] = None) -> DictConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to main config file
        override_config: Optional override configuration
        
    Returns:
        DictConfig: Loaded and merged configuration
    """
    config_manager = ConfigManager(config_path)
    
    if override_config:
        config_manager.merge_config(override_config)
    
    return config_manager.config


def create_experiment_config(
    base_config_path: str,
    experiment_name: str,
    overrides: Dict[str, Any],
    save_dir: str
) -> str:
    """
    Create experiment-specific configuration.
    
    Args:
        base_config_path: Path to base configuration
        experiment_name: Name of the experiment
        overrides: Configuration overrides for the experiment
        save_dir: Directory to save experiment config
        
    Returns:
        str: Path to saved experiment config
    """
    config_manager = ConfigManager(base_config_path)
    
    # Add experiment metadata
    experiment_metadata = {
        'experiment': {
            'name': experiment_name,
            'base_config': str(base_config_path),
            'overrides': overrides
        }
    }
    
    config_manager.merge_config(experiment_metadata)
    config_manager.merge_config(overrides)
    
    # Save experiment config
    save_path = Path(save_dir) / f"{experiment_name}_config.yaml"
    config_manager.save_config(str(save_path))
    
    return str(save_path)


class ExperimentTracker:
    """
    Track experiments and their configurations.
    
    Maintains a registry of experiments with their configs,
    results, and metadata.
    """
    
    def __init__(self, experiments_dir: str = "./experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_path = self.experiments_dir / "registry.yaml"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _save_registry(self):
        """Save experiment registry."""
        with open(self.registry_path, 'w') as f:
            yaml.dump(self.registry, f, default_flow_style=False)
    
    def register_experiment(
        self,
        experiment_name: str,
        config: DictConfig,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            metadata: Optional metadata
        """
        experiment_dir = self.experiments_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(config=config, f=f)
        
        # Update registry
        self.registry[experiment_name] = {
            'config_path': str(config_path),
            'experiment_dir': str(experiment_dir),
            'metadata': metadata or {},
            'status': 'registered'
        }
        
        self._save_registry()
        logger.info(f"Experiment registered: {experiment_name}")
    
    def get_experiment(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get experiment information."""
        return self.registry.get(experiment_name)
    
    def list_experiments(self) -> List[str]:
        """List all registered experiments."""
        return list(self.registry.keys())
    
    def update_experiment_status(self, experiment_name: str, status: str, results: Optional[Dict] = None):
        """
        Update experiment status and results.
        
        Args:
            experiment_name: Name of the experiment
            status: New status ('running', 'completed', 'failed')
            results: Optional results dictionary
        """
        if experiment_name in self.registry:
            self.registry[experiment_name]['status'] = status
            if results:
                self.registry[experiment_name]['results'] = results
            self._save_registry()


# Environment variable configuration
def load_config_from_env() -> Dict[str, Any]:
    """
    Load configuration overrides from environment variables.
    
    Environment variables should be prefixed with 'BLOOMWATCH_'
    and use underscores to separate nested keys.
    
    Example: BLOOMWATCH_MODEL_NUM_CLASSES=5
    
    Returns:
        Dict with configuration overrides
    """
    env_config = {}
    
    for key, value in os.environ.items():
        if key.startswith('BLOOMWATCH_'):
            # Remove prefix and convert to lowercase
            config_key = key[11:].lower()
            
            # Convert underscore notation to nested dict
            keys = config_key.split('_')
            current_dict = env_config
            
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            
            # Try to convert value to appropriate type
            try:
                if value.lower() in ['true', 'false']:
                    current_dict[keys[-1]] = value.lower() == 'true'
                elif value.isdigit():
                    current_dict[keys[-1]] = int(value)
                elif '.' in value and all(part.isdigit() for part in value.split('.')):
                    current_dict[keys[-1]] = float(value)
                else:
                    current_dict[keys[-1]] = value
            except (ValueError, AttributeError):
                current_dict[keys[-1]] = value
    
    return env_config
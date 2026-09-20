"""
Configuration Loader Utility
Loads and manages configuration from YAML file
"""

import yaml
import os
from pathlib import Path


class Config:
    """Configuration manager for ISL Recognition System"""
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
    
    def load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            'data/ISL_dataset',
            'data/processed',
            'data/landmarks',
            'data/cache',
            'models/saved',
            'models/checkpoints',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, *keys, default=None):
        """Get configuration value using dot notation"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]


# Global config instance
_config = None


def get_config(config_path='config.yaml'):
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

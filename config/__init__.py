"""
Configuration Module
Loads and provides access to application configuration
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration loader and accessor"""
    
    _config: Dict[str, Any] = None
    _config_path = Path(__file__).parent / "config.yaml"
    
    @classmethod
    def load(cls) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if cls._config is None:
            if not cls._config_path.exists():
                raise FileNotFoundError(f"Config file not found: {cls._config_path}")
            
            with open(cls._config_path, 'r') as f:
                cls._config = yaml.safe_load(f)
        
        return cls._config
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key
        """
        if cls._config is None:
            cls.load()
        
        keys = key.split('.')
        value = cls._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @classmethod
    def get_paths(cls) -> Dict[str, Any]:
        """Get all path configurations"""
        return cls.get('paths', {})
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration"""
        return cls.get('model', {})
    
    @classmethod
    def get_nlp_config(cls) -> Dict[str, Any]:
        """Get NLP configuration"""
        return cls.get('nlp', {})
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration"""
        return cls.get('api', {})
    
    @classmethod
    def reload(cls):
        """Reload configuration from file"""
        cls._config = None
        cls.load()


# Convenience function
def get_config(key: str = None, default: Any = None) -> Any:
    """
    Get configuration value
    """
    if key is None:
        return Config.load()
    return Config.get(key, default)


# Auto-load config on import
Config.load()
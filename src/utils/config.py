import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import torch

class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path (str, optional): Path to config YAML file. If None, uses default.
        """
        # Default configuration
        self._config = {
            'data': {
                'root_dir': 'data/',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'input_size': [512, 512],
                'volume_depth': 32
            },
            'model': {
                'detection': {
                    'model_name': 'yolov8n.pt',
                    'confidence_threshold': 0.25,
                    'iou_threshold': 0.45
                },
                'segmentation': {
                    'model_name': 'unetr',
                    'in_channels': 1,
                    'out_channels': 2,
                    'img_size': [96, 96, 96],
                    'feature_size': 16,
                    'hidden_size': 768,
                    'mlp_dim': 3072,
                    'num_heads': 12,
                    'pos_embed': 'perceptron',
                    'norm_name': 'instance',
                    'res_block': True,
                    'dropout_rate': 0.0
                },
                'classification': {
                    'model_name': 'resnet50',
                    'num_classes': 3,
                    'pretrained': True
                }
            },
            'training': {
                'batch_size': 8,
                'num_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'patience': 15,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'detection_loss': 'yolo_loss',
                'segmentation_loss': 'dice_ce',
                'classification_loss': 'cross_entropy',
                'optimizer': 'adamw',
                'scheduler': 'reduce_on_plateau',
                'scheduler_params': {
                    'mode': 'max',
                    'factor': 0.5,
                    'patience': 5,
                    'verbose': True
                }
            },
            'paths': {
                'checkpoints': 'checkpoints/',
                'logs': 'logs/',
                'outputs': 'outputs/'
            },
            'augmentations': {
                'train': [
                    {'name': 'RandomRotate90', 'p': 0.5},
                    {'name': 'HorizontalFlip', 'p': 0.5},
                    {'name': 'VerticalFlip', 'p': 0.5},
                    {'name': 'ShiftScaleRotate', 'shift_limit': 0.1, 'scale_limit': 0.1, 'rotate_limit': 15, 'p': 0.5},
                    {'name': 'RandomBrightnessContrast', 'brightness_limit': 0.1, 'contrast_limit': 0.1, 'p': 0.3},
                    {'name': 'GaussianBlur', 'blur_limit': 3, 'p': 0.2}
                ],
                'val_test': [
                    {'name': 'Resize', 'height': 512, 'width': 512, 'always_apply': True}
                ]
            },
            'evaluation': {
                'metrics': {
                    'detection': ['map', 'map50', 'map75', 'map_small', 'map_medium', 'map_large'],
                    'segmentation': ['dice', 'iou', 'precision', 'recall', 'specificity'],
                    'classification': ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
                },
                'nodule_detection': {
                    'iou_threshold': 0.5,
                    'max_detections': 100
                },
                'classification_threshold': 0.5
            }
        }
        
        # Load from file if provided
        if config_path is not None:
            self.load(config_path)
    
    def load(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            self._update_dict(self._config, loaded_config)
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def _update_dict(self, original: Dict, new: Dict) -> None:
        """Recursively update dictionary."""
        for key, value in new.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._update_dict(original[key], value)
            else:
                original[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def to_dict(self) -> Dict:
        """Return the configuration as a dictionary."""
        return self._config
    
    def update(self, config_dict: Dict) -> None:
        """Update configuration with values from a dictionary."""
        self._update_dict(self._config, config_dict)
    
    def create_dirs(self) -> None:
        """Create necessary directories specified in the configuration."""
        paths = [
            self['paths.checkpoints'],
            self['paths.logs'],
            self['paths.outputs']
        ]
        
        for path in paths:
            if path:
                os.makedirs(path, exist_ok=True)


# Global configuration instance
cfg = Config()


def load_config(config_path: Optional[str] = None) -> 'Config':
    """Load configuration from file or use default.
    
    Args:
        config_path (str, optional): Path to config YAML file.
        
    Returns:
        Config: Configuration object
    """
    if config_path:
        cfg.load(config_path)
    return cfg

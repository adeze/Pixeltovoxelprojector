"""
Configuration management system with YAML/JSON validation.

This module provides a structured way to manage configurations for
the pixeltovoxelprojector pipeline with validation and type checking.
"""

from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
import torch


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class GridConfig:
    """Configuration for voxel grid parameters."""
    size: List[int] = field(default_factory=lambda: [500, 500, 500])
    voxel_size: float = 6.0
    center: List[float] = field(default_factory=lambda: [0.0, 0.0, 500.0])
    dtype: str = "float32"
    
    def __post_init__(self):
        if len(self.size) != 3:
            raise ConfigValidationError("Grid size must have exactly 3 dimensions")
        if self.voxel_size <= 0:
            raise ConfigValidationError("Voxel size must be positive")
        if len(self.center) != 3:
            raise ConfigValidationError("Grid center must have exactly 3 coordinates")


@dataclass
class MotionDetectionConfig:
    """Configuration for motion detection parameters."""
    algorithm: str = "frame_difference"
    threshold: float = 2.0
    noise_std: float = 1.0
    blur_kernel: int = 3
    enhance_motion: bool = True
    
    def __post_init__(self):
        if self.threshold <= 0:
            raise ConfigValidationError("Motion threshold must be positive")
        if self.noise_std < 0:
            raise ConfigValidationError("Noise standard deviation must be non-negative")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ConfigValidationError("Blur kernel must be odd and positive")


@dataclass
class RayCastingConfig:
    """Configuration for ray casting parameters."""
    algorithm: str = "dda"
    max_steps: int = 20000
    step_size: float = 1.0
    early_termination: bool = True
    
    def __post_init__(self):
        if self.max_steps <= 0:
            raise ConfigValidationError("Max steps must be positive")
        if self.step_size <= 0:
            raise ConfigValidationError("Step size must be positive")


@dataclass
class AccumulationConfig:
    """Configuration for voxel accumulation parameters."""
    algorithm: str = "weighted"
    alpha: float = 0.1  # Distance attenuation factor
    max_value: Optional[float] = None
    normalize: bool = False
    
    def __post_init__(self):
        if self.alpha < 0:
            raise ConfigValidationError("Alpha must be non-negative")
        if self.max_value is not None and self.max_value <= 0:
            raise ConfigValidationError("Max value must be positive if specified")


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline."""
    batch_size: int = 4
    num_workers: int = 0
    sequence_length: int = 2
    stride: int = 1
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ConfigValidationError("Batch size must be positive")
        if self.num_workers < 0:
            raise ConfigValidationError("Number of workers must be non-negative")
        if self.sequence_length < 2:
            raise ConfigValidationError("Sequence length must be at least 2")
        if self.stride <= 0:
            raise ConfigValidationError("Stride must be positive")


@dataclass
class TransformConfig:
    """Configuration for data transforms."""
    enabled_transforms: List[str] = field(default_factory=lambda: ["noise_injection"])
    noise_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    resize_size: Optional[List[int]] = None
    rotation_degrees: float = 15.0
    flip_probability: float = 0.5
    brightness_factor: float = 1.2
    contrast_factor: float = 1.2
    
    def __post_init__(self):
        if len(self.noise_range) != 2:
            raise ConfigValidationError("Noise range must have exactly 2 values")
        if self.noise_range[0] >= self.noise_range[1]:
            raise ConfigValidationError("Noise range min must be less than max")
        if self.flip_probability < 0 or self.flip_probability > 1:
            raise ConfigValidationError("Flip probability must be between 0 and 1")


@dataclass
class IOConfig:
    """Configuration for input/output operations."""
    input_format: str = "image_sequence"  # "image_sequence", "video", "point_cloud"
    output_format: str = "bin"  # "bin", "npy", "hdf5", "json"
    compression: bool = True
    save_intermediate: bool = False
    export_mesh: bool = False
    mesh_threshold: float = 0.5
    mesh_format: str = "obj"  # "obj", "ply"
    
    def __post_init__(self):
        valid_input_formats = ["image_sequence", "video", "point_cloud"]
        if self.input_format not in valid_input_formats:
            raise ConfigValidationError(f"Input format must be one of {valid_input_formats}")
        
        valid_output_formats = ["bin", "npy", "hdf5", "json"]
        if self.output_format not in valid_output_formats:
            raise ConfigValidationError(f"Output format must be one of {valid_output_formats}")
        
        valid_mesh_formats = ["obj", "ply"]
        if self.mesh_format not in valid_mesh_formats:
            raise ConfigValidationError(f"Mesh format must be one of {valid_mesh_formats}")


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    backend: str = "pyvista"  # "pyvista", "matplotlib", "plotly"
    threshold_percentile: float = 99.0
    colormap: str = "viridis"
    show_axes: bool = True
    background_color: str = "white"
    point_size: float = 2.0
    opacity: float = 1.0
    
    def __post_init__(self):
        valid_backends = ["pyvista", "matplotlib", "plotly"]
        if self.backend not in valid_backends:
            raise ConfigValidationError(f"Visualization backend must be one of {valid_backends}")
        
        if self.threshold_percentile < 0 or self.threshold_percentile > 100:
            raise ConfigValidationError("Threshold percentile must be between 0 and 100")
        
        if self.opacity < 0 or self.opacity > 1:
            raise ConfigValidationError("Opacity must be between 0 and 1")


@dataclass
class PipelineConfig:
    """Main configuration class containing all sub-configurations."""
    grid: GridConfig = field(default_factory=GridConfig)
    motion_detection: MotionDetectionConfig = field(default_factory=MotionDetectionConfig)
    ray_casting: RayCastingConfig = field(default_factory=RayCastingConfig)
    accumulation: AccumulationConfig = field(default_factory=AccumulationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    io: IOConfig = field(default_factory=IOConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Pipeline-level settings
    name: str = "pixeltovoxel_pipeline"
    description: str = "Pixel to voxel projection pipeline"
    version: str = "1.0.0"
    debug: bool = False
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        # Individual configs validate themselves in __post_init__
        # Add cross-config validation here if needed
        
        if self.processing.use_gpu and not torch.cuda.is_available():
            if self.debug:
                print("Warning: GPU requested but not available, using CPU")
            self.processing.use_gpu = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if hasattr(value, '__dict__'):  # It's a dataclass
                result[field_info.name] = {
                    f.name: getattr(value, f.name) for f in fields(value)
                }
            else:
                result[field_info.name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        # Extract sub-config data
        field_types = {f.name: f.type for f in fields(cls)}
        
        kwargs = {}
        for key, value in data.items():
            if key in field_types and hasattr(field_types[key], '__dataclass_fields__'):
                # It's a sub-config dataclass
                kwargs[key] = field_types[key](**value)
            else:
                kwargs[key] = value
        
        return cls(**kwargs)


class ConfigManager:
    """Manager class for loading, saving, and validating configurations."""
    
    def __init__(self, default_config: Optional[PipelineConfig] = None):
        self.default_config = default_config or PipelineConfig()
    
    def load_config(self, config_path: Union[str, Path]) -> PipelineConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    data = json.load(f)
            else:
                raise ConfigValidationError(f"Unsupported config format: {config_path.suffix}")
            
            config = PipelineConfig.from_dict(data)
            config.validate()
            return config
            
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration: {e}")
    
    def save_config(self, config: PipelineConfig, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        
        try:
            config.validate()
            data = config.to_dict()
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ConfigValidationError(f"Unsupported config format: {config_path.suffix}")
                
        except Exception as e:
            raise ConfigValidationError(f"Failed to save configuration: {e}")
    
    def create_template(self, template_path: Union[str, Path]) -> None:
        """Create a template configuration file."""
        template_config = PipelineConfig()
        self.save_config(template_config, template_path)
    
    def merge_configs(self, base_config: PipelineConfig, override_config: Dict[str, Any]) -> PipelineConfig:
        """Merge override configuration into base configuration."""
        base_dict = base_config.to_dict()
        
        # Deep merge dictionaries
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(base_dict, override_config)
        return PipelineConfig.from_dict(merged_dict)
    
    def validate_config_file(self, config_path: Union[str, Path]) -> bool:
        """Validate configuration file without loading it."""
        try:
            self.load_config(config_path)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Convenience functions
def create_default_config() -> PipelineConfig:
    """Create default configuration."""
    return PipelineConfig()


def load_config_from_file(config_path: Union[str, Path]) -> PipelineConfig:
    """Load configuration from file (convenience function)."""
    manager = ConfigManager()
    return manager.load_config(config_path)


def save_config_to_file(config: PipelineConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to file (convenience function)."""
    manager = ConfigManager()
    manager.save_config(config, config_path)


def create_config_template(template_path: Union[str, Path]) -> None:
    """Create configuration template file (convenience function)."""
    manager = ConfigManager()
    manager.create_template(template_path)


# Example usage and preset configurations
def create_high_quality_config() -> PipelineConfig:
    """Create high-quality processing configuration."""
    config = PipelineConfig()
    
    # High resolution grid
    config.grid.size = [800, 800, 800]
    config.grid.voxel_size = 3.0
    
    # Enhanced motion detection
    config.motion_detection.algorithm = "optical_flow"
    config.motion_detection.threshold = 1.0
    config.motion_detection.enhance_motion = True
    
    # More accurate ray casting
    config.ray_casting.max_steps = 50000
    config.ray_casting.step_size = 0.5
    
    # Better accumulation
    config.accumulation.algorithm = "weighted"
    config.accumulation.alpha = 0.05
    config.accumulation.normalize = True
    
    return config


def create_fast_config() -> PipelineConfig:
    """Create fast processing configuration for real-time use."""
    config = PipelineConfig()
    
    # Lower resolution
    config.grid.size = [200, 200, 200]
    config.grid.voxel_size = 10.0
    
    # Simple motion detection
    config.motion_detection.algorithm = "frame_difference"
    config.motion_detection.threshold = 5.0
    
    # Fewer ray casting steps
    config.ray_casting.max_steps = 5000
    config.ray_casting.step_size = 2.0
    
    # Simple accumulation
    config.accumulation.algorithm = "additive"
    
    # Larger batch size
    config.processing.batch_size = 16
    
    return config


def create_astronomical_config() -> PipelineConfig:
    """Create configuration optimized for astronomical data."""
    config = PipelineConfig()
    
    # Large grid for space observations
    config.grid.size = [1000, 1000, 1000]
    config.grid.voxel_size = 1e10  # 10 billion meters per voxel
    config.grid.center = [0.0, 0.0, 0.0]
    
    # Sensitive motion detection
    config.motion_detection.threshold = 0.5
    config.motion_detection.enhance_motion = True
    
    # Long-range ray casting
    config.ray_casting.max_steps = 100000
    
    # Distance-weighted accumulation
    config.accumulation.algorithm = "weighted"
    config.accumulation.alpha = 1e-12  # Very weak attenuation for space
    
    return config
import json
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


# --- Pydantic Models for Typed Configuration ---

class GridConfig(BaseModel):
    """Voxel grid configuration."""
    size: List[int] = Field(default=[500, 500, 500], description="Size of the voxel grid (X, Y, Z)")
    voxel_size: float = Field(default=6.0, description="Size of each voxel in real-world units")
    center: List[float] = Field(default=[0.0, 0.0, 500.0], description="Center of the voxel grid")

    @validator('size', 'center')
    def check_list_length(cls, v):
        if len(v) != 3:
            raise ValueError("must be a list of 3 numbers")
        return v

class MotionDetectionConfig(BaseModel):
    """Motion detection configuration."""
    algorithm: str = Field(default="frame_difference", description="Motion detection algorithm")
    threshold: float = Field(default=2.0, description="Threshold for motion detection")
    enhance_motion: bool = Field(default=False, description="Apply motion enhancement filter")

class ProcessingConfig(BaseModel):
    """Processing pipeline configuration."""
    batch_size: int = Field(default=32, description="Batch size for processing")
    num_workers: int = Field(default=4, description="Number of worker processes")
    use_gpu: bool = Field(default=True, description="Use GPU for processing if available")

class IOConfig(BaseModel):
    """Input/output configuration."""
    output_format: str = Field(default="bin", description="Output format for voxel grid (bin, npy, hdf5)")
    export_mesh: bool = Field(default=True, description="Export a 3D mesh from the voxel grid")
    mesh_threshold: float = Field(default=0.5, description="Threshold for marching cubes algorithm")

class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    grid: GridConfig = Field(default_factory=GridConfig)
    motion_detection: MotionDetectionConfig = Field(default_factory=MotionDetectionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    io: IOConfig = Field(default_factory=IOConfig)

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'PipelineConfig':
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def validate(self):
        # Pydantic automatically validates on model creation, so this is for compatibility.
        pass

# --- Configuration Management ---

class ConfigManager:
    """Manages loading, saving, and validating configurations."""

    def load_config(self, path: str) -> PipelineConfig:
        """Load configuration from YAML or JSON file."""
        with open(path, 'r') as f:
            if path.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f)
            elif path.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use .yaml or .json")
        return PipelineConfig.from_dict(data)

    def save_config(self, config: PipelineConfig, path: str) -> None:
        """Save configuration to YAML or JSON file."""
        with open(path, 'w') as f:
            if path.endswith(('.yaml', '.yml')):
                yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            elif path.endswith('.json'):
                json.dump(config.to_dict(), f, indent=2)
            else:
                raise ValueError("Unsupported config file format. Use .yaml or .json")

    def validate_config_file(self, path: str) -> bool:
        """Validate a configuration file."""
        try:
            self.load_config(path)
            return True
        except Exception:
            return False

# --- Preset Factory Functions ---

def create_default_config() -> PipelineConfig:
    """Create a default configuration."""
    return PipelineConfig()

def create_high_quality_config() -> PipelineConfig:
    """Create a high-quality configuration."""
    return PipelineConfig(
        grid=GridConfig(size=[1000, 1000, 1000], voxel_size=3.0),
        motion_detection=MotionDetectionConfig(threshold=1.0, enhance_motion=True),
        processing=ProcessingConfig(batch_size=16, use_gpu=True),
        io=IOConfig(mesh_threshold=0.4)
    )

def create_fast_config() -> PipelineConfig:
    """Create a fast processing configuration."""
    return PipelineConfig(
        grid=GridConfig(size=[250, 250, 250], voxel_size=10.0),
        motion_detection=MotionDetectionConfig(threshold=3.0),
        processing=ProcessingConfig(batch_size=64, num_workers=8, use_gpu=True),
        io=IOConfig(export_mesh=False)
    )

def create_astronomical_config() -> PipelineConfig:
    """Create a configuration for astronomical data."""
    return PipelineConfig(
        grid=GridConfig(size=[2048, 2048, 2048], voxel_size=1.0, center=[0, 0, 0]),
        motion_detection=MotionDetectionConfig(algorithm="optical_flow", threshold=0.5),
        processing=ProcessingConfig(use_gpu=True),
        io=IOConfig(output_format="hdf5", mesh_threshold=0.6)
    )
"""
Core interfaces and abstract base classes for pixeltovoxelprojector.

This module defines the key abstractions that enable modularity and
interoperability with other libraries.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from pathlib import Path

from data_models import FrameInfo


class DataSource(ABC):
    """Abstract interface for data sources (images, videos, etc.)."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the data source."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, FrameInfo]:
        """
        Get item at index.
        
        Returns:
            Tuple of (image_tensor, metadata_dict)
        """
        pass
    
    @abstractmethod
    def get_frame_info(self, idx: int) -> FrameInfo:
        """Get frame metadata for given index."""
        pass


class MotionDetector(ABC):
    """Abstract interface for motion detection algorithms."""
    
    @abstractmethod
    def detect(
        self, 
        prev_frame: torch.Tensor, 
        curr_frame: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect motion between two frames.
        
        Args:
            prev_frame: Previous frame tensor
            curr_frame: Current frame tensor
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Tuple of (motion_mask, motion_magnitude)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass


class RayCaster(ABC):
    """Abstract interface for ray casting algorithms."""
    
    @abstractmethod
    def cast_rays(
        self,
        camera_pos: torch.Tensor,
        ray_directions: torch.Tensor,
        grid_params: Dict[str, Any],
        **kwargs
    ) -> List[Tuple[int, int, int, float]]:
        """
        Cast rays through voxel grid.
        
        Args:
            camera_pos: Camera position tensor
            ray_directions: Ray direction tensors
            grid_params: Grid configuration parameters
            **kwargs: Algorithm-specific parameters
            
        Returns:
            List of (x, y, z, distance) tuples for voxel intersections
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass


class VoxelAccumulator(ABC):
    """Abstract interface for voxel value accumulation strategies."""
    
    @abstractmethod
    def accumulate(
        self,
        voxel_grid: torch.Tensor,
        voxel_coords: torch.Tensor,
        values: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Accumulate values into voxel grid.
        
        Args:
            voxel_grid: Current voxel grid
            voxel_coords: Voxel coordinates to update
            values: Values to accumulate
            distances: Optional distances for attenuation
            **kwargs: Strategy-specific parameters
            
        Returns:
            Updated voxel grid
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class VoxelGrid(ABC):
    """Abstract interface for voxel grid representations."""
    
    @abstractmethod
    def __init__(self, size: Tuple[int, int, int], voxel_size: float, center: torch.Tensor):
        """Initialize voxel grid."""
        pass
    
    @abstractmethod
    def get_data(self) -> torch.Tensor:
        """Get the underlying voxel data tensor."""
        pass
    
    @abstractmethod
    def set_data(self, data: torch.Tensor) -> None:
        """Set the underlying voxel data tensor."""
        pass
    
    @abstractmethod
    def to_point_cloud(self, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to point cloud representation.
        
        Returns:
            Tuple of (points, values)
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path], format: str = "bin") -> None:
        """Save voxel grid to file."""
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load voxel grid from file."""
        pass


class Transform(ABC):
    """Abstract interface for data transforms (following scikit-image pattern)."""
    
    @abstractmethod
    def __call__(self, data: Any, **kwargs) -> Any:
        """Apply transform to data."""
        pass
    
    @property
    def name(self) -> str:
        """Return transform name."""
        return self.__class__.__name__


class Renderer(ABC):
    """Abstract interface for visualization renderers."""
    
    @abstractmethod
    def render(
        self,
        voxel_grid: VoxelGrid,
        **kwargs
    ) -> Any:
        """Render voxel grid."""
        pass
    
    @abstractmethod
    def save_image(self, output_path: Union[str, Path]) -> None:
        """Save rendered image."""
        pass
    
    @property
    @abstractmethod
    def backend(self) -> str:
        """Return renderer backend name."""
        pass


class ProcessorFactory(ABC):
    """Abstract factory for creating processors."""
    
    @abstractmethod
    def create_motion_detector(self, name: str, **kwargs) -> MotionDetector:
        """Create motion detector by name."""
        pass
    
    @abstractmethod
    def create_ray_caster(self, name: str, **kwargs) -> RayCaster:
        """Create ray caster by name."""
        pass
    
    @abstractmethod
    def create_accumulator(self, name: str, **kwargs) -> VoxelAccumulator:
        """Create accumulator by name."""
        pass
    
    @abstractmethod
    def list_available(self, processor_type: str) -> List[str]:
        """List available processors of given type."""
        pass


class Pipeline(ABC):
    """Abstract pipeline interface for chaining operations."""
    
    @abstractmethod
    def add_step(self, name: str, processor: Any, **kwargs) -> 'Pipeline':
        """Add processing step to pipeline."""
        pass
    
    @abstractmethod
    def run(self, data: Any) -> Any:
        """Execute pipeline on data."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate pipeline configuration."""
        pass
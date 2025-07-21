"""
Registry system for managing different algorithm implementations.

This module provides a centralized way to register and retrieve
different implementations of core interfaces.
"""

from typing import Dict, List, Type, Any, Callable
from interfaces import MotionDetector, RayCaster, VoxelAccumulator, Transform, Renderer
import logging

logger = logging.getLogger(__name__)


class Registry:
    """Central registry for algorithm implementations."""
    
    def __init__(self):
        self._motion_detectors: Dict[str, Type[MotionDetector]] = {}
        self._ray_casters: Dict[str, Type[RayCaster]] = {}
        self._accumulators: Dict[str, Type[VoxelAccumulator]] = {}
        self._transforms: Dict[str, Type[Transform]] = {}
        self._renderers: Dict[str, Type[Renderer]] = {}
        
    def register_motion_detector(self, name: str, cls: Type[MotionDetector]) -> None:
        """Register a motion detector implementation."""
        if name in self._motion_detectors:
            logger.warning(f"Overwriting existing motion detector: {name}")
        self._motion_detectors[name] = cls
        logger.info(f"Registered motion detector: {name}")
        
    def register_ray_caster(self, name: str, cls: Type[RayCaster]) -> None:
        """Register a ray caster implementation."""
        if name in self._ray_casters:
            logger.warning(f"Overwriting existing ray caster: {name}")
        self._ray_casters[name] = cls
        logger.info(f"Registered ray caster: {name}")
        
    def register_accumulator(self, name: str, cls: Type[VoxelAccumulator]) -> None:
        """Register a voxel accumulator implementation."""
        if name in self._accumulators:
            logger.warning(f"Overwriting existing accumulator: {name}")
        self._accumulators[name] = cls
        logger.info(f"Registered accumulator: {name}")
        
    def register_transform(self, name: str, cls: Type[Transform]) -> None:
        """Register a transform implementation."""
        if name in self._transforms:
            logger.warning(f"Overwriting existing transform: {name}")
        self._transforms[name] = cls
        logger.info(f"Registered transform: {name}")
        
    def register_renderer(self, name: str, cls: Type[Renderer]) -> None:
        """Register a renderer implementation."""
        if name in self._renderers:
            logger.warning(f"Overwriting existing renderer: {name}")
        self._renderers[name] = cls
        logger.info(f"Registered renderer: {name}")
        
    def get_motion_detector(self, name: str, **kwargs) -> MotionDetector:
        """Get motion detector instance by name."""
        if name not in self._motion_detectors:
            raise ValueError(f"Unknown motion detector: {name}")
        return self._motion_detectors[name](**kwargs)
        
    def get_ray_caster(self, name: str, **kwargs) -> RayCaster:
        """Get ray caster instance by name."""
        if name not in self._ray_casters:
            raise ValueError(f"Unknown ray caster: {name}")
        return self._ray_casters[name](**kwargs)
        
    def get_accumulator(self, name: str, **kwargs) -> VoxelAccumulator:
        """Get accumulator instance by name."""
        if name not in self._accumulators:
            raise ValueError(f"Unknown accumulator: {name}")
        return self._accumulators[name](**kwargs)
        
    def get_transform(self, name: str, **kwargs) -> Transform:
        """Get transform instance by name."""
        if name not in self._transforms:
            raise ValueError(f"Unknown transform: {name}")
        return self._transforms[name](**kwargs)
        
    def get_renderer(self, name: str, **kwargs) -> Renderer:
        """Get renderer instance by name."""
        if name not in self._renderers:
            raise ValueError(f"Unknown renderer: {name}")
        return self._renderers[name](**kwargs)
        
    def list_motion_detectors(self) -> List[str]:
        """List available motion detectors."""
        return list(self._motion_detectors.keys())
        
    def list_ray_casters(self) -> List[str]:
        """List available ray casters."""
        return list(self._ray_casters.keys())
        
    def list_accumulators(self) -> List[str]:
        """List available accumulators."""
        return list(self._accumulators.keys())
        
    def list_transforms(self) -> List[str]:
        """List available transforms."""
        return list(self._transforms.keys())
        
    def list_renderers(self) -> List[str]:
        """List available renderers."""
        return list(self._renderers.keys())


# Global registry instance
_REGISTRY = Registry()


# Decorator functions for easy registration
def register_motion_detector(name: str):
    """Decorator to register motion detector."""
    def decorator(cls: Type[MotionDetector]):
        _REGISTRY.register_motion_detector(name, cls)
        return cls
    return decorator


def register_ray_caster(name: str):
    """Decorator to register ray caster."""
    def decorator(cls: Type[RayCaster]):
        _REGISTRY.register_ray_caster(name, cls)
        return cls
    return decorator


def register_accumulator(name: str):
    """Decorator to register accumulator."""
    def decorator(cls: Type[VoxelAccumulator]):
        _REGISTRY.register_accumulator(name, cls)
        return cls
    return decorator


def register_transform(name: str):
    """Decorator to register transform."""
    def decorator(cls: Type[Transform]):
        _REGISTRY.register_transform(name, cls)
        return cls
    return decorator


def register_renderer(name: str):
    """Decorator to register renderer."""
    def decorator(cls: Type[Renderer]):
        _REGISTRY.register_renderer(name, cls)
        return cls
    return decorator


# Convenience functions
def get_motion_detector(name: str, **kwargs) -> MotionDetector:
    """Get motion detector instance."""
    return _REGISTRY.get_motion_detector(name, **kwargs)


def get_ray_caster(name: str, **kwargs) -> RayCaster:
    """Get ray caster instance."""
    return _REGISTRY.get_ray_caster(name, **kwargs)


def get_accumulator(name: str, **kwargs) -> VoxelAccumulator:
    """Get accumulator instance."""
    return _REGISTRY.get_accumulator(name, **kwargs)


def get_transform(name: str, **kwargs) -> Transform:
    """Get transform instance."""
    return _REGISTRY.get_transform(name, **kwargs)


def get_renderer(name: str, **kwargs) -> Renderer:
    """Get renderer instance."""
    return _REGISTRY.get_renderer(name, **kwargs)


def list_available(component_type: str) -> List[str]:
    """List available components of given type."""
    if component_type == "motion_detector":
        return _REGISTRY.list_motion_detectors()
    elif component_type == "ray_caster":
        return _REGISTRY.list_ray_casters()
    elif component_type == "accumulator":
        return _REGISTRY.list_accumulators()
    elif component_type == "transform":
        return _REGISTRY.list_transforms()
    elif component_type == "renderer":
        return _REGISTRY.list_renderers()
    else:
        raise ValueError(f"Unknown component type: {component_type}")
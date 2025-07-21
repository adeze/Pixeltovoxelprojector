"""
Transform pipeline compatible with scikit-image and torchvision patterns.

This module provides chainable transforms for both image and voxel data,
following established patterns from scikit-image and torchvision.
"""

from typing import Any, List, Optional, Tuple, Union, Callable
import torch
import numpy as np
from abc import ABC, abstractmethod
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF

from interfaces import Transform
from registry import register_transform


class Compose:
    """Compose several transforms together (following torchvision pattern)."""
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


# Image Transforms
@register_transform("resize")
class Resize(Transform):
    """Resize image to given size."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: str = "bilinear"):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(data.shape) == 2:
            # Add channel dimension for torchvision
            data = data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Convert interpolation string to torchvision constant
        interp_map = {
            "nearest": TF.InterpolationMode.NEAREST,
            "bilinear": TF.InterpolationMode.BILINEAR,
            "bicubic": TF.InterpolationMode.BICUBIC
        }
        
        resized = TF.resize(data, self.size, interp_map.get(self.interpolation, TF.InterpolationMode.BILINEAR))
        
        if squeeze_output:
            resized = resized.squeeze(0)
        
        return resized


@register_transform("crop")
class RandomCrop(Transform):
    """Random crop of given size."""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        cropped = TF.crop(data, *TF.get_image_size(data), *self.size)
        
        if squeeze_output:
            cropped = cropped.squeeze(0)
        
        return cropped


@register_transform("gaussian_blur")
class GaussianBlur(Transform):
    """Apply Gaussian blur to image."""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], sigma: Union[float, Tuple[float, float]] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        blurred = TF.gaussian_blur(data, self.kernel_size, self.sigma)
        
        if squeeze_output:
            blurred = blurred.squeeze(0)
        
        return blurred


@register_transform("rotation")
class RandomRotation(Transform):
    """Random rotation of the image."""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]], fill: float = 0):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.fill = fill
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        angle = torch.FloatTensor(1).uniform_(*self.degrees).item()
        
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        rotated = TF.rotate(data, angle, fill=self.fill)
        
        if squeeze_output:
            rotated = rotated.squeeze(0)
        
        return rotated


@register_transform("flip")
class RandomHorizontalFlip(Transform):
    """Random horizontal flip of the image."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return TF.hflip(data)
        return data


@register_transform("contrast")
class AdjustContrast(Transform):
    """Adjust image contrast."""
    
    def __init__(self, contrast_factor: float = 1.0):
        self.contrast_factor = contrast_factor
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        contrast = kwargs.get('contrast_factor', self.contrast_factor)
        return TF.adjust_contrast(data, contrast)


@register_transform("brightness")
class AdjustBrightness(Transform):
    """Adjust image brightness."""
    
    def __init__(self, brightness_factor: float = 1.0):
        self.brightness_factor = brightness_factor
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        brightness = kwargs.get('brightness_factor', self.brightness_factor)
        return TF.adjust_brightness(data, brightness)


# Motion-specific transforms
@register_transform("motion_enhancement")
class MotionEnhancement(Transform):
    """Enhance motion detection sensitivity through preprocessing."""
    
    def __init__(self, blur_kernel: int = 3, noise_std: float = 1.0):
        self.blur_kernel = blur_kernel
        self.noise_std = noise_std
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        # Apply slight blur to reduce high-frequency noise
        if self.blur_kernel > 1:
            blurred = TF.gaussian_blur(
                data.unsqueeze(0) if len(data.shape) == 2 else data,
                self.blur_kernel
            )
            if len(data.shape) == 2:
                blurred = blurred.squeeze(0)
            data = blurred
        
        # Add small amount of noise for motion sensitivity
        if self.noise_std > 0:
            noise = torch.randn_like(data) * self.noise_std
            data = torch.clamp(data + noise, 0, 255)
        
        return data


@register_transform("edge_enhancement")
class EdgeEnhancement(Transform):
    """Enhance edges for better motion detection."""
    
    def __init__(self, kernel_type: str = "sobel"):
        self.kernel_type = kernel_type
        
        if kernel_type == "sobel":
            self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        elif kernel_type == "prewitt":
            self.kernel_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
            self.kernel_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        # Add batch and channel dimensions for conv2d
        if len(data.shape) == 2:
            data_padded = data.unsqueeze(0).unsqueeze(0)
        else:
            data_padded = data.unsqueeze(0)
        
        # Apply edge detection kernels
        kernel_x = self.kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = self.kernel_y.unsqueeze(0).unsqueeze(0)
        
        grad_x = torch.nn.functional.conv2d(data_padded, kernel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(data_padded, kernel_y, padding=1)
        
        # Compute gradient magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Remove extra dimensions
        if len(data.shape) == 2:
            magnitude = magnitude.squeeze(0).squeeze(0)
        else:
            magnitude = magnitude.squeeze(0)
        
        return magnitude


# Voxel-specific transforms
@register_transform("voxel_threshold")
class VoxelThreshold(Transform):
    """Apply threshold to voxel grid."""
    
    def __init__(self, threshold: float = 0.5, mode: str = "binary"):
        self.threshold = threshold
        self.mode = mode  # "binary", "zero", "clamp"
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        threshold = kwargs.get('threshold', self.threshold)
        mode = kwargs.get('mode', self.mode)
        
        if mode == "binary":
            return (data > threshold).float()
        elif mode == "zero":
            return torch.where(data > threshold, data, torch.zeros_like(data))
        elif mode == "clamp":
            return torch.clamp(data, min=threshold)
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")


@register_transform("voxel_smooth")
class VoxelSmooth(Transform):
    """Apply 3D smoothing to voxel grid."""
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        # Create 3D Gaussian kernel
        kernel_size = kwargs.get('kernel_size', self.kernel_size)
        sigma = kwargs.get('sigma', self.sigma)
        
        # For simplicity, use separable 1D convolutions
        kernel_1d = self._gaussian_kernel_1d(kernel_size, sigma)
        kernel_3d = kernel_1d.view(-1, 1, 1) * kernel_1d.view(1, -1, 1) * kernel_1d.view(1, 1, -1)
        
        # Add batch and channel dimensions
        data_padded = data.unsqueeze(0).unsqueeze(0)
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)
        
        # Apply 3D convolution
        padding = kernel_size // 2
        smoothed = torch.nn.functional.conv3d(data_padded, kernel_3d, padding=padding)
        
        return smoothed.squeeze(0).squeeze(0)
    
    def _gaussian_kernel_1d(self, size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        return kernel / kernel.sum()


@register_transform("voxel_normalize")
class VoxelNormalize(Transform):
    """Normalize voxel grid values."""
    
    def __init__(self, method: str = "minmax", percentile: float = 99.0):
        self.method = method  # "minmax", "zscore", "percentile"
        self.percentile = percentile
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        method = kwargs.get('method', self.method)
        
        if method == "minmax":
            min_val = data.min()
            max_val = data.max()
            return (data - min_val) / (max_val - min_val + 1e-8)
        elif method == "zscore":
            mean_val = data.mean()
            std_val = data.std()
            return (data - mean_val) / (std_val + 1e-8)
        elif method == "percentile":
            percentile = kwargs.get('percentile', self.percentile)
            threshold = torch.quantile(data, percentile / 100.0)
            return torch.clamp(data / threshold, 0, 1)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


# Utility functions for creating common transform pipelines
def create_motion_detection_pipeline(
    enhance_motion: bool = True,
    add_noise: bool = True,
    blur: bool = False
) -> Compose:
    """Create standard motion detection preprocessing pipeline."""
    transforms = []
    
    if blur:
        transforms.append(GaussianBlur(kernel_size=3, sigma=1.0))
    
    if enhance_motion:
        transforms.append(MotionEnhancement(blur_kernel=3, noise_std=1.0))
    
    if add_noise and not enhance_motion:  # Avoid double noise
        from implementations import NoiseInjection
        transforms.append(NoiseInjection((-1.0, 1.0)))
    
    return Compose(transforms)


def create_voxel_processing_pipeline(
    threshold: Optional[float] = None,
    smooth: bool = False,
    normalize: bool = True
) -> Compose:
    """Create standard voxel processing pipeline."""
    transforms = []
    
    if smooth:
        transforms.append(VoxelSmooth(kernel_size=3, sigma=1.0))
    
    if threshold is not None:
        transforms.append(VoxelThreshold(threshold=threshold, mode="zero"))
    
    if normalize:
        transforms.append(VoxelNormalize(method="percentile", percentile=99.0))
    
    return Compose(transforms)


def create_augmentation_pipeline(
    rotation: bool = True,
    flip: bool = True,
    brightness: bool = True,
    contrast: bool = True
) -> Compose:
    """Create data augmentation pipeline."""
    transforms = []
    
    if rotation:
        transforms.append(RandomRotation(degrees=15))
    
    if flip:
        transforms.append(RandomHorizontalFlip(p=0.5))
    
    if brightness:
        transforms.append(AdjustBrightness(brightness_factor=1.2))
    
    if contrast:
        transforms.append(AdjustContrast(contrast_factor=1.2))
    
    return Compose(transforms)
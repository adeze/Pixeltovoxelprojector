"""
Concrete implementations of the core interfaces.

This module provides the default implementations that were previously
embedded in the main processing files.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from PIL import Image
import cv2
import json
from pathlib import Path

from interfaces import (
    DataSource, MotionDetector, RayCaster, VoxelAccumulator, 
    VoxelGrid, Transform
)
from registry import (
    register_motion_detector, register_ray_caster, 
    register_accumulator, register_transform
)


# Data Source Implementations
class ImageSequenceDataSource(DataSource):
    """Data source for image sequences with metadata."""
    
    def __init__(self, metadata_path: str, images_folder: str):
        self.metadata_path = Path(metadata_path)
        self.images_folder = Path(images_folder)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Group by camera and sort by frame
        self.frames_by_camera = {}
        for entry in self.metadata:
            camera_idx = entry.get("camera_index", 0)
            if camera_idx not in self.frames_by_camera:
                self.frames_by_camera[camera_idx] = []
            self.frames_by_camera[camera_idx].append(entry)
            
        for frames in self.frames_by_camera.values():
            frames.sort(key=lambda x: x.get("frame_index", 0))
            
        # Create flat list for indexing
        self.flat_frames = []
        for camera_frames in self.frames_by_camera.values():
            self.flat_frames.extend(camera_frames)
    
    def __len__(self) -> int:
        return len(self.flat_frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        frame_info = self.flat_frames[idx]
        image_path = self.images_folder / frame_info["image_file"]
        
        # Load image
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float32)
        # Add noise for motion detection sensitivity
        noise = np.random.uniform(-1, 1, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        image_tensor = torch.from_numpy(arr)
        
        return image_tensor, frame_info
    
    def get_frame_info(self, idx: int) -> Dict[str, Any]:
        return self.flat_frames[idx]


# Motion Detector Implementations
@register_motion_detector("frame_difference")
class FrameDifferenceDetector(MotionDetector):
    """Simple frame differencing motion detection."""
    
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
    
    def detect(
        self, 
        prev_frame: torch.Tensor, 
        curr_frame: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        threshold = kwargs.get('threshold', self.threshold)
        diff = torch.abs(curr_frame - prev_frame)
        changed = diff > threshold
        return changed, diff
    
    @property
    def name(self) -> str:
        return "frame_difference"


@register_motion_detector("optical_flow")
class OpticalFlowDetector(MotionDetector):
    """OpenCV-based optical flow motion detection."""
    
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
    
    def detect(
        self, 
        prev_frame: torch.Tensor, 
        curr_frame: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to numpy for OpenCV
        prev_np = prev_frame.numpy().astype(np.uint8)
        curr_np = curr_frame.numpy().astype(np.uint8)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            prev_np, curr_np, None, None
        )[1]  # Get flow vectors
        
        if flow is not None:
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            magnitude_tensor = torch.from_numpy(magnitude).float()
            changed = magnitude_tensor > self.threshold
        else:
            # Fallback to frame difference
            diff = torch.abs(curr_frame - prev_frame)
            changed = diff > self.threshold
            magnitude_tensor = diff
            
        return changed, magnitude_tensor
    
    @property
    def name(self) -> str:
        return "optical_flow"


# Ray Caster Implementations
@register_ray_caster("dda")
class DDATraversal(RayCaster):
    """Digital Differential Analyzer ray traversal."""
    
    def cast_rays(
        self,
        camera_pos: torch.Tensor,
        ray_directions: torch.Tensor,
        grid_params: Dict[str, Any],
        **kwargs
    ) -> List[Tuple[int, int, int, float]]:
        """Cast multiple rays using DDA algorithm."""
        all_steps = []
        
        for i in range(ray_directions.shape[0]):
            ray_dir = ray_directions[i]
            steps = self._cast_single_ray(camera_pos, ray_dir, grid_params)
            all_steps.extend(steps)
            
        return all_steps
    
    def _cast_single_ray(
        self,
        camera_pos: torch.Tensor,
        dir_normalized: torch.Tensor,
        grid_params: Dict[str, Any]
    ) -> List[Tuple[int, int, int, float]]:
        """Cast single ray through voxel grid."""
        N = grid_params['size']
        voxel_size = grid_params['voxel_size']
        grid_center = grid_params['center']
        
        steps = []
        half_size = 0.5 * (N * voxel_size)
        grid_min = grid_center - half_size
        grid_max = grid_center + half_size

        t_min = 0.0
        t_max = float("inf")
        origin = camera_pos
        direction = dir_normalized

        # Ray-box intersection
        for i in range(3):
            o = origin[i].item()
            d = direction[i].item()
            mn = grid_min[i].item()
            mx = grid_max[i].item()
            if abs(d) < 1e-12:
                if o < mn or o > mx:
                    return steps
            else:
                t1 = (mn - o) / d
                t2 = (mx - o) / d
                t_near = min(t1, t2)
                t_far = max(t1, t2)
                if t_near > t_min:
                    t_min = t_near
                if t_far < t_max:
                    t_max = t_far
                if t_min > t_max:
                    return steps

        if t_min < 0.0:
            t_min = 0.0

        # Start voxel
        start_world = origin + direction * t_min
        fx = (start_world[0] - grid_min[0]) / voxel_size
        fy = (start_world[1] - grid_min[1]) / voxel_size
        fz = (start_world[2] - grid_min[2]) / voxel_size
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)
        if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
            return steps

        # Step direction
        step_x = 1 if direction[0] >= 0 else -1
        step_y = 1 if direction[1] >= 0 else -1
        step_z = 1 if direction[2] >= 0 else -1

        def boundary_in_world(i, axis):
            return grid_min[axis] + i * voxel_size

        nx_x = ix + (1 if step_x > 0 else 0)
        nx_y = iy + (1 if step_y > 0 else 0)
        nx_z = iz + (1 if step_z > 0 else 0)

        next_bx = boundary_in_world(nx_x, 0)
        next_by = boundary_in_world(nx_y, 1)
        next_bz = boundary_in_world(nx_z, 2)

        def safe_div(num, den):
            eps = 1e-12
            if abs(den) < eps:
                return float("inf")
            return num / den

        t_max_x = safe_div(next_bx - origin[0], direction[0])
        t_max_y = safe_div(next_by - origin[1], direction[1])
        t_max_z = safe_div(next_bz - origin[2], direction[2])

        t_delta_x = safe_div(voxel_size, abs(direction[0]))
        t_delta_y = safe_div(voxel_size, abs(direction[1]))
        t_delta_z = safe_div(voxel_size, abs(direction[2]))

        t_current = t_min
        step_count = 0

        while t_current <= t_max:
            steps.append((ix, iy, iz, t_current))
            if t_max_x < t_max_y and t_max_x < t_max_z:
                ix += step_x
                t_current = t_max_x
                t_max_x += t_delta_x
            elif t_max_y < t_max_z:
                iy += step_y
                t_current = t_max_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t_current = t_max_z
                t_max_z += t_delta_z
            step_count += 1
            if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
                break
        return steps
    
    @property
    def name(self) -> str:
        return "dda"


# Voxel Accumulator Implementations
@register_accumulator("weighted")
class WeightedAccumulator(VoxelAccumulator):
    """Distance-weighted accumulation strategy."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    def accumulate(
        self,
        voxel_grid: torch.Tensor,
        voxel_coords: torch.Tensor,
        values: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        alpha = kwargs.get('alpha', self.alpha)
        
        for i in range(len(voxel_coords)):
            ix, iy, iz = voxel_coords[i]
            value = values[i]
            
            if distances is not None:
                dist = distances[i]
                attenuation = 1.0 / (1.0 + alpha * dist)
                value = value * attenuation
                
            if 0 <= ix < voxel_grid.shape[0] and 0 <= iy < voxel_grid.shape[1] and 0 <= iz < voxel_grid.shape[2]:
                voxel_grid[ix, iy, iz] += value
                
        return voxel_grid
    
    @property
    def name(self) -> str:
        return "weighted"


@register_accumulator("additive")
class AdditiveAccumulator(VoxelAccumulator):
    """Simple additive accumulation strategy."""
    
    def accumulate(
        self,
        voxel_grid: torch.Tensor,
        voxel_coords: torch.Tensor,
        values: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        for i in range(len(voxel_coords)):
            ix, iy, iz = voxel_coords[i]
            value = values[i]
                
            if 0 <= ix < voxel_grid.shape[0] and 0 <= iy < voxel_grid.shape[1] and 0 <= iz < voxel_grid.shape[2]:
                voxel_grid[ix, iy, iz] += value
                
        return voxel_grid
    
    @property
    def name(self) -> str:
        return "additive"


# VoxelGrid Implementation
class StandardVoxelGrid(VoxelGrid):
    """Standard voxel grid implementation."""
    
    def __init__(self, size: Tuple[int, int, int], voxel_size: float, center: torch.Tensor):
        self.size = size
        self.voxel_size = voxel_size
        self.center = center
        self.data = torch.zeros(size, dtype=torch.float32)
        
    def get_data(self) -> torch.Tensor:
        return self.data
        
    def set_data(self, data: torch.Tensor) -> None:
        self.data = data
        
    def to_point_cloud(self, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to point cloud representation."""
        # Find non-zero voxels
        indices = torch.nonzero(self.data > threshold)
        values = self.data[self.data > threshold]
        
        # Convert indices to world coordinates
        half_size = 0.5 * (self.size[0] * self.voxel_size)
        grid_min = self.center - half_size
        
        points = torch.zeros(indices.shape[0], 3, dtype=torch.float32)
        points[:, 0] = grid_min[0] + indices[:, 0] * self.voxel_size
        points[:, 1] = grid_min[1] + indices[:, 1] * self.voxel_size
        points[:, 2] = grid_min[2] + indices[:, 2] * self.voxel_size
        
        return points, values
        
    def save(self, path: Path, format: str = "bin") -> None:
        """Save voxel grid to file."""
        path = Path(path)
        
        if format == "bin":
            with open(path, 'wb') as f:
                f.write(np.array(self.size, dtype=np.int32).tobytes())
                f.write(np.array([self.voxel_size], dtype=np.float32).tobytes())
                f.write(self.data.numpy().astype(np.float32).tobytes())
        elif format == "npy":
            np.save(path, self.data.numpy())
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def load(self, path: Path) -> None:
        """Load voxel grid from file."""
        path = Path(path)
        
        if path.suffix == ".bin":
            with open(path, 'rb') as f:
                size_data = f.read(3 * 4)  # 3 int32s
                size = np.frombuffer(size_data, dtype=np.int32)
                self.size = tuple(size)
                
                voxel_size_data = f.read(4)  # 1 float32
                self.voxel_size = np.frombuffer(voxel_size_data, dtype=np.float32)[0]
                
                grid_data = f.read()
                grid = np.frombuffer(grid_data, dtype=np.float32).reshape(self.size)
                self.data = torch.from_numpy(grid)
        elif path.suffix == ".npy":
            self.data = torch.from_numpy(np.load(path))
            self.size = tuple(self.data.shape)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


# Transform Implementations
@register_transform("noise_injection")
class NoiseInjection(Transform):
    """Add random noise to images for motion detection sensitivity."""
    
    def __init__(self, noise_range: Tuple[float, float] = (-1.0, 1.0)):
        self.noise_range = noise_range
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        noise_range = kwargs.get('noise_range', self.noise_range)
        noise = torch.FloatTensor(*data.shape).uniform_(*noise_range)
        return torch.clamp(data + noise, 0, 255)


@register_transform("normalize")
class Normalize(Transform):
    """Normalize tensor values."""
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        min_val = kwargs.get('min_val', self.min_val)
        max_val = kwargs.get('max_val', self.max_val)
        
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        
        if data_range > 0:
            normalized = (data - data_min) / data_range
            return normalized * (max_val - min_val) + min_val
        else:
            return data
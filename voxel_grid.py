"""
VoxelGrid implementation with Open3D and trimesh compatibility.

This module provides a comprehensive VoxelGrid class that can interchange
data with Open3D and trimesh libraries for enhanced 3D processing capabilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from pathlib import Path
import struct
import json
import h5py

from interfaces import VoxelGrid

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import mcubes
    HAS_MCUBES = True
except ImportError:
    HAS_MCUBES = False


class StandardVoxelGrid(VoxelGrid):
    """
    Standard voxel grid implementation with library interoperability.
    
    Supports interchange with Open3D VoxelGrid and trimesh.voxel.VoxelGrid.
    """
    
    def __init__(
        self, 
        size: Tuple[int, int, int], 
        voxel_size: float, 
        center: torch.Tensor,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize voxel grid.
        
        Args:
            size: Grid dimensions (nx, ny, nz)
            voxel_size: Size of each voxel in world units
            center: World center point of the grid
            dtype: Data type for voxel values
        """
        self.size = size
        self.voxel_size = voxel_size
        self.center = center.clone() if isinstance(center, torch.Tensor) else torch.tensor(center, dtype=torch.float32)
        self.dtype = dtype
        self.data = torch.zeros(size, dtype=dtype)
        
        # Calculate grid bounds
        self.half_extent = 0.5 * torch.tensor(size, dtype=torch.float32) * voxel_size
        self.min_bound = self.center - self.half_extent
        self.max_bound = self.center + self.half_extent
    
    def get_data(self) -> torch.Tensor:
        """Get the underlying voxel data tensor."""
        return self.data
    
    def set_data(self, data: torch.Tensor) -> None:
        """Set the underlying voxel data tensor."""
        if data.shape != self.size:
            raise ValueError(f"Data shape {data.shape} does not match grid size {self.size}")
        self.data = data.to(self.dtype)
    
    def get_voxel_coordinates(self) -> torch.Tensor:
        """Get world coordinates of all voxel centers."""
        indices = torch.stack(torch.meshgrid(
            torch.arange(self.size[0]),
            torch.arange(self.size[1]),
            torch.arange(self.size[2]),
            indexing='ij'
        ), dim=-1).float()
        
        # Convert indices to world coordinates
        coords = self.min_bound + (indices + 0.5) * self.voxel_size
        return coords
    
    def world_to_voxel(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel indices."""
        voxel_coords = (world_coords - self.min_bound) / self.voxel_size
        return torch.floor(voxel_coords).long()
    
    def voxel_to_world(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """Convert voxel indices to world coordinates (voxel centers)."""
        return self.min_bound + (voxel_indices.float() + 0.5) * self.voxel_size
    
    def get_occupied_voxels(self, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get indices and values of occupied voxels.
        
        Args:
            threshold: Minimum value to consider occupied
            
        Returns:
            Tuple of (voxel_indices, voxel_values)
        """
        mask = self.data > threshold
        indices = torch.nonzero(mask)
        values = self.data[mask]
        return indices, values
    
    def to_point_cloud(self, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to point cloud representation.
        
        Args:
            threshold: Minimum value to include in point cloud
            
        Returns:
            Tuple of (points, values)
        """
        indices, values = self.get_occupied_voxels(threshold)
        if len(indices) == 0:
            return torch.empty((0, 3)), torch.empty(0)
        
        points = self.voxel_to_world(indices)
        return points, values
    
    def from_point_cloud(
        self, 
        points: torch.Tensor, 
        values: Optional[torch.Tensor] = None,
        accumulate: bool = True
    ) -> None:
        """
        Fill voxel grid from point cloud.
        
        Args:
            points: Point coordinates (N, 3)
            values: Point values (N,). If None, uses 1.0 for all points
            accumulate: Whether to accumulate values or overwrite
        """
        if values is None:
            values = torch.ones(points.shape[0])
        
        voxel_indices = self.world_to_voxel(points)
        
        # Filter valid indices
        valid_mask = (
            (voxel_indices >= 0).all(dim=1) &
            (voxel_indices[:, 0] < self.size[0]) &
            (voxel_indices[:, 1] < self.size[1]) &
            (voxel_indices[:, 2] < self.size[2])
        )
        
        valid_indices = voxel_indices[valid_mask]
        valid_values = values[valid_mask]
        
        if not accumulate:
            self.data.zero_()
        
        # Accumulate values
        for i in range(len(valid_indices)):
            idx = valid_indices[i]
            self.data[idx[0], idx[1], idx[2]] += valid_values[i]
    
    # Open3D Integration
    def to_open3d(self) -> 'o3d.geometry.VoxelGrid':
        """Convert to Open3D VoxelGrid."""
        if not HAS_OPEN3D:
            raise ImportError("Open3D is not installed")
        
        # Create Open3D voxel grid
        o3d_voxel_grid = o3d.geometry.VoxelGrid()
        o3d_voxel_grid.voxel_size = self.voxel_size
        o3d_voxel_grid.origin = self.min_bound.numpy()
        
        # Convert occupied voxels to Open3D format
        indices, values = self.get_occupied_voxels()
        
        for i in range(len(indices)):
            idx = indices[i].numpy()
            value = values[i].item()
            
            # Create voxel
            voxel = o3d.geometry.Voxel()
            voxel.grid_index = idx
            voxel.color = [value, value, value]  # Grayscale color
            
            o3d_voxel_grid.add_voxel(voxel)
        
        return o3d_voxel_grid
    
    def from_open3d(self, o3d_voxel_grid: 'o3d.geometry.VoxelGrid') -> None:
        """Load from Open3D VoxelGrid."""
        if not HAS_OPEN3D:
            raise ImportError("Open3D is not installed")
        
        # Extract voxels
        voxels = o3d_voxel_grid.get_voxels()
        
        self.data.zero_()
        
        for voxel in voxels:
            idx = voxel.grid_index
            # Use average of RGB as value
            value = np.mean(voxel.color)
            
            if (0 <= idx[0] < self.size[0] and 
                0 <= idx[1] < self.size[1] and 
                0 <= idx[2] < self.size[2]):
                self.data[idx[0], idx[1], idx[2]] = value
    
    # trimesh Integration
    def to_trimesh(self) -> 'trimesh.voxel.VoxelGrid':
        """Convert to trimesh VoxelGrid."""
        if not HAS_TRIMESH:
            raise ImportError("trimesh is not installed")
        
        # Convert to numpy array (trimesh expects numpy)
        data_np = self.data.numpy()
        
        # Create transform matrix
        transform = np.eye(4)
        transform[:3, 3] = self.min_bound.numpy()
        transform[:3, :3] *= self.voxel_size
        
        return trimesh.voxel.VoxelGrid(
            encoding=data_np,
            transform=transform
        )
    
    def from_trimesh(self, trimesh_voxel_grid: 'trimesh.voxel.VoxelGrid') -> None:
        """Load from trimesh VoxelGrid."""
        if not HAS_TRIMESH:
            raise ImportError("trimesh is not installed")
        
        # Extract data and transform
        data_np = trimesh_voxel_grid.encoding
        self.data = torch.from_numpy(data_np).to(self.dtype)
        self.size = tuple(self.data.shape)
        
        # Extract voxel size from transform
        transform = trimesh_voxel_grid.transform
        self.voxel_size = float(transform[0, 0])  # Assuming uniform scaling
        
        # Extract origin
        origin = torch.tensor(transform[:3, 3], dtype=torch.float32)
        self.min_bound = origin
        
        # Recalculate center and bounds
        self.half_extent = 0.5 * torch.tensor(self.size, dtype=torch.float32) * self.voxel_size
        self.center = origin + self.half_extent
        self.max_bound = origin + 2 * self.half_extent
    
    # Mesh Extraction
    def extract_mesh(self, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mesh using Marching Cubes algorithm.
        
        Args:
            threshold: Iso-surface threshold
            
        Returns:
            Tuple of (vertices, faces)
        """
        if not HAS_MCUBES:
            raise ImportError("PyMCubes is not installed")
        
        # Extract mesh using marching cubes
        vertices, triangles = mcubes.marching_cubes(self.data.numpy(), threshold)
        
        # Convert to world coordinates
        vertices = torch.from_numpy(vertices).float()
        vertices = self.min_bound + vertices * self.voxel_size
        
        triangles = torch.from_numpy(triangles).long()
        
        return vertices, triangles
    
    def save_mesh(self, path: Union[str, Path], threshold: float = 0.5, format: str = "obj") -> None:
        """Save extracted mesh to file."""
        vertices, triangles = self.extract_mesh(threshold)
        
        path = Path(path)
        
        if format.lower() == "obj" or path.suffix.lower() == ".obj":
            self._save_obj(vertices, triangles, path)
        elif format.lower() == "ply" or path.suffix.lower() == ".ply":
            self._save_ply(vertices, triangles, path)
        else:
            raise ValueError(f"Unsupported mesh format: {format}")
    
    def _save_obj(self, vertices: torch.Tensor, triangles: torch.Tensor, path: Path) -> None:
        """Save mesh in OBJ format."""
        with open(path, 'w') as f:
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for triangle in triangles:
                f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
    
    def _save_ply(self, vertices: torch.Tensor, triangles: torch.Tensor, path: Path) -> None:
        """Save mesh in PLY format."""
        with open(path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(triangles)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces
            for triangle in triangles:
                f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")
    
    # File I/O
    def save(self, path: Union[str, Path], format: str = "bin") -> None:
        """Save voxel grid to file."""
        path = Path(path)
        
        if format == "bin":
            self._save_binary(path)
        elif format == "npy":
            np.save(path, self.data.numpy())
        elif format == "hdf5":
            self._save_hdf5(path)
        elif format == "json":
            self._save_json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load voxel grid from file."""
        path = Path(path)
        
        if path.suffix == ".bin":
            self._load_binary(path)
        elif path.suffix == ".npy":
            self.data = torch.from_numpy(np.load(path)).to(self.dtype)
            self.size = tuple(self.data.shape)
        elif path.suffix in [".hdf5", ".h5"]:
            self._load_hdf5(path)
        elif path.suffix == ".json":
            self._load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _save_binary(self, path: Path) -> None:
        """Save in custom binary format."""
        with open(path, 'wb') as f:
            # Write header
            f.write(struct.pack('III', *self.size))  # Grid size
            f.write(struct.pack('f', self.voxel_size))  # Voxel size
            f.write(struct.pack('fff', *self.center.tolist()))  # Center
            
            # Write data
            f.write(self.data.numpy().astype(np.float32).tobytes())
    
    def _load_binary(self, path: Path) -> None:
        """Load from custom binary format."""
        with open(path, 'rb') as f:
            # Read header
            size_data = f.read(3 * 4)  # 3 uint32s
            self.size = struct.unpack('III', size_data)
            
            voxel_size_data = f.read(4)  # 1 float32
            self.voxel_size = struct.unpack('f', voxel_size_data)[0]
            
            center_data = f.read(3 * 4)  # 3 float32s
            center_list = struct.unpack('fff', center_data)
            self.center = torch.tensor(center_list, dtype=torch.float32)
            
            # Read data
            grid_data = f.read()
            grid = np.frombuffer(grid_data, dtype=np.float32).reshape(self.size)
            self.data = torch.from_numpy(grid).to(self.dtype)
            
            # Recalculate bounds
            self.half_extent = 0.5 * torch.tensor(self.size, dtype=torch.float32) * self.voxel_size
            self.min_bound = self.center - self.half_extent
            self.max_bound = self.center + self.half_extent
    
    def _save_hdf5(self, path: Path) -> None:
        """Save in HDF5 format."""
        with h5py.File(path, 'w') as f:
            # Save metadata
            f.attrs['size'] = self.size
            f.attrs['voxel_size'] = self.voxel_size
            f.attrs['center'] = self.center.numpy()
            
            # Save data
            f.create_dataset('data', data=self.data.numpy(), compression='gzip')
    
    def _load_hdf5(self, path: Path) -> None:
        """Load from HDF5 format."""
        with h5py.File(path, 'r') as f:
            # Load metadata
            self.size = tuple(f.attrs['size'])
            self.voxel_size = float(f.attrs['voxel_size'])
            self.center = torch.tensor(f.attrs['center'], dtype=torch.float32)
            
            # Load data
            self.data = torch.from_numpy(f['data'][:]).to(self.dtype)
            
            # Recalculate bounds
            self.half_extent = 0.5 * torch.tensor(self.size, dtype=torch.float32) * self.voxel_size
            self.min_bound = self.center - self.half_extent
            self.max_bound = self.center + self.half_extent
    
    def _save_json(self, path: Path) -> None:
        """Save metadata and sparse data in JSON format."""
        indices, values = self.get_occupied_voxels()
        
        data_dict = {
            'metadata': {
                'size': list(self.size),
                'voxel_size': float(self.voxel_size),
                'center': self.center.tolist(),
                'dtype': str(self.dtype)
            },
            'voxels': [
                {
                    'index': indices[i].tolist(),
                    'value': float(values[i])
                }
                for i in range(len(indices))
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    def _load_json(self, path: Path) -> None:
        """Load from JSON format."""
        with open(path, 'r') as f:
            data_dict = json.load(f)
        
        # Load metadata
        metadata = data_dict['metadata']
        self.size = tuple(metadata['size'])
        self.voxel_size = float(metadata['voxel_size'])
        self.center = torch.tensor(metadata['center'], dtype=torch.float32)
        
        # Initialize data
        self.data = torch.zeros(self.size, dtype=self.dtype)
        
        # Load voxel data
        for voxel_data in data_dict['voxels']:
            idx = tuple(voxel_data['index'])
            value = float(voxel_data['value'])
            self.data[idx] = value
        
        # Recalculate bounds
        self.half_extent = 0.5 * torch.tensor(self.size, dtype=torch.float32) * self.voxel_size
        self.min_bound = self.center - self.half_extent
        self.max_bound = self.center + self.half_extent
    
    def __repr__(self) -> str:
        return (f"StandardVoxelGrid(size={self.size}, voxel_size={self.voxel_size:.3f}, "
                f"center={self.center.tolist()}, occupied_voxels={torch.sum(self.data > 0).item()})")


# Convenience functions
def create_voxel_grid_from_point_cloud(
    points: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    voxel_size: float = 1.0,
    size: Optional[Tuple[int, int, int]] = None
) -> StandardVoxelGrid:
    """Create voxel grid from point cloud."""
    if size is None:
        # Estimate size from point cloud bounds
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        extent = max_coords - min_coords
        size = tuple((extent / voxel_size).ceil().int().tolist())
    
    center = points.mean(dim=0)
    voxel_grid = StandardVoxelGrid(size, voxel_size, center)
    voxel_grid.from_point_cloud(points, values)
    
    return voxel_grid
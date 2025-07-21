"""
I/O plugins for various data formats.

This module provides pluggable I/O interfaces for different file formats
including HDF5, NetCDF, PLY, and standard 3D formats.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import torch
import json
import struct

from interfaces import VoxelGrid
from voxel_grid import StandardVoxelGrid


class IOPlugin(ABC):
    """Abstract base class for I/O plugins."""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return human-readable format name."""
        pass
    
    @abstractmethod
    def can_read(self) -> bool:
        """Return True if plugin supports reading."""
        pass
    
    @abstractmethod
    def can_write(self) -> bool:
        """Return True if plugin supports writing."""
        pass


class VoxelIOPlugin(IOPlugin):
    """Base class for voxel grid I/O plugins."""
    
    @abstractmethod
    def load_voxel_grid(self, path: Path) -> StandardVoxelGrid:
        """Load voxel grid from file."""
        pass
    
    @abstractmethod
    def save_voxel_grid(self, voxel_grid: StandardVoxelGrid, path: Path, **kwargs) -> None:
        """Save voxel grid to file."""
        pass


class PointCloudIOPlugin(IOPlugin):
    """Base class for point cloud I/O plugins."""
    
    @abstractmethod
    def load_point_cloud(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud from file. Returns (points, values)."""
        pass
    
    @abstractmethod
    def save_point_cloud(self, points: torch.Tensor, values: torch.Tensor, path: Path, **kwargs) -> None:
        """Save point cloud to file."""
        pass


class MeshIOPlugin(IOPlugin):
    """Base class for mesh I/O plugins."""
    
    @abstractmethod
    def load_mesh(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load mesh from file. Returns (vertices, faces)."""
        pass
    
    @abstractmethod
    def save_mesh(self, vertices: torch.Tensor, faces: torch.Tensor, path: Path, **kwargs) -> None:
        """Save mesh to file."""
        pass


# HDF5 Plugin
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False


class HDF5VoxelPlugin(VoxelIOPlugin):
    """HDF5 format plugin for voxel grids."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.h5', '.hdf5']
    
    @property
    def format_name(self) -> str:
        return "HDF5"
    
    def can_read(self) -> bool:
        return HAS_HDF5
    
    def can_write(self) -> bool:
        return HAS_HDF5
    
    def load_voxel_grid(self, path: Path) -> StandardVoxelGrid:
        """Load voxel grid from HDF5 file."""
        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 support")
        
        with h5py.File(path, 'r') as f:
            # Load metadata
            size = tuple(f.attrs['size'])
            voxel_size = float(f.attrs['voxel_size'])
            center = torch.tensor(f.attrs['center'], dtype=torch.float32)
            
            # Create voxel grid
            voxel_grid = StandardVoxelGrid(size, voxel_size, center)
            
            # Load data
            data = torch.from_numpy(f['data'][:])
            voxel_grid.set_data(data)
            
        return voxel_grid
    
    def save_voxel_grid(self, voxel_grid: StandardVoxelGrid, path: Path, **kwargs) -> None:
        """Save voxel grid to HDF5 file."""
        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 support")
        
        compression = kwargs.get('compression', 'gzip')
        
        with h5py.File(path, 'w') as f:
            # Save metadata
            f.attrs['size'] = voxel_grid.size
            f.attrs['voxel_size'] = voxel_grid.voxel_size
            f.attrs['center'] = voxel_grid.center.numpy()
            f.attrs['format_version'] = '1.0'
            
            # Save data with compression
            f.create_dataset('data', data=voxel_grid.get_data().numpy(), 
                           compression=compression, chunks=True)


# NetCDF Plugin
try:
    import netCDF4
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False


class NetCDFVoxelPlugin(VoxelIOPlugin):
    """NetCDF format plugin for voxel grids."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.nc', '.netcdf']
    
    @property
    def format_name(self) -> str:
        return "NetCDF"
    
    def can_read(self) -> bool:
        return HAS_NETCDF
    
    def can_write(self) -> bool:
        return HAS_NETCDF
    
    def load_voxel_grid(self, path: Path) -> StandardVoxelGrid:
        """Load voxel grid from NetCDF file."""
        if not HAS_NETCDF:
            raise ImportError("netCDF4 is required for NetCDF support")
        
        with netCDF4.Dataset(path, 'r') as ds:
            # Load metadata
            size = tuple(ds.getncattr('size'))
            voxel_size = float(ds.getncattr('voxel_size'))
            center = torch.tensor(ds.getncattr('center'), dtype=torch.float32)
            
            # Create voxel grid
            voxel_grid = StandardVoxelGrid(size, voxel_size, center)
            
            # Load data
            data = torch.from_numpy(ds.variables['data'][:])
            voxel_grid.set_data(data)
            
        return voxel_grid
    
    def save_voxel_grid(self, voxel_grid: StandardVoxelGrid, path: Path, **kwargs) -> None:
        """Save voxel grid to NetCDF file."""
        if not HAS_NETCDF:
            raise ImportError("netCDF4 is required for NetCDF support")
        
        with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
            # Set metadata
            ds.setncattr('size', voxel_grid.size)
            ds.setncattr('voxel_size', voxel_grid.voxel_size)
            ds.setncattr('center', voxel_grid.center.numpy())
            ds.setncattr('format_version', '1.0')
            
            # Create dimensions
            ds.createDimension('x', voxel_grid.size[0])
            ds.createDimension('y', voxel_grid.size[1])
            ds.createDimension('z', voxel_grid.size[2])
            
            # Create variable
            data_var = ds.createVariable('data', 'f4', ('x', 'y', 'z'), 
                                       zlib=True, complevel=6)
            data_var[:] = voxel_grid.get_data().numpy()
            
            # Add coordinate variables
            x_var = ds.createVariable('x', 'f4', ('x',))
            y_var = ds.createVariable('y', 'f4', ('y',))
            z_var = ds.createVariable('z', 'f4', ('z',))
            
            # Set coordinate values
            coords = voxel_grid.get_voxel_coordinates()
            x_var[:] = coords[:, 0, 0, 0].numpy()
            y_var[:] = coords[0, :, 0, 1].numpy()
            z_var[:] = coords[0, 0, :, 2].numpy()


# PLY Plugin for Point Clouds
class PLYPointCloudPlugin(PointCloudIOPlugin):
    """PLY format plugin for point clouds."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.ply']
    
    @property
    def format_name(self) -> str:
        return "PLY Point Cloud"
    
    def can_read(self) -> bool:
        return True
    
    def can_write(self) -> bool:
        return True
    
    def load_point_cloud(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud from PLY file."""
        points = []
        values = []
        
        with open(path, 'r') as f:
            # Read header
            line = f.readline().strip()
            if line != 'ply':
                raise ValueError("Invalid PLY file")
            
            vertex_count = 0
            header_ended = False
            
            while not header_ended:
                line = f.readline().strip()
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line == 'end_header':
                    header_ended = True
            
            # Read vertices
            for _ in range(vertex_count):
                line = f.readline().strip()
                parts = line.split()
                x, y, z = map(float, parts[:3])
                value = float(parts[3]) if len(parts) > 3 else 1.0
                
                points.append([x, y, z])
                values.append(value)
        
        return torch.tensor(points, dtype=torch.float32), torch.tensor(values, dtype=torch.float32)
    
    def save_point_cloud(self, points: torch.Tensor, values: torch.Tensor, path: Path, **kwargs) -> None:
        """Save point cloud to PLY file."""
        with open(path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float value\n")
            f.write("end_header\n")
            
            # Write vertices
            for i in range(len(points)):
                point = points[i]
                value = values[i]
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {value:.6f}\n")


# XYZ Plugin for Point Clouds
class XYZPointCloudPlugin(PointCloudIOPlugin):
    """XYZ format plugin for point clouds."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.xyz']
    
    @property
    def format_name(self) -> str:
        return "XYZ Point Cloud"
    
    def can_read(self) -> bool:
        return True
    
    def can_write(self) -> bool:
        return True
    
    def load_point_cloud(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud from XYZ file."""
        points = []
        values = []
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                x, y, z = map(float, parts[:3])
                value = float(parts[3]) if len(parts) > 3 else 1.0
                
                points.append([x, y, z])
                values.append(value)
        
        return torch.tensor(points, dtype=torch.float32), torch.tensor(values, dtype=torch.float32)
    
    def save_point_cloud(self, points: torch.Tensor, values: torch.Tensor, path: Path, **kwargs) -> None:
        """Save point cloud to XYZ file."""
        with open(path, 'w') as f:
            f.write("# X Y Z Value\n")
            for i in range(len(points)):
                point = points[i]
                value = values[i]
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {value:.6f}\n")


# OBJ Plugin for Meshes
class OBJMeshPlugin(MeshIOPlugin):
    """OBJ format plugin for meshes."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.obj']
    
    @property
    def format_name(self) -> str:
        return "Wavefront OBJ"
    
    def can_read(self) -> bool:
        return True
    
    def can_write(self) -> bool:
        return True
    
    def load_mesh(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load mesh from OBJ file."""
        vertices = []
        faces = []
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    # Vertex
                    parts = line.split()[1:]
                    vertices.append([float(x) for x in parts[:3]])
                elif line.startswith('f '):
                    # Face (convert from 1-based to 0-based indexing)
                    parts = line.split()[1:]
                    face = []
                    for part in parts:
                        # Handle format like "1/1/1" or "1//1" or just "1"
                        vertex_idx = int(part.split('/')[0]) - 1
                        face.append(vertex_idx)
                    if len(face) >= 3:
                        faces.append(face[:3])  # Take first 3 vertices for triangle
        
        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)
    
    def save_mesh(self, vertices: torch.Tensor, faces: torch.Tensor, path: Path, **kwargs) -> None:
        """Save mesh to OBJ file."""
        with open(path, 'w') as f:
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces (convert to 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# PLY Plugin for Meshes
class PLYMeshPlugin(MeshIOPlugin):
    """PLY format plugin for meshes."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.ply']
    
    @property
    def format_name(self) -> str:
        return "PLY Mesh"
    
    def can_read(self) -> bool:
        return True
    
    def can_write(self) -> bool:
        return True
    
    def load_mesh(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load mesh from PLY file."""
        vertices = []
        faces = []
        
        with open(path, 'r') as f:
            # Read header
            line = f.readline().strip()
            if line != 'ply':
                raise ValueError("Invalid PLY file")
            
            vertex_count = 0
            face_count = 0
            header_ended = False
            
            while not header_ended:
                line = f.readline().strip()
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])
                elif line == 'end_header':
                    header_ended = True
            
            # Read vertices
            for _ in range(vertex_count):
                line = f.readline().strip()
                parts = line.split()
                x, y, z = map(float, parts[:3])
                vertices.append([x, y, z])
            
            # Read faces
            for _ in range(face_count):
                line = f.readline().strip()
                parts = line.split()
                face_size = int(parts[0])
                if face_size >= 3:
                    face = [int(parts[i+1]) for i in range(3)]  # Take first 3 vertices
                    faces.append(face)
        
        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)
    
    def save_mesh(self, vertices: torch.Tensor, faces: torch.Tensor, path: Path, **kwargs) -> None:
        """Save mesh to PLY file."""
        with open(path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# Plugin Manager
class IOPluginManager:
    """Manager for I/O plugins."""
    
    def __init__(self):
        self.voxel_plugins: Dict[str, VoxelIOPlugin] = {}
        self.point_cloud_plugins: Dict[str, PointCloudIOPlugin] = {}
        self.mesh_plugins: Dict[str, MeshIOPlugin] = {}
        
        # Register default plugins
        self._register_default_plugins()
    
    def _register_default_plugins(self):
        """Register default I/O plugins."""
        # Voxel plugins
        if HAS_HDF5:
            self.register_voxel_plugin(HDF5VoxelPlugin())
        if HAS_NETCDF:
            self.register_voxel_plugin(NetCDFVoxelPlugin())
        
        # Point cloud plugins
        self.register_point_cloud_plugin(PLYPointCloudPlugin())
        self.register_point_cloud_plugin(XYZPointCloudPlugin())
        
        # Mesh plugins
        self.register_mesh_plugin(OBJMeshPlugin())
        self.register_mesh_plugin(PLYMeshPlugin())
    
    def register_voxel_plugin(self, plugin: VoxelIOPlugin):
        """Register a voxel I/O plugin."""
        for ext in plugin.supported_extensions:
            self.voxel_plugins[ext] = plugin
    
    def register_point_cloud_plugin(self, plugin: PointCloudIOPlugin):
        """Register a point cloud I/O plugin."""
        for ext in plugin.supported_extensions:
            self.point_cloud_plugins[ext] = plugin
    
    def register_mesh_plugin(self, plugin: MeshIOPlugin):
        """Register a mesh I/O plugin."""
        for ext in plugin.supported_extensions:
            self.mesh_plugins[ext] = plugin
    
    def get_voxel_plugin(self, path: Path) -> Optional[VoxelIOPlugin]:
        """Get voxel plugin for file extension."""
        ext = path.suffix.lower()
        return self.voxel_plugins.get(ext)
    
    def get_point_cloud_plugin(self, path: Path) -> Optional[PointCloudIOPlugin]:
        """Get point cloud plugin for file extension."""
        ext = path.suffix.lower()
        return self.point_cloud_plugins.get(ext)
    
    def get_mesh_plugin(self, path: Path) -> Optional[MeshIOPlugin]:
        """Get mesh plugin for file extension."""
        ext = path.suffix.lower()
        return self.mesh_plugins.get(ext)
    
    def load_voxel_grid(self, path: Union[str, Path]) -> StandardVoxelGrid:
        """Load voxel grid using appropriate plugin."""
        path = Path(path)
        plugin = self.get_voxel_plugin(path)
        
        if plugin is None:
            raise ValueError(f"No plugin available for file extension: {path.suffix}")
        if not plugin.can_read():
            raise ValueError(f"Plugin {plugin.format_name} cannot read files")
        
        return plugin.load_voxel_grid(path)
    
    def save_voxel_grid(self, voxel_grid: StandardVoxelGrid, path: Union[str, Path], **kwargs):
        """Save voxel grid using appropriate plugin."""
        path = Path(path)
        plugin = self.get_voxel_plugin(path)
        
        if plugin is None:
            raise ValueError(f"No plugin available for file extension: {path.suffix}")
        if not plugin.can_write():
            raise ValueError(f"Plugin {plugin.format_name} cannot write files")
        
        plugin.save_voxel_grid(voxel_grid, path, **kwargs)
    
    def list_supported_formats(self) -> Dict[str, List[str]]:
        """List all supported formats."""
        return {
            'voxel_grids': list(self.voxel_plugins.keys()),
            'point_clouds': list(self.point_cloud_plugins.keys()),
            'meshes': list(self.mesh_plugins.keys())
        }


# Global plugin manager instance
_PLUGIN_MANAGER = IOPluginManager()


# Convenience functions
def load_voxel_grid(path: Union[str, Path]) -> StandardVoxelGrid:
    """Load voxel grid from file."""
    return _PLUGIN_MANAGER.load_voxel_grid(path)


def save_voxel_grid(voxel_grid: StandardVoxelGrid, path: Union[str, Path], **kwargs):
    """Save voxel grid to file."""
    _PLUGIN_MANAGER.save_voxel_grid(voxel_grid, path, **kwargs)


def load_point_cloud(path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load point cloud from file."""
    path = Path(path)
    plugin = _PLUGIN_MANAGER.get_point_cloud_plugin(path)
    
    if plugin is None:
        raise ValueError(f"No plugin available for file extension: {path.suffix}")
    
    return plugin.load_point_cloud(path)


def save_point_cloud(points: torch.Tensor, values: torch.Tensor, path: Union[str, Path], **kwargs):
    """Save point cloud to file."""
    path = Path(path)
    plugin = _PLUGIN_MANAGER.get_point_cloud_plugin(path)
    
    if plugin is None:
        raise ValueError(f"No plugin available for file extension: {path.suffix}")
    
    plugin.save_point_cloud(points, values, path, **kwargs)


def load_mesh(path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load mesh from file."""
    path = Path(path)
    plugin = _PLUGIN_MANAGER.get_mesh_plugin(path)
    
    if plugin is None:
        raise ValueError(f"No plugin available for file extension: {path.suffix}")
    
    return plugin.load_mesh(path)


def save_mesh(vertices: torch.Tensor, faces: torch.Tensor, path: Union[str, Path], **kwargs):
    """Save mesh to file."""
    path = Path(path)
    plugin = _PLUGIN_MANAGER.get_mesh_plugin(path)
    
    if plugin is None:
        raise ValueError(f"No plugin available for file extension: {path.suffix}")
    
    plugin.save_mesh(vertices, faces, path, **kwargs)


def list_supported_formats() -> Dict[str, List[str]]:
    """List all supported file formats."""
    return _PLUGIN_MANAGER.list_supported_formats()
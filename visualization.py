"""
Enhanced visualization module with multiple backend support.

This module provides comprehensive visualization capabilities using
PyVista, Matplotlib, and Plotly backends with consistent interfaces.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from pathlib import Path
from abc import ABC, abstractmethod

from interfaces import Renderer, VoxelGrid
from voxel_grid import StandardVoxelGrid
from registry import register_renderer

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@register_renderer("pyvista")
class PyVistaRenderer(Renderer):
    """Enhanced PyVista renderer with advanced visualization options."""
    
    def __init__(self, **kwargs):
        if not HAS_PYVISTA:
            raise ImportError("PyVista is required for PyVistaRenderer")
        
        self.plotter = None
        self.mesh = None
        self.last_render_path = None
        
        # Default rendering options
        self.options = {
            'threshold_percentile': 99.0,
            'colormap': 'viridis',
            'show_axes': True,
            'background_color': 'white',
            'point_size': 2.0,
            'opacity': 1.0,
            'lighting': True,
            'smooth_shading': True,
            'show_edges': False,
            'edge_color': 'black',
            'window_size': (1024, 768),
            'off_screen': False
        }
        self.options.update(kwargs)
    
    def render(self, voxel_grid: VoxelGrid, **kwargs) -> pv.Plotter:
        """Render voxel grid using PyVista."""
        options = self.options.copy()
        options.update(kwargs)
        
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=options['window_size'],
            off_screen=options['off_screen']
        )
        
        # Set background
        self.plotter.background_color = options['background_color']
        
        # Convert voxel grid to point cloud
        points, values = voxel_grid.to_point_cloud()
        
        if len(points) == 0:
            print("Warning: No occupied voxels to render")
            return self.plotter
        
        # Apply threshold
        if options.get('threshold_percentile') is not None:
            threshold = torch.quantile(values, options['threshold_percentile'] / 100.0)
            mask = values >= threshold
            points = points[mask]
            values = values[mask]
        
        # Create point cloud
        point_cloud = pv.PolyData(points.numpy())
        point_cloud['values'] = values.numpy()
        
        # Render based on rendering mode
        render_mode = options.get('render_mode', 'points')
        
        if render_mode == 'points':
            self._render_points(point_cloud, options)
        elif render_mode == 'cubes':
            self._render_cubes(voxel_grid, points, values, options)
        elif render_mode == 'surface':
            self._render_surface(voxel_grid, options)
        elif render_mode == 'volume':
            self._render_volume(voxel_grid, options)
        else:
            self._render_points(point_cloud, options)  # Default
        
        # Add axes if requested
        if options['show_axes']:
            self.plotter.add_axes(
                xlabel='X',
                ylabel='Y',
                zlabel='Z',
                line_width=5,
                labels_off=False
            )
        
        # Set camera
        if 'camera_position' in options:
            self.plotter.camera_position = options['camera_position']
        else:
            self.plotter.show_bounds(
                grid='back',
                location='outer',
                all_edges=True
            )
        
        return self.plotter
    
    def _render_points(self, point_cloud: pv.PolyData, options: Dict):
        """Render as points."""
        self.plotter.add_mesh(
            point_cloud,
            scalars='values',
            cmap=options['colormap'],
            point_size=options['point_size'],
            opacity=options['opacity'],
            render_points_as_spheres=True
        )
    
    def _render_cubes(self, voxel_grid: VoxelGrid, points: torch.Tensor, values: torch.Tensor, options: Dict):
        """Render as individual cubes."""
        # Create cube mesh for each occupied voxel
        cube = pv.Cube()
        
        # Scale cube to voxel size
        if hasattr(voxel_grid, 'voxel_size'):
            cube.scale(voxel_grid.voxel_size * 0.9)  # Slightly smaller for gaps
        
        # Create multi-block dataset
        blocks = pv.MultiBlock()
        
        for i in range(len(points)):
            cube_copy = cube.copy()
            cube_copy.translate(points[i].numpy())
            cube_copy['value'] = values[i].item()
            blocks.append(cube_copy)
        
        # Render
        self.plotter.add_mesh(
            blocks,
            scalars='value',
            cmap=options['colormap'],
            opacity=options['opacity'],
            show_edges=options['show_edges'],
            edge_color=options['edge_color']
        )
    
    def _render_surface(self, voxel_grid: VoxelGrid, options: Dict):
        """Render as extracted surface mesh."""
        if hasattr(voxel_grid, 'extract_mesh'):
            try:
                vertices, faces = voxel_grid.extract_mesh(options.get('surface_threshold', 0.5))
                
                # Create mesh
                mesh_points = vertices.numpy()
                mesh_faces = faces.numpy()
                
                # Create faces array for PyVista (add leading count)
                pv_faces = np.column_stack([
                    np.full(len(mesh_faces), 3),  # Triangle count
                    mesh_faces
                ]).flatten()
                
                surface_mesh = pv.PolyData(mesh_points, pv_faces)
                
                # Add surface mesh
                self.plotter.add_mesh(
                    surface_mesh,
                    color='lightblue',
                    opacity=options['opacity'],
                    smooth_shading=options['smooth_shading'],
                    lighting=options['lighting'],
                    show_edges=options['show_edges'],
                    edge_color=options['edge_color']
                )
                
            except Exception as e:
                print(f"Surface extraction failed: {e}")
                # Fallback to point rendering
                points, values = voxel_grid.to_point_cloud()
                point_cloud = pv.PolyData(points.numpy())
                point_cloud['values'] = values.numpy()
                self._render_points(point_cloud, options)
    
    def _render_volume(self, voxel_grid: VoxelGrid, options: Dict):
        """Render as volumetric data."""
        if hasattr(voxel_grid, 'get_data'):
            # Create structured grid
            data = voxel_grid.get_data().numpy()
            
            # Create coordinate arrays
            nx, ny, nz = data.shape
            x = np.arange(nx)
            y = np.arange(ny) 
            z = np.arange(nz)
            
            # Create structured grid
            grid = pv.StructuredGrid()
            grid.dimensions = [nx, ny, nz]
            
            # Set coordinates
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            grid.points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
            
            # Add scalar data
            grid['voxel_values'] = data.ravel()
            
            # Volume render
            self.plotter.add_volume(
                grid,
                scalars='voxel_values',
                cmap=options['colormap'],
                opacity=options.get('volume_opacity', 'sigmoid')
            )
    
    def save_image(self, output_path: Union[str, Path], **kwargs) -> None:
        """Save rendered image."""
        if self.plotter is None:
            raise RuntimeError("Must call render() before saving image")
        
        self.plotter.screenshot(str(output_path), **kwargs)
        self.last_render_path = Path(output_path)
    
    def show(self, **kwargs):
        """Show interactive visualization."""
        if self.plotter is None:
            raise RuntimeError("Must call render() before showing")
        
        return self.plotter.show(**kwargs)
    
    @property
    def backend(self) -> str:
        return "pyvista"


@register_renderer("matplotlib")
class MatplotlibRenderer(Renderer):
    """Matplotlib renderer for basic 3D visualization."""
    
    def __init__(self, **kwargs):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for MatplotlibRenderer")
        
        self.fig = None
        self.ax = None
        
        # Default options
        self.options = {
            'figure_size': (10, 8),
            'colormap': 'viridis',
            'point_size': 20,
            'alpha': 0.6,
            'show_axes': True,
            'grid': True
        }
        self.options.update(kwargs)
    
    def render(self, voxel_grid: VoxelGrid, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Render voxel grid using Matplotlib."""
        options = self.options.copy()
        options.update(kwargs)
        
        # Create figure
        self.fig = plt.figure(figsize=options['figure_size'])
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Convert to point cloud
        points, values = voxel_grid.to_point_cloud()
        
        if len(points) == 0:
            print("Warning: No occupied voxels to render")
            return self.fig, self.ax
        
        # Plot points
        points_np = points.numpy()
        values_np = values.numpy()
        
        scatter = self.ax.scatter(
            points_np[:, 0],
            points_np[:, 1], 
            points_np[:, 2],
            c=values_np,
            cmap=options['colormap'],
            s=options['point_size'],
            alpha=options['alpha']
        )
        
        # Add colorbar
        self.fig.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=5)
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Configure axes
        if options['show_axes']:
            self.ax.grid(options['grid'])
        
        # Set title
        title = options.get('title', 'Voxel Grid Visualization')
        self.ax.set_title(title)
        
        return self.fig, self.ax
    
    def save_image(self, output_path: Union[str, Path], **kwargs) -> None:
        """Save rendered image."""
        if self.fig is None:
            raise RuntimeError("Must call render() before saving image")
        
        self.fig.savefig(str(output_path), **kwargs)
    
    def show(self, **kwargs):
        """Show interactive visualization."""
        if self.fig is None:
            raise RuntimeError("Must call render() before showing")
        
        plt.show(**kwargs)
    
    @property
    def backend(self) -> str:
        return "matplotlib"


@register_renderer("plotly")
class PlotlyRenderer(Renderer):
    """Plotly renderer for interactive web-based visualization."""
    
    def __init__(self, **kwargs):
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for PlotlyRenderer")
        
        self.fig = None
        
        # Default options
        self.options = {
            'colorscale': 'viridis',
            'marker_size': 2,
            'opacity': 0.8,
            'show_axes': True,
            'title': 'Voxel Grid Visualization',
            'width': 1000,
            'height': 800
        }
        self.options.update(kwargs)
    
    def render(self, voxel_grid: VoxelGrid, **kwargs) -> go.Figure:
        """Render voxel grid using Plotly."""
        options = self.options.copy()
        options.update(kwargs)
        
        # Convert to point cloud
        points, values = voxel_grid.to_point_cloud()
        
        if len(points) == 0:
            print("Warning: No occupied voxels to render")
            self.fig = go.Figure()
            return self.fig
        
        # Create 3D scatter plot
        points_np = points.numpy()
        values_np = values.numpy()
        
        trace = go.Scatter3d(
            x=points_np[:, 0],
            y=points_np[:, 1],
            z=points_np[:, 2],
            mode='markers',
            marker=dict(
                size=options['marker_size'],
                color=values_np,
                colorscale=options['colorscale'],
                opacity=options['opacity'],
                colorbar=dict(title="Value")
            ),
            text=values_np,
            hovertemplate='<b>Position:</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         '<b>Value:</b> %{text:.4f}<extra></extra>'
        )
        
        # Create figure
        self.fig = go.Figure(data=[trace])
        
        # Update layout
        self.fig.update_layout(
            title=options['title'],
            width=options['width'],
            height=options['height'],
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )
        
        if not options['show_axes']:
            self.fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                )
            )
        
        return self.fig
    
    def save_image(self, output_path: Union[str, Path], **kwargs) -> None:
        """Save rendered image."""
        if self.fig is None:
            raise RuntimeError("Must call render() before saving image")
        
        output_path = Path(output_path)
        
        # Determine format from extension
        if output_path.suffix.lower() == '.html':
            self.fig.write_html(str(output_path), **kwargs)
        elif output_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
            self.fig.write_image(str(output_path), **kwargs)
        else:
            # Default to HTML
            self.fig.write_html(str(output_path.with_suffix('.html')), **kwargs)
    
    def show(self, **kwargs):
        """Show interactive visualization."""
        if self.fig is None:
            raise RuntimeError("Must call render() before showing")
        
        self.fig.show(**kwargs)
    
    @property
    def backend(self) -> str:
        return "plotly"


# Visualization Manager
class VisualizationManager:
    """Manager for different visualization backends."""
    
    def __init__(self):
        self.available_backends = self._get_available_backends()
    
    def _get_available_backends(self) -> List[str]:
        """Get list of available visualization backends."""
        backends = []
        if HAS_PYVISTA:
            backends.append('pyvista')
        if HAS_MATPLOTLIB:
            backends.append('matplotlib')
        if HAS_PLOTLY:
            backends.append('plotly')
        return backends
    
    def create_renderer(self, backend: str, **kwargs) -> Renderer:
        """Create renderer for specified backend."""
        if backend not in self.available_backends:
            raise ValueError(f"Backend '{backend}' not available. Available: {self.available_backends}")
        
        if backend == 'pyvista':
            return PyVistaRenderer(**kwargs)
        elif backend == 'matplotlib':
            return MatplotlibRenderer(**kwargs)
        elif backend == 'plotly':
            return PlotlyRenderer(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def render_voxel_grid(
        self,
        voxel_grid: VoxelGrid,
        backend: str = 'pyvista',
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        **kwargs
    ) -> Any:
        """Convenience function to render voxel grid."""
        renderer = self.create_renderer(backend, **kwargs)
        result = renderer.render(voxel_grid, **kwargs)
        
        if save_path:
            renderer.save_image(save_path)
        
        if show:
            renderer.show()
        
        return result
    
    def compare_renderers(
        self,
        voxel_grid: VoxelGrid,
        backends: Optional[List[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Compare different rendering backends."""
        if backends is None:
            backends = self.available_backends
        
        results = {}
        
        for backend in backends:
            if backend not in self.available_backends:
                print(f"Skipping unavailable backend: {backend}")
                continue
            
            try:
                renderer = self.create_renderer(backend, **kwargs)
                result = renderer.render(voxel_grid, **kwargs)
                results[backend] = result
                
                if save_dir:
                    save_path = Path(save_dir) / f"render_{backend}.png"
                    renderer.save_image(save_path)
                
            except Exception as e:
                print(f"Error with backend {backend}: {e}")
                results[backend] = None
        
        return results


# Convenience functions
def create_visualization_manager() -> VisualizationManager:
    """Create visualization manager."""
    return VisualizationManager()


def quick_render(
    voxel_grid: VoxelGrid,
    backend: str = 'pyvista',
    **kwargs
) -> Any:
    """Quick render function for simple visualization."""
    manager = VisualizationManager()
    return manager.render_voxel_grid(voxel_grid, backend=backend, **kwargs)


def create_animation(
    voxel_grids: List[VoxelGrid],
    output_path: Union[str, Path],
    backend: str = 'pyvista',
    fps: int = 10,
    **kwargs
) -> None:
    """Create animation from sequence of voxel grids."""
    if backend != 'pyvista':
        raise NotImplementedError("Animation only supported with PyVista backend")
    
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required for animation")
    
    # Create temporary image files
    temp_dir = Path(output_path).parent / "temp_animation"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        renderer = PyVistaRenderer(off_screen=True, **kwargs)
        
        for i, voxel_grid in enumerate(voxel_grids):
            renderer.render(voxel_grid, **kwargs)
            temp_path = temp_dir / f"frame_{i:06d}.png"
            renderer.save_image(temp_path)
        
        # Create animation using imageio (if available)
        try:
            import imageio
            
            images = []
            for i in range(len(voxel_grids)):
                temp_path = temp_dir / f"frame_{i:06d}.png"
                images.append(imageio.imread(temp_path))
            
            # Save as GIF or MP4
            output_path = Path(output_path)
            if output_path.suffix.lower() == '.gif':
                imageio.mimsave(output_path, images, fps=fps)
            else:
                imageio.mimsave(output_path, images, fps=fps, codec='libx264')
            
        except ImportError:
            print("imageio not available. Individual frames saved in:", temp_dir)
            return
        
    finally:
        # Cleanup temporary files
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
"""
Comprehensive CLI module with advanced argument parsing and validation.

This module provides a feature-rich command-line interface for the
pixeltovoxelprojector with subcommands, configuration management, and validation.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml

from config import (
    PipelineConfig, ConfigManager, create_default_config,
    create_high_quality_config, create_fast_config, create_astronomical_config,
    create_rtsp_config
)
from implementations import ImageSequenceDataSource
from datasets import create_motion_detection_dataloader
from voxel_grid import StandardVoxelGrid
from visualization import VisualizationManager
from io_plugins import list_supported_formats
from registry import list_available


class CLIError(Exception):
    """Base exception for CLI errors."""
    pass


class ArgumentValidator:
    """Validates command-line arguments."""
    
    @staticmethod
    def validate_file_exists(path: str) -> Path:
        """Validate that file exists."""
        file_path = Path(path)
        if not file_path.exists():
            raise argparse.ArgumentTypeError(f"File does not exist: {path}")
        return file_path
    
    @staticmethod
    def validate_directory_exists(path: str) -> Path:
        """Validate that directory exists."""
        dir_path = Path(path)
        if not dir_path.is_dir():
            raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
        return dir_path
    
    @staticmethod
    def validate_output_path(path: str) -> Path:
        """Validate output path (create parent directories if needed)."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    
    @staticmethod
    def validate_positive_int(value: str) -> int:
        """Validate positive integer."""
        try:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(f"Value must be positive: {value}")
            return ivalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer: {value}")
    
    @staticmethod
    def validate_positive_float(value: str) -> float:
        """Validate positive float."""
        try:
            fvalue = float(value)
            if fvalue <= 0:
                raise argparse.ArgumentTypeError(f"Value must be positive: {value}")
            return fvalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float: {value}")
    
    @staticmethod
    def validate_probability(value: str) -> float:
        """Validate probability (0-1)."""
        try:
            fvalue = float(value)
            if not 0.0 <= fvalue <= 1.0:
                raise argparse.ArgumentTypeError(f"Probability must be between 0 and 1: {value}")
            return fvalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float: {value}")
    
    @staticmethod
    def validate_input_source(path: str) -> str:
        """Validate input source (directory, file, or RTSP URL)."""
        # RTSP URL
        if path.lower().startswith(('rtsp://', 'rtmp://')):
            return path
        
        # Check if it's a file
        file_path = Path(path)
        if file_path.is_file():
            return str(file_path)
        
        # Check if it's a directory
        if file_path.is_dir():
            return str(file_path)
        
        raise argparse.ArgumentTypeError(f"Invalid input source: {path} (not a directory, file, or valid stream URL)")


def create_main_parser() -> argparse.ArgumentParser:
    """Create main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="pixeltovoxel",
        description="Pixel to voxel projection pipeline with motion detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing with image folder
  pixeltovoxel process metadata.json images/ output.bin
  
  # Process RTSP stream
  pixeltovoxel process metadata.json rtsp://user:pass@192.168.1.100:554/stream output.bin --max-frames 50
  
  # RTSP with custom camera settings
  pixeltovoxel process metadata.json rtsp://camera.local/stream output.bin --camera-position 10 5 2 --camera-rotation 30 -10 0
  
  # With configuration file
  pixeltovoxel process metadata.json images/ output.bin --config config.yaml
  
  # High quality processing with mesh extraction
  pixeltovoxel process metadata.json images/ output.bin --preset high-quality --extract-mesh
  
  # Visualize existing voxel grid
  pixeltovoxel visualize output.bin --backend pyvista --render-mode surface
  
  # Create configuration template
  pixeltovoxel config create-template config.yaml
        """
    )
    
    # Global options
    parser.add_argument(
        "--version", 
        action="version", 
        version="pixeltovoxelprojector 1.0.0"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Process subcommand
    create_process_parser(subparsers)
    
    # Visualize subcommand
    create_visualize_parser(subparsers)
    
    # Config subcommand
    create_config_parser(subparsers)
    
    # Info subcommand
    create_info_parser(subparsers)
    
    # Convert subcommand
    create_convert_parser(subparsers)
    
    return parser


def create_process_parser(subparsers):
    """Create process subcommand parser."""
    parser = subparsers.add_parser(
        "process",
        help="Process images and generate voxel grid",
        description="Process image sequences or videos to generate 3D voxel grids"
    )
    
    # Required arguments
    parser.add_argument(
        "metadata_path",
        type=ArgumentValidator.validate_file_exists,
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "input_source", 
        type=ArgumentValidator.validate_input_source,
        help="Input source: folder with images, video file, or RTSP URL (rtsp://...)"
    )
    parser.add_argument(
        "output_path",
        type=ArgumentValidator.validate_output_path,
        help="Output voxel grid file path"
    )
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config", "-c",
        type=ArgumentValidator.validate_file_exists,
        help="Configuration file (YAML or JSON)"
    )
    config_group.add_argument(
        "--preset",
        choices=["default", "high-quality", "fast", "astronomical", "rtsp"],
        default="default",
        help="Use preset configuration"
    )
    
    # Grid options
    grid_group = parser.add_argument_group("Grid Parameters")
    grid_group.add_argument(
        "--grid-size",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Grid dimensions (default: 500 500 500)"
    )
    grid_group.add_argument(
        "--voxel-size",
        type=ArgumentValidator.validate_positive_float,
        help="Voxel size in world units (default: 6.0)"
    )
    grid_group.add_argument(
        "--grid-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Grid center coordinates (default: 0 0 500)"
    )
    
    # Motion detection options
    motion_group = parser.add_argument_group("Motion Detection")
    motion_group.add_argument(
        "--motion-algorithm",
        choices=list_available("motion_detector"),
        help="Motion detection algorithm"
    )
    motion_group.add_argument(
        "--motion-threshold",
        type=ArgumentValidator.validate_positive_float,
        help="Motion detection threshold (default: 2.0)"
    )
    motion_group.add_argument(
        "--enhance-motion",
        action="store_true",
        help="Enable motion enhancement preprocessing"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "--batch-size",
        type=ArgumentValidator.validate_positive_int,
        help="Batch size for processing (default: 4)"
    )
    proc_group.add_argument(
        "--num-workers",
        type=int,
        help="Number of worker processes (default: 0)"
    )
    proc_group.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available"
    )
    proc_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU processing"
    )
    
    # RTSP/Stream options
    stream_group = parser.add_argument_group("Stream Processing (RTSP/Video)")
    stream_group.add_argument(
        "--max-frames",
        type=ArgumentValidator.validate_positive_int,
        help="Maximum number of frames to process from stream/video"
    )
    stream_group.add_argument(
        "--rtsp-buffer-size",
        type=ArgumentValidator.validate_positive_int,
        default=1,
        help="RTSP buffer size for low latency (1-3 recommended, default: 1)"
    )
    stream_group.add_argument(
        "--rtsp-timeout",
        type=ArgumentValidator.validate_positive_int,
        default=10000,
        help="RTSP connection timeout in milliseconds (default: 10000)"
    )
    stream_group.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.0],
        help="Camera position for stream processing (default: 0 0 0)"
    )
    stream_group.add_argument(
        "--camera-rotation",
        type=float,
        nargs=3,
        metavar=("YAW", "PITCH", "ROLL"),
        default=[0.0, 0.0, 0.0],
        help="Camera rotation angles in degrees (default: 0 0 0)"
    )
    stream_group.add_argument(
        "--camera-fov",
        type=ArgumentValidator.validate_positive_float,
        default=60.0,
        help="Camera field of view in degrees (default: 60.0)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-format",
        choices=["bin", "npy", "hdf5", "json"],
        default="bin",
        help="Output format (default: bin)"
    )
    output_group.add_argument(
        "--extract-mesh",
        action="store_true",
        help="Extract mesh using Marching Cubes"
    )
    output_group.add_argument(
        "--mesh-output",
        type=ArgumentValidator.validate_output_path,
        help="Mesh output file path"
    )
    output_group.add_argument(
        "--mesh-threshold",
        type=ArgumentValidator.validate_positive_float,
        default=0.5,
        help="Mesh extraction threshold (default: 0.5)"
    )
    output_group.add_argument(
        "--save-config",
        type=ArgumentValidator.validate_output_path,
        help="Save used configuration to file"
    )


def create_visualize_parser(subparsers):
    """Create visualize subcommand parser."""
    parser = subparsers.add_parser(
        "visualize",
        help="Visualize voxel grids",
        description="Visualize voxel grids using various rendering backends"
    )
    
    # Required arguments
    parser.add_argument(
        "voxel_path",
        type=ArgumentValidator.validate_file_exists,
        help="Path to voxel grid file"
    )
    
    # Visualization options
    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument(
        "--backend",
        choices=["pyvista", "matplotlib", "plotly"],
        default="pyvista",
        help="Visualization backend (default: pyvista)"
    )
    vis_group.add_argument(
        "--render-mode",
        choices=["points", "cubes", "surface", "volume"],
        default="points",
        help="Rendering mode (default: points)"
    )
    vis_group.add_argument(
        "--colormap",
        default="viridis",
        help="Color map for visualization (default: viridis)"
    )
    vis_group.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Threshold percentile for display (default: 99.0)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save-image",
        type=ArgumentValidator.validate_output_path,
        help="Save visualization to image file"
    )
    output_group.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show interactive visualization"
    )
    output_group.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[1024, 768],
        help="Window size for visualization (default: 1024 768)"
    )


def create_config_parser(subparsers):
    """Create config subcommand parser."""
    parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        description="Create, validate, and manage configuration files"
    )
    
    config_subparsers = parser.add_subparsers(
        dest="config_action",
        help="Configuration actions"
    )
    
    # Create template
    create_parser = config_subparsers.add_parser(
        "create-template",
        help="Create configuration template"
    )
    create_parser.add_argument(
        "output_path",
        type=ArgumentValidator.validate_output_path,
        help="Output configuration file path"
    )
    create_parser.add_argument(
        "--preset",
        choices=["default", "high-quality", "fast", "astronomical", "rtsp"],
        default="default",
        help="Configuration preset to use"
    )
    
    # Validate config
    validate_parser = config_subparsers.add_parser(
        "validate",
        help="Validate configuration file"
    )
    validate_parser.add_argument(
        "config_path",
        type=ArgumentValidator.validate_file_exists,
        help="Configuration file to validate"
    )
    
    # Show config
    show_parser = config_subparsers.add_parser(
        "show",
        help="Show configuration"
    )
    show_parser.add_argument(
        "config_path",
        type=ArgumentValidator.validate_file_exists,
        help="Configuration file to display"
    )


def create_info_parser(subparsers):
    """Create info subcommand parser."""
    parser = subparsers.add_parser(
        "info",
        help="Show system and format information",
        description="Display information about available algorithms, formats, and system capabilities"
    )
    
    parser.add_argument(
        "--formats",
        action="store_true",
        help="Show supported file formats"
    )
    parser.add_argument(
        "--algorithms",
        action="store_true",
        help="Show available algorithms"
    )
    parser.add_argument(
        "--backends",
        action="store_true",
        help="Show visualization backends"
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Show system information"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all information"
    )


def create_convert_parser(subparsers):
    """Create convert subcommand parser."""
    parser = subparsers.add_parser(
        "convert",
        help="Convert between file formats",
        description="Convert voxel grids, point clouds, and meshes between different formats"
    )
    
    parser.add_argument(
        "input_path",
        type=ArgumentValidator.validate_file_exists,
        help="Input file path"
    )
    parser.add_argument(
        "output_path",
        type=ArgumentValidator.validate_output_path,
        help="Output file path"
    )
    parser.add_argument(
        "--data-type",
        choices=["voxel", "pointcloud", "mesh"],
        help="Data type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Enable compression for supported formats"
    )


def execute_process_command(args: argparse.Namespace) -> None:
    """Execute the process command."""
    try:
        # Load or create configuration
        if args.config:
            config_manager = ConfigManager()
            config = config_manager.load_config(args.config)
        else:
            # Use preset
            if args.preset == "high-quality":
                config = create_high_quality_config()
            elif args.preset == "fast":
                config = create_fast_config()
            elif args.preset == "astronomical":
                config = create_astronomical_config()
            elif args.preset == "rtsp":
                config = create_rtsp_config()
            else:
                config = create_default_config()
        
        # Override configuration with command-line arguments
        if args.grid_size:
            config.grid.size = args.grid_size
        if args.voxel_size:
            config.grid.voxel_size = args.voxel_size
        if args.grid_center:
            config.grid.center = args.grid_center
        if args.motion_algorithm:
            config.motion_detection.algorithm = args.motion_algorithm
        if args.motion_threshold:
            config.motion_detection.threshold = args.motion_threshold
        if args.enhance_motion:
            config.motion_detection.enhance_motion = True
        if args.batch_size:
            config.processing.batch_size = args.batch_size
        if args.num_workers is not None:
            config.processing.num_workers = args.num_workers
        if args.use_gpu:
            config.processing.use_gpu = True
        if args.no_gpu:
            config.processing.use_gpu = False
        if args.output_format:
            config.io.output_format = args.output_format
        if args.extract_mesh:
            config.io.export_mesh = True
        if args.mesh_threshold:
            config.io.mesh_threshold = args.mesh_threshold
        
        # Validate configuration
        config.validate()
        
        # Save configuration if requested
        if args.save_config:
            config_manager = ConfigManager()
            config_manager.save_config(config, args.save_config)
            print(f"Configuration saved to: {args.save_config}")
        
        # Determine input type and process accordingly
        input_source = str(args.input_source)
        
        if input_source.lower().startswith(('rtsp://', 'rtmp://')):
            # RTSP stream processing
            print(f"Processing RTSP stream: {input_source}")
            
            from process_image import process_rtsp_stream_realtime
            
            # Override RTSP config with command-line arguments
            if args.max_frames:
                config.rtsp.max_frames = args.max_frames
            if args.rtsp_buffer_size != 1:  # Only override if different from default
                config.rtsp.buffer_size = args.rtsp_buffer_size
            if args.rtsp_timeout != 10000:  # Only override if different from default
                config.rtsp.timeout_ms = args.rtsp_timeout
            if args.camera_position != [0.0, 0.0, 0.0]:  # Only override if different from default
                config.rtsp.camera_position = args.camera_position
            if args.camera_rotation != [0.0, 0.0, 0.0]:  # Only override if different from default
                config.rtsp.camera_rotation = args.camera_rotation
            if args.camera_fov != 60.0:  # Only override if different from default
                config.rtsp.fov_degrees = args.camera_fov
                
            # Use stream processing
            voxel_grid = process_rtsp_stream_realtime(
                rtsp_url=input_source,
                camera_position=config.rtsp.camera_position,
                yaw=config.rtsp.camera_rotation[0],
                pitch=config.rtsp.camera_rotation[1], 
                roll=config.rtsp.camera_rotation[2],
                fov_degrees=config.rtsp.fov_degrees,
                max_frames=config.rtsp.max_frames or 100,
                N=config.grid.size[0],
                voxel_size=config.grid.voxel_size,
                grid_center=config.grid.center,
                motion_threshold=config.motion_detection.threshold,
                alpha=0.1,
                rtsp_buffer_size=config.rtsp.buffer_size,
                rtsp_timeout=config.rtsp.timeout_ms
            )
            
            # Save voxel grid
            import torch
            import numpy as np
            with open(args.output_path, "wb") as f:
                N = voxel_grid.shape[0]
                voxel_size = config.grid.voxel_size
                f.write(np.array([N], dtype=np.int32).tobytes())
                f.write(np.array([voxel_size], dtype=np.float32).tobytes()) 
                f.write(voxel_grid.numpy().astype(np.float32).tobytes())
                
        elif Path(input_source).is_file() and input_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            # Video file processing
            print(f"Processing video file: {input_source}")
            
            # For video files, we still need metadata for camera information
            # This is a limitation - video files need separate metadata
            from ray_voxel import process_all
            
            # Use existing process_all but note this expects image folder structure
            print("Warning: Video file processing requires adaptation of existing pipeline")
            print("Consider using RTSP stream or converting video to image sequence")
            
        else:
            # Image folder processing (existing functionality)
            print(f"Processing images from {input_source} using {args.metadata_path}")
            
            # Import processing functions
            from ray_voxel import process_all
            
            # Execute processing
            process_all(
                str(args.metadata_path),
                input_source,
                str(args.output_path),
                use_mcubes=config.io.export_mesh if hasattr(config.io, 'export_mesh') else False,
                output_mesh=str(args.mesh_output) if args.mesh_output else None,
                config=config
            )
        
        print(f"Voxel grid saved to: {args.output_path}")
        
    except Exception as e:
        raise CLIError(f"Processing failed: {e}")


def execute_visualize_command(args: argparse.Namespace) -> None:
    """Execute the visualize command."""
    try:
        from io_plugins import load_voxel_grid
        from visualization import VisualizationManager
        
        # Load voxel grid
        print(f"Loading voxel grid from: {args.voxel_path}")
        voxel_grid = load_voxel_grid(args.voxel_path)
        
        # Create visualization manager
        vis_manager = VisualizationManager()
        
        # Set visualization options
        vis_options = {
            'render_mode': args.render_mode,
            'colormap': args.colormap,
            'threshold_percentile': args.threshold_percentile,
            'window_size': tuple(args.window_size)
        }
        
        # Render
        print(f"Rendering with {args.backend} backend...")
        vis_manager.render_voxel_grid(
            voxel_grid,
            backend=args.backend,
            save_path=args.save_image,
            show=not args.no_show,
            **vis_options
        )
        
        if args.save_image:
            print(f"Image saved to: {args.save_image}")
        
    except Exception as e:
        raise CLIError(f"Visualization failed: {e}")


def execute_config_command(args: argparse.Namespace) -> None:
    """Execute the config command."""
    try:
        config_manager = ConfigManager()
        
        if args.config_action == "create-template":
            # Create configuration template
            if args.preset == "high-quality":
                config = create_high_quality_config()
            elif args.preset == "fast":
                config = create_fast_config()
            elif args.preset == "astronomical":
                config = create_astronomical_config()
            elif args.preset == "rtsp":
                config = create_rtsp_config()
            else:
                config = create_default_config()
            
            config_manager.save_config(config, args.output_path)
            print(f"Configuration template created: {args.output_path}")
            
        elif args.config_action == "validate":
            # Validate configuration
            is_valid = config_manager.validate_config_file(args.config_path)
            if is_valid:
                print(f"Configuration is valid: {args.config_path}")
            else:
                raise CLIError(f"Configuration validation failed: {args.config_path}")
                
        elif args.config_action == "show":
            # Show configuration
            config = config_manager.load_config(args.config_path)
            config_dict = config.to_dict()
            
            if args.config_path.suffix.lower() in ['.yaml', '.yml']:
                print(yaml.dump(config_dict, default_flow_style=False, indent=2))
            else:
                print(json.dumps(config_dict, indent=2))
    
    except Exception as e:
        raise CLIError(f"Configuration operation failed: {e}")


def execute_info_command(args: argparse.Namespace) -> None:
    """Execute the info command."""
    show_all = args.all
    
    if args.formats or show_all:
        print("=== Supported File Formats ===")
        formats = list_supported_formats()
        for category, extensions in formats.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for ext in sorted(extensions):
                print(f"  {ext}")
    
    if args.algorithms or show_all:
        print("\n=== Available Algorithms ===")
        for algo_type in ["motion_detector", "ray_caster", "accumulator"]:
            algorithms = list_available(algo_type)
            print(f"\n{algo_type.replace('_', ' ').title()}:")
            for algo in algorithms:
                print(f"  {algo}")
    
    if args.backends or show_all:
        print("\n=== Visualization Backends ===")
        vis_manager = VisualizationManager()
        for backend in vis_manager.available_backends:
            print(f"  {backend}")
    
    if args.system or show_all:
        print("\n=== System Information ===")
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
        
        # Check optional dependencies
        optional_deps = {
            "Open3D": "open3d",
            "trimesh": "trimesh",
            "PyMCubes": "mcubes",
            "PyVista": "pyvista",
            "Plotly": "plotly",
            "H5PY": "h5py",
            "NetCDF4": "netCDF4"
        }
        
        print("\nOptional Dependencies:")
        for name, module in optional_deps.items():
            try:
                __import__(module)
                print(f"  {name}: ✓ Available")
            except ImportError:
                print(f"  {name}: ✗ Not available")


def execute_convert_command(args: argparse.Namespace) -> None:
    """Execute the convert command."""
    try:
        from io_plugins import load_voxel_grid, save_voxel_grid
        
        print(f"Converting {args.input_path} to {args.output_path}")
        
        # Auto-detect data type if not specified
        if not args.data_type:
            # Simple heuristic based on file extension
            input_ext = args.input_path.suffix.lower()
            if input_ext in ['.bin', '.npy', '.h5', '.hdf5', '.nc']:
                args.data_type = 'voxel'
            elif input_ext in ['.ply', '.xyz']:
                args.data_type = 'pointcloud'
            elif input_ext in ['.obj']:
                args.data_type = 'mesh'
            else:
                args.data_type = 'voxel'  # Default
        
        if args.data_type == 'voxel':
            voxel_grid = load_voxel_grid(args.input_path)
            save_options = {}
            if args.compression:
                save_options['compression'] = 'gzip'
            save_voxel_grid(voxel_grid, args.output_path, **save_options)
        else:
            raise CLIError(f"Conversion for {args.data_type} not yet implemented")
        
        print(f"Conversion completed: {args.output_path}")
        
    except Exception as e:
        raise CLIError(f"Conversion failed: {e}")


def main():
    """Main CLI entry point."""
    parser = create_main_parser()
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Handle global options
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        import logging
        logging.basicConfig(level=logging.ERROR)
    
    try:
        # Execute command
        if args.command == "process":
            execute_process_command(args)
        elif args.command == "visualize":
            execute_visualize_command(args)
        elif args.command == "config":
            execute_config_command(args)
        elif args.command == "info":
            execute_info_command(args)
        elif args.command == "convert":
            execute_convert_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
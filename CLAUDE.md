# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixeltovoxelprojector is a pure Python voxel-based motion detection and 3D visualization system that projects pixel motion from images into a 3D voxel grid. The project uses PyTorch for high-performance computation and supports both image sequence processing and video input, with optional mesh extraction capabilities using PyMCubes.

## Build Commands

### Python Dependencies (using uv)
```bash
uv sync
```

### Alternative pip installation
```bash
pip install -e .
```

### Legacy C++ Extensions (if needed)
```bash
python setup.py build_ext --inplace
```

### Legacy Standalone C++ Ray Tracer (if needed)
```bash
g++ -std=c++17 -O2 ray_voxel.cpp -o ray_voxel
```

Or use the provided batch file:
```bash
./examplebuildvoxelgridfrommotion.bat
```

## Core Architecture

### Primary Python Components
- **main.py**: Command-line entry point with argument parsing for metadata processing, voxel grid generation, and optional mesh extraction
- **ray_voxel.py**: Main pipeline for processing image metadata, loading images, detecting motion, casting rays using DDA, and accumulating results in a voxel grid. Includes PyMCubes integration for mesh extraction.
- **process_image.py**: Enhanced core logic with type hints, vectorized operations, OpenCV/PIL image loading options, video processing capabilities, and optimized DDA ray traversal algorithms using PyTorch.

### Visualization Components
- **voxelmotionviewer.py**: PyVista-based 3D visualization tool for interactive viewing of voxel grids with rotation history
- **spacevoxelviewer.py**: Specialized astronomical processor that handles FITS files, calculates celestial coordinates, and builds voxel grids from space-based observations

### Legacy C++ Components (maintained for reference)
- **process_image.cpp**: Legacy pybind11 module providing ray-AABB intersection and voxel traversal algorithms
- **ray_voxel.cpp**: Legacy standalone motion detection system

### Data Flow
1. Input processing: Image sequences, video files, or FITS format for astronomy
2. Motion detection between consecutive frames using PyTorch tensor operations
3. Vectorized ray casting from pixel positions into 3D voxel space using DDA algorithm
4. Accumulation of brightness values in voxel grid with distance-based attenuation
5. Binary voxel grid output with optional mesh extraction using Marching Cubes
6. 3D visualization and analysis

### Key Dependencies
- **Core**: torch, numpy, pillow, opencv-python, pymcubes
- **Visualization**: pyvista, matplotlib  
- **Astronomy**: astropy
- **Web/API**: fastapi, gradio, mcp
- **Media**: ffmpeg-python
- **Legacy C++**: pybind11, nlohmann/json, stb_image (if using C++ components)

### Voxel Grid Configuration
The voxel grid parameters are configurable:
- **ray_voxel.py**: Grid size 500x500x500 voxels, voxel size 6.0, grid center [0,0,500]
- **spacevoxelviewer.py**: Grid size 400x400x400 voxels, spatial extent 3e12 meters half-width

### Build System
- **Primary**: Uses pyproject.toml with uv for dependency management
- **Legacy**: setuptools with custom BuildExt class for C++ compilation

## Usage

### Command Line Interface
The project now includes a comprehensive CLI with multiple subcommands:

```bash
# Basic processing
python main.py process metadata.json images_folder output.bin

# With configuration file
python main.py process metadata.json images_folder output.bin --config config.yaml

# High quality processing with mesh extraction
python main.py process metadata.json images_folder output.bin --preset high-quality --extract-mesh

# Visualize existing voxel grid
python main.py visualize output.bin --backend pyvista --render-mode surface

# Create configuration template
python main.py config create-template config.yaml

# Show system information
python main.py info --all
```

### Modular Architecture Features
- **Abstract interfaces**: Pluggable algorithms for motion detection, ray casting, and accumulation
- **Registry system**: Dynamic algorithm selection and configuration
- **PyTorch Dataset integration**: Efficient batch processing with DataLoader support
- **Transform pipeline**: Chainable transforms compatible with torchvision and scikit-image
- **Multi-format I/O**: Support for HDF5, NetCDF, PLY, OBJ, and other standard formats
- **Library integration**: Compatible with Open3D, trimesh, and PyVista ecosystems
- **Configuration management**: YAML/JSON configuration with validation and presets
- **Multi-backend visualization**: PyVista, Matplotlib, and Plotly rendering options

### Processing Capabilities
- **Image sequences**: Process frame-by-frame motion detection from metadata JSON
- **Video files**: Direct video processing with OpenCV
- **Batch processing**: Vectorized operations with configurable batch sizes
- **Mesh extraction**: Marching Cubes mesh generation with multiple output formats
- **Point cloud processing**: Convert between voxel grids and point clouds
- **Multi-format output**: Binary, NumPy, HDF5, JSON, and compressed formats
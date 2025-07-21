# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixeltovoxelprojector is a pure Python voxel-based motion detection and 3D visualization system that projects pixel motion from images into a 3D voxel grid. The project has been refactored to use PyTorch for high-performance computation, eliminating the need for C++ extensions.

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
- **ray_voxel.py**: Main pipeline for processing image metadata, loading images, detecting motion, casting rays using DDA, and accumulating results in a voxel grid. Outputs binary voxel grid files.
- **process_image.py**: Refactored version containing core logic for image processing, motion detection, and DDA ray traversal algorithms using PyTorch.
- **main.py**: Entry point for the application

### Visualization Components
- **voxelmotionviewer.py**: PyVista-based 3D visualization tool for interactive viewing of voxel grids with rotation history
- **spacevoxelviewer.py**: Specialized astronomical processor that handles FITS files, calculates celestial coordinates, and builds voxel grids from space-based observations

### Legacy C++ Components (maintained for reference)
- **process_image.cpp**: Legacy pybind11 module providing ray-AABB intersection and voxel traversal algorithms
- **ray_voxel.cpp**: Legacy standalone motion detection system

### Data Flow
1. Input images (standard formats for motion detection, FITS format for astronomy)
2. Motion detection between consecutive frames (PyTorch)
3. Ray casting from pixel positions into 3D voxel space using DDA algorithm
4. Accumulation of brightness values in voxel grid
5. Binary output and 3D visualization

### Key Dependencies
- **Python**: torch, numpy, pillow, pyvista, astropy, matplotlib
- **Legacy C++**: pybind11, nlohmann/json, stb_image (if using C++ components)

### Voxel Grid Configuration
The voxel grid parameters are configurable:
- **ray_voxel.py**: Grid size 500x500x500 voxels, voxel size 6.0, grid center [0,0,500]
- **spacevoxelviewer.py**: Grid size 400x400x400 voxels, spatial extent 3e12 meters half-width

### Build System
- **Primary**: Uses pyproject.toml with uv for dependency management
- **Legacy**: setuptools with custom BuildExt class for C++ compilation
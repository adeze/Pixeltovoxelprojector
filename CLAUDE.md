# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixeltovoxelprojector is a voxel-based motion detection and 3D visualization system that projects pixel motion from images into a 3D voxel grid. The project combines C++ for high-performance ray casting and Python for astronomical calculations and visualization.

## Build Commands

### C++ Extensions (Python bindings)
```bash
python setup.py build_ext --inplace
```

### Standalone C++ Ray Tracer
```bash
g++ -std=c++17 -O2 ray_voxel.cpp -o ray_voxel
```

Or use the provided batch file:
```bash
./examplebuildvoxelgridfrommotion.bat
```

## Core Architecture

### C++ Components
- **process_image.cpp**: Pybind11 module providing ray-AABB intersection and voxel traversal algorithms for Python
- **ray_voxel.cpp**: Standalone motion detection system that processes image sequences, detects motion between frames, and accumulates results in a 3D voxel grid

### Python Components
- **spacevoxelviewer.py**: Main astronomical processor that handles FITS files, calculates celestial coordinates, and builds voxel grids from space-based observations
- **voxelmotionviewer.py**: PyVista-based 3D visualization tool for interactive viewing of voxel grids with rotation history

### Data Flow
1. Input images (FITS format for astronomy, standard formats for motion detection)
2. Motion detection between consecutive frames (C++)
3. Ray casting from pixel positions into 3D voxel space
4. Accumulation of brightness values in voxel grid
5. Visualization and analysis of resulting 3D data

### Key Dependencies
- **C++**: pybind11, nlohmann/json, stb_image
- **Python**: astropy, numpy, matplotlib, pyvista

### Voxel Grid Configuration
The voxel grid parameters are configurable in spacevoxelviewer.py:
- Grid size: 400x400x400 voxels (default)
- Spatial extent: 3e12 meters half-width
- Ray casting: 20,000 steps with configurable max distance

### Build System
Uses setuptools with custom BuildExt class to handle OpenMP compilation flags across different compilers (MSVC/Unix).
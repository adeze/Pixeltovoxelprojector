# Pixeltovoxelprojector

## Overview

Pixeltovoxelprojector is a pure Python pipeline for projecting pixel motion from images into a 3D voxel grid, enabling volumetric reconstruction and visualization. The project uses PyTorch, NumPy, Pillow, PyVista, and Astropy for efficient processing and visualization.

## Main Components

- **ray_voxel.py**: Main pipeline for processing image metadata, loading images, detecting motion, casting rays (DDA), and accumulating results in a voxel grid. Outputs a binary voxel grid file for visualization.

- **process_image.py**: Contains core logic for image processing, motion detection, and DDA ray traversal. Used by `ray_voxel.py` for efficient grid updates.

- **voxelmotionviewer.py**: Loads the binary voxel grid and visualizes the top percentile of bright voxels interactively using PyVista. Supports rotation, zoom, and screenshot history.

- **spacevoxelviewer.py**: Specialized viewer for astronomical data. Loads FITS files, computes Earth/telescope positions, projects image data into a spatial voxel grid, and visualizes results in 3D using Matplotlib.

- **setup.py**: Installs Python dependencies for the pipeline. No C++ or pybind11 required; all processing is pure Python.

## Typical Workflow

1. Prepare a metadata JSON and a folder of images.
2. Run `ray_voxel.py` to process images and generate a voxel grid binary file.
3. Visualize results with `voxelmotionviewer.py` (for general 3D data) or `spacevoxelviewer.py` (for astronomical/FITS data).

## Extensibility

- The pipeline is modular and can be refactored for batch processing, plugin support, or integration with other Python frameworks.
- All core logic is in Python, making it easy to extend or optimize.

## Requirements

- Python 3.8+
- torch
- numpy
- pillow
- pyvista
- astropy
- matplotlib

## Getting Started

Install dependencies:

```bash
pip install torch numpy pillow pyvista astropy matplotlib
```

Run the main pipeline:

```bash
python ray_voxel.py
```

Visualize results:

```bash
python voxelmotionviewer.py
```

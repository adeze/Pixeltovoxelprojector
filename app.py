import os
import shutil
import tempfile
import zipfile

import gradio as gr
import numpy as np
import torch
from PIL import Image

# Import the main pipeline from ray_voxel.py
from ray_voxel import process_all


def run_pipeline(metadata_json, images_zip):
    # Create a temp directory for images
    with tempfile.TemporaryDirectory() as tmpdir:
        images_folder = os.path.join(tmpdir, "images")
        os.makedirs(images_folder, exist_ok=True)
        # Save metadata
        metadata_path = os.path.join(tmpdir, "metadata.json")
        with open(metadata_path, "wb") as f:
            f.write(metadata_json.read())
        # Unzip images
        with zipfile.ZipFile(images_zip, "r") as zip_ref:
            zip_ref.extractall(images_folder)
        # Output voxel grid file
        output_bin = os.path.join(tmpdir, "output_voxel.bin")
        # Run pipeline
        process_all(metadata_path, images_folder, output_bin)
        # Load voxel grid for summary
        with open(output_bin, "rb") as f:
            N = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            voxel_size = float(np.frombuffer(f.read(4), dtype=np.float32)[0])
            data = np.frombuffer(f.read(), dtype=np.float32)
            voxel_grid = data.reshape((N, N, N))
        max_val = float(voxel_grid.max())
        num_nonzero = int(np.count_nonzero(voxel_grid))
        summary = f"Voxel grid shape: {voxel_grid.shape}\nVoxel size: {voxel_size}\nMax value: {max_val}\nNonzero voxels: {num_nonzero}"
        # Optionally, create a 2D slice image for preview
        z_slice = voxel_grid[:, :, N // 2]
        img = (
            Image.fromarray((255 * (z_slice / z_slice.max())).astype(np.uint8))
            if z_slice.max() > 0
            else None
        )
        return summary, img


demo = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.File(label="Metadata JSON"),
        gr.File(label="Images ZIP (grayscale images)"),
    ],
    outputs=[
        gr.Textbox(label="Voxel Grid Summary"),
        gr.Image(label="Voxel Grid Slice (Z plane)"),
    ],
    title="Pixel-to-Voxel Projector",
    description="Upload a metadata JSON and a ZIP of images to run the pixel-to-voxel pipeline. View a summary and a slice of the output voxel grid.",
)

if __name__ == "__main__":
    demo.launch()

"""
Declarative Gradio Web Interface for Pixeltovoxelprojector.

This module provides a clean, modular, and declarative web interface
that separates UI definition from event handling logic.
"""

import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pydantic
import torch
import yaml
from PIL import Image

# Import modular components
from config import (
    ConfigManager,
    PipelineConfig,
    create_astronomical_config,
    create_default_config,
    create_fast_config,
    create_high_quality_config,
)
from io_plugins import list_supported_formats, load_voxel_grid, save_voxel_grid
from ray_voxel import FrameInfo, process_all, process_video_stream
from registry import list_available
from visualization import VisualizationManager

# --- Constants and Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(module)s] - %(message)s"
)
CONFIGS_DIR = "configs"
MAX_CAMERAS = 8
FIELDS_PER_CAMERA = 8  # enabled, pos_x, pos_y, pos_z, yaw, pitch, roll, fov

# --- Handler Classes (Business Logic) ---


class WebAppState:
    """Centralized state management for the web application."""

    def __init__(self):
        logging.info("Initializing WebAppState...")
        self.vis_manager = VisualizationManager()
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        logging.info(f"Temp dir: {self.temp_dir}, Configs dir: {CONFIGS_DIR}")

    def cleanup(self):
        logging.info(f"Cleaning up temp directory: {self.temp_dir}")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def get_preset_config(self, preset_name: str) -> PipelineConfig:
        presets = {
            "High Quality": create_high_quality_config,
            "Fast Processing": create_fast_config,
            "Astronomical": create_astronomical_config,
            "Default": create_default_config,
        }
        return presets.get(preset_name, create_default_config)()


class MetadataHandler:
    """Handles metadata generation, saving, and loading."""

    def _get_camera_data_from_inputs(self, num_cameras, *args) -> List[Dict]:
        metadata, camera_inputs = [], args[3:]
        for i in range(int(num_cameras)):
            cam_data = camera_inputs[
                i * FIELDS_PER_CAMERA : (i + 1) * FIELDS_PER_CAMERA
            ]
            if not cam_data[0]:
                continue
            metadata.append(
                {
                    "camera_index": i,
                    "frame_index": 0,
                    "camera_position": [
                        float(cam_data[1]),
                        float(cam_data[2]),
                        float(cam_data[3]),
                    ],
                    "yaw": float(cam_data[4]),
                    "pitch": float(cam_data[5]),
                    "roll": float(cam_data[6]),
                    "fov_degrees": float(cam_data[7]),
                    "image_file": f"camera_{i}_frame_0000.png",
                }
            )
        return metadata

    def generate_metadata(self, *args) -> Tuple[str, str, str]:
        logging.info("Generating 'metadata.json'.")
        try:
            metadata = self._get_camera_data_from_inputs(args[0], *args)
            if not metadata:
                return ["❌ Error: No cameras enabled.", "", gr.update()]

            output_path = "metadata.json"
            with open(output_path, "w") as f:
                json.dump(metadata, f, indent=4)

            success_msg = f"✅ Saved to {output_path} and loaded into Processing tab."
            logging.info(success_msg)
            return success_msg, json.dumps(metadata, indent=4), output_path
        except Exception as e:
            logging.error(f"Metadata generation failed: {e}", exc_info=True)
            return f"❌ Error: {e}", "", gr.update()

    def save_configuration(
        self, config_name, *args
    ) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
        logging.info(f"Saving configuration: '{config_name}'")
        try:
            if not config_name or not config_name.strip():
                raise ValueError("Config name is empty.")
            metadata = self._get_camera_data_from_inputs(args[0], *args)
            if not metadata:
                raise ValueError("No cameras enabled.")
            filename = f"{config_name.strip().replace(' ', '_')}.json"
            output_path = os.path.join(CONFIGS_DIR, filename)
            with open(output_path, "w") as f:
                json.dump(metadata, f, indent=4)

            updated_choices = gr.update(choices=self.list_saved_configs())
            return f"✅ Saved to {output_path}", updated_choices, updated_choices
        except Exception as e:
            logging.error(f"Save failed: {e}", exc_info=True)
            return f"❌ Error: {e}", gr.update(), gr.update()

    def load_configuration(self, config_name: str):
        logging.info(f"Loading configuration: '{config_name}'")
        try:
            if not config_name:
                raise ValueError("Config name not provided.")
            with open(os.path.join(CONFIGS_DIR, config_name), "r") as f:
                metadata = json.load(f)
            num_cams = len(metadata)
            updates = [num_cams, Path(config_name).stem, gr.update()]
            cam_data = []
            for i in range(MAX_CAMERAS):
                if i < num_cams:
                    info, pos = metadata[i], metadata[i]["camera_position"]
                    cam_data.extend(
                        [
                            True,
                            pos[0],
                            pos[1],
                            pos[2],
                            info["yaw"],
                            info["pitch"],
                            info["roll"],
                            info["fov_degrees"],
                        ]
                    )
                else:
                    cam_data.extend([False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0])
            updates.extend(cam_data)
            return tuple(updates + self.update_camera_visibility(num_cams))
        except Exception as e:
            logging.error(f"Load failed: {e}", exc_info=True)
            return tuple(
                [gr.update() for _ in range(3 + MAX_CAMERAS * (FIELDS_PER_CAMERA + 1))]
            )

    def load_config_to_processing_tab(self, config_name: str) -> str:
        """Returns the full path to a saved config file for the File input."""
        if not config_name:
            return gr.update()
        path = os.path.join(CONFIGS_DIR, config_name)
        logging.info(f"Loading '{path}' into processing tab file input.")
        return path

    def list_saved_configs(self) -> List[str]:
        return [f for f in os.listdir(CONFIGS_DIR) if f.endswith(".json")]

    def update_camera_visibility(self, num_cameras_str: str) -> Tuple:
        num_cameras = int(num_cameras_str)
        return tuple(gr.update(visible=i < num_cameras) for i in range(MAX_CAMERAS))


class ProcessingHandler:
    """Handles the main processing pipeline for both file and URL inputs."""

    def __init__(self, app_state: WebAppState):
        self.app_state = app_state

    def run_pipeline_stream(self, *args):
        logging.info("--- Starting Processing Pipeline (Streaming) ---")
        (
            input_source,
            metadata_file,
            images_file,
            camera_url,
            duration,
            live_pos_x,
            live_pos_y,
            live_pos_z,
            live_yaw,
            live_pitch,
            live_roll,
            live_fov,
            preset_name,
            grid_x,
            grid_y,
            grid_z,
            voxel_size,
            motion_algo,
            motion_thresh,
            extract_mesh,
            mesh_thresh,
            output_format,
        ) = args
        run_dir = self._setup_run_directory()
        config = self._create_config(
            preset_name,
            (grid_x, grid_y, grid_z),
            voxel_size,
            motion_thresh,
            motion_algo,
            extract_mesh,
            mesh_thresh,
            output_format,
        )
        output_path = os.path.join(run_dir, f"output.{config.io.output_format}")
        mesh_path = os.path.join(run_dir, "mesh.obj") if extract_mesh else None

        try:
            if input_source == "File Upload":
                if not metadata_file or not images_file:
                    raise ValueError("Metadata/images files required.")
                metadata_path, images_folder = self._extract_files(
                    metadata_file, images_file, run_dir
                )
                yield "Processing files...", None, None, None, None, gr.State(
                    value=None
                )
                process_all(
                    metadata_path,
                    images_folder,
                    output_path,
                    use_mcubes=extract_mesh,
                    output_mesh=mesh_path,
                    config=config,
                )

            elif input_source == "Live URL":
                if not camera_url:
                    raise ValueError("Camera URL required.")
                cam_info = FrameInfo(
                    {
                        "camera_position": [live_pos_x, live_pos_y, live_pos_z],
                        "yaw": live_yaw,
                        "pitch": live_pitch,
                        "roll": live_roll,
                        "fov_degrees": live_fov,
                    }
                )
                for result in process_video_stream(
                    camera_url,
                    cam_info,
                    config,
                    output_path,
                    duration,
                    use_mcubes=extract_mesh,
                    output_mesh=mesh_path,
                ):
                    motion_frame = result.get("motion_frame")
                    if motion_frame is not None and motion_frame.max() > 0:
                        motion_frame = (255 * motion_frame / motion_frame.max()).astype(
                            np.uint8
                        )
                    yield "Processing stream...", result.get(
                        "live_frame"
                    ), motion_frame, None, None, gr.State(value=None)

            logging.info("--- Pipeline Completed Successfully ---")
            summary, _, final_voxel_path, final_mesh_path = self._generate_results(
                output_path, mesh_path
            )
            yield summary, None, None, final_voxel_path, final_mesh_path, gr.State(
                value=final_mesh_path
            )

        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            yield f"❌ Error: {e}", None, None, None, None, gr.State(value=None)

    def _setup_run_directory(self) -> str:
        run_dir = os.path.join(self.app_state.temp_dir, "current_run")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _extract_files(self, md_file, img_file, run_dir):
        md_path = os.path.join(run_dir, "metadata.json")
        img_folder = os.path.join(run_dir, "images")
        os.makedirs(img_folder, exist_ok=True)
        shutil.copy2(md_file.name, md_path)
        with zipfile.ZipFile(img_file.name, "r") as z:
            z.extractall(img_folder)
        return md_path, img_folder

    def _create_config(
        self, preset, grid_size, vs, mt, ma, em, mth, of
    ) -> PipelineConfig:
        config = self.app_state.get_preset_config(preset)
        config.grid.size, config.grid.voxel_size = list(grid_size), vs
        config.motion_detection.threshold, config.motion_detection.algorithm = (
            mt,
            ma.lower().replace(" ", "_"),
        )
        config.io.export_mesh, config.io.mesh_threshold, config.io.output_format = (
            em,
            mth,
            of.lower(),
        )
        config.validate()
        return config

    def _generate_results(self, output_path, mesh_path):
        voxel_grid = load_voxel_grid(output_path)
        self.app_state.current_voxel_grid = voxel_grid
        data = voxel_grid.get_data()
        summary = f"✅ Complete! Occupied Voxels: {torch.sum(data > 0).item():,}"
        preview = self._create_preview_image(data, voxel_grid.size)
        return summary, preview, output_path, mesh_path

    def _create_preview_image(self, data, grid_size):
        if data.numel() == 0:
            return None
        slice_data = data[:, :, grid_size[2] // 2].numpy()
        if slice_data.max() > 0:
            norm = (255 * slice_data / slice_data.max()).astype(np.uint8)
            return Image.fromarray(norm, mode="L").resize(
                (512, 512), Image.Resampling.NEAREST
            )
        return None


# --- UI Builder Class ---


class WebApp:
    """Declarative UI builder for the Gradio application."""

    def __init__(self):
        self.app_state = WebAppState()
        self.metadata_handler = MetadataHandler()
        self.processing_handler = ProcessingHandler(self.app_state)
        self.components = {}

    def build(self) -> gr.Blocks:
        """Constructs the entire Gradio UI."""
        with gr.Blocks(title="Pixeltovoxelprojector", theme=gr.themes.Soft()) as demo:
            self.components["shared_mesh_path"] = gr.State(None)
            gr.Markdown(
                "# 🎯 Pixeltovoxelprojector\n*3D Voxel Grid Generation from Images or Live Streams*"
            )

            with gr.Tabs():
                self._build_processing_tab()
                self._build_visualization_tab()
                self._build_metadata_tab()

            self._bind_events()
            demo.cleanup_fn = self.app_state.cleanup
        return demo

    def _build_processing_tab(self):
        with gr.TabItem("🚀 Processing") as tab:
            gr.Markdown("## Process Images or Live Stream to Voxel Grid")
            with gr.Row():
                with gr.Column(scale=2):
                    self.components["live_video_preview"] = gr.Image(
                        label="Live Feed Preview", interactive=False
                    )
                    self.components["motion_preview"] = gr.Image(
                        label="Detected Motion", interactive=False
                    )
                with gr.Column(scale=1):
                    self.components["preset"] = gr.Dropdown(
                        ["Default", "High Quality", "Fast Processing", "Astronomical"],
                        value="Default",
                        label="Preset",
                    )
                    self.components["input_source"] = gr.Radio(
                        ["File Upload", "Live URL"],
                        label="Input Source",
                        value="File Upload",
                    )
                    self.components["motion_algo"] = gr.Dropdown(
                        ["Frame Difference"],
                        value="Frame Difference",
                        label="Algorithm",
                    )
                    self.components["motion_thresh"] = gr.Slider(
                        0.1, 10.0, 2.0, label="Threshold"
                    )

            with gr.Group(visible=True) as file_upload_group:
                gr.Markdown("### 📁 File-Based Input")
                with gr.Row():
                    with gr.Column():
                        self.components["metadata_file"] = gr.File(
                            label="Metadata JSON", file_types=[".json"]
                        )
                        self.components["images_file"] = gr.File(
                            label="Images ZIP", file_types=[".zip"]
                        )
                    with gr.Column():
                        self.components["proc_saved_configs"] = gr.Dropdown(
                            label="Load Saved Config",
                            choices=self.metadata_handler.list_saved_configs(),
                        )
                        self.components["proc_load_button"] = gr.Button(
                            "📂 Load Config to Input"
                        )

            with gr.Group(visible=False) as live_url_group:
                self.components["camera_url"] = gr.Textbox(
                    label="Camera Stream URL", placeholder="https://..."
                )
                self.components["duration"] = gr.Slider(
                    5, 120, 10, step=5, label="Duration (s)"
                )
                with gr.Accordion("Live Camera Parameters", open=True):
                    (
                        self.components["live_pos_x"],
                        self.components["live_pos_y"],
                        self.components["live_pos_z"],
                    ) = (
                        gr.Number(label="X", value=0),
                        gr.Number(label="Y", value=0),
                        gr.Number(label="Z", value=0),
                    )
                    (
                        self.components["live_yaw"],
                        self.components["live_pitch"],
                        self.components["live_roll"],
                    ) = (
                        gr.Slider(label="Yaw", minimum=-180, maximum=180, value=0),
                        gr.Slider(label="Pitch", minimum=-90, maximum=90, value=0),
                        gr.Slider(label="Roll", minimum=-180, maximum=180, value=0),
                    )
                    self.components["live_fov"] = gr.Slider(
                        label="FOV (deg)", minimum=10, maximum=120, value=60
                    )

            with gr.Row():
                (
                    self.components["grid_x"],
                    self.components["grid_y"],
                    self.components["grid_z"],
                ) = (
                    gr.Slider(50, 1000, 500, step=10, label="Grid X"),
                    gr.Slider(50, 1000, 500, step=10, label="Grid Y"),
                    gr.Slider(50, 1000, 500, step=10, label="Grid Z"),
                )
                self.components["voxel_size"] = gr.Slider(
                    0.1, 50.0, 6.0, label="Voxel Size"
                )
                self.components["extract_mesh"] = gr.Checkbox(
                    label="Extract Mesh for 3D View", value=True
                )
                self.components["mesh_thresh"] = gr.Slider(
                    0.1, 2.0, 0.5, label="Mesh Threshold"
                )
                self.components["output_format"] = gr.Dropdown(
                    ["bin", "npy", "hdf5"], value="bin", label="Voxel Format"
                )

            self.components["process_button"] = gr.Button(
                "🚀 Process", variant="primary"
            )
            self.components["results_text"] = gr.Textbox(
                label="Results", lines=2, interactive=False
            )
            self.components["download_voxel"] = gr.File(
                label="Download Voxel Grid", visible=False
            )
            self.components["download_mesh"] = gr.File(
                label="Download Mesh", visible=False
            )
            self.components["file_upload_group"] = file_upload_group
            self.components["live_url_group"] = live_url_group

    def _build_visualization_tab(self):
        with gr.TabItem("🎨 3D Viewer"):
            gr.Markdown("## Interactive 3D Voxel Mesh Viewer")
            self.components["model_3d_viewer"] = gr.Model3D(
                label="Voxel Grid Mesh", interactive=True
            )
            self.components["refresh_3d_button"] = gr.Button(
                "🔄 Refresh Viewer with Last Result"
            )

    def _build_metadata_tab(self):
        with gr.TabItem("📝 Metadata Generator"):
            gr.Markdown("## Camera Metadata Generator")
            with gr.Row():
                self.components["config_name"] = gr.Textbox(
                    label="Config Name", placeholder="e.g., 'traffic_cam_setup'"
                )
                self.components["meta_saved_configs"] = gr.Dropdown(
                    label="Saved Configs",
                    choices=self.metadata_handler.list_saved_configs(),
                )
                self.components["save_button"] = gr.Button("💾 Save")
                self.components["load_button"] = gr.Button("📂 Load")
            self.components["num_cameras_slider"] = gr.Slider(
                1, MAX_CAMERAS, 2, step=1, label="Number of Cameras"
            )

            (
                self.components["camera_accordions"],
                self.components["all_camera_inputs"],
            ) = ([], [])
            for i in range(MAX_CAMERAS):
                with gr.Accordion(
                    f"Camera {i+1}", open=i < 2, visible=i < 2
                ) as accordion:
                    inputs = [
                        gr.Checkbox(label="Enable", value=i < 2),
                        gr.Number(label="X", value=0),
                        gr.Number(label="Y", value=0),
                        gr.Number(label="Z", value=0),
                        gr.Slider(label="Yaw", minimum=-180, maximum=180, value=0),
                        gr.Slider(label="Pitch", minimum=-90, maximum=90, value=0),
                        gr.Slider(label="Roll", minimum=-180, maximum=180, value=0),
                        gr.Slider(label="FOV", minimum=10, maximum=120, value=60),
                    ]
                    self.components["camera_accordions"].append(accordion)
                    self.components["all_camera_inputs"].extend(inputs)

            self.components["generate_meta_button"] = gr.Button(
                "📝 Generate & Load metadata.json", variant="primary"
            )
            self.components["gen_status"] = gr.Textbox(
                label="Status", interactive=False
            )
            self.components["gen_json_output"] = gr.Code(
                label="Generated JSON", language="json", interactive=False
            )

    def _bind_events(self):
        # Processing Tab Events
        self.components["input_source"].change(
            lambda s: (
                gr.update(visible=s == "File Upload"),
                gr.update(visible=s == "Live URL"),
            ),
            self.components["input_source"],
            [self.components["file_upload_group"], self.components["live_url_group"]],
        )

        proc_inputs = [
            self.components[k]
            for k in [
                "input_source",
                "metadata_file",
                "images_file",
                "camera_url",
                "duration",
                "live_pos_x",
                "live_pos_y",
                "live_pos_z",
                "live_yaw",
                "live_pitch",
                "live_roll",
                "live_fov",
                "preset",
                "grid_x",
                "grid_y",
                "grid_z",
                "voxel_size",
                "motion_algo",
                "motion_thresh",
                "extract_mesh",
                "mesh_thresh",
                "output_format",
            ]
        ]
        proc_outputs = [
            self.components[k]
            for k in [
                "results_text",
                "live_video_preview",
                "motion_preview",
                "download_voxel",
                "download_mesh",
                "shared_mesh_path",
            ]
        ]
        self.components["process_button"].click(
            self.processing_handler.run_pipeline_stream, proc_inputs, proc_outputs
        )

        self.components["download_mesh"].change(
            lambda v, m: (
                gr.update(visible=v is not None),
                gr.update(visible=m is not None),
            ),
            [self.components["download_voxel"], self.components["download_mesh"]],
            [self.components["download_voxel"], self.components["download_mesh"]],
        )

        self.components["proc_load_button"].click(
            self.metadata_handler.load_config_to_processing_tab,
            self.components["proc_saved_configs"],
            self.components["metadata_file"],
        )

        # Visualization Tab Events
        self.components["refresh_3d_button"].click(
            lambda path: path,
            self.components["shared_mesh_path"],
            self.components["model_3d_viewer"],
        )

        # Metadata Tab Events
        self.components["num_cameras_slider"].change(
            self.metadata_handler.update_camera_visibility,
            self.components["num_cameras_slider"],
            self.components["camera_accordions"],
        )

        meta_gen_inputs = [
            self.components["num_cameras_slider"],
            self.components["config_name"],
            self.components["meta_saved_configs"],
        ] + self.components["all_camera_inputs"]
        meta_gen_outputs = [
            self.components["gen_status"],
            self.components["gen_json_output"],
            self.components["metadata_file"],
        ]
        self.components["generate_meta_button"].click(
            self.metadata_handler.generate_metadata, meta_gen_inputs, meta_gen_outputs
        )

        save_outputs = [
            self.components["gen_status"],
            self.components["meta_saved_configs"],
            self.components["proc_saved_configs"],
        ]
        self.components["save_button"].click(
            self.metadata_handler.save_configuration,
            [self.components["config_name"]] + meta_gen_inputs,
            save_outputs,
        )

        load_outputs = (
            [
                self.components["num_cameras_slider"],
                self.components["config_name"],
                self.components["meta_saved_configs"],
            ]
            + self.components["all_camera_inputs"]
            + self.components["camera_accordions"]
        )
        self.components["load_button"].click(
            self.metadata_handler.load_configuration,
            self.components["meta_saved_configs"],
            load_outputs,
        )


# --- Main Execution ---
def main():
    logging.info("Starting Gradio application...")
    web_app = WebApp()
    demo = web_app.build()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, show_error=True)
    logging.info("Gradio application launched.")


if __name__ == "__main__":
    main()

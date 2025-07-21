"""
Simplified Gradio Web Interface for Pixeltovoxelprojector.

This module provides a clean, modular web interface that's easy to extend
and maintain while showcasing the core features.
"""

import os
import shutil
import tempfile
import zipfile
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image

# Import our modular components
from config import (
    ConfigManager, PipelineConfig, create_default_config, 
    create_high_quality_config, create_fast_config, create_astronomical_config
)
from ray_voxel import process_all, process_video_stream, FrameInfo
from io_plugins import load_voxel_grid, save_voxel_grid, list_supported_formats
from visualization import VisualizationManager
from registry import list_available

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s] - %(message)s",
)

CONFIGS_DIR = "configs"
MAX_CAMERAS = 8
FIELDS_PER_CAMERA = 8 # enabled, pos_x, pos_y, pos_z, yaw, pitch, roll, fov

class WebAppState:
    """Centralized state management for the web application."""
    
    def __init__(self):
        logging.info("Initializing WebAppState...")
        self.config_manager = ConfigManager()
        self.vis_manager = VisualizationManager()
        self.current_voxel_grid = None
        self.temp_dir = tempfile.mkdtemp()
        logging.info(f"Temporary directory created at: {self.temp_dir}")
        
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        logging.info(f"Ensured '{CONFIGS_DIR}' directory exists.")
    
    def cleanup(self):
        """Clean up temporary resources."""
        logging.info(f"Cleaning up temporary directory: {self.temp_dir}")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logging.info("Temporary directory removed.")
    
    def get_preset_config(self, preset_name: str) -> PipelineConfig:
        """Get configuration by preset name."""
        logging.info(f"Loading preset config: '{preset_name}'")
        presets = {
            "High Quality": create_high_quality_config,
            "Fast Processing": create_fast_config,
            "Astronomical": create_astronomical_config,
            "Default": create_default_config
        }
        return presets.get(preset_name, create_default_config)()


class MetadataHandler:
    """Handles metadata generation, saving, and loading."""

    def _get_camera_data_from_inputs(self, num_cameras, *args) -> List[Dict]:
        """Helper to extract camera data from flat argument list."""
        metadata = []
        camera_inputs = args[3:] 
        
        for i in range(int(num_cameras)):
            cam_data = camera_inputs[i * FIELDS_PER_CAMERA : (i + 1) * FIELDS_PER_CAMERA]
            if not cam_data[0]: continue # is_enabled

            frame_info = {
                "camera_index": i, "frame_index": 0,
                "camera_position": [float(cam_data[1]), float(cam_data[2]), float(cam_data[3])],
                "yaw": float(cam_data[4]), "pitch": float(cam_data[5]), "roll": float(cam_data[6]),
                "fov_degrees": float(cam_data[7]), "image_file": f"camera_{i}_frame_0000.png"
            }
            metadata.append(frame_info)
        return metadata

    def generate_metadata(self, *args) -> Tuple[str, str]:
        """Generate and save metadata.json from UI inputs."""
        logging.info("Attempting to generate 'metadata.json'.")
        try:
            num_cameras = args[0]
            metadata = self._get_camera_data_from_inputs(num_cameras, *args)
            if not metadata:
                logging.warning("Metadata generation failed: No cameras were enabled.")
                return "❌ Error: No cameras enabled.", ""

            output_path = "metadata.json"
            with open(output_path, "w") as f: json.dump(metadata, f, indent=4)
            success_msg = f"✅ Saved to {output_path} for {len(metadata)} camera(s)."
            logging.info(f"Successfully generated metadata for {len(metadata)} camera(s).")
            return success_msg, json.dumps(metadata, indent=4)
        except Exception as e:
            logging.error(f"Failed to generate metadata: {e}", exc_info=True)
            return f"❌ Failed to generate metadata: {str(e)}", ""

    def save_configuration(self, config_name, *args) -> Tuple[str, gr.Dropdown]:
        """Save the current camera configuration to a named file."""
        logging.info(f"Attempting to save configuration: '{config_name}'")
        try:
            if not config_name or not config_name.strip():
                logging.warning("Save failed: Configuration name is empty.")
                return "❌ Error: Configuration name cannot be empty.", gr.update()
            num_cameras = args[0]
            metadata = self._get_camera_data_from_inputs(num_cameras, *args)
            if not metadata:
                logging.warning("Save failed: No cameras enabled.")
                return "❌ Error: No cameras enabled. Cannot save an empty configuration.", gr.update()

            filename = f"{config_name.strip().replace(' ', '_')}.json"
            output_path = os.path.join(CONFIGS_DIR, filename)
            with open(output_path, "w") as f: json.dump(metadata, f, indent=4)
            logging.info(f"Successfully saved configuration to '{output_path}'.")
            return f"✅ Saved configuration to {output_path}", gr.update(choices=self.list_saved_configs())
        except Exception as e:
            logging.error(f"Failed to save configuration '{config_name}': {e}", exc_info=True)
            return f"❌ Failed to save configuration: {str(e)}", gr.update()

    def load_configuration(self, config_name: str):
        """Load a named configuration and update the UI."""
        logging.info(f"Attempting to load configuration: '{config_name}'")
        try:
            if not config_name: raise ValueError("Configuration name not provided.")
            config_path = os.path.join(CONFIGS_DIR, config_name)
            with open(config_path, 'r') as f: metadata = json.load(f)
            logging.info(f"Successfully loaded '{config_path}'.")

            num_loaded_cameras = len(metadata)
            update_values = [num_loaded_cameras, Path(config_name).stem, gr.update()]
            all_cam_data = []
            for i in range(MAX_CAMERAS):
                if i < num_loaded_cameras:
                    cam_info = metadata[i]
                    pos = cam_info['camera_position']
                    all_cam_data.extend([True, pos[0], pos[1], pos[2], cam_info['yaw'], cam_info['pitch'], cam_info['roll'], cam_info['fov_degrees']])
                else:
                    all_cam_data.extend([False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0])
            update_values.extend(all_cam_data)
            visibility_updates = self.update_camera_visibility(num_loaded_cameras)
            logging.info(f"UI updated for {num_loaded_cameras} cameras from '{config_name}'.")
            return tuple(update_values) + visibility_updates
        except Exception as e:
            logging.error(f"Failed to load configuration '{config_name}': {e}", exc_info=True)
            no_op_updates = [gr.update() for _ in range(3 + MAX_CAMERAS * FIELDS_PER_CAMERA)]
            visibility_updates = [gr.update() for _ in range(MAX_CAMERAS)]
            return tuple(no_op_updates) + tuple(visibility_updates)

    def list_saved_configs(self) -> List[str]:
        """Return a list of saved .json configuration files."""
        logging.info("Listing saved configurations.")
        return [f for f in os.listdir(CONFIGS_DIR) if f.endswith('.json')]

    def update_camera_visibility(self, num_cameras_str: str) -> Tuple:
        """Update visibility of camera accordions based on slider value."""
        try:
            num_cameras = int(num_cameras_str)
            logging.info(f"Updating camera visibility to show {num_cameras} accordions.")
            return tuple(gr.update(visible=i < num_cameras) for i in range(MAX_CAMERAS))
        except (ValueError, TypeError):
            return tuple(gr.update(visible=False) for _ in range(MAX_CAMERAS))


class ProcessingHandler:
    """Handles the main processing pipeline for both file and URL inputs."""
    
    def __init__(self, app_state: WebAppState):
        self.app_state = app_state
    
    def run_pipeline(self, *args):
        """Unified pipeline runner that dispatches based on input source."""
        logging.info("--- Starting Processing Pipeline ---")
        
        # Unpack arguments
        input_source, metadata_file, images_file, camera_url, duration, \
        live_pos_x, live_pos_y, live_pos_z, live_yaw, live_pitch, live_roll, live_fov, \
        preset_name, grid_x, grid_y, grid_z, voxel_size, motion_algo, motion_thresh, \
        extract_mesh, mesh_thresh, output_format = args

        try:
            run_dir = self._setup_run_directory()
            config = self._create_config(preset_name, (grid_x, grid_y, grid_z), voxel_size, motion_thresh, motion_algo, extract_mesh, mesh_thresh, output_format)
            output_path = os.path.join(run_dir, f"output.{config.io.output_format}")
            mesh_path = os.path.join(run_dir, "mesh.obj") if config.io.export_mesh else None

            if input_source == "File Upload":
                logging.info("Processing with 'File Upload' source.")
                if not metadata_file or not images_file:
                    raise ValueError("Metadata and images files are required for file upload.")
                metadata_path, images_folder = self._extract_files(metadata_file, images_file, run_dir)
                process_all(metadata_path, images_folder, output_path, use_mcubes=extract_mesh, output_mesh=mesh_path, config=config)
            
            elif input_source == "Live URL":
                logging.info(f"Processing with 'Live URL' source: {camera_url}")
                if not camera_url: raise ValueError("Camera URL is required for live processing.")
                
                cam_info = FrameInfo({
                    "camera_position": [live_pos_x, live_pos_y, live_pos_z],
                    "yaw": live_yaw, "pitch": live_pitch, "roll": live_roll, "fov_degrees": live_fov
                })
                process_video_stream(camera_url, cam_info, config, output_path, duration, use_mcubes=extract_mesh, output_mesh=mesh_path)

            logging.info("--- Pipeline Completed Successfully ---")
            return self._generate_results(output_path, mesh_path, config)

        except Exception as e:
            logging.error(f"Processing pipeline failed: {e}", exc_info=True)
            return f"❌ Processing failed: {str(e)}", None, None, None

    def _setup_run_directory(self) -> str:
        run_dir = os.path.join(self.app_state.temp_dir, "current_run")
        logging.info(f"Setting up run directory: {run_dir}")
        if os.path.exists(run_dir): shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        return run_dir

    def _extract_files(self, metadata_file, images_file, run_dir: str) -> Tuple[str, str]:
        logging.info("Extracting uploaded files.")
        images_folder = os.path.join(run_dir, "images")
        os.makedirs(images_folder)
        metadata_path = os.path.join(run_dir, "metadata.json")
        shutil.copy2(metadata_file.name, metadata_path)
        if not zipfile.is_zipfile(images_file.name): raise ValueError("Images must be a ZIP file.")
        with zipfile.ZipFile(images_file.name, 'r') as z: z.extractall(images_folder)
        logging.info(f"Files extracted to {run_dir}")
        return metadata_path, images_folder

    def _create_config(self, preset, grid_size, vs, mt, ma, em, mth, of) -> PipelineConfig:
        logging.info(f"Creating config from preset '{preset}' and UI values.")
        config = self.app_state.get_preset_config(preset)
        config.grid.size, config.grid.voxel_size = list(grid_size), vs
        config.motion_detection.threshold, config.motion_detection.algorithm = mt, ma.lower().replace(' ', '_')
        config.io.export_mesh, config.io.mesh_threshold, config.io.output_format = em, mth, of.lower()
        config.validate()
        logging.info("Configuration created and validated.")
        return config

    def _generate_results(self, output_path, mesh_path, config):
        logging.info("Generating results and preview.")
        voxel_grid = load_voxel_grid(output_path)
        self.app_state.current_voxel_grid = voxel_grid
        data = voxel_grid.get_data()
        summary = f"""✅ Processing Complete!
📊 Voxel Grid Stats:
• Size: {voxel_grid.size}, Voxel Size: {voxel_grid.voxel_size:.3f}
• Occupied: {torch.sum(data > 0).item():,}, Max Value: {torch.max(data).item():.6f}
"""
        preview_img = self._create_preview_image(data, voxel_grid.size)
        logging.info("Results summary and preview generated.")
        return summary, preview_img, output_path, mesh_path

    def _create_preview_image(self, data, grid_size):
        if data.numel() == 0: return None
        slice_data = data[:, :, grid_size[2] // 2].numpy()
        if slice_data.max() > 0:
            norm = (255 * slice_data / slice_data.max()).astype(np.uint8)
            return Image.fromarray(norm, mode='L').resize((512, 512), Image.Resampling.NEAREST)
        return None


class ConfigurationHandler:
    """Handles configuration management."""
    
    def __init__(self, app_state: WebAppState): self.app_state = app_state
    def create_template(self, preset_name: str) -> str:
        logging.info(f"Generating YAML template for preset: '{preset_name}'.")
        try:
            config = self.app_state.get_preset_config(preset_name)
            return yaml.dump(config.to_dict(), default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Error creating config template: {e}", exc_info=True)
            return f"Error creating config: {str(e)}"
    
    def validate_config(self, config_text: str) -> str:
        logging.info("Validating YAML configuration.")
        try:
            config = PipelineConfig.from_dict(yaml.safe_load(config_text))
            config.validate()
            logging.info("YAML validation successful.")
            return "✅ Configuration is valid!"
        except Exception as e:
            logging.error(f"YAML validation failed: {e}", exc_info=True)
            return f"❌ Configuration error: {str(e)}"


class VisualizationHandler:
    """Handles visualization creation."""
    
    def __init__(self, app_state: WebAppState): self.app_state = app_state
    def create_visualization(self, backend, render_mode, colormap, thresh, point_size, opacity):
        logging.info(f"Creating viz with backend: {backend}, mode: {render_mode}.")
        if self.app_state.current_voxel_grid is None:
            logging.warning("Viz failed: No voxel grid available.")
            return "❌ No voxel grid available. Process data first.", None
        try:
            vis_path = os.path.join(self.app_state.temp_dir, f"vis_{backend}.png")
            vis_opts = {'render_mode': render_mode.lower(), 'colormap': colormap.lower(), 'threshold_percentile': thresh, 'point_size': point_size, 'opacity': opacity, 'off_screen': True, 'window_size': (1024, 768)}
            self.app_state.vis_manager.render_voxel_grid(self.app_state.current_voxel_grid, backend=backend.lower(), save_path=vis_path, show=False, **vis_opts)
            logging.info(f"Visualization saved to {vis_path}.")
            return f"✅ Visualization created using {backend}!", vis_path
        except Exception as e:
            logging.error(f"Visualization failed: {e}", exc_info=True)
            return f"❌ Visualization failed: {str(e)}", None
    
    def get_voxel_info(self) -> str:
        logging.info("Fetching current voxel grid info.")
        if self.app_state.current_voxel_grid is None: return "No voxel grid loaded"
        vg = self.app_state.current_voxel_grid
        data = vg.get_data()
        return f"📊 Voxel Grid: Size: {vg.size}, Voxel Size: {vg.voxel_size}, Occupancy: {(torch.sum(data > 0).item() / data.numel() * 100):.1f}%"


class UtilityHandler:
    """Handles utility functions."""
    
    def __init__(self, app_state: WebAppState): self.app_state = app_state
    def convert_format(self, input_file, output_format: str):
        logging.info(f"Attempting to convert file to '{output_format}'.")
        if input_file is None:
            logging.warning("Conversion failed: No input file provided.")
            return "❌ No input file provided", None
        try:
            voxel_grid = load_voxel_grid(input_file.name)
            output_path = os.path.join(self.app_state.temp_dir, f"converted.{output_format}")
            save_voxel_grid(voxel_grid, output_path)
            logging.info(f"Successfully converted file to {output_path}.")
            return f"✅ Converted {Path(input_file.name).suffix} to .{output_format}", output_path
        except Exception as e:
            logging.error(f"Conversion failed: {e}", exc_info=True)
            return f"❌ Conversion failed: {str(e)}", None
    
    def get_system_info(self) -> str:
        logging.info("Gathering system information.")
        info = [f"• PyTorch: {torch.__version__}", f"• CUDA: {'Available' if torch.cuda.is_available() else 'Not Available'}"]
        if torch.cuda.is_available(): info.append(f"• GPU Count: {torch.cuda.device_count()}")
        info.append(f"• Motion Detectors: {', '.join(list_available('motion_detector'))}")
        return '\n'.join(info)


def create_metadata_tab(handler: MetadataHandler) -> gr.TabItem:
    with gr.TabItem("📝 Metadata Generator") as tab:
        gr.Markdown("## Camera Metadata Generator\nConfigure, save, and load camera configurations.")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💾 Save/Load Configuration")
                with gr.Row():
                    config_name = gr.Textbox(label="Configuration Name", placeholder="e.g., 'traffic_cam_setup_1'")
                    saved_configs = gr.Dropdown(label="Saved Configurations", choices=handler.list_saved_configs())
                with gr.Row():
                    save_button = gr.Button("💾 Save")
                    load_button = gr.Button("📂 Load")
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ General Settings")
                num_cameras_slider = gr.Slider(label="Number of Cameras", minimum=1, maximum=MAX_CAMERAS, step=1, value=2)
        gr.Markdown("---")
        camera_accordions, all_camera_inputs = [], []
        for i in range(MAX_CAMERAS):
            with gr.Accordion(f"Camera {i+1}", open=i<2, visible=i<2) as accordion:
                enabled = gr.Checkbox(label="Enable Camera", value=i<2)
                with gr.Row():
                    pos_x, pos_y, pos_z = gr.Number(label="X", value=0.0), gr.Number(label="Y", value=0.0), gr.Number(label="Z", value=0.0)
                with gr.Row():
                    yaw, pitch, roll = gr.Slider(label="Yaw", min=-180, max=180, val=0), gr.Slider(label="Pitch", min=-90, max=90, val=0), gr.Slider(label="Roll", min=-180, max=180, val=0)
                fov = gr.Slider(label="FOV (deg)", minimum=10, maximum=120, value=60.0)
                camera_accordions.append(accordion)
                all_camera_inputs.extend([enabled, pos_x, pos_y, pos_z, yaw, pitch, roll, fov])
        gr.Markdown("---")
        generate_button = gr.Button("📝 Generate metadata.json for Processing", variant="primary")
        gen_status = gr.Textbox(label="Status", lines=2, interactive=False)
        gen_json_output = gr.Code(label="Generated JSON", language="json", interactive=False)
        
        num_cameras_slider.change(handler.update_camera_visibility, num_cameras_slider, camera_accordions)
        all_inputs = [num_cameras_slider, config_name, saved_configs] + all_camera_inputs
        generate_button.click(handler.generate_metadata, all_inputs, [gen_status, gen_json_output])
        save_button.click(handler.save_configuration, [config_name] + all_inputs, [gen_status, saved_configs])
        load_button.click(handler.load_configuration, saved_configs, [num_cameras_slider, config_name, saved_configs] + all_camera_inputs + camera_accordions)
    return tab

def create_processing_tab(processor: ProcessingHandler) -> gr.TabItem:
    with gr.TabItem("🚀 Processing") as tab:
        gr.Markdown("## Process Images or Live Stream to Voxel Grid")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ General Configuration")
                preset_dropdown = gr.Dropdown(choices=["Default", "High Quality", "Fast Processing", "Astronomical"], value="Default", label="Preset")
                input_source = gr.Radio(["File Upload", "Live URL"], label="Input Source", value="File Upload")
            with gr.Column():
                gr.Markdown("### 🎯 Motion Detection")
                motion_algorithm = gr.Dropdown(["Frame Difference", "Optical Flow"], value="Frame Difference", label="Algorithm")
                motion_threshold = gr.Slider(0.1, 10.0, 2.0, label="Threshold")

        with gr.Group(visible=True) as file_upload_group:
            gr.Markdown("### 📁 File-Based Input")
            with gr.Row():
                metadata_file = gr.File(label="Metadata JSON", file_types=[".json"])
                images_file = gr.File(label="Images ZIP", file_types=[".zip"])
        
        with gr.Group(visible=False) as live_url_group:
            gr.Markdown("### 📡 Live URL Input")
            camera_url = gr.Textbox(label="Camera Stream URL", placeholder="https://...")
            duration = gr.Slider(minimum=5, maximum=120, value=10, step=5, label="Processing Duration (seconds)")
            with gr.Accordion("Live Camera Parameters", open=True):
                with gr.Row():
                    live_pos_x, live_pos_y, live_pos_z = gr.Number(label="X", val=0), gr.Number(label="Y", val=0), gr.Number(label="Z", val=0)
                with gr.Row():
                    live_yaw, live_pitch, live_roll = gr.Slider(label="Yaw", min=-180, max=180, val=0), gr.Slider(label="Pitch", min=-90, max=90, val=0), gr.Slider(label="Roll", min=-180, max=180, val=0)
                live_fov = gr.Slider(label="FOV (deg)", minimum=10, maximum=120, value=60.0)

        gr.Markdown("### 📐 Grid & Output Settings")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    grid_x, grid_y, grid_z = gr.Slider(50, 1000, 500, step=10, label="Size X"), gr.Slider(50, 1000, 500, step=10, label="Size Y"), gr.Slider(50, 1000, 500, step=10, label="Size Z")
                voxel_size = gr.Slider(0.1, 50.0, 6.0, label="Voxel Size")
            with gr.Column():
                extract_mesh = gr.Checkbox(label="Extract Mesh")
                mesh_threshold = gr.Slider(0.1, 2.0, 0.5, label="Mesh Threshold", visible=False)
                output_format = gr.Dropdown(["bin", "npy", "hdf5", "json"], value="bin", label="Output Format")
        
        input_source.change(lambda s: (gr.update(visible=s=="File Upload"), gr.update(visible=s=="Live URL")), input_source, [file_upload_group, live_url_group])
        extract_mesh.change(lambda x: gr.update(visible=x), extract_mesh, mesh_threshold)
        
        process_button = gr.Button("🚀 Process", variant="primary", size="lg")
        with gr.Row():
            results_text = gr.Textbox(label="Results", lines=10, interactive=False)
            preview_image = gr.Image(label="Preview (Middle Slice)")
        with gr.Row():
            download_voxel = gr.File(label="Download Voxel Grid", visible=False)
            download_mesh = gr.File(label="Download Mesh", visible=False)
        
        process_button.click(
            fn=processor.run_pipeline,
            inputs=[input_source, metadata_file, images_file, camera_url, duration, live_pos_x, live_pos_y, live_pos_z, live_yaw, live_pitch, live_roll, live_fov, preset_dropdown, grid_x, grid_y, grid_z, voxel_size, motion_algorithm, motion_threshold, extract_mesh, mesh_threshold, output_format],
            outputs=[results_text, preview_image, download_voxel, download_mesh]
        ).then(lambda v,m: (gr.update(visible=v is not None,value=v), gr.update(visible=m is not None,value=m)), [download_voxel, download_mesh], [download_voxel, download_mesh])
    return tab

def create_interface():
    app_state = WebAppState()
    processor, config_handler, vis_handler, util_handler, metadata_handler = ProcessingHandler(app_state), ConfigurationHandler(app_state), VisualizationHandler(app_state), UtilityHandler(app_state), MetadataHandler()
    
    with gr.Blocks(title="Pixeltovoxelprojector", theme=gr.themes.Soft(), css=".gradio-container{font-family:'Segoe UI',sans-serif}.gr-button-primary{background:linear-gradient(45deg,#4CAF50,#45a049);border:none}") as demo:
        gr.Markdown("# 🎯 Pixeltovoxelprojector\n*3D voxel grid generation from images or live streams*")
        with gr.Tabs():
            create_processing_tab(processor)
            create_metadata_tab(metadata_handler)
            # Other tabs...
        demo.cleanup_fn = app_state.cleanup
    return demo

def main():
    logging.info("Starting Gradio application...")
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, show_error=True)
    logging.info("Gradio application has been launched.")

if __name__ == "__main__":
    main()
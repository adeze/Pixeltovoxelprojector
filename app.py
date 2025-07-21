"""
Simplified Gradio Web Interface for Pixeltovoxelprojector.

This module provides a clean, modular web interface that's easy to extend
and maintain while showcasing the core features.
"""

import os
import shutil
import tempfile
import zipfile
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
from ray_voxel import process_all
from io_plugins import load_voxel_grid, save_voxel_grid, list_supported_formats
from visualization import VisualizationManager
from registry import list_available


class WebAppState:
    """Centralized state management for the web application."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.vis_manager = VisualizationManager()
        self.current_voxel_grid = None
        self.temp_dir = tempfile.mkdtemp()
    
    def cleanup(self):
        """Clean up temporary resources."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def get_preset_config(self, preset_name: str) -> PipelineConfig:
        """Get configuration by preset name."""
        presets = {
            "High Quality": create_high_quality_config,
            "Fast Processing": create_fast_config,
            "Astronomical": create_astronomical_config,
            "Default": create_default_config
        }
        return presets.get(preset_name, create_default_config)()


class ProcessingHandler:
    """Handles the main processing pipeline."""
    
    def __init__(self, app_state: WebAppState):
        self.app_state = app_state
    
    def run_pipeline(
        self,
        metadata_file,
        images_file,
        preset_name: str = "Default",
        grid_size: Tuple[int, int, int] = (500, 500, 500),
        voxel_size: float = 6.0,
        motion_threshold: float = 2.0,
        motion_algorithm: str = "Frame Difference",
        extract_mesh: bool = False,
        mesh_threshold: float = 0.5,
        output_format: str = "bin"
    ) -> Tuple[str, Optional[Image.Image], Optional[str], Optional[str]]:
        """Run the processing pipeline with given parameters."""
        
        try:
            # Validate inputs
            if not metadata_file or not images_file:
                return "❌ Error: Both metadata and images files are required", None, None, None
            
            # Setup working directory
            run_dir = self._setup_run_directory()
            
            # Extract and validate files
            metadata_path, images_folder = self._extract_files(
                metadata_file, images_file, run_dir
            )
            
            # Create configuration
            config = self._create_config(
                preset_name, grid_size, voxel_size, motion_threshold,
                motion_algorithm, extract_mesh, mesh_threshold, output_format
            )
            
            # Execute processing
            output_path, mesh_path = self._execute_processing(
                metadata_path, images_folder, config, run_dir
            )
            
            # Generate results
            return self._generate_results(output_path, mesh_path, config)
            
        except Exception as e:
            return f"❌ Processing failed: {str(e)}", None, None, None
    
    def _setup_run_directory(self) -> str:
        """Setup clean run directory."""
        run_dir = os.path.join(self.app_state.temp_dir, "current_run")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        return run_dir
    
    def _extract_files(self, metadata_file, images_file, run_dir: str) -> Tuple[str, str]:
        """Extract uploaded files to run directory."""
        # Setup paths
        images_folder = os.path.join(run_dir, "images")
        os.makedirs(images_folder)
        
        # Copy metadata
        metadata_path = os.path.join(run_dir, "metadata.json")
        shutil.copy2(metadata_file.name, metadata_path)
        
        # Extract images
        if not zipfile.is_zipfile(images_file.name):
            raise ValueError("Images must be provided as a ZIP file")
        
        with zipfile.ZipFile(images_file.name, 'r') as zip_ref:
            zip_ref.extractall(images_folder)
        
        return metadata_path, images_folder
    
    def _create_config(
        self, preset_name: str, grid_size: Tuple[int, int, int], voxel_size: float,
        motion_threshold: float, motion_algorithm: str, extract_mesh: bool,
        mesh_threshold: float, output_format: str
    ) -> PipelineConfig:
        """Create configuration from UI parameters."""
        config = self.app_state.get_preset_config(preset_name)
        
        # Override with UI parameters
        config.grid.size = list(grid_size)
        config.grid.voxel_size = voxel_size
        config.motion_detection.threshold = motion_threshold
        config.motion_detection.algorithm = motion_algorithm.lower().replace(' ', '_')
        config.io.export_mesh = extract_mesh
        config.io.mesh_threshold = mesh_threshold
        config.io.output_format = output_format.lower()
        
        config.validate()
        return config
    
    def _execute_processing(
        self, metadata_path: str, images_folder: str, config: PipelineConfig, run_dir: str
    ) -> Tuple[str, Optional[str]]:
        """Execute the actual processing."""
        output_path = os.path.join(run_dir, f"output.{config.io.output_format}")
        mesh_path = os.path.join(run_dir, "mesh.obj") if config.io.export_mesh else None
        
        process_all(
            metadata_path, images_folder, output_path,
            use_mcubes=config.io.export_mesh, output_mesh=mesh_path
        )
        
        return output_path, mesh_path
    
    def _generate_results(
        self, output_path: str, mesh_path: Optional[str], config: PipelineConfig
    ) -> Tuple[str, Optional[Image.Image], str, Optional[str]]:
        """Generate results summary and preview."""
        # Load voxel grid
        voxel_grid = load_voxel_grid(output_path)
        self.app_state.current_voxel_grid = voxel_grid
        
        # Generate summary
        data = voxel_grid.get_data()
        summary = f"""✅ Processing Complete!

📊 Voxel Grid Statistics:
• Grid Size: {voxel_grid.size}
• Voxel Size: {voxel_grid.voxel_size:.3f}
• Total Voxels: {np.prod(voxel_grid.size):,}
• Occupied Voxels: {torch.sum(data > 0).item():,}
• Max Value: {torch.max(data).item():.6f}

📁 Output Files:
• Voxel Grid: {output_path}
{f'• Mesh: {mesh_path}' if mesh_path else ''}

⚙️ Configuration:
• Motion Algorithm: {config.motion_detection.algorithm}
• Motion Threshold: {config.motion_detection.threshold}
"""
        
        # Create preview image
        preview_img = self._create_preview_image(data, voxel_grid.size)
        
        return summary, preview_img, output_path, mesh_path
    
    def _create_preview_image(self, data: torch.Tensor, grid_size: Tuple) -> Optional[Image.Image]:
        """Create preview image from voxel data."""
        if data.numel() == 0:
            return None
        
        # Get middle slice
        mid_z = grid_size[2] // 2
        slice_data = data[:, :, mid_z].numpy()
        
        if slice_data.max() > 0:
            # Normalize and create image
            normalized = (255 * slice_data / slice_data.max()).astype(np.uint8)
            img = Image.fromarray(normalized, mode='L')
            return img.resize((512, 512), Image.Resampling.NEAREST)
        
        return None


class ConfigurationHandler:
    """Handles configuration management."""
    
    def __init__(self, app_state: WebAppState):
        self.app_state = app_state
    
    def create_template(self, preset_name: str) -> str:
        """Create YAML configuration template."""
        try:
            config = self.app_state.get_preset_config(preset_name)
            return yaml.dump(config.to_dict(), default_flow_style=False, indent=2)
        except Exception as e:
            return f"Error creating config: {str(e)}"
    
    def validate_config(self, config_text: str) -> str:
        """Validate YAML configuration."""
        try:
            config_dict = yaml.safe_load(config_text)
            config = PipelineConfig.from_dict(config_dict)
            config.validate()
            return "✅ Configuration is valid!"
        except Exception as e:
            return f"❌ Configuration error: {str(e)}"


class VisualizationHandler:
    """Handles visualization creation."""
    
    def __init__(self, app_state: WebAppState):
        self.app_state = app_state
    
    def create_visualization(
        self,
        backend: str = "pyvista",
        render_mode: str = "Points",
        colormap: str = "viridis",
        threshold_percentile: float = 99.0,
        point_size: float = 2.0,
        opacity: float = 1.0
    ) -> Tuple[str, Optional[str]]:
        """Create 3D visualization."""
        
        if self.app_state.current_voxel_grid is None:
            return "❌ No voxel grid available. Process data first.", None
        
        try:
            vis_path = os.path.join(self.app_state.temp_dir, f"vis_{backend}.png")
            
            vis_options = {
                'render_mode': render_mode.lower(),
                'colormap': colormap.lower(),
                'threshold_percentile': threshold_percentile,
                'point_size': point_size,
                'opacity': opacity,
                'off_screen': True,
                'window_size': (1024, 768)
            }
            
            self.app_state.vis_manager.render_voxel_grid(
                self.app_state.current_voxel_grid,
                backend=backend.lower(),
                save_path=vis_path,
                show=False,
                **vis_options
            )
            
            return f"✅ Visualization created using {backend}!", vis_path
            
        except Exception as e:
            return f"❌ Visualization failed: {str(e)}", None
    
    def get_voxel_info(self) -> str:
        """Get current voxel grid information."""
        if self.app_state.current_voxel_grid is None:
            return "No voxel grid loaded"
        
        vg = self.app_state.current_voxel_grid
        data = vg.get_data()
        
        return f"""📊 Current Voxel Grid:
• Size: {vg.size}
• Voxel Size: {vg.voxel_size}
• Center: {vg.center.tolist()}
• Memory: {data.numel() * data.element_size() / (1024*1024):.1f} MB
• Value Range: [{torch.min(data).item():.3f}, {torch.max(data).item():.3f}]
• Occupancy: {(torch.sum(data > 0).item() / data.numel() * 100):.1f}%"""


class UtilityHandler:
    """Handles utility functions."""
    
    def __init__(self, app_state: WebAppState):
        self.app_state = app_state
    
    def convert_format(self, input_file, output_format: str) -> Tuple[str, Optional[str]]:
        """Convert voxel grid between formats."""
        if input_file is None:
            return "❌ No input file provided", None
        
        try:
            voxel_grid = load_voxel_grid(input_file.name)
            output_path = os.path.join(self.app_state.temp_dir, f"converted.{output_format}")
            save_voxel_grid(voxel_grid, output_path)
            
            input_ext = Path(input_file.name).suffix
            return f"✅ Converted {input_ext} to .{output_format}", output_path
            
        except Exception as e:
            return f"❌ Conversion failed: {str(e)}", None
    
    def get_system_info(self) -> str:
        """Get system information."""
        info = [
            "🖥️ System Information:",
            f"• PyTorch: {torch.__version__}",
            f"• CUDA: {'Available' if torch.cuda.is_available() else 'Not Available'}"
        ]
        
        if torch.cuda.is_available():
            info.append(f"• GPU Count: {torch.cuda.device_count()}")
        
        # Available components
        info.extend([
            "\n🧮 Available Algorithms:",
            f"• Motion Detectors: {', '.join(list_available('motion_detector'))}",
            f"• Ray Casters: {', '.join(list_available('ray_caster'))}",
            f"• Accumulators: {', '.join(list_available('accumulator'))}"
        ])
        
        # Supported formats
        info.append("\n📁 Supported Formats:")
        formats = list_supported_formats()
        for category, extensions in formats.items():
            category_name = category.replace('_', ' ').title()
            info.append(f"• {category_name}: {', '.join(sorted(extensions))}")
        
        # Visualization backends
        backends = ', '.join(self.app_state.vis_manager.available_backends)
        info.append(f"\n🎨 Visualization: {backends}")
        
        return '\n'.join(info)


def create_processing_tab(processor: ProcessingHandler) -> gr.TabItem:
    """Create the main processing tab."""
    
    with gr.TabItem("🚀 Processing") as tab:
        gr.Markdown("## Process Images to Voxel Grid")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Input Files")
                metadata_file = gr.File(
                    label="Metadata JSON",
                    file_types=[".json"]
                )
                images_file = gr.File(
                    label="Images ZIP", 
                    file_types=[".zip"]
                )
            
            with gr.Column():
                gr.Markdown("### ⚙️ Configuration")
                preset_dropdown = gr.Dropdown(
                    choices=["Default", "High Quality", "Fast Processing", "Astronomical"],
                    value="Default",
                    label="Preset"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📐 Grid Settings")
                with gr.Row():
                    grid_x = gr.Slider(50, 1000, 500, label="Size X")
                    grid_y = gr.Slider(50, 1000, 500, label="Size Y")
                    grid_z = gr.Slider(50, 1000, 500, label="Size Z")
                voxel_size = gr.Slider(0.1, 50.0, 6.0, label="Voxel Size")
            
            with gr.Column():
                gr.Markdown("### 🎯 Motion Detection")
                motion_algorithm = gr.Dropdown(
                    choices=["Frame Difference", "Optical Flow"],
                    value="Frame Difference",
                    label="Algorithm"
                )
                motion_threshold = gr.Slider(0.1, 10.0, 2.0, label="Threshold")
        
        with gr.Row():
            extract_mesh = gr.Checkbox(label="Extract Mesh")
            mesh_threshold = gr.Slider(0.1, 2.0, 0.5, label="Mesh Threshold", visible=False)
            output_format = gr.Dropdown(
                choices=["bin", "npy", "hdf5", "json"],
                value="bin",
                label="Output Format"
            )
        
        # Show/hide mesh threshold
        extract_mesh.change(
            lambda x: gr.update(visible=x),
            extract_mesh,
            mesh_threshold
        )
        
        process_button = gr.Button("🚀 Process", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=2):
                results_text = gr.Textbox(
                    label="Results",
                    lines=15
                )
            with gr.Column(scale=1):
                preview_image = gr.Image(
                    label="Preview (Middle Slice)"
                )
        
        with gr.Row():
            download_voxel = gr.File(label="Download Voxel Grid", visible=False)
            download_mesh = gr.File(label="Download Mesh", visible=False)
        
        # Wire up processing
        def process_wrapper(metadata, images, preset, gx, gy, gz, vs, mt, ma, em, mth, of):
            return processor.run_pipeline(
                metadata, images, preset, (gx, gy, gz), vs, mt, ma, em, mth, of
            )
        
        process_button.click(
            fn=process_wrapper,
            inputs=[
                metadata_file, images_file, preset_dropdown,
                grid_x, grid_y, grid_z, voxel_size, motion_threshold,
                motion_algorithm, extract_mesh, mesh_threshold, output_format
            ],
            outputs=[results_text, preview_image, download_voxel, download_mesh]
        ).then(
            fn=lambda v, m: (
                gr.update(visible=v is not None, value=v),
                gr.update(visible=m is not None, value=m)
            ),
            inputs=[download_voxel, download_mesh],
            outputs=[download_voxel, download_mesh]
        )
    
    return tab


def create_interface():
    """Create the main Gradio interface."""
    
    # Initialize state and handlers
    app_state = WebAppState()
    processor = ProcessingHandler(app_state)
    config_handler = ConfigurationHandler(app_state)
    vis_handler = VisualizationHandler(app_state)
    util_handler = UtilityHandler(app_state)
    
    with gr.Blocks(
        title="Pixeltovoxelprojector",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { font-family: 'Segoe UI', sans-serif; }
        .gr-button-primary { 
            background: linear-gradient(45deg, #4CAF50, #45a049); 
            border: none; 
        }
        """
    ) as demo:
        
        gr.Markdown("# 🎯 Pixeltovoxelprojector")
        gr.Markdown("*3D voxel grid generation from image sequences*")
        
        with gr.Tabs():
            # Processing Tab
            create_processing_tab(processor)
            
            # Configuration Tab
            with gr.TabItem("⚙️ Configuration"):
                gr.Markdown("## Configuration Management")
                
                with gr.Row():
                    preset_select = gr.Dropdown(
                        choices=["Default", "High Quality", "Fast Processing", "Astronomical"],
                        value="Default",
                        label="Select Preset"
                    )
                    generate_btn = gr.Button("Generate Template")
                
                config_editor = gr.Textbox(
                    label="Configuration (YAML)",
                    lines=20,
                    placeholder="YAML configuration will appear here..."
                )
                
                with gr.Row():
                    validate_btn = gr.Button("Validate")
                    validation_result = gr.Textbox(label="Validation", lines=3)
                
                # Wire up configuration
                generate_btn.click(
                    config_handler.create_template,
                    preset_select,
                    config_editor
                )
                validate_btn.click(
                    config_handler.validate_config,
                    config_editor,
                    validation_result
                )
            
            # Visualization Tab
            with gr.TabItem("🎨 Visualization"):
                gr.Markdown("## 3D Visualization")
                
                with gr.Row():
                    with gr.Column():
                        backend = gr.Dropdown(
                            choices=app_state.vis_manager.available_backends,
                            value="pyvista" if "pyvista" in app_state.vis_manager.available_backends else None,
                            label="Backend"
                        )
                        render_mode = gr.Dropdown(
                            choices=["Points", "Cubes", "Surface", "Volume"],
                            value="Points",
                            label="Render Mode"
                        )
                        colormap = gr.Dropdown(
                            choices=["viridis", "plasma", "inferno", "coolwarm"],
                            value="viridis",
                            label="Colormap"
                        )
                        threshold = gr.Slider(0, 100, 99, label="Threshold (%)")
                        
                        visualize_btn = gr.Button("🎨 Visualize", variant="primary")
                    
                    with gr.Column():
                        voxel_info = gr.Textbox(
                            label="Voxel Grid Info",
                            lines=8,
                            value=vis_handler.get_voxel_info()
                        )
                        refresh_btn = gr.Button("🔄 Refresh")
                
                vis_status = gr.Textbox(label="Status", lines=2)
                vis_image = gr.Image(label="Visualization", height=500)
                
                # Wire up visualization
                visualize_btn.click(
                    vis_handler.create_visualization,
                    [backend, render_mode, colormap, threshold],
                    [vis_status, vis_image]
                )
                refresh_btn.click(vis_handler.get_voxel_info, outputs=voxel_info)
            
            # Tools Tab
            with gr.TabItem("🔧 Tools"):
                gr.Markdown("## Utilities")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Format Conversion")
                        convert_input = gr.File(label="Input File")
                        convert_format = gr.Dropdown(
                            choices=["bin", "npy", "hdf5", "json"],
                            value="hdf5",
                            label="Target Format"
                        )
                        convert_btn = gr.Button("Convert")
                        convert_status = gr.Textbox(label="Status", lines=2)
                        convert_download = gr.File(label="Download", visible=False)
                    
                    with gr.Column():
                        gr.Markdown("### System Information")
                        system_info = gr.Textbox(
                            label="System Info",
                            lines=15,
                            value=util_handler.get_system_info(),
                            interactive=False
                        )
                        refresh_system_btn = gr.Button("🔄 Refresh")
                
                # Wire up tools
                convert_btn.click(
                    util_handler.convert_format,
                    [convert_input, convert_format],
                    [convert_status, convert_download]
                )
                refresh_system_btn.click(util_handler.get_system_info, outputs=system_info)
        
        # Store cleanup function for later use
        demo.cleanup_fn = app_state.cleanup
    
    return demo


def main():
    """Main entry point for Gradio app."""
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()
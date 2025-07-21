#!/usr/bin/env python3
"""
MCP Server for Pixeltovoxelprojector.

This module provides an MCP (Model Context Protocol) server interface for the
pixeltovoxelprojector system, allowing it to be used as a tool by LLM assistants.
"""

import asyncio
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Import our modular components
from config import (
    ConfigManager, create_default_config, create_high_quality_config,
    create_fast_config, create_astronomical_config
)
from ray_voxel import process_all
from io_plugins import load_voxel_grid, save_voxel_grid, list_supported_formats
from visualization import VisualizationManager
from registry import list_available
from voxel_grid import StandardVoxelGrid
import torch
import numpy as np
from PIL import Image
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixeltovoxel-mcp")

# Initialize the MCP server
server = Server("pixeltovoxelprojector")

# Global state for the MCP server
class MCPState:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.vis_manager = VisualizationManager()
        self.temp_dir = tempfile.mkdtemp()
        self.current_voxel_grid: Optional[StandardVoxelGrid] = None
        
    def cleanup(self):
        """Clean up temporary resources."""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

mcp_state = MCPState()

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def create_image_content(image: Image.Image, alt_text: str = "") -> ImageContent:
    """Create ImageContent from PIL Image."""
    img_b64 = image_to_base64(image)
    return ImageContent(
        type="image",
        data=img_b64,
        mimeType="image/png"
    )

@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available resources."""
    resources = []
    
    # System information resource
    resources.append(
        Resource(
            uri="pixeltovoxel://system/info",
            name="System Information",
            description="System capabilities and available algorithms",
            mimeType="application/json"
        )
    )
    
    # Supported formats resource
    resources.append(
        Resource(
            uri="pixeltovoxel://system/formats",
            name="Supported Formats",
            description="List of supported file formats",
            mimeType="application/json"
        )
    )
    
    # Current voxel grid info (if available)
    if mcp_state.current_voxel_grid is not None:
        resources.append(
            Resource(
                uri="pixeltovoxel://voxel/current",
                name="Current Voxel Grid",
                description="Information about the currently loaded voxel grid",
                mimeType="application/json"
            )
        )
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content."""
    
    if uri == "pixeltovoxel://system/info":
        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "available_algorithms": {
                "motion_detectors": list_available('motion_detector'),
                "ray_casters": list_available('ray_caster'),
                "accumulators": list_available('accumulator')
            },
            "visualization_backends": mcp_state.vis_manager.available_backends
        }
        if torch.cuda.is_available():
            info["cuda_devices"] = torch.cuda.device_count()
        return json.dumps(info, indent=2)
    
    elif uri == "pixeltovoxel://system/formats":
        formats = list_supported_formats()
        return json.dumps(formats, indent=2)
    
    elif uri == "pixeltovoxel://voxel/current":
        if mcp_state.current_voxel_grid is None:
            return json.dumps({"error": "No voxel grid currently loaded"})
        
        vg = mcp_state.current_voxel_grid
        data = vg.get_data()
        info = {
            "size": list(vg.size),
            "voxel_size": float(vg.voxel_size),
            "center": vg.center.tolist(),
            "data_type": str(data.dtype),
            "memory_usage_mb": float(data.numel() * data.element_size() / (1024*1024)),
            "value_range": [float(torch.min(data).item()), float(torch.max(data).item())],
            "total_voxels": int(data.numel()),
            "occupied_voxels": int(torch.sum(data > 0).item()),
            "occupancy_ratio": float(torch.sum(data > 0).item() / data.numel())
        }
        return json.dumps(info, indent=2)
    
    else:
        raise ValueError(f"Unknown resource URI: {uri}")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="process_images",
            description="Process image sequences to generate 3D voxel grids with motion detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "metadata_json": {
                        "type": "string",
                        "description": "JSON string containing camera positions, orientations, and frame information"
                    },
                    "images_zip_path": {
                        "type": "string",
                        "description": "Path to ZIP file containing input images"
                    },
                    "preset": {
                        "type": "string",
                        "enum": ["default", "high_quality", "fast", "astronomical"],
                        "default": "default",
                        "description": "Configuration preset to use"
                    },
                    "grid_size": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 3,
                        "maxItems": 3,
                        "default": [500, 500, 500],
                        "description": "Voxel grid dimensions [x, y, z]"
                    },
                    "voxel_size": {
                        "type": "number",
                        "default": 6.0,
                        "description": "Size of each voxel in world units"
                    },
                    "motion_threshold": {
                        "type": "number",
                        "default": 2.0,
                        "description": "Threshold for motion detection"
                    },
                    "motion_algorithm": {
                        "type": "string",
                        "enum": ["frame_difference", "optical_flow"],
                        "default": "frame_difference",
                        "description": "Motion detection algorithm"
                    },
                    "extract_mesh": {
                        "type": "boolean",
                        "default": False,
                        "description": "Extract 3D mesh using Marching Cubes"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["bin", "npy", "hdf5", "json"],
                        "default": "bin",
                        "description": "Output file format"
                    }
                },
                "required": ["metadata_json", "images_zip_path"]
            }
        ),
        Tool(
            name="visualize_voxel_grid",
            description="Create 3D visualization of the current voxel grid",
            inputSchema={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["pyvista", "matplotlib", "plotly"],
                        "default": "matplotlib",
                        "description": "Visualization backend to use"
                    },
                    "render_mode": {
                        "type": "string",
                        "enum": ["points", "cubes", "surface", "volume"],
                        "default": "points",
                        "description": "Rendering mode"
                    },
                    "colormap": {
                        "type": "string",
                        "default": "viridis",
                        "description": "Color map for visualization"
                    },
                    "threshold_percentile": {
                        "type": "number",
                        "default": 99.0,
                        "minimum": 0.0,
                        "maximum": 100.0,
                        "description": "Threshold percentile for display"
                    }
                }
            }
        ),
        Tool(
            name="create_config",
            description="Create configuration template for processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": ["default", "high_quality", "fast", "astronomical"],
                        "default": "default",
                        "description": "Configuration preset"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["yaml", "json"],
                        "default": "yaml",
                        "description": "Output format"
                    }
                }
            }
        ),
        Tool(
            name="convert_voxel_format",
            description="Convert voxel grid between different file formats",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input voxel grid file"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["bin", "npy", "hdf5", "json"],
                        "description": "Target output format"
                    }
                },
                "required": ["input_path", "output_format"]
            }
        ),
        Tool(
            name="get_voxel_info",
            description="Get detailed information about the current voxel grid",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    
    if name == "process_images":
        return await handle_process_images(arguments)
    elif name == "visualize_voxel_grid":
        return await handle_visualize_voxel_grid(arguments)
    elif name == "create_config":
        return await handle_create_config(arguments)
    elif name == "convert_voxel_format":
        return await handle_convert_voxel_format(arguments)
    elif name == "get_voxel_info":
        return await handle_get_voxel_info(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_process_images(arguments: Dict[str, Any]) -> Sequence[TextContent | ImageContent]:
    """Handle image processing tool call."""
    try:
        # Extract arguments
        metadata_json = arguments["metadata_json"]
        images_zip_path = arguments["images_zip_path"]
        preset = arguments.get("preset", "default")
        grid_size = arguments.get("grid_size", [500, 500, 500])
        voxel_size = arguments.get("voxel_size", 6.0)
        motion_threshold = arguments.get("motion_threshold", 2.0)
        motion_algorithm = arguments.get("motion_algorithm", "frame_difference")
        extract_mesh = arguments.get("extract_mesh", False)
        output_format = arguments.get("output_format", "bin")
        
        # Setup working directory
        import shutil
        work_dir = Path(mcp_state.temp_dir) / "processing"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        
        # Save metadata
        metadata_path = work_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            if isinstance(metadata_json, str):
                # Try to parse as JSON string first
                try:
                    metadata_dict = json.loads(metadata_json)
                    json.dump(metadata_dict, f, indent=2)
                except json.JSONDecodeError:
                    # If it fails, treat as raw string
                    f.write(metadata_json)
            else:
                json.dump(metadata_json, f, indent=2)
        
        # Extract images
        images_folder = work_dir / "images"
        images_folder.mkdir()
        
        if not Path(images_zip_path).exists():
            return [TextContent(type="text", text=f"Error: Images ZIP file not found: {images_zip_path}")]
        
        with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_folder)
        
        # Create configuration
        preset_functions = {
            "high_quality": create_high_quality_config,
            "fast": create_fast_config,
            "astronomical": create_astronomical_config,
            "default": create_default_config
        }
        
        config = preset_functions.get(preset, create_default_config)()
        
        # Override with parameters
        config.grid.size = grid_size
        config.grid.voxel_size = voxel_size
        config.motion_detection.threshold = motion_threshold
        config.motion_detection.algorithm = motion_algorithm
        config.io.export_mesh = extract_mesh
        config.io.output_format = output_format
        
        config.validate()
        
        # Execute processing
        output_path = work_dir / f"output.{output_format}"
        mesh_path = work_dir / "mesh.obj" if extract_mesh else None
        
        process_all(
            str(metadata_path),
            str(images_folder),
            str(output_path),
            use_mcubes=extract_mesh,
            output_mesh=str(mesh_path) if mesh_path else None,
            config=config
        )
        
        # Load and analyze results
        voxel_grid = load_voxel_grid(output_path)
        mcp_state.current_voxel_grid = voxel_grid
        
        data = voxel_grid.get_data()
        
        # Generate summary
        summary = f"""✅ Processing Complete!

📊 Voxel Grid Statistics:
• Grid Size: {voxel_grid.size}
• Voxel Size: {voxel_grid.voxel_size:.3f}
• Total Voxels: {np.prod(voxel_grid.size):,}
• Occupied Voxels: {torch.sum(data > 0).item():,}
• Max Value: {torch.max(data).item():.6f}
• Occupancy: {(torch.sum(data > 0).item() / data.numel() * 100):.2f}%

📁 Output Files:
• Voxel Grid: {output_path}
{f'• Mesh: {mesh_path}' if mesh_path else ''}

⚙️ Configuration:
• Preset: {preset}
• Motion Algorithm: {motion_algorithm}
• Motion Threshold: {motion_threshold}
"""
        
        result = [TextContent(type="text", text=summary)]
        
        # Create preview image
        if data.numel() > 0:
            mid_z = voxel_grid.size[2] // 2
            slice_data = data[:, :, mid_z].numpy()
            
            if slice_data.max() > 0:
                normalized = (255 * slice_data / slice_data.max()).astype(np.uint8)
                preview_img = Image.fromarray(normalized, mode='L')
                preview_img = preview_img.resize((512, 512), Image.Resampling.NEAREST)
                
                result.append(create_image_content(
                    preview_img, 
                    f"Voxel grid preview (middle slice, z={mid_z})"
                ))
        
        return result
        
    except Exception as e:
        logger.exception("Error in process_images")
        return [TextContent(type="text", text=f"❌ Processing failed: {str(e)}")]

async def handle_visualize_voxel_grid(arguments: Dict[str, Any]) -> Sequence[TextContent | ImageContent]:
    """Handle voxel grid visualization tool call."""
    try:
        if mcp_state.current_voxel_grid is None:
            return [TextContent(type="text", text="❌ No voxel grid available. Please process images first.")]
        
        backend = arguments.get("backend", "matplotlib")
        render_mode = arguments.get("render_mode", "points")
        colormap = arguments.get("colormap", "viridis")
        threshold_percentile = arguments.get("threshold_percentile", 99.0)
        
        # Create visualization
        vis_path = Path(mcp_state.temp_dir) / f"visualization_{backend}.png"
        
        vis_options = {
            'render_mode': render_mode.lower(),
            'colormap': colormap.lower(),
            'threshold_percentile': threshold_percentile,
            'point_size': 2.0,
            'opacity': 1.0,
            'off_screen': True,
            'window_size': (1024, 768)
        }
        
        mcp_state.vis_manager.render_voxel_grid(
            mcp_state.current_voxel_grid,
            backend=backend.lower(),
            save_path=str(vis_path),
            show=False,
            **vis_options
        )
        
        # Load and return the image
        vis_img = Image.open(vis_path)
        
        return [
            TextContent(type="text", text=f"✅ Visualization created using {backend} backend with {render_mode} mode"),
            create_image_content(vis_img, f"3D visualization using {backend}")
        ]
        
    except Exception as e:
        logger.exception("Error in visualize_voxel_grid")
        return [TextContent(type="text", text=f"❌ Visualization failed: {str(e)}")]

async def handle_create_config(arguments: Dict[str, Any]) -> Sequence[TextContent]:
    """Handle configuration creation tool call."""
    try:
        preset = arguments.get("preset", "default")
        format_type = arguments.get("format", "yaml")
        
        # Get configuration
        preset_functions = {
            "high_quality": create_high_quality_config,
            "fast": create_fast_config, 
            "astronomical": create_astronomical_config,
            "default": create_default_config
        }
        
        config = preset_functions.get(preset, create_default_config)()
        config_dict = config.to_dict()
        
        if format_type == "yaml":
            import yaml
            config_text = yaml.dump(config_dict, default_flow_style=False, indent=2)
        else:
            config_text = json.dumps(config_dict, indent=2)
        
        return [TextContent(
            type="text", 
            text=f"✅ {preset.title()} configuration template ({format_type.upper()}):\n\n```{format_type}\n{config_text}\n```"
        )]
        
    except Exception as e:
        logger.exception("Error in create_config")
        return [TextContent(type="text", text=f"❌ Config creation failed: {str(e)}")]

async def handle_convert_voxel_format(arguments: Dict[str, Any]) -> Sequence[TextContent]:
    """Handle voxel format conversion tool call."""
    try:
        input_path = Path(arguments["input_path"])
        output_format = arguments["output_format"]
        
        if not input_path.exists():
            return [TextContent(type="text", text=f"❌ Input file not found: {input_path}")]
        
        # Load voxel grid
        voxel_grid = load_voxel_grid(input_path)
        
        # Save in new format
        output_path = Path(mcp_state.temp_dir) / f"converted.{output_format}"
        save_voxel_grid(voxel_grid, output_path)
        
        return [TextContent(
            type="text", 
            text=f"✅ Successfully converted {input_path.suffix} to .{output_format}\nOutput saved to: {output_path}"
        )]
        
    except Exception as e:
        logger.exception("Error in convert_voxel_format")
        return [TextContent(type="text", text=f"❌ Conversion failed: {str(e)}")]

async def handle_get_voxel_info(arguments: Dict[str, Any]) -> Sequence[TextContent]:
    """Handle get voxel info tool call."""
    try:
        if mcp_state.current_voxel_grid is None:
            return [TextContent(type="text", text="❌ No voxel grid currently loaded")]
        
        vg = mcp_state.current_voxel_grid
        data = vg.get_data()
        
        info = f"""📊 Current Voxel Grid Information:
• Size: {vg.size}
• Voxel Size: {vg.voxel_size}
• Center: {vg.center.tolist()}
• Data Type: {data.dtype}
• Memory Usage: {data.numel() * data.element_size() / (1024*1024):.2f} MB
• Value Range: [{torch.min(data).item():.6f}, {torch.max(data).item():.6f}]
• Total Voxels: {data.numel():,}
• Occupied Voxels: {torch.sum(data > 0).item():,}
• Occupancy Ratio: {(torch.sum(data > 0).item() / data.numel() * 100):.2f}%
"""
        
        return [TextContent(type="text", text=info)]
        
    except Exception as e:
        logger.exception("Error in get_voxel_info")
        return [TextContent(type="text", text=f"❌ Failed to get voxel info: {str(e)}")]

async def main():
    """Main entry point for the MCP server."""
    try:
        # Use stdio for MCP communication
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="pixeltovoxelprojector",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None
                    )
                )
            )
    except Exception as e:
        logger.exception("Server error")
        raise
    finally:
        # Cleanup
        mcp_state.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
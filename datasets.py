"""
PyTorch Dataset implementations for batch processing.

This module provides Dataset classes compatible with PyTorch DataLoader
for efficient batch processing of image sequences and voxel data.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from PIL import Image
import cv2

from interfaces import DataSource, Transform
from implementations import ImageSequenceDataSource


class MotionDetectionDataset(Dataset):
    """PyTorch Dataset for motion detection tasks."""
    
    def __init__(
        self,
        data_source: DataSource,
        sequence_length: int = 2,
        stride: int = 1,
        transform: Optional[Transform] = None
    ):
        """
        Initialize motion detection dataset.
        
        Args:
            data_source: Source of image data
            sequence_length: Number of consecutive frames to return
            stride: Stride between sequences
            transform: Optional transform to apply to frames
        """
        self.data_source = data_source
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        # Calculate valid sequence indices
        self.valid_indices = []
        total_frames = len(data_source)
        
        for i in range(0, total_frames - sequence_length + 1, stride):
            # Check if frames are from same camera
            frame_infos = [data_source.get_frame_info(j) for j in range(i, i + sequence_length)]
            camera_indices = [info.get("camera_index", 0) for info in frame_infos]
            
            if all(cam == camera_indices[0] for cam in camera_indices):
                self.valid_indices.append(i)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_idx = self.valid_indices[idx]
        
        frames = []
        frame_infos = []
        
        for i in range(start_idx, start_idx + self.sequence_length):
            frame, frame_info = self.data_source[i]
            
            if self.transform:
                frame = self.transform(frame)
                
            frames.append(frame)
            frame_infos.append(frame_info)
        
        return {
            'frames': torch.stack(frames),
            'frame_infos': frame_infos,
            'sequence_idx': idx
        }


class VoxelDataset(Dataset):
    """PyTorch Dataset for voxel grid data."""
    
    def __init__(
        self,
        voxel_files: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Initialize voxel dataset.
        
        Args:
            voxel_files: List of paths to voxel grid files
            transform: Optional transform to apply to voxel grids
        """
        self.voxel_files = [Path(f) for f in voxel_files]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.voxel_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        voxel_path = self.voxel_files[idx]
        
        # Load voxel grid
        if voxel_path.suffix == '.bin':
            voxel_grid = self._load_bin_file(voxel_path)
        elif voxel_path.suffix == '.npy':
            voxel_grid = torch.from_numpy(np.load(voxel_path))
        else:
            raise ValueError(f"Unsupported voxel file format: {voxel_path.suffix}")
        
        if self.transform:
            voxel_grid = self.transform(voxel_grid)
        
        return {
            'voxel_grid': voxel_grid,
            'path': str(voxel_path),
            'filename': voxel_path.stem
        }
    
    def _load_bin_file(self, path: Path) -> torch.Tensor:
        """Load binary voxel grid file."""
        with open(path, 'rb') as f:
            # Read size (3 int32s)
            size_data = f.read(3 * 4)
            size = np.frombuffer(size_data, dtype=np.int32)
            
            # Read voxel size (1 float32)
            voxel_size_data = f.read(4)
            voxel_size = np.frombuffer(voxel_size_data, dtype=np.float32)[0]
            
            # Read grid data
            grid_data = f.read()
            grid = np.frombuffer(grid_data, dtype=np.float32).reshape(tuple(size))
            
        return torch.from_numpy(grid)


class VideoDataset(Dataset):
    """PyTorch Dataset for video processing."""
    
    def __init__(
        self,
        video_path: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        transform: Optional[Transform] = None
    ):
        """
        Initialize video dataset.
        
        Args:
            video_path: Path to video file
            frame_skip: Number of frames to skip between samples
            max_frames: Maximum number of frames to load (None for all)
            transform: Optional transform to apply to frames
        """
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.transform = transform
        
        # Load video and extract frames
        self.frames = self._load_video_frames()
    
    def _load_video_frames(self) -> List[torch.Tensor]:
        """Load frames from video file."""
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % (self.frame_skip + 1) == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_tensor = torch.from_numpy(gray.astype(np.float32))
                frames.append(frame_tensor)
                
                if self.max_frames and len(frames) >= self.max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self.frames[idx]
        
        if self.transform:
            frame = self.transform(frame)
        
        return {
            'frame': frame,
            'frame_idx': idx,
            'video_path': str(self.video_path)
        }


# Collate functions for DataLoader
def motion_detection_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for motion detection batches."""
    # Stack frames
    frames = torch.stack([item['frames'] for item in batch])
    
    # Collect frame infos
    frame_infos = [item['frame_infos'] for item in batch]
    
    # Collect sequence indices
    sequence_indices = torch.tensor([item['sequence_idx'] for item in batch])
    
    return {
        'frames': frames,
        'frame_infos': frame_infos,
        'sequence_indices': sequence_indices
    }


def voxel_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for voxel grid batches."""
    voxel_grids = torch.stack([item['voxel_grid'] for item in batch])
    paths = [item['path'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'voxel_grids': voxel_grids,
        'paths': paths,
        'filenames': filenames
    }


# Convenience functions
def create_motion_detection_dataloader(
    metadata_path: str,
    images_folder: str,
    batch_size: int = 4,
    sequence_length: int = 2,
    stride: int = 1,
    num_workers: int = 0,
    transform: Optional[Transform] = None
) -> DataLoader:
    """Create DataLoader for motion detection."""
    data_source = ImageSequenceDataSource(metadata_path, images_folder)
    dataset = MotionDetectionDataset(
        data_source=data_source,
        sequence_length=sequence_length,
        stride=stride,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=motion_detection_collate_fn
    )


def create_voxel_dataloader(
    voxel_files: List[str],
    batch_size: int = 4,
    num_workers: int = 0,
    transform: Optional[Callable] = None
) -> DataLoader:
    """Create DataLoader for voxel grids."""
    dataset = VoxelDataset(voxel_files, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=voxel_collate_fn
    )


def create_video_dataloader(
    video_path: str,
    batch_size: int = 4,
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    num_workers: int = 0,
    transform: Optional[Transform] = None
) -> DataLoader:
    """Create DataLoader for video processing."""
    dataset = VideoDataset(
        video_path=video_path,
        frame_skip=frame_skip,
        max_frames=max_frames,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep temporal order
        num_workers=num_workers
    )
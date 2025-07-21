"""
Pydantic models for data structures used in the pixeltovoxelprojector application.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class FrameInfo(BaseModel):
    """
    Represents metadata for a single frame in a sequence.
    """
    image_file: str = Field(..., description="The filename of the image for this frame.")
    frame_index: int = Field(..., description="The index of this frame in the sequence.")
    camera_index: int = Field(..., description="The index of the camera that captured this frame.")
    
    # Optional camera calibration parameters
    camera_matrix: Optional[List[List[float]]] = Field(None, description="3x3 camera intrinsic matrix.")
    dist_coeffs: Optional[List[float]] = Field(None, description="Distortion coefficients.")
    rotation: Optional[List[float]] = Field(None, description="Rotation vector (e.g., Rodrigues).")
    translation: Optional[List[float]] = Field(None, description="Translation vector.")

    class Config:
        # Allows the model to be used with ORMs or other class instances
        orm_mode = True

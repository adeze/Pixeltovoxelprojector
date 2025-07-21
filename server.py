import json
import os
import subprocess

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()
UPLOAD_DIR = "uploads"
METADATA_FILE = "metadata.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_image_metadata(filepath, filename):
    with Image.open(filepath) as img:
        width, height = img.size
    projection_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    camera_position = [0, 0, 10]
    voxel_file = filename.rsplit(".", 1)[0] + ".bin"
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "projection_matrix": projection_matrix,
        "camera_position": camera_position,
        "voxel_file": voxel_file,
    }


def extract_frame_from_video(video_path, output_image_path):
    # Extract the first frame using ffmpeg
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            "select=eq(n\\,0)",
            "-q:v",
            "3",
            output_image_path,
            "-y",
        ],
        check=True,
    )


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        metadata = get_image_metadata(filepath, file.filename)
    elif ext in [".mp4", ".mov", ".avi"]:
        # Extract first frame as PNG
        frame_path = filepath + "_frame.png"
        extract_frame_from_video(filepath, frame_path)
        metadata = get_image_metadata(frame_path, file.filename + "_frame.png")
        metadata["source_video"] = file.filename
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    # Load existing metadata or create new
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"images": []}
    data["images"].append(metadata)
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return JSONResponse(content=metadata)

import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


# 1) Data Structures
class FrameInfo:
    def __init__(
        self,
        camera_index: int,
        frame_index: int,
        camera_position: List[float],
        yaw: float,
        pitch: float,
        roll: float,
        fov_degrees: float,
        image_file: str,
    ):
        self.camera_index = camera_index
        self.frame_index = frame_index
        self.camera_position = torch.tensor(camera_position, dtype=torch.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.fov_degrees = fov_degrees
        self.image_file = image_file


# 2) Math Helpers
def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def normalize(v: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(v)
    return v / norm if norm > 1e-12 else torch.zeros_like(v)


def rotation_matrix_yaw_pitch_roll(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> torch.Tensor:
    y, p, r = map(deg2rad, [yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float32)
    Ry = torch.tensor([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]], dtype=torch.float32)
    Rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float32)
    return Rz @ Ry @ Rx


# 3) Load JSON Metadata
def load_metadata(json_path: str) -> List[FrameInfo]:
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = []
    for entry in data:
        frames.append(
            FrameInfo(
                entry.get("camera_index", 0),
                entry.get("frame_index", 0),
                entry.get("camera_position", [0, 0, 0]),
                entry.get("yaw", 0.0),
                entry.get("pitch", 0.0),
                entry.get("roll", 0.0),
                entry.get("fov_degrees", 60.0),
                entry.get("image_file", ""),
            )
        )
    return frames


# 4) Image Loading (Gray) & Motion Detection
def load_image_gray(img_path: str, use_opencv: bool = False) -> torch.Tensor:
    if use_opencv:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        arr = img.astype(np.float32)
    else:
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float32)
    noise = np.random.uniform(-1, 1, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return torch.from_numpy(arr)


def detect_motion(
    prev: torch.Tensor, curr: torch.Tensor, threshold: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = torch.abs(curr - prev)
    changed = diff > threshold
    return changed, diff


def load_frames_from_video(video_path: str) -> List[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arr = gray.astype(np.float32)
        noise = np.random.uniform(-1, 1, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        frames.append(torch.from_numpy(arr))
    cap.release()
    return frames


# 5) Voxel DDA (Ray Casting)
def cast_ray_into_grid(
    camera_pos: torch.Tensor,
    dir_normalized: torch.Tensor,
    N: int,
    voxel_size: float,
    grid_center: torch.Tensor,
):
    # Full DDA implementation: returns a list of (ix, iy, iz, distance)
    steps = []
    half_size = 0.5 * (N * voxel_size)
    grid_min = grid_center - half_size
    grid_max = grid_center + half_size

    t_min = 0.0
    t_max = float("inf")
    origin = camera_pos
    direction = dir_normalized

    # Ray-box intersection
    for i in range(3):
        o = origin[i].item()
        d = direction[i].item()
        mn = grid_min[i].item()
        mx = grid_max[i].item()
        if abs(d) < 1e-12:
            if o < mn or o > mx:
                return steps
        else:
            t1 = (mn - o) / d
            t2 = (mx - o) / d
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            if t_near > t_min:
                t_min = t_near
            if t_far < t_max:
                t_max = t_far
            if t_min > t_max:
                return steps

    if t_min < 0.0:
        t_min = 0.0

    # Start voxel
    start_world = origin + direction * t_min
    fx = (start_world[0] - grid_min[0]) / voxel_size
    fy = (start_world[1] - grid_min[1]) / voxel_size
    fz = (start_world[2] - grid_min[2]) / voxel_size
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
        return steps

    # Step direction
    step_x = 1 if direction[0] >= 0 else -1
    step_y = 1 if direction[1] >= 0 else -1
    step_z = 1 if direction[2] >= 0 else -1

    def boundary_in_world(i, axis):
        return grid_min[axis] + i * voxel_size

    nx_x = ix + (1 if step_x > 0 else 0)
    nx_y = iy + (1 if step_y > 0 else 0)
    nx_z = iz + (1 if step_z > 0 else 0)

    next_bx = boundary_in_world(nx_x, 0)
    next_by = boundary_in_world(nx_y, 1)
    next_bz = boundary_in_world(nx_z, 2)

    def safe_div(num, den):
        eps = 1e-12
        if abs(den) < eps:
            return float("inf")
        return num / den

    t_max_x = safe_div(next_bx - origin[0], direction[0])
    t_max_y = safe_div(next_by - origin[1], direction[1])
    t_max_z = safe_div(next_bz - origin[2], direction[2])

    t_delta_x = safe_div(voxel_size, abs(direction[0]))
    t_delta_y = safe_div(voxel_size, abs(direction[1]))
    t_delta_z = safe_div(voxel_size, abs(direction[2]))

    t_current = t_min
    step_count = 0

    while t_current <= t_max:
        steps.append((ix, iy, iz, t_current))
        if t_max_x < t_max_y and t_max_x < t_max_z:
            ix += step_x
            t_current = t_max_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_z:
            iy += step_y
            t_current = t_max_y
            t_max_y += t_delta_y
        else:
            iz += step_z
            t_current = t_max_z
            t_max_z += t_delta_z
        step_count += 1
        if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
            break
    return steps


# 6) Main Pipeline (Vectorized)
def process_all(
    metadata_path: str, images_folder: str, output_bin: str, use_opencv: bool = False
):
    frames = load_metadata(metadata_path)
    frames_by_cam = {}
    for f in frames:
        frames_by_cam.setdefault(f.camera_index, []).append(f)
    for v in frames_by_cam.values():
        v.sort(key=lambda x: x.frame_index)

    N = 500
    voxel_size = 6.0
    grid_center = torch.tensor([0.0, 0.0, 500.0], dtype=torch.float32)
    voxel_grid = torch.zeros((N, N, N), dtype=torch.float32)

    motion_threshold = 2.0
    alpha = 0.1

    for cam_id, cam_frames in frames_by_cam.items():
        if len(cam_frames) < 2:
            continue
        prev_img = None
        for i, curr_info in enumerate(cam_frames):
            img_path = os.path.join(images_folder, curr_info.image_file)
            curr_img = load_image_gray(img_path, use_opencv=use_opencv)
            if prev_img is None:
                prev_img = curr_img
                continue
            changed, diff = detect_motion(prev_img, curr_img, motion_threshold)
            cam_pos = curr_info.camera_position
            cam_rot = rotation_matrix_yaw_pitch_roll(
                curr_info.yaw, curr_info.pitch, curr_info.roll
            )
            fov_rad = deg2rad(curr_info.fov_degrees)
            focal_len = (curr_img.shape[1] * 0.5) / np.tan(fov_rad * 0.5)

            # Vectorized pixel selection
            idxs = torch.nonzero(changed)
            if idxs.numel() == 0:
                prev_img = curr_img
                continue

            u = idxs[:, 1].float()
            v = idxs[:, 0].float()
            pix_val = diff[idxs[:, 0], idxs[:, 1]]

            mask = pix_val >= 1e-3
            u = u[mask]
            v = v[mask]
            pix_val = pix_val[mask]

            # Generate rays in batch
            x = u - 0.5 * curr_img.shape[1]
            y = -(v - 0.5 * curr_img.shape[0])
            z = torch.full_like(x, -focal_len)
            rays_cam = torch.stack([x, y, z], dim=1)
            rays_cam = torch.nn.functional.normalize(rays_cam, dim=1)
            rays_world = torch.matmul(rays_cam, cam_rot.T)
            rays_world = torch.nn.functional.normalize(rays_world, dim=1)

            for idx in range(rays_world.shape[0]):
                ray_world = rays_world[idx]
                val = pix_val[idx]
                steps = cast_ray_into_grid(
                    cam_pos, ray_world, N, voxel_size, grid_center
                )
                for rs in steps:
                    ix, iy, iz, dist = rs
                    attenuation = 1.0 / (1.0 + alpha * dist)
                    vval = val * attenuation
                    if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
                        voxel_grid[ix, iy, iz] += vval
            prev_img = curr_img

    # Save voxel grid to .bin
    with open(output_bin, "wb") as f:
        f.write(np.array([N], dtype=np.int32).tobytes())
        f.write(np.array([voxel_size], dtype=np.float32).tobytes())
        f.write(voxel_grid.numpy().astype(np.float32).tobytes())
    print(f"Saved voxel grid to {output_bin}")


# Example usage:
# process_all('metadata.json', 'images_folder', 'output_voxel_grid.bin', use_opencv=True)
# process_all('metadata.json', 'images_folder', 'output_voxel_grid.bin', use_opencv=True)

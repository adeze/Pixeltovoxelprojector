import json
import torch
import numpy as np
from PIL import Image

# 1. Data Structures
class FrameInfo:
    def __init__(self, entry):
        self.camera_index = entry.get("camera_index", 0)
        self.frame_index = entry.get("frame_index", 0)
        self.yaw = entry.get("yaw", 0.0)
        self.pitch = entry.get("pitch", 0.0)
        self.roll = entry.get("roll", 0.0)
        self.fov_degrees = entry.get("fov_degrees", 60.0)
        self.image_file = entry.get("image_file", "")
        cp = entry.get("camera_position", [0.0, 0.0, 0.0])
        self.camera_position = torch.tensor(cp, dtype=torch.float32)

# 2. Math Helpers
def deg2rad(deg):
    return deg * np.pi / 180.0

def normalize(v):
    norm = torch.norm(v)
    return v / norm if norm > 1e-12 else v

def rotation_matrix_yaw_pitch_roll(yaw, pitch, roll):
    y, p, r = map(deg2rad, [yaw, pitch, roll])
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float32)
    Ry = torch.tensor([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]], dtype=torch.float32)
    Rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float32)
    return Rz @ Ry @ Rx

# 3. Load Metadata
def load_metadata(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [FrameInfo(entry) for entry in data]

# 4. Image Loading (Gray)
def load_image_gray(img_path):
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    noise = np.random.uniform(-1.0, 1.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return torch.from_numpy(arr)

# 5. Motion Detection
def detect_motion(prev, curr, threshold):
    diff = torch.abs(curr - prev)
    changed = diff > threshold
    return changed, diff

# 6. DDA Ray Traversal (see previous implementation in process_image.py)
def cast_ray_into_grid(camera_pos, dir_normalized, N, voxel_size, grid_center):
    steps = []
    half_size = 0.5 * (N * voxel_size)
    grid_min = grid_center - half_size
    grid_max = grid_center + half_size

    t_min = 0.0
    t_max = float('inf')

    # Ray-box intersection
    for i in range(3):
        origin = camera_pos[i].item()
        d = dir_normalized[i].item()
        mn = grid_min[i].item()
        mx = grid_max[i].item()
        if abs(d) < 1e-12:
            if origin < mn or origin > mx:
                return steps
        else:
            t1 = (mn - origin) / d
            t2 = (mx - origin) / d
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
    start_world = camera_pos + t_min * dir_normalized
    fx = (start_world[0] - grid_min[0]) / voxel_size
    fy = (start_world[1] - grid_min[1]) / voxel_size
    fz = (start_world[2] - grid_min[2]) / voxel_size
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
        return steps

    step_x = 1 if dir_normalized[0] >= 0.0 else -1
    step_y = 1 if dir_normalized[1] >= 0.0 else -1
    step_z = 1 if dir_normalized[2] >= 0.0 else -1

    def boundary_in_world_x(i_x):
        return grid_min[0] + i_x * voxel_size
    def boundary_in_world_y(i_y):
        return grid_min[1] + i_y * voxel_size
    def boundary_in_world_z(i_z):
        return grid_min[2] + i_z * voxel_size

    nx_x = ix + (1 if step_x > 0 else 0)
    nx_y = iy + (1 if step_y > 0 else 0)
    nx_z = iz + (1 if step_z > 0 else 0)

    def safe_div(num, den):
        eps = 1e-12
        if abs(den) < eps:
            return float('inf')
        return num / den

    next_bx = boundary_in_world_x(nx_x)
    next_by = boundary_in_world_y(nx_y)
    next_bz = boundary_in_world_z(nx_z)

    t_max_x = safe_div(next_bx - camera_pos[0].item(), dir_normalized[0].item())
    t_max_y = safe_div(next_by - camera_pos[1].item(), dir_normalized[1].item())
    t_max_z = safe_div(next_bz - camera_pos[2].item(), dir_normalized[2].item())

    t_delta_x = safe_div(voxel_size, abs(dir_normalized[0].item()))
    t_delta_y = safe_div(voxel_size, abs(dir_normalized[1].item()))
    t_delta_z = safe_div(voxel_size, abs(dir_normalized[2].item()))

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

# 7. Main Pipeline
def process_all(metadata_path, images_folder, output_bin):
    frames = load_metadata(metadata_path)
    # Group by camera_index
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

    for cam_frames in frames_by_cam.values():
        if len(cam_frames) < 2:
            continue
        prev_img = load_image_gray(f"{images_folder}/{cam_frames[0].image_file}")
        for i in range(1, len(cam_frames)):
            curr_info = cam_frames[i]
            curr_img = load_image_gray(f"{images_folder}/{curr_info.image_file}")
            changed, diff = detect_motion(prev_img, curr_img, motion_threshold)
            cam_pos = curr_info.camera_position
            cam_rot = rotation_matrix_yaw_pitch_roll(curr_info.yaw, curr_info.pitch, curr_info.roll)
            fov_rad = deg2rad(curr_info.fov_degrees)
            focal_len = (curr_img.shape[1] * 0.5) / np.tan(fov_rad * 0.5)
            for v in range(curr_img.shape[0]):
                for u in range(curr_img.shape[1]):
                    if not changed[v, u]:
                        continue
                    pix_val = diff[v, u].item()
                    if pix_val < 1e-3:
                        continue
                    x = float(u) - 0.5 * curr_img.shape[1]
                    y = - (float(v) - 0.5 * curr_img.shape[0])
                    z = -focal_len
                    ray_cam = torch.tensor([x, y, z], dtype=torch.float32)
                    ray_cam = normalize(ray_cam)
                    ray_world = cam_rot @ ray_cam
                    ray_world = normalize(ray_world)
                    steps = cast_ray_into_grid(cam_pos, ray_world, N, voxel_size, grid_center)
                    for rs in steps:
                        ix, iy, iz, dist = rs
                        attenuation = 1.0 / (1.0 + alpha * dist)
                        val = pix_val * attenuation
                        if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
                            voxel_grid[ix, iy, iz] += val
            prev_img = curr_img

    # Save voxel grid to .bin
    with open(output_bin, "wb") as f:
        f.write(np.array([N], dtype=np.int32).tobytes())
        f.write(np.array([voxel_size], dtype=np.float32).tobytes())
        f.write(voxel_grid.numpy().astype(np.float32).tobytes())
    print(f"Saved voxel grid to {output_bin}")

# Usage:
# process_all("metadata.json", "image_folder", "output_voxel.bin")
import json
import time
import cv2
import mcubes
import numpy as np
import torch
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


# 6. DDA Ray Traversal
def cast_ray_into_grid(camera_pos, dir_normalized, N, voxel_size, grid_center):
    steps = []
    half_size = 0.5 * (N * voxel_size)
    grid_min = grid_center - half_size
    grid_max = grid_center + half_size
    t_min, t_max = 0.0, float("inf")

    for i in range(3):
        origin, d = camera_pos[i].item(), dir_normalized[i].item()
        mn, mx = grid_min[i].item(), grid_max[i].item()
        if abs(d) < 1e-12:
            if origin < mn or origin > mx: return steps
        else:
            t1, t2 = (mn - origin) / d, (mx - origin) / d
            t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
            if t_min > t_max: return steps

    if t_min < 0.0: t_min = 0.0
    start_world = camera_pos + t_min * dir_normalized
    ix, iy, iz = (int((start_world[i] - grid_min[i]) / voxel_size) for i in range(3))
    if not (0 <= ix < N and 0 <= iy < N and 0 <= iz < N): return steps

    step = torch.sign(dir_normalized)
    t_delta = torch.abs(voxel_size / dir_normalized)
    next_bound = grid_min + (torch.floor((start_world - grid_min) / voxel_size) + (step + 1) / 2) * voxel_size
    t_max_v = (next_bound - camera_pos) / dir_normalized
    
    t_current = t_min
    while t_current <= t_max:
        steps.append((ix, iy, iz, t_current))
        t_max_min = torch.min(t_max_v)
        if t_max_v[0] == t_max_min:
            ix += int(step[0].item())
            t_current = t_max_v[0].item()
            t_max_v[0] += t_delta[0]
        elif t_max_v[1] == t_max_min:
            iy += int(step[1].item())
            t_current = t_max_v[1].item()
            t_max_v[1] += t_delta[1]
        else:
            iz += int(step[2].item())
            t_current = t_max_v[2].item()
            t_max_v[2] += t_delta[2]
        
        if not (0 <= ix < N and 0 <= iy < N and 0 <= iz < N): break
    return steps


def _project_motion_to_grid(voxel_grid, changed, diff, cam_info, focal_len, N, voxel_size, grid_center, alpha):
    """Helper to project detected motion from a single frame into the voxel grid."""
    cam_pos = cam_info.camera_position
    cam_rot = rotation_matrix_yaw_pitch_roll(cam_info.yaw, cam_info.pitch, cam_info.roll)
    
    # Vectorized pixel processing
    changed_pixels = torch.nonzero(changed)
    if changed_pixels.numel() == 0:
        return

    v, u = changed_pixels[:, 0], changed_pixels[:, 1]
    pix_val = diff[v, u]

    # Create rays in camera space
    x = u.float() - 0.5 * diff.shape[1]
    y = -(v.float() - 0.5 * diff.shape[0])
    z = torch.full_like(x, -focal_len)
    rays_cam = torch.stack([x, y, z], dim=1)
    rays_cam = torch.nn.functional.normalize(rays_cam, dim=1)

    # Transform rays to world space
    rays_world = torch.matmul(rays_cam, cam_rot.T)

    for i in range(rays_world.shape[0]):
        ray_world = rays_world[i]
        val = pix_val[i].item()
        if val < 1e-3: continue
        
        steps = cast_ray_into_grid(cam_pos, ray_world, N, voxel_size, grid_center)
        for ix, iy, iz, dist in steps:
            attenuation = 1.0 / (1.0 + alpha * dist)
            voxel_grid[ix, iy, iz] += val * attenuation

# 7. Main Pipeline (File-based)
def process_all(
    metadata_path, images_folder, output_bin,
    use_mcubes=False, output_mesh="output_mesh.obj", config=None
):
    if config is None:
        raise ValueError("Configuration must be provided.")
    frames = load_metadata(metadata_path)
    frames_by_cam = {}
    for f in frames:
        frames_by_cam.setdefault(f.camera_index, []).append(f)
    for v in frames_by_cam.values():
        v.sort(key=lambda x: x.frame_index)

    N = config.grid.size[0]
    voxel_size = config.grid.voxel_size
    grid_center = torch.tensor(config.grid.center, dtype=torch.float32)
    motion_threshold = config.motion_detection.threshold
    alpha = 0.1

    voxel_grid = torch.zeros((N, N, N), dtype=torch.float32)

    for cam_frames in frames_by_cam.values():
        if len(cam_frames) < 2: continue
        prev_img = load_image_gray(f"{images_folder}/{cam_frames[0].image_file}")
        for i in range(1, len(cam_frames)):
            curr_info = cam_frames[i]
            curr_img = load_image_gray(f"{images_folder}/{curr_info.image_file}")
            changed, diff = detect_motion(prev_img, curr_img, motion_threshold)
            
            fov_rad = deg2rad(curr_info.fov_degrees)
            focal_len = (curr_img.shape[1] * 0.5) / np.tan(fov_rad * 0.5)
            
            _project_motion_to_grid(voxel_grid, changed, diff, curr_info, focal_len, N, voxel_size, grid_center, alpha)
            prev_img = curr_img

    with open(output_bin, "wb") as f:
        f.write(np.array([N], dtype=np.int32).tobytes())
        f.write(np.array([voxel_size], dtype=np.float32).tobytes())
        f.write(voxel_grid.numpy().astype(np.float32).tobytes())
    print(f"Saved voxel grid to {output_bin}")

    if use_mcubes:
        print("Extracting mesh with PyMCubes...")
        vertices, triangles = mcubes.marching_cubes(voxel_grid.numpy(), config.io.mesh_threshold)
        mcubes.export_obj(vertices, triangles, output_mesh)
        print(f"Saved mesh to {output_mesh}")

# 8. New Pipeline (URL-based)
def process_video_stream(
    video_url: str, camera_info: FrameInfo, config,
    output_bin: str, duration_seconds: int,
    use_mcubes=False, output_mesh="output_mesh.obj"
):
    if config is None:
        raise ValueError("Configuration must be provided.")
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise IOError(f"Cannot open video stream from URL: {video_url}")

    N = config.grid.size[0]
    voxel_size = config.grid.voxel_size
    grid_center = torch.tensor(config.grid.center, dtype=torch.float32)
    motion_threshold = config.motion_detection.threshold
    alpha = 0.1
    voxel_grid = torch.zeros((N, N, N), dtype=torch.float32)

    ret, prev_frame_bgr = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read the first frame from the video stream.")
    
    prev_img = torch.from_numpy(cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    
    fov_rad = deg2rad(camera_info.fov_degrees)
    focal_len = (prev_img.shape[1] * 0.5) / np.tan(fov_rad * 0.5)

    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        ret, curr_frame_bgr = cap.read()
        if not ret:
            print("Video stream ended.")
            break
        
        curr_img = torch.from_numpy(cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        changed, diff = detect_motion(prev_img, curr_img, motion_threshold)
        
        _project_motion_to_grid(voxel_grid, changed, diff, camera_info, focal_len, N, voxel_size, grid_center, alpha)
        prev_img = curr_img

    cap.release()

    with open(output_bin, "wb") as f:
        f.write(np.array([N], dtype=np.int32).tobytes())
        f.write(np.array([voxel_size], dtype=np.float32).tobytes())
        f.write(voxel_grid.numpy().astype(np.float32).tobytes())
    print(f"Saved voxel grid from stream to {output_bin}")

    if use_mcubes:
        print("Extracting mesh from stream data...")
        vertices, triangles = mcubes.marching_cubes(voxel_grid.numpy(), config.io.mesh_threshold)
        mcubes.export_obj(vertices, triangles, output_mesh)
        print(f"Saved mesh to {output_mesh}")

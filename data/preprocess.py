import os
import os.path as osp
import struct

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def project_lidar(points, H=64, W=720, fov_up=3.0, fov_down=-25.0):
    """
    Generate 3-channel colorized range image (RGB).
    """
    fov_up = fov_up / 180.0 * np.pi
    fov_down = fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    depth = np.linalg.norm(points[:, :3], 2, axis=1)
    mask = depth > 0
    points = points[mask]
    depth = depth[mask]

    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(np.clip(scan_z / depth, -1.0, 1.0))

    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov

    proj_x = np.floor(proj_x * W)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)

    proj_y = np.floor(proj_y * H)
    proj_y = np.minimum(H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)

    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.full((H, W), -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    proj_range = np.clip(proj_range, 0, 100)
    proj_range_norm = (proj_range / 100.0 * 255).astype(np.uint8)

    proj_color = cv2.applyColorMap(proj_range_norm, cv2.COLORMAP_JET)
    proj_color = cv2.cvtColor(proj_color, cv2.COLOR_BGR2RGB)
    proj_color = proj_color.astype(np.float32) / 255.0
    proj_color = proj_color.transpose(2, 0, 1) # [3, H, W]

    return proj_color

def sample_points(points, max_points=4096):
    """
    Sample points to max_points and return as [max_points, 3].
    """
    num_points = points.shape[0]
    xyz = points[:, :3]
    if num_points > max_points:
        indices = np.random.choice(num_points, max_points, replace=False)
        xyz = xyz[indices]
    else:
        padding = np.zeros((max_points - num_points, 3), dtype=np.float32)
        xyz = np.concatenate([xyz, padding], axis=0)
    return xyz.astype(np.float32)

def bin_to_points_nclt(bin_path):
    points = []
    with open(bin_path, 'rb') as f:
        while True:
            header = f.read(24)
            if len(header) < 24: break
            for _ in range(384):
                data = f.read(8)
                if len(data) < 8: break
                x_raw, y_raw, z_raw, intensity, laser_id = struct.unpack('<HHHBB', data)
                if x_raw == 0 and y_raw == 0 and z_raw == 0: continue
                x = x_raw * 0.005 - 100
                y = y_raw * 0.005 - 100
                z = z_raw * 0.005 - 100
                points.append([x, y, z, intensity])
    return np.array(points, dtype=np.float32)

def preprocess_dataset(data_dir, dataset_name):
    print(f"Preprocessing {dataset_name} in {data_dir}...")
    
    if dataset_name == 'vreloc':
        full_dir = osp.join(data_dir, "full")
        if not osp.exists(full_dir): return
        seqs = [d for d in os.listdir(full_dir) if d.startswith("seq-")]
        fov_up, fov_down = 30.0, -15.0
        
        for seq in tqdm(seqs):
            seq_path = osp.join(full_dir, seq)
            out_lidar_path = osp.join(data_dir, "velodyne_left_fps_4096_3_float32_npy", seq)
            out_proj_path = osp.join(data_dir, "projected_lidar_64_720_npy", seq)
            os.makedirs(out_lidar_path, exist_ok=True)
            os.makedirs(out_proj_path, exist_ok=True)
            
            bins = [f for f in os.listdir(seq_path) if f.endswith(".bin")]
            for b in bins:
                points = np.fromfile(osp.join(seq_path, b), dtype=np.float32).reshape(-1, 4)

                sampled = sample_points(points)
                np.save(osp.join(out_lidar_path, b.replace(".bin", ".npy")), sampled)

                proj = project_lidar(points, fov_up=fov_up, fov_down=fov_down)
                np.save(osp.join(out_proj_path, b.replace(".bin", ".npy")), proj)

    elif dataset_name == 'nclt':
        velodyne_dir = osp.join(data_dir, "velodyne_data")
        if not osp.exists(velodyne_dir): return
        sessions = os.listdir(velodyne_dir)
        fov_up, fov_down = 3.0, -25.0
        
        for session in tqdm(sessions):
            session_path = osp.join(velodyne_dir, session, "velodyne_sync")
            if not osp.exists(session_path): continue
            out_lidar_path = osp.join(data_dir, "velodyne_left_fps_4096_3_float32_npy", session)
            out_proj_path = osp.join(data_dir, "projected_lidar_64_720_npy", session)
            os.makedirs(out_lidar_path, exist_ok=True)
            os.makedirs(out_proj_path, exist_ok=True)
            
            bins = [f for f in os.listdir(session_path) if f.endswith(".bin")]
            for b in bins:
                points = bin_to_points_nclt(osp.join(session_path, b))
                if points.size == 0: continue

                sampled = sample_points(points)
                np.save(osp.join(out_lidar_path, b.replace(".bin", ".npy")), sampled)

                proj = project_lidar(points, fov_up=fov_up, fov_down=fov_down)
                np.save(osp.join(out_proj_path, b.replace(".bin", ".npy")), proj)

    elif dataset_name == 'robotcar':
        seqs = [d for d in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, d)) and 'radar-oxford-10k' in d]
        fov_up, fov_down = 30.0, -15.0
        
        for seq in tqdm(seqs):
            lidar_dir = osp.join(data_dir, seq, "velodyne_left")
            if not osp.exists(lidar_dir): lidar_dir = osp.join(data_dir, seq, "velodyne_right")
            if not osp.exists(lidar_dir): continue
            
            out_lidar_path = osp.join(data_dir, seq, "velodyne_left_fps_4096_3_float32_npy")
            out_proj_path = osp.join(data_dir, seq, "projected_lidar_64_720_npy")
            os.makedirs(out_lidar_path, exist_ok=True)
            os.makedirs(out_proj_path, exist_ok=True)
            
            files = os.listdir(lidar_dir)
            for f in files:
                if f.endswith(".bin"):
                    points = np.fromfile(osp.join(lidar_dir, f), dtype=np.float32).reshape(-1, 4)
                elif f.endswith(".png"):
                    try:
                        from robotcar_dataset_sdk_pointloc.python.velodyne import load_velodyne_raw, velodyne_raw_to_pointcloud
                        ranges, intensities, angles, _ = load_velodyne_raw(osp.join(lidar_dir, f))
                        ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)
                        points = ptcld.T
                    except: continue
                else: continue
                
                sampled = sample_points(points)
                np.save(osp.join(out_lidar_path, f.replace(".bin", ".npy").replace(".png", ".npy")), sampled)

                proj = project_lidar(points, fov_up=fov_up, fov_down=fov_down)
                np.save(osp.join(out_proj_path, f.replace(".bin", ".npy").replace(".png", ".npy")), proj)


if __name__ == "__main__":
    conf_path = osp.join(osp.dirname(__file__), "..", "conf.yaml")
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)
    
    preprocess_dataset(conf['vReLoc_data_dir'], 'vreloc')
    # preprocess_dataset(conf['nclt_data_dir'], 'nclt')
    # preprocess_dataset(conf['robot_car_data_dir'], 'robotcar')

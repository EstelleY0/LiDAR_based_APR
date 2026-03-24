import os
import torch
import numpy as np
import os.path as osp
from torch.utils import data
from PIL import Image
import torchvision
from scipy.spatial.transform import Rotation as R
from utils.train_utils import set_seed
import cv2
import struct
import scipy.interpolate

set_seed(7)


class NCLT(data.Dataset):
    def __init__(self, data_dir, training):
        self.training = training

        if self.training:
            sessions = ['2012-01-15',
                        '2012-01-22',
                        '2012-02-02',
                        '2012-02-04',
                        '2012-02-05',
                        '2012-02-12',
                        '2012-02-18',
                        '2012-02-19',
                        '2012-03-17',
                        '2012-03-25',
                        '2012-03-31',
                        '2012-04-29',
                        '2012-05-11',
                        '2012-05-26',
                        '2012-06-15',
                        '2012-08-04',
                        '2012-08-20',
                        '2012-09-28',
                        '2012-10-28',
                        '2012-11-04',
                        '2012-11-16',
                        '2012-11-17',
                        '2012-12-01',
                        '2013-01-10',
                        '2013-02-23',
                        '2013-04-05']
        else:
            sessions = ['2012-01-08']

        self.lidar_paths = []
        self.poses = []

        for session in sessions:
            lidar_folder = osp.join(data_dir, 'velodyne_data', session, 'velodyne_sync')
            pose_file_name = 'groundtruth_' + session + '.csv'
            pose_path = osp.join(data_dir, 'ground_truth', pose_file_name)
            cov_file_name = 'cov_' + session + '.csv'
            cov_path = osp.join(data_dir, 'ground_truth_cov', cov_file_name)

            if not osp.exists(lidar_folder) or not osp.exists(pose_path) or not osp.exists(cov_path):
                continue

            ts_list = sorted([int(f.replace('.bin', '')) for f in os.listdir(lidar_folder) if f.endswith('.bin')])

            pose = np.loadtxt(pose_path, delimiter = ",")
            cov = np.loadtxt(cov_path, delimiter = ",")

            t_cov = cov[1:, 0]

            valid_pose = pose[~np.isnan(pose[:, 1:]).any(axis=1)]
            interp = scipy.interpolate.interp1d(valid_pose[:, 0], valid_pose[:, 1:], kind='nearest', axis=0, bounds_error=False, fill_value="extrapolate")
            pose_gt = interp(ts_list)

            x = pose_gt[:, 0]
            y = pose_gt[:, 1]
            z = pose_gt[:, 2]

            r = pose_gt[:, 3]
            p = pose_gt[:, 4]
            h = pose_gt[:, 5]

            positions = np.stack([x, y, z, r, p, h], axis=1)

            if self.training:
                self.mean_t = np.mean(positions[:, :3], axis=0)
                self.std_t = np.std(positions[:, :3], axis=0)
                np.savetxt("nclt_pose_mean_std.txt", np.vstack([self.mean_t, self.std_t]))
            else:
                mean_std = np.loadtxt("nclt_pose_mean_std.txt")
                self.mean_t, self.std_t = mean_std[0], mean_std[1]

            positions[:, :3] = (positions[:, :3] - self.mean_t) / self.std_t

            self.poses.extend(positions)

            for ts in ts_list[:len(positions)]:
                self.lidar_paths.append(osp.join(lidar_folder, f"{ts}.bin"))

        assert len(self.lidar_paths) == len(self.poses)

    def __getitem__(self, index):
        MAX_POINTS = 4096

        pose = torch.tensor(self.poses[index], dtype=torch.float32)

        lidar = self.bin_to_npy(self.lidar_paths[index])
        num_points = lidar.shape[0]

        if num_points > MAX_POINTS:
            indices = np.random.choice(num_points, MAX_POINTS, replace=False)
            lidar = lidar[indices]
        else:
            padding = np.zeros((MAX_POINTS - num_points, 5), dtype=np.float32)
            if lidar.size == 0:
                lidar = padding.copy()
            else:
                lidar = np.concatenate([lidar, padding], axis=0)

        projected_lidar = self.project_lidar(lidar)
        projected_lidar = torch.tensor(projected_lidar, dtype=torch.float32)

        return {
            "lidar_float32": lidar[:, :3],
            "projected_lidar_float32": projected_lidar,
            "image_float32": 1,
            "bev_float32": 1,
            "pose_float32": pose
        }

    def __len__(self):
        return len(self.lidar_paths)

    def bin_to_npy(self, bin_path):
        points = []
        with open(bin_path, 'rb') as f:
            while True:
                header = f.read(24)
                if len(header) < 24:
                    break  # 파일 끝
                for _ in range(384):
                    data = f.read(8)
                    if len(data) < 8:
                        break
                    x_raw, y_raw, z_raw, intensity, laser_id = struct.unpack('<HHHBB', data)
                    if x_raw == 0 and y_raw == 0 and z_raw == 0:
                        continue
                    x = x_raw * 0.005 - 100
                    y = y_raw * 0.005 - 100
                    z = z_raw * 0.005 - 100
                    points.append([x, y, z, intensity, laser_id])
        arr = np.array(points, dtype=np.float32)
        if arr.size == 0:
            return arr.reshape(0, 5)
        return arr


    def quat_to_tangent(self, position, quaternion):
        rotm = R.from_quat(quaternion).as_matrix()
        forward = rotm @ np.array([1, 0, 0])
        return np.concatenate([position, forward], axis=1)

    def project_lidar(self, points, H=64, W=720, fov_up=3.0, fov_down=-25.0):
        fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

        # get depth of all points
        depth = np.linalg.norm(points[:, :3], 2, axis=1)
        points = points[(depth > 0)]
        depth = depth[(depth > 0)]

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= W  # in [0.0, W]
        proj_y *= H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        depth = depth[order]

        proj_y = proj_y[order]
        proj_x = proj_x[order]

        proj_range = np.full((H, W), -1,
                             dtype=np.float32)  # [H,W] range (-1 is no data)

        proj_range[proj_y, proj_x] = depth

        proj_range = np.clip(proj_range, 0, 100)
        proj_range_norm = (proj_range / 100.0 * 255).astype(np.uint8)

        proj_color = cv2.applyColorMap(proj_range_norm, cv2.COLORMAP_JET)
        proj_color = cv2.cvtColor(proj_color, cv2.COLOR_BGR2RGB)
        proj_color = proj_color.astype(np.float32) / 255.0
        proj_color = proj_color.transpose(2, 0, 1)

        return proj_color

import os
import os.path as osp

import numpy as np
import torch
import transforms3d.euler as txe
import transforms3d.quaternions as txq
from torch.utils import data

from utils.train_utils import qexp
from utils.train_utils import set_seed, qlog

set_seed(7)

class VReLoc(data.Dataset):
    def __init__(self, data_dir, training, num_class_loc=10, num_class_ori=10):
        self.training = training
        self.data_dir = data_dir
        self.num_class_loc = num_class_loc
        self.num_class_ori = num_class_ori
        
        split_file = 'TrainSplit.txt' if training else 'TestSplit.txt'
        split_path = osp.join(data_dir, 'full', split_file)
        
        with open(split_path, 'r') as f:
            lines = f.readlines()
            seq_names = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
            
        self.lidar_paths = []
        self.projected_paths = []
        self.poses = []
        
        all_positions = []
        
        for seq_name in seq_names:
            # Map sequenceX to seq-XX
            seq_num = seq_name.replace('sequence', '')
            seq_folder = f"seq-{int(seq_num):02d}"
            
            pre_lidar_seq_path = osp.join(data_dir, "velodyne_left_fps_4096_3_float32_npy", seq_folder)
            pre_proj_seq_path = osp.join(data_dir, "projected_lidar_64_720_npy", seq_folder)
            
            raw_seq_path = osp.join(data_dir, 'full', seq_folder)
            if not osp.exists(raw_seq_path):
                continue

            frames = sorted([f.replace('.bin', '') for f in os.listdir(raw_seq_path) if f.endswith('.bin')])
            
            for frame in frames:
                pose_path = osp.join(raw_seq_path, f"{frame}.pose.txt")
                if not osp.exists(pose_path):
                    continue

                lidar_path = osp.join(pre_lidar_seq_path, f"{frame}.npy")
                proj_path = osp.join(pre_proj_seq_path, f"{frame}.npy")
                
                if osp.exists(lidar_path) and osp.exists(proj_path):
                    self.lidar_paths.append(lidar_path)
                    self.projected_paths.append(proj_path)
                    
                    matrix = np.loadtxt(pose_path, delimiter=',')
                    t = matrix[:3, 3]
                    R = matrix[:3, :3]
                    q = txq.mat2quat(R)
                    q *= np.sign(q[0])
                    q_l = qlog(q)
                    
                    pose_vec = np.concatenate([t, q_l])
                    self.poses.append(pose_vec)
                    all_positions.append(t)
                
        self.poses = np.array(self.poses)
        all_positions = np.array(all_positions)
        
        stats_file = osp.join(data_dir, "vreloc_pose_mean_std.txt")
        if self.training:
            self.mean_t = np.mean(all_positions, axis=0)
            self.std_t = np.std(all_positions, axis=0)
            np.savetxt(stats_file, np.vstack([self.mean_t, self.std_t]))
        else:
            if osp.exists(stats_file):
                mean_std = np.loadtxt(stats_file)
                self.mean_t, self.std_t = mean_std[0], mean_std[1]
            else:
                # Fallback if training wasn't run or stats file missing
                self.mean_t = np.mean(all_positions, axis=0)
                self.std_t = np.std(all_positions, axis=0)

        self.poses[:, :3] = (self.poses[:, :3] - self.mean_t) / (self.std_t + 1e-8)

        pose_max_min_file = osp.join(data_dir, "vreloc_pose_max_min.txt")
        if self.training:
            self.pose_max = np.max(self.poses[:, :2], axis=0)
            self.pose_min = np.min(self.poses[:, :2], axis=0)
            np.savetxt(pose_max_min_file, np.vstack([self.pose_max, self.pose_min]))
        else:
            if osp.exists(pose_max_min_file):
                max_min = np.loadtxt(pose_max_min_file)
                self.pose_max, self.pose_min = max_min[0], max_min[1]
            else:
                self.pose_max = np.max(self.poses[:, :2], axis=0)
                self.pose_min = np.min(self.poses[:, :2], axis=0)

    def __len__(self):
        return len(self.lidar_paths)

    def __getitem__(self, index):
        lidar = np.load(self.lidar_paths[index])
        lidar = lidar[:, :3]
        
        projected_lidar = np.load(self.projected_paths[index])

        projected_lidar = torch.tensor(projected_lidar, dtype=torch.float32)
        lidar = torch.tensor(lidar, dtype=torch.float32)
        pose = torch.tensor(self.poses[index], dtype=torch.float32)
        
        x = (self.poses[index][0] - self.pose_min[0]) / (self.pose_max[0] - self.pose_min[0] + 1e-8)
        y = (self.poses[index][1] - self.pose_min[1]) / (self.pose_max[1] - self.pose_min[1] + 1e-8)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        x_idx = int(np.minimum(x * self.num_class_loc, self.num_class_loc - 1))
        y_idx = int(np.minimum(y * self.num_class_loc, self.num_class_loc - 1))
        cls_loc = x_idx * self.num_class_loc + y_idx

        quat = qexp(self.poses[index][3:])
        _, _, yaw = txe.quat2euler(quat)
        theta = np.degrees(yaw)
        theta = (theta + 180) % 360 - 180
        
        cls_ori = (theta + 180) / 360.0
        cls_ori = int(np.minimum(cls_ori * self.num_class_ori, self.num_class_ori - 1))

        return {
            "lidar_float32": lidar,
            "pose_float32": pose,
            "image_float32": 1,
            "bev_float32": 1,
            "projected_lidar_float32": projected_lidar,
            "cls_loc": torch.tensor(cls_loc, dtype=torch.long),
            "cls_ori": torch.tensor(cls_ori, dtype=torch.long)
        }

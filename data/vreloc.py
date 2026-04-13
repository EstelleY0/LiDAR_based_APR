import os
import torch
import numpy as np
import os.path as osp
from torch.utils import data
from utils.train_utils import set_seed, qlog, load_config_as_namespace
import transforms3d.quaternions as txq

set_seed(7)

class VReLoc(data.Dataset):
    def __init__(self, data_dir, training):
        self.training = training
        self.data_dir = data_dir
        
        split_file = 'TrainSplit.txt' if training else 'TestSplit.txt'
        split_path = osp.join(data_dir, 'full', split_file)
        
        with open(split_path, 'r') as f:
            lines = f.readlines()
            seq_names = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
            
        self.lidar_paths = []
        self.poses = []
        
        all_positions = []
        
        for seq_name in seq_names:
            # Map sequenceX to seq-XX
            seq_num = seq_name.replace('sequence', '')
            seq_folder = f"seq-{int(seq_num):02d}"
            seq_path = osp.join(data_dir, 'full', seq_folder)
            
            if not osp.exists(seq_path):
                print(f"Warning: {seq_path} does not exist. Skipping.")
                continue
                
            frames = sorted([f.replace('.bin', '') for f in os.listdir(seq_path) if f.endswith('.bin')])
            
            for frame in frames:
                lidar_path = osp.join(seq_path, f"{frame}.bin")
                pose_path = osp.join(seq_path, f"{frame}.pose.txt")
                
                if not osp.exists(pose_path):
                    continue
                    
                self.lidar_paths.append(lidar_path)
                
                # Load pose 4x4 matrix
                matrix = np.loadtxt(pose_path, delimiter=',')
                
                # Extract translation
                t = matrix[:3, 3]
                
                # Extract rotation and convert to log quaternion
                R = matrix[:3, :3]
                q = txq.mat2quat(R)
                q *= np.sign(q[0]) # constrain to hemisphere
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

    def __len__(self):
        return len(self.lidar_paths)

    def __getitem__(self, index):
        MAX_POINTS = 4096
        
        # Load lidar [x,y,z,i] float32
        lidar_all = np.fromfile(self.lidar_paths[index], dtype=np.float32).reshape(-1, 4)
        lidar = lidar_all[:, :3] # only [x,y,z]
        
        num_points = lidar.shape[0]
        if num_points > MAX_POINTS:
            indices = np.random.choice(num_points, MAX_POINTS, replace=False)
            lidar = lidar[indices]
        else:
            padding = np.zeros((MAX_POINTS - num_points, 3), dtype=np.float32)
            lidar = np.concatenate([lidar, padding], axis=0)
            
        pose = torch.tensor(self.poses[index], dtype=torch.float32)
        lidar = torch.tensor(lidar, dtype=torch.float32)
        
        return {
            "lidar_float32": lidar,
            "pose_float32": pose,
            "image_float32": 1,
            "bev_float32": 1,
            "projected_lidar_float32": 1 # Placeholder as we don't need it for basic training
        }

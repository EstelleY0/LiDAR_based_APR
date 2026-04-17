import os
import os.path as osp

import numpy as np
import torch
import torchvision
import transforms3d.euler as txe
from PIL import Image
from torch.utils import data

from robotcar_dataset_sdk_pointloc.python.interpolate_poses import interpolate_ins_poses
from utils.train_utils import load_config_as_namespace
from utils.train_utils import process_poses
from utils.train_utils import qexp
from utils.train_utils import set_seed

set_seed(7)

conf = load_config_as_namespace('conf.yaml')

class RobotCar(data.Dataset):
    def __init__(self, data_dir, training, num_class_loc=10, num_class_ori=10):
        self.training = training
        self.num_class_loc = num_class_loc
        self.num_class_ori = num_class_ori

        if self.training:
            seqs = [
                '2019-01-11-14-02-26-radar-oxford-10k',
                '2019-01-14-12-05-52-radar-oxford-10k',
                '2019-01-14-14-48-55-radar-oxford-10k',
                '2019-01-18-15-20-12-radar-oxford-10k',
            ]
        elif not self.training:
            if conf.scene == 'full6':
                seqs = ['2019-01-10-11-46-21-radar-oxford-10k']  # full 6

            elif conf.scene == 'full7':
                seqs = ['2019-01-15-13-06-37-radar-oxford-10k']  # full 7

            elif conf.scene == 'full8':
                seqs = ['2019-01-17-14-03-00-radar-oxford-10k']  # full 8

            elif conf.scene == 'full9':
                seqs = ['2019-01-18-14-14-42-radar-oxford-10k']  # full 9
        seqs = ['2019-01-18-14-14-42-radar-oxford-10k']  # full 9

        ps = {}
        ts = {}
        self.lidar_paths = []
        self.projected_lidar_paths = []
        all_poses_length = 0
        for seq in seqs:
            lidars_folder = osp.join(data_dir, seq, 'velodyne_left_fps_4096_3_float32_npy')
            lidars_list = os.listdir(lidars_folder)
            lidars_list = sorted(lidars_list)
            lidars_list = [
                int(lidar_name.replace('.npy', '')) for lidar_name in lidars_list if lidar_name.endswith('.npy')
            ]

            ts_filename = osp.join(data_dir, seq, 'velodyne_left.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]
            assert ts[seq] == lidars_list

            ts[seq] = ts[seq][5:-5]
            lidars_list = lidars_list[5:-5]
            assert ts[seq] == lidars_list

            # poses from PointLoc
            pose_filename = osp.join(data_dir, seq, 'gps', 'ins.csv')
            p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq].copy(), ts[seq][0]))
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))

            for lidar_name_pure in lidars_list:
                lidar_path = osp.join(lidars_folder, str(lidar_name_pure) + '.npy')
                self.lidar_paths.append(lidar_path)

            projected_lidars_folder = osp.join(data_dir, seq, 'projected_lidar_64_720_shifted')
            for lidar_name_pure in lidars_list:
                projected_lidar_path = osp.join(projected_lidars_folder, str(lidar_name_pure) + '.png')
                self.projected_lidar_paths.append(projected_lidar_path)

            all_poses_length += len(ps[seq])

        assert all_poses_length == len(self.lidar_paths)
        assert all_poses_length == len(self.projected_lidar_paths)

        pose_stats_filename = osp.join(data_dir, 'pose_stats_full1234.txt')
        self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)
        self.poses = np.empty([0, 6])
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
                                align_R=np.eye(3), align_t=np.zeros(3),
                                align_s=1)
            self.poses = np.vstack((self.poses, pss))

        # Calculate pose_max and pose_min for normalized x, y
        pose_max_min_file = osp.join(data_dir, 'robotcar_pose_max_min.txt')
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
        return len(self.poses)

    def __getitem__(self, index):
        lidar = np.load(self.lidar_paths[index], allow_pickle=True)
        projected_lidar = Image.open(self.projected_lidar_paths[index]).convert('RGB')

        pose = self.poses[index].copy()

        shuffle_ids = np.random.choice(len(lidar), size=len(lidar), replace=False)
        lidar = lidar[shuffle_ids]

        lidar = torch.tensor(lidar, dtype=torch.float32)
        lidar = lidar / conf.divide_factor

        data_transform = []
        data_transform.append(torchvision.transforms.ToTensor())
        data_transform.append(torchvision.transforms.Normalize(mean=0.5, std=1))
        data_transform = torchvision.transforms.Compose(data_transform)

        projected_lidar = data_transform(projected_lidar)
        c, h, w = projected_lidar.shape

        projected_lidar = projected_lidar[:, h // 2:, :]

        pose = torch.tensor(pose, dtype=torch.float32)

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
            'lidar_float32': lidar,
            'projected_lidar_float32': projected_lidar,
            'image_float32': 1,
            'bev_float32': 1,
            'pose_float32': pose,
            'cls_loc': torch.tensor(cls_loc, dtype=torch.long),
            'cls_ori': torch.tensor(cls_ori, dtype=torch.long)
        }

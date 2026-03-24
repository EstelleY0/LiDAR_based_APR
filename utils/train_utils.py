import torch.distributed as dist
import torch
import os
import os.path as osp
import random
import shutil
import sys
from argparse import Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import transforms3d.quaternions as txq
import yaml
from torchvision.datasets.folder import default_loader

def setup(rank, world_size, local_gpu_id):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_gpu_id)

def cleanup():
    dist.destroy_process_group()

def load_config_as_namespace(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)



def load_state_dict(model, state_dict):
    model_names = [n for n, _ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

def set_seed(seed=7):
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def quaternion_angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta

def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
    return q

def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    assert poses_in.shape[-1] == 12
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out

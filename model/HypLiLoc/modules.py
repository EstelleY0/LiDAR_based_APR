import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.pointnet.pointnet_utils import PointNetSetAbstraction

class SphericalProjection(nn.Module):
    def __init__(self, h=64, w=900, fov_up=3.0, fov_down=-25.0):
        super().__init__()
        self.h = h
        self.w = w
        self.fov_up = fov_up / 180.0 * math.pi
        self.fov_down = fov_down / 180.0 * math.pi
        self.fov = self.fov_up - self.fov_down

    def forward(self, x):
        # x: [B, N, 3] (x, y, z)
        B, N, _ = x.shape
        depth = torch.norm(x, dim=-1) # [B, N]
        yaw = -torch.atan2(x[..., 1], x[..., 0]) # [B, N]
        pitch = torch.asin(torch.clamp(x[..., 2] / (depth + 1e-8), -1.0, 1.0)) # [B, N]

        # Normalize to [0, 1]
        v = (pitch - self.fov_down) / self.fov
        u = 0.5 * (yaw / math.pi + 1.0)

        # Scale to grid size
        v = torch.clamp(v * self.h, 0, self.h - 1).long()
        u = torch.clamp(u * self.w, 0, self.w - 1).long()

        # Create projection image [B, C, H, W]
        # Channels: x, y, z, depth, intensity (1.0)
        proj = torch.zeros((B, 5, self.h, self.w), device=x.device)
        
        # Batch processing for projection is tricky in pure PyTorch without loops or specialized kernels
        # For simplicity and style consistency, we use a loop over batch
        for b in range(B):
            proj[b, 0, v[b], u[b]] = x[b, :, 0]
            proj[b, 1, v[b], u[b]] = x[b, :, 1]
            proj[b, 2, v[b], u[b]] = x[b, :, 2]
            proj[b, 3, v[b], u[b]] = depth[b]
            proj[b, 4, v[b], u[b]] = 1.0 
        return proj

class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.q = nn.Linear(in_channels, out_channels)
        self.k = nn.Linear(in_channels, out_channels)
        self.v = nn.Linear(in_channels, out_channels)
        self.scale = math.sqrt(out_channels)

    def forward(self, x):
        # x: [B, N, C]
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

class SAGALayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.sa = PointNetSetAbstraction(npoint, radius, nsample, in_channel, mlp, group_all=False)
        self.ga = GraphAttention(mlp[-1], mlp[-1])

    def forward(self, xyz, features):
        # PointNetSetAbstraction expects xyz: [B, 3, N], features: [B, D, N]
        # and returns new_xyz: [B, 3, S], new_features: [B, D', S]
        new_xyz, new_features = self.sa(xyz, features)
        
        # GA expects [B, S, D']
        new_features = new_features.permute(0, 2, 1)
        new_features = self.ga(new_features)
        new_features = new_features.permute(0, 2, 1)
        
        return new_xyz, new_features

class HyperbolicFusion(nn.Module):
    def __init__(self, dim, c=1.0):
        super().__init__()
        self.c = c # Curvature
        self.proj_e = nn.Linear(dim, dim)
        self.proj_h = nn.Linear(dim, dim)

    def exp_map(self, x):
        # Euclidean to Hyperbolic (Poincaré Ball)
        norm = torch.norm(x, dim=-1, keepdim=True)
        sqrt_c = math.sqrt(self.c)
        # Handle division by zero
        res = torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm + 1e-8)
        return res

    def log_map(self, y):
        # Hyperbolic to Euclidean
        norm = torch.norm(y, dim=-1, keepdim=True)
        sqrt_c = math.sqrt(self.c)
        # Clamp norm to avoid atanh(1.0)
        norm = torch.clamp(norm * sqrt_c, -1.0 + 1e-7, 1.0 - 1e-7)
        res = torch.atanh(norm) * y / (torch.norm(y, dim=-1, keepdim=True) * sqrt_c + 1e-8)
        return res

    def forward(self, f_3d, f_2d):
        # Euclidean Interaction
        f_e = self.proj_e(f_3d + f_2d)
        
        # Hyperbolic Interaction
        h_3d = self.exp_map(f_3d)
        h_2d = self.exp_map(f_2d)
        # Simple addition in hyperbolic space (approximation)
        f_h = self.log_map(h_3d + h_2d) 
        f_h = self.proj_h(f_h)
        
        return f_e + f_h

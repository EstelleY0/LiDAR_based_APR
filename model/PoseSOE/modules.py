import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet.pointnet_utils import PointNetSetAbstraction, index_points, farthest_point_sample, query_ball_point


class PointwiseOrientationEncoding(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointwiseOrientationEncoding, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3 + 3 # features + relative_xyz + orientation_features
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points, nsample=32, radius=0.4):
        """
        Input:
            xyz: input points position data, [B, 3, N]
            points: input points data, [B, D, N]
        Return:
            new_points: encoded point features, [B, D', N]
        """
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1) # [B, N, 3]
        if points is not None:
            points_t = points.permute(0, 2, 1) # [B, N, D]
        else:
            points_t = None
            
        # For each point, find neighbors
        # We use all points as query points to keep N points
        idx = query_ball_point(radius, nsample, xyz_t, xyz_t)
        grouped_xyz = index_points(xyz_t, idx) # [B, N, nsample, 3]
        
        # Relative coordinates
        relative_xyz = grouped_xyz - xyz_t.view(B, N, 1, 3) # [B, N, nsample, 3]
        
        # Orientation features: here we use a simple projection of relative coordinates 
        # as a placeholder for more complex LRF-based features.
        # In SOE-Net, they use the projection onto the 3 axes of the LRF.
        # Since we don't compute LRF here, we'll just use relative_xyz and its norm.
        dist = torch.norm(relative_xyz, dim=-1, keepdim=True) # [B, N, nsample, 1]
        orientation_features = torch.cat([relative_xyz, dist, torch.zeros_like(dist), torch.zeros_like(dist)], dim=-1)[:, :, :, :3] # [B, N, nsample, 3]
        
        if points_t is not None:
            grouped_points = index_points(points_t, idx) # [B, N, nsample, D]
            new_points = torch.cat([relative_xyz, orientation_features, grouped_points], dim=-1) # [B, N, nsample, 3+3+D]
        else:
            new_points = torch.cat([relative_xyz, orientation_features], dim=-1) # [B, N, nsample, 3+3]
            
        new_points = new_points.permute(0, 3, 2, 1) # [B, C', nsample, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=True)
            
        new_points = torch.max(new_points, 2)[0] # [B, D', N]
        return new_points


class PointwiseAttention(nn.Module):
    def __init__(self, in_channel):
        super(PointwiseAttention, self).__init__()
        self.conv = nn.Conv1d(in_channel, 1, 1)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        """
        Input:
            x: [B, D, N]
        Return:
            x: [B, D, N] (weighted)
        """
        attn = torch.sigmoid(self.bn(self.conv(x))) # [B, 1, N]
        return x * attn


class PoseSOEEncoder(nn.Module):
    def __init__(self):
        super(PoseSOEEncoder, self).__init__()
        # Simplified SOE-Net style encoder
        self.poe1 = PointwiseOrientationEncoding(in_channel=0, mlp=[32, 32, 64])
        self.pa1 = PointwiseAttention(64)
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.4, nsample=32, in_channel=64 + 3, mlp=[64, 128, 256], group_all=False)
        self.pa2 = PointwiseAttention(256)
        
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.8, nsample=16, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=False)
        self.pa3 = PointwiseAttention(1024)
        
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024 + 3, mlp=[1024, 1024, 1024], group_all=True)

    def forward(self, x):
        # x is [B, 3, N]
        f = self.poe1(x, None) # [B, 64, N]
        f = self.pa1(f)
        
        xyz, f = self.sa1(x, f) # [B, 256, 512]
        f = self.pa2(f)
        
        xyz, f = self.sa2(xyz, f) # [B, 512, 128]
        f = self.pa3(f)
        
        _, f = self.sa3(xyz, f) # [B, 1024, 1]
        return f.squeeze(-1) # [B, 1024]

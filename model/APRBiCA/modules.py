import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet.pointnet_utils import PointNetSetAbstraction

class BiCA(nn.Module):
    """
    Bidirectional Cross-Attention (BiCA) Module
    Fuses two feature sets (e.g., spatial and semantic) bidirectionally.
    """
    def __init__(self, dim):
        super(BiCA, self).__init__()
        self.q1 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)
        
        self.q2 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)
        
        self.scale = dim ** -0.5

    def forward(self, f1, f2):
        # f1: [B, C, N1] 
        # f2: [B, C, N2]
        B, C, N1 = f1.shape
        _, _, N2 = f2.shape
        
        f1_t = f1.transpose(1, 2) # [B, N1, C]
        f2_t = f2.transpose(1, 2) # [B, N2, C]
        
        # Path 1: f1 attends to f2
        q1 = self.q1(f1_t)
        k2 = self.k1(f2_t)
        v2 = self.v1(f2_t)
        attn1 = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.matmul(attn1, v2) # [B, N1, C]
        
        # Path 2: f2 attends to f1
        q2 = self.q2(f2_t)
        k1 = self.k2(f1_t)
        v1 = self.v2(f1_t)
        attn2 = torch.matmul(q2, k1.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        out2 = torch.matmul(attn2, v1) # [B, N2, C]
        
        return out1.transpose(1, 2), out2.transpose(1, 2)

class GatingUnit(nn.Module):
    """
    Gating Unit to control the flow of information.
    """
    def __init__(self, dim):
        super(GatingUnit, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, attended):
        # x: [B, C, N]
        # attended: [B, C, N]
        B, C, N = x.shape
        x_t = x.transpose(1, 2) # [B, N, C]
        att_t = attended.transpose(1, 2) # [B, N, C]
        
        g = self.gate(torch.cat([x_t, att_t], dim=-1))
        out = g * x_t + (1.0 - g) * att_t
        return out.transpose(1, 2)

class PointNet2Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(PointNet2Encoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=16, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 512, 1024], group_all=True)

    def forward(self, xyz, points):
        # xyz: [B, 3, N], points: [B, D, N]
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        return l1_points, l2_points, l3_points, l4_points

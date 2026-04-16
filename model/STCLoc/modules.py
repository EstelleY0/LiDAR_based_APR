# Modified from PSYZ1234/STCLoc (https://github.com/PSYZ1234/STCLoc)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet.pointnet_utils import PointNetSetAbstraction


def similarity(feat1, feat2, feat3):
    B, _, D  = feat1.size()
    feat1    = feat1.view(B, -1)  # [B, D]
    feat2    = feat2.view(B, -1)  # [B, D]
    feat3    = feat3.view(B, -1)  # [B, D]
    
    w1       = F.pairwise_distance(feat1, feat2, p=2, keepdim=True)  # [B, 1]
    w2       = F.pairwise_distance(feat1, feat3, p=2, keepdim=True)  # [B, 1]
    w1       = 1 / (1 + w1)  # [B, 1] 
    w2       = 1 / (1 + w2)  # [B, 1]
    
    w3       = torch.cosine_similarity(feat1, feat2, dim=1)  # [B, 1]
    w4       = torch.cosine_similarity(feat1, feat3, dim=1)  # [B, 1]
    w3       = w3.view(B, -1)  # [B, 1]
    w4       = w4.view(B, -1)  # [B, 1]
    w3       = 0.5 + 0.5 * w3  # [B, 1]
    w4       = 0.5 + 0.5 * w4  # [B, 1]

    # weighted 
    feat_out = feat1 + (w1 + w3) * feat2  + (w2 + w4) * feat3  # [B, D]
    feat_out = feat_out.view(B, 1, D)  # [B, 1, D]

    return feat_out

class FeatureCorrelation(nn.Module):
    def __init__(self, steps, feat_size):
        super(FeatureCorrelation, self).__init__()
        self.steps = steps       
        self.feat_size = feat_size
        self.pos_embedding = nn.Parameter(torch.randn(1, steps, feat_size))   

    def forward(self, feat_in):
        # TAFA
        if self.steps <= 1:
            return feat_in
        
        B                   = feat_in.size(0) // self.steps  # B
        feat_in             = feat_in.view(B, self.steps, self.feat_size)  # [B, T, D]
        feat_in             = feat_in + self.pos_embedding   # [B, T, D]
        if self.steps == 3:
            feat1, feat2, feat3 = torch.split(feat_in, 1, dim=1)  # [B, 1, D]*2
            feat1_new           = similarity(feat1, feat2, feat3)  # [B, 1, D]
            feat2_new           = similarity(feat2, feat1, feat3)  # [B, 1, D]
            feat3_new           = similarity(feat3, feat1, feat2)  # [B, 1, D]         
            feat_out            = torch.cat((feat1_new, feat2_new, feat3_new), dim=1)  # (B, T, D)
            feat_out = feat_out.view(B*self.steps, self.feat_size)   # [B*T, D]
            return feat_out
        else:
            # Fallback for steps != 3
            return feat_in.view(B*self.steps, self.feat_size)

class STCLocEncoder(nn.Module):
    def __init__(self, steps=1, feature_correlation=False):
        super(STCLocEncoder, self).__init__()
        # Oxford settings from original STCLoc paper
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=4.0, nsample=32, in_channel=3, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=8.0, nsample=16, in_channel=64 + 3, mlp=[64, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.correlation = FeatureCorrelation(steps, 1024)
        self.feature_correlation = feature_correlation

    def forward(self, input):
        # input is [B, 3, N]
        xyz, f = self.sa1(input, None)
        xyz, f = self.sa2(xyz, f)
        _, f = self.sa3(xyz, f)
        f = f.squeeze(-1) # [B, 1024]
        
        if self.feature_correlation:
            f = self.correlation(f)
            
        return f

class STCLocDecoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(STCLocDecoder, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            x  = F.relu(bn(fc(x)))  # [B, D]
        
        return x

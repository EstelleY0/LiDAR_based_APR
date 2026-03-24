import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet.pointnet_utils import PointNetSetAbstraction


class PointCloudEncoder(nn.Module):
    def __init__(self):
        super(PointCloudEncoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=2048, radius=0.2, nsample=64,in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.4, nsample=32,in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=512, radius=0.8, nsample=16, in_channel=256+3, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=256, radius=1.2, nsample=16, in_channel=256+3, mlp=[128, 128, 256], group_all=False)

    def forward(self, input):
        xyz, f = self.sa1(input, None)
        xyz, f = self.sa2(xyz, f)
        xyz, f = self.sa3(xyz, f)
        xyz, f = self.sa4(xyz, f)
        return xyz, f

class SelfAttnModule(nn.Module):
    def __init__(self, feature_dim=256):
        super(SelfAttnModule, self).__init__()

        self.feature_dim = feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, feature):
        mask = self.mlp(feature)
        weighted_feature = feature * mask
        return weighted_feature


class GroupAllLayer(nn.Module):
    def __init__(self, in_channel=256):
        super(GroupAllLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channel, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.fc_layer = nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.mlp(x)

        x = F.max_pool1d(x, x.size(-1))

        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)

        return x

class PoseRegressor(nn.Module):
    def __init__(self):
        super(PoseRegressor, self).__init__()

        self.trans = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 3)
        )

        self.logq = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        t = self.trans(x)
        r = self.logq(x)

        return torch.cat([t, r], dim=-1)

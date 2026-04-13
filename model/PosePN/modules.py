# Modified from PSYZ1234/PosePN (https://github.com/PSYZ1234/PosePN)

import torch.nn as nn

from model.pointnet.pointnet_utils import PointNetSetAbstraction


class PosePNPPEncoder(nn.Module):
    def __init__(self):
        super(PosePNPPEncoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=2048, radius=0.2, nsample=64, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=512, radius=0.8, nsample=16, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=256, radius=1.2, nsample=16, in_channel=512 + 3, mlp=[512, 512, 1024], group_all=False)
        self.sa5 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024 + 3, mlp=[1024, 1024, 1024], group_all=True)

    def forward(self, input):
        if input.shape[-1] == 3:
            input = input.transpose(1, 2)
        xyz, f = self.sa1(input, None)
        xyz, f = self.sa2(xyz, f)
        xyz, f = self.sa3(xyz, f)
        xyz, f = self.sa4(xyz, f)
        _, f = self.sa5(xyz, f)
        return f.squeeze(-1) # [B, 1024]

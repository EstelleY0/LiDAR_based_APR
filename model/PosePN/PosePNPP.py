import torch.nn as nn

from model.PosePN.modules import PosePNPPEncoder
from model.utils import MARegressor


class PosePNPP(nn.Module):
    def __init__(self, hidden_units=512, freeze_backbone=False):
        super(PosePNPP, self).__init__()
        self.encoder = PosePNPPEncoder()
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.regressor = MARegressor(in_channel=1024, hidden_units=hidden_units)

    def forward(self, input):
        if input.shape[-1] == 3:
            input = input.transpose(1, 2)
        feature = self.encoder(input)
        pose = self.regressor(feature)
        return pose

import torch.nn as nn
from model.PosePN.modules import PosePNEncoder
from model.utils import MARegressor

class PosePN(nn.Module):
    def __init__(self, hidden_units=512, freeze_backbone=False):
        super(PosePN, self).__init__()
        self.encoder = PosePNEncoder()
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.regressor = MARegressor(in_channel=1024, hidden_units=hidden_units)

    def forward(self, input):
        # input is [B, N, 3] or [B, 3, N]
        if input.shape[-1] == 3:
            input = input.transpose(1, 2)
            
        feature = self.encoder(input)
        pose = self.regressor(feature)
        return pose

import torch.nn as nn
from model.PoseSOE.modules import PoseSOEEncoder
from model.utils import MARegressor

class PoseSOE(nn.Module):
    def __init__(self, hidden_units=512, freeze_backbone=False):
        super(PoseSOE, self).__init__()
        self.encoder = PoseSOEEncoder()
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

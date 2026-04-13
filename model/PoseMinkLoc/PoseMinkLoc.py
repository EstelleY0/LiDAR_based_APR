import torch
import torch.nn as nn
import MinkowskiEngine as ME
from model.PoseMinkLoc.modules import MinkResNet
from model.utils import MARegressor

class PoseMinkLoc(nn.Module):
    def __init__(self, hidden_units=512, freeze_backbone=False, grid_size=0.01):
        super(PoseMinkLoc, self).__init__()
        self.grid_size = grid_size
        self.encoder = MinkResNet(in_channels=1, out_channels=1024)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.regressor = MARegressor(in_channel=1024, hidden_units=hidden_units)

    def forward(self, input):
        # input is [B, N, 3] or [B, 3, N]
        if input.shape[-1] != 3:
            input = input.transpose(1, 2) # [B, N, 3]
            
        device = input.device
        batch_size = input.shape[0]
        
        # Voxelization: Convert to discrete coordinates
        coords = []
        features = []
        for i in range(batch_size):
            # Scale and cast to int for voxelization
            c = torch.floor(input[i] / self.grid_size).int()
            # Unique voxels (Minkowski Engine can handle duplicates but unique is more efficient)
            c, idx = ME.utils.sparse_quantize(c, return_index=True)
            coords.append(c)
            features.append(torch.ones((len(c), 1), device=device))
            
        batch_coords, batch_features = ME.utils.sparse_collate(coords, features)
        
        st = ME.SparseTensor(features=batch_features, coordinates=batch_coords, device=device)
        
        feature_vector = self.encoder(st) # [B, 1024]
        pose = self.regressor(feature_vector)
        return pose

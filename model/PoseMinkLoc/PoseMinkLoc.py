import torch
import torch.nn as nn
from model.utils import MARegressor

class PoseMinkLoc(nn.Module):
    def __init__(self, hidden_units=512, freeze_backbone=False, grid_size=0.01, sparse_engine='spconv'):
        super(PoseMinkLoc, self).__init__()
        self.grid_size = grid_size
        self.sparse_engine = sparse_engine
        
        if sparse_engine == 'minkowski':
            try:
                import MinkowskiEngine as ME
                from model.PoseMinkLoc.modules import MinkResNet
                self.encoder = MinkResNet(in_channels=1, out_channels=1024)
            except ImportError:
                raise ImportError("MinkowskiEngine is required for sparse_engine='minkowski'")
        else: # Default spconv
            try:
                import spconv.pytorch as spconv
                from model.PoseMinkLoc.modules_spconv import MinkResNet_spconv
                self.encoder = MinkResNet_spconv(in_channels=1, out_channels=1024)
            except ImportError:
                raise ImportError("spconv is required for sparse_engine='spconv'")
                
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
        
        if self.sparse_engine == 'minkowski':
            import MinkowskiEngine as ME
            coords = []
            features = []
            for i in range(batch_size):
                c = torch.floor(input[i] / self.grid_size).int()
                c, idx = ME.utils.sparse_quantize(c, return_index=True)
                coords.append(c)
                features.append(torch.ones((len(c), 1), device=device))
            batch_coords, batch_features = ME.utils.sparse_collate(coords, features)
            st = ME.SparseTensor(features=batch_features, coordinates=batch_coords, device=device)
        else:
            import spconv.pytorch as spconv
            coords_list = []
            features_list = []
            for i in range(batch_size):
                # Voxelize
                c = torch.floor(input[i] / self.grid_size).int()
                # Unique voxels
                # Simplified: use torch.unique to simulate spconv/ME quantize
                c_unique, idx = torch.unique(c, dim=0, return_inverse=True)
                
                # spconv expects indices with batch dimension [N, 4] (batch, z, y, x)
                # Note: original order was (x, y, z), we'll keep it for consistency but spconv usually uses z,y,x
                b_idx = torch.full((c_unique.shape[0], 1), i, dtype=torch.int32, device=device)
                c_with_batch = torch.cat([b_idx, c_unique], dim=1)
                
                coords_list.append(c_with_batch)
                features_list.append(torch.ones((c_unique.shape[0], 1), device=device))
                
            batch_coords = torch.cat(coords_list, dim=0)
            batch_features = torch.cat(features_list, dim=0)
            
            # spatial_shape is needed for spconv. [z, y, x]
            # Since our space is not bounded, we use a large enough shape or dynamic
            # Or just use spconv.SparseConvTensor without strict spatial_shape if possible
            # Actually spconv needs spatial_shape for some ops.
            # We use the max coords as a hint or a very large value
            spatial_shape = [1000000, 1000000, 1000000] 
            
            st = spconv.SparseConvTensor(features=batch_features, indices=batch_coords, 
                                        spatial_shape=spatial_shape, batch_size=batch_size)
        
        feature_vector = self.encoder(st) # [B, 1024]
        pose = self.regressor(feature_vector)
        return pose

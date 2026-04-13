import torch
import torch.nn as nn
import spconv.pytorch as spconv

class GeM_spconv(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_spconv, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x is a spconv.SparseConvTensor
        # Features: [N, C], Indices: [N, 4] (batch_idx, z, y, x)
        features = x.features
        features = torch.clamp(features, min=self.eps)
        features = features.pow(self.p)
        
        # spconv doesn't have a direct GlobalAvgPooling like ME for all cases
        # We can use scatter_mean or just manual mean per batch
        batch_idx = x.indices[:, 0]
        batch_size = x.batch_size
        
        # Manual global average pooling over batch
        # We'll use torch_scatter if available, but for style consistency we use basic torch
        pooled_feat = []
        for i in range(batch_size):
            mask = (batch_idx == i)
            if mask.any():
                pooled_feat.append(features[mask].mean(dim=0))
            else:
                pooled_feat.append(torch.zeros(features.shape[1], device=features.device))
        
        res = torch.stack(pooled_feat, dim=0) # [Batch, C]
        res = res.pow(1./self.p)
        return res

class MinkResNet_spconv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1024):
        super(MinkResNet_spconv, self).__init__()
        
        # Initial convolution
        self.conv1 = spconv.SparseConv3d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Basic Blocks
        self.block1 = self._make_layer(64, 128, stride=2)
        self.block2 = self._make_layer(128, 256, stride=2)
        self.block3 = self._make_layer(256, 512, stride=2)
        self.block4 = self._make_layer(512, out_channels, stride=2)
        
        self.gem = GeM_spconv()

    def _make_layer(self, in_planes, out_planes, stride):
        layers = []
        # Downsample conv
        layers.append(spconv.SparseConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm1d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        
        # Submanifold conv (keeps sparsity pattern)
        layers.append(spconv.SubMConv3d(out_planes, out_planes, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return spconv.SparseSequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # BatchNorm1d expects [N, C]
        x = x.replace_feature(self.relu(self.bn1(x.features)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.gem(x) # Returns [Batch, out_channels]
        return x
